"""Run the paired RQ2/RQ3 evaluation for both agent architectures."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.agents.orchestrator import OrchestratorAgent, classify_runtime_error
from app.core.agents.single_agent import SingleAgent
from app.core.config import get_settings
from app.core.eval_metrics import (
    aggregate_metrics,
    build_case_metrics,
    compare_modes,
    load_ground_truth,
    validate_experiment_design,
)
from app.core.html_generator import HTMLGenerator
from app.core.product_repository import ProductRepository
from app.core.product_repository_pg import build_product_repository
from app.schemas.analysis import (
    AgentMode,
    AggregateMetrics,
    CaseMetrics,
    CaseRecord,
    ExecutionTrace,
    FacialAgentOutput,
    GroundTruthRow,
    PairedCaseDelta,
    PipelineStage,
    RQ2TraceLog,
    SeasonRule,
    StepStatus,
    TraceStep,
)


ALL_MODES = (AgentMode.MULTI_AGENT, AgentMode.SINGLE_AGENT)


def _failed_trace(mode: AgentMode, case_id: str, error: Exception) -> ExecutionTrace:
    category = classify_runtime_error(error)
    stage = (
        PipelineStage.MONOLITHIC
        if mode == AgentMode.SINGLE_AGENT
        else PipelineStage.FACIAL_ANALYSIS
    )
    return ExecutionTrace(
        trace_id=f"trace_{mode.value}_{case_id}_{time.time_ns()}",
        case_id=case_id,
        mode=mode,
        steps=[
            TraceStep(
                step_index=1,
                stage=stage,
                agent="single_agent" if mode == AgentMode.SINGLE_AGENT else "facial_agent",
                status=StepStatus.FAILED,
                input_summary={"case_id": case_id},
                output_summary={},
                error_message=str(error),
                failure_category=category,
            )
        ],
        final_status=StepStatus.FAILED,
        failure_category=category,
        error_message=str(error),
    )


def _checkpoint_key(case_id: str, mode: AgentMode) -> tuple[str, AgentMode]:
    return case_id, mode


def _ordered_modes(case_index: int, modes: tuple[AgentMode, ...]) -> tuple[AgentMode, ...]:
    order = tuple(reversed(ALL_MODES)) if case_index % 2 == 0 else ALL_MODES
    return tuple(mode for mode in order if mode in modes)


def _load_checkpoint(path: Path) -> dict[tuple[str, AgentMode], CaseRecord]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid evaluation checkpoint format: {path}")
    raw_records = payload.get("records")
    if not isinstance(raw_records, dict):
        raise ValueError(f"Invalid evaluation checkpoint format: {path}")

    records: dict[tuple[str, AgentMode], CaseRecord] = {}
    for case_id, case_modes in raw_records.items():
        if not isinstance(case_modes, dict):
            raise ValueError(f"Invalid checkpoint entries for {case_id!r}: {path}")
        for mode_name, raw_record in case_modes.items():
            mode = AgentMode(mode_name)
            record = CaseRecord.model_validate(raw_record)
            if record.trace.case_id != case_id or record.trace.mode != mode:
                raise ValueError(f"Checkpoint key does not match its result: {case_id}/{mode}")
            records[_checkpoint_key(case_id, mode)] = record
    return records


def _write_checkpoint(
    path: Path, records: dict[tuple[str, AgentMode], CaseRecord]
) -> None:
    nested: dict[str, dict[str, dict[str, object]]] = {}
    for (case_id, mode), record in records.items():
        nested.setdefault(case_id, {})[mode.value] = record.model_dump(mode="json")
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps({"records": nested}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def _append_progress_record(path: Path, record: CaseRecord) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(record.model_dump_json() + "\n")
        file.flush()


def _write_summary_csv(
    path: Path, aggregate_by_mode: dict[AgentMode, AggregateMetrics]
) -> None:
    fields = [
        "mode",
        "n_cases",
        "esr",
        "tc",
        "kgr",
        "season_accuracy",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p95_ms",
        "latency_std_ms",
        "input_tokens",
        "output_tokens",
        "wrong_season_search_calls",
        "failure_counts",
        "grounding_failure_counts",
    ]
    rows: list[dict[str, object]] = []
    for mode in ALL_MODES:
        aggregate = aggregate_by_mode.get(mode)
        if aggregate is None:
            continue
        rows.append(
            {
                "mode": mode.value,
                "n_cases": aggregate.n_cases,
                "esr": aggregate.esr,
                "tc": aggregate.tc,
                "kgr": aggregate.kgr,
                "season_accuracy": aggregate.season_accuracy,
                "latency_mean_ms": aggregate.latency_mean_ms,
                "latency_median_ms": aggregate.latency_median_ms,
                "latency_p95_ms": aggregate.latency_p95_ms,
                "latency_std_ms": aggregate.latency_std_ms,
                "input_tokens": aggregate.input_tokens,
                "output_tokens": aggregate.output_tokens,
                "wrong_season_search_calls": aggregate.wrong_season_search_calls,
                "failure_counts": json.dumps(aggregate.failure_counts, ensure_ascii=False),
                "grounding_failure_counts": json.dumps(
                    aggregate.grounding_failure_counts, ensure_ascii=False
                ),
            }
        )
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _paired_deltas(
    multi: dict[str, CaseMetrics], single: dict[str, CaseMetrics]
) -> list[PairedCaseDelta]:
    deltas: list[PairedCaseDelta] = []
    for case_id in sorted(set(multi) & set(single)):
        multi_metrics = multi[case_id]
        single_metrics = single[case_id]
        season_delta = (
            (100.0 if multi_metrics.season_correct else 0.0)
            - (100.0 if single_metrics.season_correct else 0.0)
            if multi_metrics.season_correct is not None
            and single_metrics.season_correct is not None
            else None
        )
        deltas.append(
            PairedCaseDelta(
                case_id=case_id,
                esr=100.0
                * (multi_metrics.execution_success - single_metrics.execution_success),
                tc=multi_metrics.trace_completeness - single_metrics.trace_completeness,
                kgr=(
                    multi_metrics.kgr - single_metrics.kgr
                    if multi_metrics.grounding.total_components
                    and single_metrics.grounding.total_components
                    else None
                ),
                season_accuracy=season_delta,
                latency_ms=multi_metrics.latency_ms - single_metrics.latency_ms,
                input_tokens=multi_metrics.input_tokens - single_metrics.input_tokens,
                output_tokens=multi_metrics.output_tokens - single_metrics.output_tokens,
            )
        )
    return deltas


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the paired RQ2/RQ3 evaluation.")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--case-id",
        nargs="+",
        type=int,
        metavar="ID",
        help="Run selected cases by numeric IDs from the ground-truth CSV (for example, --case-id 510 515).",
    )
    selection.add_argument(
        "--limit",
        "--cases",
        dest="limit",
        type=int,
        metavar="N",
        help="Run the first N cases in sorted filename order (default: all).",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=[mode.value for mode in ALL_MODES],
        default=[mode.value for mode in ALL_MODES],
        help="Architectures to run (default: both).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the planned run count without calling agents.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue from outputs/rq2_checkpoint.json.",
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1.")
    if args.case_id is not None and len(args.case_id) != len(set(args.case_id)):
        parser.error("--case-id cannot contain duplicates.")
    if len(args.modes) != len(set(args.modes)):
        parser.error("--modes cannot contain duplicates.")
    args.modes = tuple(AgentMode(mode) for mode in args.modes)
    return args


def _filter_cases(
    images: list[Path],
    rows: list[GroundTruthRow],
    case_ids: list[int],
) -> tuple[list[Path], list[GroundTruthRow]]:
    available_case_ids = {row.no for row in rows}
    missing_case_ids = sorted(set(case_ids) - available_case_ids)
    if missing_case_ids:
        missing = ", ".join(str(case_id) for case_id in missing_case_ids)
        raise ValueError(f"Case ID tidak ditemukan pada data uji aktif: {missing}.")
    selected_case_ids = set(case_ids)
    selected = [
        (image, row)
        for image, row in zip(images, rows)
        if row.no in selected_case_ids
    ]
    selected_images, selected_rows = zip(*selected)
    return list(selected_images), list(selected_rows)


def _save_multi_agent_report(
    html_generator: HTMLGenerator,
    record: CaseRecord,
    image_path: Path,
    report_filename: str,
    catalog_data: list[dict] | None = None,
) -> None:
    if record.result is None or record.trace.final_status != StepStatus.SUCCESS:
        return
    report_path = html_generator.save_report(
        record.result,
        client_name=f"{record.trace.case_id} · Multi-Agent",
        input_image_path=image_path,
        report_filename=report_filename,
        catalog_data=catalog_data,
    )
    record.report_file = report_path.name


def main() -> None:
    args = _parse_args()
    settings = get_settings()
    settings.ensure_output_dirs()

    problems = settings.validate_runtime()
    if args.dry_run:
        problems = [problem for problem in problems if "GEMINI_API_KEY" not in problem]
    if problems:
        raise RuntimeError("Runtime validation failed:\n - " + "\n - ".join(problems))

    ground_truth_rows = load_ground_truth(
        settings.GROUND_TRUTH_CSV, encoding=settings.GROUND_TRUTH_ENCODING
    )
    images = settings.list_test_images()
    image_names = {path.name for path in images}
    ground_truth_by_name = {row.file_name.strip(): row for row in ground_truth_rows}
    missing_ground_truth = sorted(image_names - set(ground_truth_by_name))
    if missing_ground_truth:
        raise FileNotFoundError(
            "Ground truth untuk foto uji tidak ditemukan: "
            + ", ".join(missing_ground_truth)
        )
    rows = [ground_truth_by_name[path.name] for path in images]
    with settings.KB_RULES_PATH.open("r", encoding="utf-8") as file:
        raw_kb = json.load(file)
    kb = {
        code: SeasonRule.model_validate(
            {**rule, "season_code": rule.get("season_code", code)}
        )
        for code, rule in raw_kb.items()
    }
    repository = build_product_repository(settings)
    catalog = repository.list_products()
    categories = sorted({product.category for product in catalog})
    seasons = sorted({product.target_season for product in catalog})
    print("Nilai category katalog:", categories)
    print("Nilai target_season katalog:", seasons)
    empty_target_season = [
        product.product_id for product in catalog if not product.target_season.strip()
    ]
    if empty_target_season:
        raise ValueError(
            "Produk dengan target_season kosong: " + ", ".join(empty_target_season)
        )
    catalog_by_id = {product.product_id: product for product in catalog}

    design_problems = validate_experiment_design(
        rows,
        set(kb),
        expected_n=settings.EXPECTED_N_CASES,
    )
    if design_problems:
        raise ValueError(
            "Experiment design validation failed:\n - " + "\n - ".join(design_problems)
        )
    if args.case_id is not None:
        images, rows = _filter_cases(images, rows, args.case_id)
    if args.limit is not None:
        if args.limit > len(images):
            raise ValueError(f"--limit cannot exceed the {len(images)} available test cases.")
        images = images[: args.limit]
        rows = rows[: args.limit]

    modes = args.modes
    total_runs = len(images) * len(modes)
    print(f"Jumlah test case yang dijalankan: {len(images)}")
    print(f"Total runs: {total_runs}")

    if args.dry_run:
        print("Dry run selesai; tidak ada agent yang dipanggil.")
        return

    if args.resume:
        if not settings.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint tidak ditemukan untuk --resume: {settings.checkpoint_path}"
            )
        completed = _load_checkpoint(settings.checkpoint_path)
    else:
        completed = {}
        _write_checkpoint(settings.checkpoint_path, completed)

    gt_by_name = {row.file_name.strip(): row for row in rows}
    case_ids = {
        path.name: f"case_{gt_by_name[path.name].no:02d}"
        for path in images
    }
    active_keys = {
        _checkpoint_key(case_ids[path.name], mode)
        for path in images
        for mode in modes
    }

    multi_agent = (
        OrchestratorAgent(
            settings=settings,
            products=repository,
        )
        if AgentMode.MULTI_AGENT in modes
        else None
    )
    single_agent = (
        SingleAgent(settings=settings, kb_rules=raw_kb, products=repository)
        if AgentMode.SINGLE_AGENT in modes
        else None
    )
    html_generator = HTMLGenerator()
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns()}"
    progress_log_path = settings.OUTPUTS_DIR / f"rq2_trace_progress_{run_id}.jsonl"
    reports_written: set[str] = set()
    report_errors: dict[str, str] = {}

    images_by_case_id = {case_ids[path.name]: path for path in images}
    for (case_id, mode), record in list(completed.items()):
        if (case_id, mode) not in active_keys:
            continue
        if mode != AgentMode.MULTI_AGENT or record.result is None:
            continue
        if record.trace.final_status != StepStatus.SUCCESS:
            continue
        report_exists = record.report_file and (
            settings.reports_dir / record.report_file
        ).exists()
        if not report_exists:
            filename = f"report_{run_id}_{case_id}_multi_agent.html"
            try:
                _save_multi_agent_report(
                    html_generator,
                    record,
                    images_by_case_id[case_id],
                    filename,
                    catalog_data=[p.model_dump(mode="json") for p in catalog],
                )
                _write_checkpoint(settings.checkpoint_path, completed)
            except Exception as error:
                report_errors[case_id] = str(error)
        if record.report_file and (
            settings.reports_dir / record.report_file
        ).exists():
            reports_written.add(case_id)

    remaining_runs = total_runs - sum(key in completed for key in active_keys)
    print(f"Runs already present in checkpoint: {total_runs - remaining_runs}")
    progress = tqdm(
        total=remaining_runs,
        desc="RQ2/RQ3",
        unit="run",
        dynamic_ncols=True,
        mininterval=0.5,
    )
    try:
        for case_index, image_path in enumerate(images):
            ground_truth = gt_by_name[image_path.name]
            case_id = case_ids[image_path.name]
            for mode in _ordered_modes(case_index, modes):
                key = _checkpoint_key(case_id, mode)
                if key in completed:
                    continue
                try:
                    if mode == AgentMode.MULTI_AGENT:
                        facial_output = (
                            FacialAgentOutput(
                                attributes=ground_truth.attributes,
                                probabilities={},
                                model_name="ground_truth_ablation",
                                source="ground_truth",
                            )
                            if settings.ATTRIBUTE_SOURCE == "ground_truth"
                            else None
                        )
                        trace, result = multi_agent.analyze(
                            case_id,
                            str(image_path),
                            facial_output=facial_output,
                        )
                    else:
                        trace, result = single_agent.analyze(
                            case_id,
                            str(image_path),
                            attributes_override=(
                                ground_truth.attributes
                                if settings.ATTRIBUTE_SOURCE == "ground_truth"
                                else None
                            ),
                        )
                except Exception as error:
                    trace = _failed_trace(mode, case_id, error)
                    result = None

                metrics = build_case_metrics(
                    trace, result, kb, catalog_by_id, ground_truth
                )
                record = CaseRecord(trace=trace, metrics=metrics, result=result)
                if (
                    mode == AgentMode.MULTI_AGENT
                    and result is not None
                    and trace.final_status == StepStatus.SUCCESS
                ):
                    report_filename = f"report_{run_id}_{case_id}_multi_agent.html"
                    try:
                        _save_multi_agent_report(
                            html_generator,
                            record,
                            image_path,
                            report_filename,
                            catalog_data=[p.model_dump(mode="json") for p in catalog],
                        )
                        reports_written.add(case_id)
                        progress.write(
                            f"Report dibuat: {settings.reports_dir / report_filename}"
                        )
                    except Exception as error:
                        report_errors[case_id] = str(error)
                        progress.write(f"Gagal membuat report {report_filename}: {error}")
                if result is None or trace.final_status != StepStatus.SUCCESS:
                    progress.write(
                        f"Run gagal: {mode.value} {case_id}; "
                        f"status={trace.final_status.value}; "
                        f"error={trace.error_message or 'hasil tidak tersedia'}"
                    )

                completed[key] = record
                _append_progress_record(progress_log_path, record)
                _write_checkpoint(settings.checkpoint_path, completed)
                progress.set_postfix_str(
                    f"{mode.value} {case_id} "
                    f"{'sukses' if result is not None else 'gagal'}"
                )
                progress.update(1)
    finally:
        progress.close()

    records_by_mode = {
        mode: [
            completed[_checkpoint_key(case_ids[path.name], mode)]
            for path in images
            if _checkpoint_key(case_ids[path.name], mode) in completed
        ]
        for mode in modes
    }
    aggregate_by_mode = {
        mode: aggregate_metrics(mode, [record.metrics for record in mode_records])
        for mode, mode_records in records_by_mode.items()
    }
    comparison = (
        compare_modes(
            aggregate_by_mode[AgentMode.MULTI_AGENT],
            aggregate_by_mode[AgentMode.SINGLE_AGENT],
        )
        if all(mode in aggregate_by_mode for mode in ALL_MODES)
        else None
    )
    paired = (
        _paired_deltas(
            {
                record.metrics.case_id: record.metrics
                for record in records_by_mode[AgentMode.MULTI_AGENT]
            },
            {
                record.metrics.case_id: record.metrics
                for record in records_by_mode[AgentMode.SINGLE_AGENT]
            },
        )
        if all(mode in records_by_mode for mode in ALL_MODES)
        else []
    )
    ordered_records = [
        record
        for mode in ALL_MODES
        if mode in records_by_mode
        for record in records_by_mode[mode]
    ]
    trace_log = RQ2TraceLog(
        llm_model=settings.LLM_MODEL,
        attribute_source=";".join(
            f"{mode.value}:"
            f"{'cnn_independent' if settings.ATTRIBUTE_SOURCE == 'cnn' else 'ground_truth'}"
            for mode in ALL_MODES
            if mode in modes
        ),
        n_cases=len(rows),
        records=ordered_records,
        aggregates=[aggregate_by_mode[mode] for mode in ALL_MODES if mode in aggregate_by_mode],
        paired_case_deltas=paired,
        comparison=comparison,
    )
    settings.trace_log_path.write_text(
        trace_log.model_dump_json(indent=2), encoding="utf-8"
    )
    _write_summary_csv(settings.summary_csv_path, aggregate_by_mode)

    successful_multi_cases = {
        record.trace.case_id
        for record in records_by_mode.get(AgentMode.MULTI_AGENT, [])
        if record.result is not None and record.trace.final_status == StepStatus.SUCCESS
    }
    missing_reports = sorted(successful_multi_cases - reports_written)
    print("\n=== RQ2 ringkasan per mode ===")
    print(
        f"{'Mode':<14} {'ESR':>8} {'TC':>8} {'KGR':>8} {'SeasonAcc':>11} "
        f"{'Latency(ms)':>13} {'Tokens(in/out)':>18} {'Wrong searches':>15} {'Failures':>15}"
    )
    for mode in ALL_MODES:
        aggregate = aggregate_by_mode.get(mode)
        if aggregate is None:
            continue
        print(
            f"{mode.value:<14} {aggregate.esr:>7.2f}% {aggregate.tc:>7.2f}% "
            f"{aggregate.kgr:>7.2f}% {str(aggregate.season_accuracy):>11} "
            f"{aggregate.latency_mean_ms:>13.2f} "
            f"{aggregate.input_tokens}/{aggregate.output_tokens:>13} "
            f"{aggregate.wrong_season_search_calls:>15} "
            f"{str(aggregate.failure_counts):>15}"
        )
    print(f"\nTrace log: {settings.trace_log_path}")
    print(f"Progress trace (JSONL): {progress_log_path}")
    print(f"Summary CSV: {settings.summary_csv_path}")
    print(f"Multi-Agent reports dibuat: {len(reports_written)}")
    print(f"Folder reports: {settings.reports_dir}")
    print("Kasus sukses tanpa report:", missing_reports)
    if report_errors:
        print("Kesalahan pembuatan report:", report_errors)


if __name__ == "__main__":
    main()
