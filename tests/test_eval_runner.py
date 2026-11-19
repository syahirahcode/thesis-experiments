import csv

from app.core.eval_metrics import aggregate_metrics
from app.schemas.analysis import (
    AgentMode,
    CaseMetrics,
    CaseRecord,
    ExecutionTrace,
    GroundTruthRow,
    StepStatus,
)
from experiments.run_eval_rq2_rq3 import (
    _filter_cases,
    _load_checkpoint,
    _ordered_modes,
    _paired_deltas,
    _parse_args,
    _write_checkpoint,
    _write_summary_csv,
)


def _metrics(
    case_id: str,
    mode: AgentMode,
    latency_ms: float,
    *,
    success: bool = True,
    completed_steps: int = 0,
    season_correct: bool | None = None,
) -> CaseMetrics:
    return CaseMetrics(
        case_id=case_id,
        mode=mode,
        latency_ms=latency_ms,
        execution_success=success,
        completed_steps=completed_steps,
        season_correct=season_correct,
        input_tokens=10,
        output_tokens=5,
    )


def test_runner_parses_modes_and_case_limit() -> None:
    args = _parse_args(
        ["--limit", "2", "--modes", "multi_agent", "single_agent", "--dry-run"]
    )

    assert args.limit == 2
    assert args.modes == (AgentMode.MULTI_AGENT, AgentMode.SINGLE_AGENT)
    assert args.dry_run


def test_runner_accepts_previous_case_limit_option() -> None:
    args = _parse_args(["--cases", "3", "--modes", "single_agent"])

    assert args.limit == 3
    assert args.modes == (AgentMode.SINGLE_AGENT,)


def test_runner_parses_case_id_filter() -> None:
    args = _parse_args(
        ["--case-id", "510", "515", "--modes", "multi_agent", "--dry-run"]
    )

    assert args.case_id == [510, 515]
    assert args.modes == (AgentMode.MULTI_AGENT,)


def test_case_id_filter_selects_requested_images_and_rows(tmp_path) -> None:
    images = [tmp_path / "first.jpg", tmp_path / "second.jpg", tmp_path / "third.jpg"]
    rows = [
        GroundTruthRow(
            no=1,
            file_name="first.jpg",
            skintone="Dark",
            undertone="Warm",
            saturation="High",
            season_name="Autumn Dark",
        ),
        GroundTruthRow(
            no=515,
            file_name="second.jpg",
            skintone="Dark",
            undertone="Warm",
            saturation="High",
            season_name="Autumn Dark",
        ),
        GroundTruthRow(
            no=510,
            file_name="third.jpg",
            skintone="Dark",
            undertone="Warm",
            saturation="High",
            season_name="Autumn Dark",
        ),
    ]

    selected_images, selected_rows = _filter_cases(images, rows, [510, 515])

    assert selected_images == [images[1], images[2]]
    assert selected_rows == [rows[1], rows[2]]


def test_modes_alternate_order_by_case() -> None:
    both_modes = (AgentMode.MULTI_AGENT, AgentMode.SINGLE_AGENT)

    assert _ordered_modes(0, both_modes) == (
        AgentMode.SINGLE_AGENT,
        AgentMode.MULTI_AGENT,
    )
    assert _ordered_modes(1, both_modes) == both_modes


def test_checkpoint_round_trip_uses_case_and_mode(tmp_path) -> None:
    mode = AgentMode.MULTI_AGENT
    trace = ExecutionTrace(
        trace_id="trace_case_01",
        case_id="case_01",
        mode=mode,
        final_status=StepStatus.SUCCESS,
    )
    record = CaseRecord(
        trace=trace,
        metrics=_metrics("case_01", mode, 125.0),
    )
    checkpoint_path = tmp_path / "checkpoint.json"

    _write_checkpoint(checkpoint_path, {("case_01", mode): record})
    restored = _load_checkpoint(checkpoint_path)

    assert list(restored) == [("case_01", mode)]
    assert restored[("case_01", mode)].metrics.latency_ms == 125.0


def test_case_aggregates_and_summary_include_each_case_once(tmp_path) -> None:
    mode = AgentMode.MULTI_AGENT
    cases = [
        _metrics("case_01", mode, 100.0, completed_steps=3, season_correct=True),
        _metrics("case_02", mode, 300.0, success=False, completed_steps=1),
    ]

    aggregate = aggregate_metrics(mode, cases)
    summary_path = tmp_path / "summary.csv"
    _write_summary_csv(summary_path, {mode: aggregate})

    assert aggregate.n_cases == 2
    assert aggregate.esr == 50.0
    assert aggregate.latency_mean_ms == 200.0
    assert aggregate.latency_median_ms == 200.0
    with summary_path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["mode"] == mode.value
    assert rows[0]["n_cases"] == "2"


def test_paired_delta_matches_by_case_id() -> None:
    case_id = "case_01"
    multi = {
        case_id: _metrics(
            case_id,
            AgentMode.MULTI_AGENT,
            150.0,
            completed_steps=3,
            season_correct=True,
        )
    }
    single = {
        case_id: _metrics(
            case_id,
            AgentMode.SINGLE_AGENT,
            250.0,
            completed_steps=1,
            season_correct=False,
        )
    }

    delta = _paired_deltas(multi, single)

    assert len(delta) == 1
    assert delta[0].case_id == case_id
    assert delta[0].tc == 66.67
    assert delta[0].season_accuracy == 100.0
    assert delta[0].latency_ms == -100.0
