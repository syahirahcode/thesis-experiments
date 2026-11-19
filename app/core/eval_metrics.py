"""
app/core/eval_metrics.py

Logika perhitungan metrik RQ2 (Bab 3.5.2), dipisah dari runner agar modular & mudah diuji.

  ESR = test case sukses eksekusi sekuensial / total test case x 100
  TC  = tahapan terekam lengkap / total tahapan ideal x 100
  KGR = komponen rekomendasi terbukti dari KB atau Katalog / total komponen x 100

Keputusan desain:
  * ESR murni teknis/runtime (API timeout, JSON parsing error, crash). Halusinasi KB
    dan invalid product mapping BUKAN bagian ESR; itu dinilai lewat KGR dan dilaporkan
    di `grounding_failure_counts`.
  * KGR menilai kualitas konten, termasuk kesesuaian season: season ada di KB, atribut
    input memenuhi aturan season tsb (SEASON_RULE_MATCH), serta look/warna/produk.
    Season Accuracy (vs ground truth) dilaporkan sebagai metrik pendamping.
  * Kasus gagal total (tanpa output) dikeluarkan dari penyebut KGR & Season Accuracy;
    efeknya terwakili di ESR dan TC.
  * TC = Modular Trace Completeness: (sum S_i)/N, N = 3 (facial, reasoning, product);
    satu step monolitik Single-Agent tidak dihitung sebagai salah satu tahap modular.
  * Produk grounded bila ada di katalog, metadata konsisten, DAN target_season produk
    = season hasil reasoning (mismatch = Invalid Product Mapping).
  * Setiap mode menjalankan inferensi CNN sendiri untuk setiap kasus.
  * TC dan KGR diagregasi micro (jumlah / jumlah).
"""

import csv
from collections import Counter
from pathlib import Path
from statistics import mean, median, pstdev

from app.schemas.analysis import (
    REQUIRED_STAGES,
    season_to_code,
    AgentMode,
    AggregateMetrics,
    CaseMetrics,
    ComponentCheck,
    ComponentType,
    ExecutionTrace,
    FailureCategory,
    FinalAnalysisResult,
    GroundingBreakdown,
    GroundTruthRow,
    KnowledgeSource,
    MetricDelta,
    PipelineStage,
    ProductAgentOutput,
    Product,
    ReasoningAgentOutput,
    RQ2Comparison,
    SeasonRule,
    StepStatus,
    VisualAttributes,
)

# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


def load_ground_truth(
    csv_path: Path, encoding: str = "utf-8-sig", include_rejected: bool = False
) -> list[GroundTruthRow]:
    """Baca `Final Dataset - Data.csv`; baris kosong dilewati, baris Rejected? dibuang."""
    rows: list[GroundTruthRow] = []
    with open(csv_path, newline="", encoding=encoding) as f:
        for raw in csv.DictReader(f):
            raw = {(k or "").strip(): v for k, v in raw.items()}
            if not any((v or "").strip() for v in raw.values()):
                continue
            row = GroundTruthRow.model_validate(raw)
            if row.rejected and not include_rejected:
                continue
            rows.append(row)
    return rows


def validate_experiment_design(
    rows: list[GroundTruthRow],
    kb_season_codes: set[str],
    expected_n: int = 24,
    per_season: int = 2,
) -> list[str]:
    """Cek desain: 24 kasus, 2 per season, 11 season (1 F + 1 M) + 1 season (2 F)."""
    problems: list[str] = []
    if len(rows) != expected_n:
        problems.append(f"Jumlah kasus aktif = {len(rows)}, diharapkan {expected_n}")

    by_season: dict[str, list[GroundTruthRow]] = {}
    for r in rows:
        by_season.setdefault(r.season_code, []).append(r)

    unknown = sorted(set(by_season) - kb_season_codes)
    if unknown:
        problems.append(f"Season di CSV tidak ada di kb_rules.json: {unknown}")

    for code, items in sorted(by_season.items()):
        if len(items) != per_season:
            problems.append(f"{code}: {len(items)} kasus (diharapkan {per_season})")

    mixed = sum(1 for items in by_season.values() if sorted(i.gender for i in items) == ["Female", "Male"])
    two_f = sum(1 for items in by_season.values() if [i.gender for i in items] == ["Female", "Female"])
    if len(by_season) == 12 and not (mixed == 11 and two_f == 1):
        problems.append(
            f"Komposisi gender: {mixed} season (1F+1M) dan {two_f} season (2F); diharapkan 11 dan 1"
        )

    dup = [n for n, c in Counter(r.file_name for r in rows).items() if c > 1]
    if dup:
        problems.append(f"File Name duplikat: {dup}")
    return problems


# --------------------------------------------------------------------------- #
# ESR & TC
# --------------------------------------------------------------------------- #


def is_sequential_success(trace: ExecutionTrace) -> bool:
    """
    ESR (teknis): tidak ada error/crash dan urutan step sesuai desain mode.
      * Multi-Agent : facial -> reasoning -> product semuanya SUCCESS, berurutan
        (step explanation dari Orchestrator, bila ada, juga harus SUCCESS).
      * Single-Agent: tepat 1 step MONOLITHIC berstatus SUCCESS.
    """
    if trace.final_status != StepStatus.SUCCESS or trace.failure_category is not None:
        return False
    if any(s.status != StepStatus.SUCCESS for s in trace.steps):
        return False
    ordered = sorted(trace.steps, key=lambda s: s.step_index)
    if trace.mode == AgentMode.SINGLE_AGENT:
        return len(ordered) == 1 and ordered[0].stage == PipelineStage.MONOLITHIC
    seq = [s.stage for s in ordered if s.stage in REQUIRED_STAGES]
    return seq == list(REQUIRED_STAGES)


def count_complete_stages(trace: ExecutionTrace) -> int:
    """sum S_i; step monolitik tidak dihitung sebagai tahap modul manapun."""
    return sum(stage_scores(trace).values())


def stage_scores(trace: ExecutionTrace) -> dict[str, int]:
    """S_i per tahap modular: 1 bila log tahap itu lengkap & sukses, selain itu 0."""
    return {st.value: int(trace.stage_complete(st)) for st in REQUIRED_STAGES}


# --------------------------------------------------------------------------- #
# KGR
# --------------------------------------------------------------------------- #


def _hx(value: str) -> str:
    return value.strip().upper()


def _rule_mismatches(rule: SeasonRule, attrs: VisualAttributes) -> list[str]:
    """Daftar atribut input yang tidak memenuhi aturan season di KB (kosong = sesuai)."""
    low = lambda xs: {x.strip().lower() for x in xs}  # noqa: E731
    bad: list[str] = []
    if attrs.undertone.value.lower() != rule.undertone.strip().lower():
        bad.append(f"undertone {attrs.undertone.value} != {rule.undertone}")
    if attrs.skintone.value.lower() not in low(rule.target_skintone):
        bad.append(f"skintone {attrs.skintone.value} not in {rule.target_skintone}")
    if attrs.saturation.value.lower() not in low(rule.target_saturation):
        bad.append(f"saturation {attrs.saturation.value} not in {rule.target_saturation}")
    return bad


def check_grounding(
    reasoning: ReasoningAgentOutput,
    attributes: VisualAttributes,
    products: ProductAgentOutput,
    kb: dict[str, SeasonRule],
    catalog: dict[str, Product],
) -> GroundingBreakdown:
    """Periksa tiap komponen rekomendasi terhadap KB (season, look, warna) dan katalog (produk)."""
    checks: list[ComponentCheck] = []
    rule = kb.get(reasoning.season_code)

    checks.append(
        ComponentCheck(
            component_type=ComponentType.SEASON,
            value=reasoning.season_code,
            source=KnowledgeSource.KB,
            grounded=rule is not None,
            note="" if rule else "season_code tidak ada di kb_rules.json",
        )
    )
    mismatches = _rule_mismatches(rule, attributes) if rule else ["season tidak ada di KB"]
    checks.append(
        ComponentCheck(
            component_type=ComponentType.SEASON_RULE_MATCH,
            value=f"{attributes.skintone.value}/{attributes.undertone.value}/{attributes.saturation.value}"
            f" -> {reasoning.season_code}",
            source=KnowledgeSource.KB,
            grounded=not mismatches,
            note="; ".join(mismatches),
        )
    )
    look_ok = (
        rule is not None
        and reasoning.makeup_look.strip() == rule.makeup_look.strip()
        and reasoning.look_file_name.strip() == rule.look_file_name.strip()
    )
    checks.append(
        ComponentCheck(
            component_type=ComponentType.MAKEUP_LOOK,
            value=f"{reasoning.makeup_look} | {reasoning.look_file_name}",
            source=KnowledgeSource.KB,
            grounded=look_ok,
            note="" if look_ok else "makeup look / file tidak sama dengan KB season tsb",
        )
    )

    color_groups = (
        (ComponentType.EYE_COLOR, reasoning.eye_hex_colors, rule.eye_makeup if rule else None),
        (ComponentType.BLUSH_COLOR, reasoning.blush_hex_colors, rule.blush_makeup if rule else None),
        (ComponentType.LIP_COLOR, reasoning.lip_hex_colors, rule.lip_makeup if rule else None),
    )
    for ctype, hexes, guide in color_groups:
        allowed = {_hx(h) for h in guide.hex_colors} if guide else set()
        for h in hexes:
            ok = _hx(h) in allowed
            checks.append(
                ComponentCheck(
                    component_type=ctype,
                    value=h,
                    source=KnowledgeSource.KB,
                    grounded=ok,
                    note="" if ok else "HEX tidak ada di palet KB untuk season/area ini",
                )
            )

    for pick in products.recommendations:
        item = catalog.get(pick.product_id)
        note = ""
        if item is None:
            note = "product_id tidak ada di katalog"
        elif pick.product_name and pick.product_name.strip().lower() != item.product_name.strip().lower():
            note = "product_name tidak sesuai katalog"
        elif pick.category and pick.category.strip().lower() != item.category.strip().lower():
            note = "category tidak sesuai katalog"
        elif season_to_code(item.target_season) != season_to_code(reasoning.season_code):
            note = (
                f"target_season produk {item.target_season} != season hasil reasoning "
                f"{reasoning.season_code}"
            )
        checks.append(
            ComponentCheck(
                component_type=ComponentType.PRODUCT,
                value=pick.product_id,
                source=KnowledgeSource.CATALOG,
                grounded=not note,
                note=note,
            )
        )
    return GroundingBreakdown(checks=checks)


def classify_grounding_failures(breakdown: GroundingBreakdown) -> list[FailureCategory]:
    """Kategori kegagalan pasca-hoc dari komponen yang tidak terbukti."""
    cats: list[FailureCategory] = []
    bad = [c for c in breakdown.checks if not c.grounded]
    if any(c.source == KnowledgeSource.KB for c in bad):
        cats.append(FailureCategory.KB_HALLUCINATION)
    if any(c.source == KnowledgeSource.CATALOG for c in bad):
        cats.append(FailureCategory.INVALID_PRODUCT_MAPPING)
    return cats


def _wrong_season_search_calls(
    trace: ExecutionTrace, final_season_code: str | None
) -> int:
    if final_season_code is None:
        return 0
    return sum(
        call.name == "search_products"
        and call.args.get("season_code") != final_season_code
        for step in trace.steps
        for call in step.tool_calls
    )


# --------------------------------------------------------------------------- #
# Per kasus & agregat
# --------------------------------------------------------------------------- #


def build_case_metrics(
    trace: ExecutionTrace,
    result: FinalAnalysisResult | None,
    kb: dict[str, SeasonRule],
    catalog: dict[str, Product],
    ground_truth: GroundTruthRow | None = None,
) -> CaseMetrics:
    grounding = (
        check_grounding(result.reasoning, result.attributes, result.products, kb, catalog)
        if result is not None
        else GroundingBreakdown()
    )

    attrs_ok: dict[str, bool] = {}
    expected = predicted = None
    season_ok: bool | None = None
    if ground_truth is not None:
        expected = ground_truth.season_code
        if result is not None:
            predicted = result.reasoning.season_code
            season_ok = predicted == expected
            attrs_ok = {
                "skintone": result.attributes.skintone == ground_truth.skintone,
                "undertone": result.attributes.undertone == ground_truth.undertone,
                "saturation": result.attributes.saturation == ground_truth.saturation,
            }
    return CaseMetrics(
        case_id=trace.case_id,
        mode=trace.mode,
        latency_ms=round(trace.total_latency_ms, 2),
        execution_success=is_sequential_success(trace),
        expected_steps=len(REQUIRED_STAGES),
        completed_steps=count_complete_stages(trace),
        stage_scores=stage_scores(trace),
        grounding=grounding,
        failure_category=trace.failure_category,
        grounding_failures=classify_grounding_failures(grounding),
        expected_season_code=expected,
        predicted_season_code=predicted if result is not None else None,
        season_correct=season_ok,
        attributes_correct=attrs_ok,
        input_tokens=trace.total_input_tokens,
        output_tokens=trace.total_output_tokens,
        wrong_season_search_calls=_wrong_season_search_calls(
            trace, result.reasoning.season_code if result is not None else None
        ),
    )


def aggregate_metrics(mode: AgentMode, cases: list[CaseMetrics]) -> AggregateMetrics:
    n = len(cases)
    if n == 0:
        return AggregateMetrics(
            mode=mode, n_cases=0, esr=0, tc=0, kgr=0,
            n_unique_cases=0,
            latency_mean_ms=0, latency_median_ms=0, latency_p95_ms=0, latency_std_ms=0,
        )
    lat = sorted(c.latency_ms for c in cases)
    p95 = lat[min(n - 1, max(0, -(-95 * n // 100) - 1))]  # nearest-rank
    comps = sum(c.grounding.total_components for c in cases)
    grounded = sum(c.grounding.grounded_components for c in cases)
    steps_total = sum(c.expected_steps for c in cases)
    steps_done = sum(c.completed_steps for c in cases)
    scored = [c.season_correct for c in cases if c.season_correct is not None]
    return AggregateMetrics(
        mode=mode,
        n_cases=n,
        n_unique_cases=len({case.case_id for case in cases}),
        esr=round(100 * sum(c.execution_success for c in cases) / n, 2),
        tc=round(100 * steps_done / steps_total, 2) if steps_total else 0.0,
        kgr=round(100 * grounded / comps, 2) if comps else 0.0,
        latency_mean_ms=round(mean(lat), 2),
        latency_median_ms=round(median(lat), 2),
        latency_p95_ms=round(p95, 2),
        latency_std_ms=round(pstdev(lat), 2) if n > 1 else 0.0,
        failure_counts=dict(
            Counter(
                (c.failure_category.value if c.failure_category else "other_runtime_error")
                for c in cases
                if not c.execution_success
            )
        ),
        grounding_failure_counts=dict(
            Counter(f.value for c in cases for f in c.grounding_failures)
        ),
        n_graded=len(scored),
        season_accuracy=round(100 * sum(scored) / len(scored), 2) if scored else None,
        input_tokens=sum(c.input_tokens for c in cases),
        output_tokens=sum(c.output_tokens for c in cases),
        wrong_season_search_calls=sum(c.wrong_season_search_calls for c in cases),
    )


def compare_modes(multi: AggregateMetrics, single: AggregateMetrics) -> RQ2Comparison:
    return RQ2Comparison(
        multi_agent=multi,
        single_agent=single,
        delta=MetricDelta(
            esr=round(multi.esr - single.esr, 2),
            tc=round(multi.tc - single.tc, 2),
            kgr=round(multi.kgr - single.kgr, 2),
            season_accuracy=(
                round(multi.season_accuracy - single.season_accuracy, 2)
                if multi.season_accuracy is not None and single.season_accuracy is not None
                else None
            ),
            latency_mean_ms=round(multi.latency_mean_ms - single.latency_mean_ms, 2),
            input_tokens=multi.input_tokens - single.input_tokens,
            output_tokens=multi.output_tokens - single.output_tokens,
        ),
    )