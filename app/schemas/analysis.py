"""
app/schemas/analysis.py

Skema Pydantic (v2) untuk pipeline eksperimen RQ2 & RQ3 (Bab 3.5 - 3.6).

  1. Knowledge Base & Katalog   (kb_rules.json, products.json)
  2. Input & ground truth       (Final Dataset - Data.csv, 24 kasus uji)
  3. Output tiap agen           (Facial -> Reasoning -> Product -> Explanation)
  4. Execution trace log
  5. Metrik RQ2                 (Latensi, ESR, TC, KGR, kategori kegagalan)
  6. Data laporan HTML (RQ3; penilaian pakar dicatat manual, tidak ada skema di sini)

Definisi metrik (Bab 3.5.2):
  ESR = test case sukses eksekusi sekuensial / total test case x 100
        (murni teknis/runtime: API timeout, JSON error, crash)
  TC  = Modular Trace Completeness (MTC) = (sum S_i) / N x 100, N = 3 (facial, reasoning, product)
        S_i = 1 bila log input, status eksekusi, dan output modul ke-i terekam lengkap & sukses.
        Single-Agent tidak memiliki batas modul, sehingga seluruh S_i = 0.
  KGR = komponen rekomendasi terbukti dari KB atau Katalog / total komponen x 100
        (kualitas konten; produk harus ada di katalog DAN target_season-nya = season hasil
        reasoning; kasus tanpa output tidak masuk penyebut)
  Season Accuracy = season prediksi == season ground truth (metrik tambahan, kasus ber-output)
Logika perhitungannya ada di app/core/eval_metrics.py (modul ini hanya skema data).
"""

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    StringConstraints,
    computed_field,
    field_validator,
    model_validator,
)

# --------------------------------------------------------------------------- #
# Tipe dasar, helper, enum
# --------------------------------------------------------------------------- #

def _normalize_hex_color(value: Any) -> Any:
    if isinstance(value, str):
        match = re.fullmatch(r"#?\[?([0-9A-Fa-f]{6})\]?", value.strip())
        if match:
            return f"#{match.group(1)}"
    return value


HexColor = Annotated[
    str,
    BeforeValidator(_normalize_hex_color),
    StringConstraints(pattern=r"^#[0-9A-Fa-f]{6}$"),
]
LikertScore = Annotated[int, Field(ge=1, le=5)]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def season_to_code(name: str) -> str:
    """'Winter Dark' -> 'WINTER_DARK' (format key di kb_rules.json)."""
    return re.sub(r"[\s\-]+", "_", name.strip()).upper()


def _norm_label(value: Any) -> Any:
    """'warm ' -> 'Warm'; 'tan' -> 'Tanned' (alias umum)."""
    if isinstance(value, str):
        value = value.strip().title()
        return {"Tan": "Tanned"}.get(value, value)
    return value


def _none_to_empty(value: Any) -> Any:
    return "" if value is None else value


class AgentMode(str, Enum):
    MULTI_AGENT = "multi_agent"
    SINGLE_AGENT = "single_agent"


class AgentName(str, Enum):
    FACIAL = "facial_agent"
    REASONING = "reasoning_agent"
    PRODUCT = "product_agent"
    ORCHESTRATOR = "orchestrator_agent"
    SINGLE = "single_agent"


class PipelineStage(str, Enum):
    FACIAL_ANALYSIS = "facial_analysis"
    SEASON_REASONING = "season_reasoning"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    EXPLANATION = "explanation"  # disusun Orchestrator; dicatat tetapi tidak dihitung di TC
    MONOLITHIC = "monolithic_single_step"  # satu-satunya step milik Single-Agent


class ToolCallRecord(BaseModel):
    """Satu pemanggilan tool oleh LLM, termasuk ringkasan hasil dan latensi."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result_summary: str = ""
    latency_ms: float = 0.0


# Tahapan ideal per kasus -> penyebut TC (jejak antar-agen Multi-Agent)
REQUIRED_STAGES: tuple[PipelineStage, ...] = (
    PipelineStage.FACIAL_ANALYSIS,
    PipelineStage.SEASON_REASONING,
    PipelineStage.PRODUCT_RECOMMENDATION,
)


class StepStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class FailureCategory(str, Enum):
    """4 kategori kegagalan agen (Bab 3.5.2)."""

    API_TIMEOUT = "api_timeout_connection_error"
    JSON_PARSING = "json_parsing_formatting_error"
    KB_HALLUCINATION = "knowledge_base_hallucination"
    INVALID_PRODUCT_MAPPING = "invalid_product_mapping"


# Kelas atribut visual (Bab 3.3.3)
class SkinTone(str, Enum):
    FAIR = "Fair"
    MEDIUM = "Medium"
    TANNED = "Tanned"
    DEEP = "Deep"
    DARK = "Dark"


class Undertone(str, Enum):
    COOL = "Cool"
    WARM = "Warm"


class Saturation(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


SkinToneLabel = Annotated[SkinTone, BeforeValidator(_norm_label)]
UndertoneLabel = Annotated[Undertone, BeforeValidator(_norm_label)]
SaturationLabel = Annotated[Saturation, BeforeValidator(_norm_label)]


class ComponentType(str, Enum):
    """Jenis komponen rekomendasi yang dicek untuk KGR."""

    SEASON = "season"  # season_code ada di KB
    SEASON_RULE_MATCH = "season_rule_match"  # atribut input memenuhi aturan season tsb di KB
    MAKEUP_LOOK = "makeup_look"
    EYE_COLOR = "eye_color"
    BLUSH_COLOR = "blush_color"
    LIP_COLOR = "lip_color"
    PRODUCT = "product"


class ProductCategory(str, Enum):
    COMPLEXION = "complexion"
    EYE = "eye"
    BLUSH = "blush"
    LIP = "lip"


class TemplateProductCategory(str, Enum):
    COMPLEXION = "complexion"
    EYE = "eye"
    BLUSH = "blush"
    LIP = "lip"


class KnowledgeSource(str, Enum):
    KB = "knowledge_base"
    CATALOG = "product_catalog"


# --------------------------------------------------------------------------- #
# 1. Knowledge Base & Katalog Produk
# --------------------------------------------------------------------------- #


class EyeMakeupGuide(BaseModel):
    model_config = ConfigDict(extra="allow")
    technique: str = ""
    direction: str = ""
    hex_colors: list[HexColor] = Field(default_factory=list)


class BlushMakeupGuide(BaseModel):
    model_config = ConfigDict(extra="allow")
    application: str = ""
    direction: str = ""
    hex_colors: list[HexColor] = Field(default_factory=list)


class LipMakeupGuide(BaseModel):
    model_config = ConfigDict(extra="allow")
    finish: str = ""
    direction: str = ""
    hex_colors: list[HexColor] = Field(default_factory=list)


class SeasonRule(BaseModel):
    """Satu entri kb_rules.json (key = season_code, mis. SPRING_LIGHT)."""

    model_config = ConfigDict(extra="allow")

    season_code: str
    season_name: str
    undertone: str
    target_skintone: list[str]
    target_saturation: list[str]
    target_combinations: list[list[str]] = Field(default_factory=list)
    makeup_look: str
    look_file_name: str
    eye_makeup: EyeMakeupGuide
    blush_makeup: BlushMakeupGuide
    lip_makeup: LipMakeupGuide
    complexion_finish: str = ""


class Product(BaseModel):
    """Satu baris products.json."""

    model_config = ConfigDict(extra="allow")

    product_id: str
    product_name: str
    brand: str
    category: str  # complexion | eyeshadow | blush | lipstick (sesuai katalog)
    raw_category: str = ""
    shade: str = ""
    target_season: str
    target_undertone: str
    target_skintone: str
    target_saturation: str
    image_file: str = ""
    notes: str = ""
    image_url: str = ""
    url: str = ""
    link: str = ""


# --------------------------------------------------------------------------- #
# 2. Input & ground truth
# --------------------------------------------------------------------------- #


class VisualAttributes(BaseModel):
    """Tiga atribut visual hasil klasifikasi CNN."""

    skintone: SkinToneLabel
    undertone: UndertoneLabel
    saturation: SaturationLabel


class ExplanationItem(BaseModel):
    label: str
    text: str


class ReasonGroup(BaseModel):
    title: str
    items: list[ExplanationItem] = Field(default_factory=list)


class GroundTruthRow(BaseModel):
    """
    Satu baris `experiments/Final Dataset - Data.csv`.
    Header: No,Age,Age Bin,Gender,Ethnic,Link,File Name,Skin tone,Under tone,
            Skin saturation,Season,Rejected?,Reason
    Baca CSV dengan encoding 'utf-8-sig' (Google Sheets export sering membawa BOM).
    """

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    no: int = Field(alias="No")
    age: int | None = Field(default=None, alias="Age")
    age_bin: str = Field(default="", alias="Age Bin")
    gender: str = Field(default="", alias="Gender")
    ethnic: str = Field(default="", alias="Ethnic")
    link: str = Field(default="", alias="Link")
    file_name: str = Field(alias="File Name")
    skintone: SkinToneLabel = Field(alias="Skin tone")
    undertone: UndertoneLabel = Field(alias="Under tone")
    saturation: SaturationLabel = Field(alias="Skin saturation")
    season_name: str = Field(alias="Season")  # mis. "Winter Dark"
    rejected: bool = Field(default=False, alias="Rejected?")
    reason: str = Field(default="", alias="Reason")

    @field_validator("age_bin", "gender", "ethnic", "link", "reason", mode="before")
    @classmethod
    def _blank(cls, v: Any) -> Any:
        v = _none_to_empty(v)
        return v.strip() if isinstance(v, str) else v

    @field_validator("age", mode="before")
    @classmethod
    def _age(cls, v: Any) -> Any:
        return None if v is None or (isinstance(v, str) and not v.strip()) else v

    @field_validator("rejected", mode="before")
    @classmethod
    def _rejected(cls, v: Any) -> Any:
        if v is None:
            return False
        if isinstance(v, str):
            return v.strip().lower() in {"1", "y", "yes", "ya", "true", "x", "rejected", "v"}
        return v

    @field_validator("gender", mode="after")
    @classmethod
    def _gender(cls, v: str) -> str:
        return v.title()

    @computed_field
    @property
    def season_code(self) -> str:
        return season_to_code(self.season_name)

    @property
    def attributes(self) -> VisualAttributes:
        return VisualAttributes(
            skintone=self.skintone, undertone=self.undertone, saturation=self.saturation
        )


class EvalCase(BaseModel):
    """Satu kasus uji: foto + label acuan."""

    case_id: str  # mis. "case_01"
    file_name: str
    image_path: str
    ground_truth: GroundTruthRow


class AnalyzeRequest(BaseModel):
    """Payload endpoint /api/v1/analyze (juga dipakai runner)."""

    case_id: str
    image_path: str
    mode: AgentMode = AgentMode.MULTI_AGENT


# --------------------------------------------------------------------------- #
# 3. Output agen
# --------------------------------------------------------------------------- #


class FacialAgentOutput(BaseModel):
    attributes: VisualAttributes
    probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)
    model_name: str = ""
    source: Literal["cnn", "ground_truth"] = "cnn"


class ReasoningAgentOutput(BaseModel):
    season_code: str
    season_name: str
    secondary_season_code: str
    makeup_look: str
    look_file_name: str
    eye_hex_colors: list[HexColor]
    blush_hex_colors: list[HexColor]
    lip_hex_colors: list[HexColor]
    eye_technique: str
    eye_direction: str
    blush_application: str
    blush_direction: str
    lip_finish: str
    lip_direction: str
    complexion_finish: str
    kb_rules_referenced: list[str]
    is_ambiguous: bool
    ambiguity_note: str
    rationale: str
    attribute_rationale: str
    season_rationale: str
    palette_rationale: str
    avoidance_rationale: str
    makeup_style_rationale: str
    complexion_rationale: str
    eye_rationale: str
    blush_rationale: str
    lip_rationale: str
    explanation_groups: list[ReasonGroup]

    @model_validator(mode="after")
    def validate_distinct_seasons(self) -> "ReasoningAgentOutput":
        if self.secondary_season_code == self.season_code:
            raise ValueError("Primary season and sister season must be different.")
        return self


class ProductPick(BaseModel):
    product_id: str
    product_name: str = ""
    category: ProductCategory
    shade: str = ""
    reason: str = ""
    matched_criteria: list[str] = Field(default_factory=list)  # mis. ["season", "undertone"]


class ProductAgentOutput(BaseModel):
    recommendations: list[ProductPick] = Field(default_factory=list)
    category_rationales: dict[str, str] = Field(default_factory=dict)


class NaturalLanguageExplanation(BaseModel):
    summary: str
    season_rationale: str = ""
    style_rationale: str = ""
    product_rationale: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    groups: list[ReasonGroup]


class SingleAgentOutput(BaseModel):
    """Complete output generated by the independent monolithic baseline."""

    reasoning: ReasoningAgentOutput
    products: ProductAgentOutput
    explanation: NaturalLanguageExplanation


class FinalAnalysisResult(BaseModel):
    """Final System Report (Bab 3.6.1) dalam bentuk terstruktur."""

    case_id: str
    mode: AgentMode
    attributes: VisualAttributes
    attribute_probabilities: dict[str, dict[str, float]] = Field(default_factory=dict)
    reasoning: ReasoningAgentOutput
    products: ProductAgentOutput
    explanation: NaturalLanguageExplanation


# --------------------------------------------------------------------------- #
# 4. Execution trace log
# --------------------------------------------------------------------------- #


class TraceStep(BaseModel):
    step_index: int
    stage: PipelineStage
    agent: AgentName
    status: StepStatus = StepStatus.SUCCESS
    started_at: datetime = Field(default_factory=_utcnow)
    latency_ms: float = 0.0
    input_summary: dict[str, Any] = Field(default_factory=dict)
    output_summary: dict[str, Any] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)  # season_code / product_id yang dipakai
    retries: int = 0
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    llm_rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: str | None = None
    failure_category: FailureCategory | None = None

    @property
    def is_complete(self) -> bool:
        """S_i = 1 (MTC): input, status eksekusi, dan output terekam, dan status = SUCCESS."""
        return (
            self.status == StepStatus.SUCCESS
            and bool(self.input_summary)
            and bool(self.output_summary)
        )


class ExecutionTrace(BaseModel):
    trace_id: str
    case_id: str
    mode: AgentMode
    steps: list[TraceStep] = Field(default_factory=list)
    llm_calls: int = 0
    total_llm_rounds: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_latency_ms: float = 0.0
    final_status: StepStatus = StepStatus.SUCCESS
    failure_category: FailureCategory | None = None
    error_message: str | None = None

    def stage_complete(self, stage: PipelineStage) -> bool:
        return any(s.stage == stage and s.is_complete for s in self.steps)


# --------------------------------------------------------------------------- #
# 5. Metrik RQ2
# --------------------------------------------------------------------------- #


class ComponentCheck(BaseModel):
    """Satu komponen rekomendasi yang diperiksa terhadap KB / katalog (dasar KGR)."""

    component_type: ComponentType
    value: str
    source: KnowledgeSource
    grounded: bool
    note: str = ""


class GroundingBreakdown(BaseModel):
    checks: list[ComponentCheck] = Field(default_factory=list)

    @computed_field
    @property
    def total_components(self) -> int:
        return len(self.checks)

    @computed_field
    @property
    def grounded_components(self) -> int:
        return sum(c.grounded for c in self.checks)

    @computed_field
    @property
    def kgr(self) -> float:
        n = len(self.checks)
        return round(100 * self.grounded_components / n, 2) if n else 0.0


class CaseMetrics(BaseModel):
    case_id: str
    mode: AgentMode
    latency_ms: float
    execution_success: bool  # sukses eksekusi sekuensial (untuk ESR)
    expected_steps: int = len(REQUIRED_STAGES)
    completed_steps: int = 0  # sum S_i
    stage_scores: dict[str, int] = Field(default_factory=dict)  # S_i per tahap (0/1)
    grounding: GroundingBreakdown = Field(default_factory=GroundingBreakdown)
    failure_category: FailureCategory | None = None  # kegagalan eksekusi (API / JSON)
    grounding_failures: list[FailureCategory] = Field(default_factory=list)  # hallucination / mapping
    expected_season_code: str | None = None
    predicted_season_code: str | None = None
    season_correct: bool | None = None  # None bila tidak ada output (kasus gagal total)
    attributes_correct: dict[str, bool] = Field(default_factory=dict)  # vs ground truth
    input_tokens: int = 0
    output_tokens: int = 0
    wrong_season_search_calls: int = 0

    @computed_field
    @property
    def trace_completeness(self) -> float:
        return round(100 * self.completed_steps / self.expected_steps, 2) if self.expected_steps else 0.0

    @computed_field
    @property
    def kgr(self) -> float:
        return self.grounding.kgr


class AggregateMetrics(BaseModel):
    mode: AgentMode
    n_cases: int
    n_unique_cases: int = 0
    esr: float  # %
    tc: float  # %  (total tahapan lengkap / total tahapan ideal, seluruh kasus)
    kgr: float  # %  (total komponen terbukti / total komponen, seluruh kasus)
    latency_mean_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_std_ms: float
    failure_counts: dict[str, int] = Field(default_factory=dict)  # kegagalan eksekusi
    grounding_failure_counts: dict[str, int] = Field(default_factory=dict)  # hallucination / mapping
    n_graded: int = 0  # jumlah kasus ber-output yang dinilai kesesuaian season-nya
    season_accuracy: float | None = None  # % season prediksi == ground truth (kasus ber-output)
    input_tokens: int = 0
    output_tokens: int = 0
    wrong_season_search_calls: int = 0


class MetricDelta(BaseModel):
    """Multi-agent dikurangi single-agent."""

    esr: float
    tc: float
    kgr: float
    season_accuracy: float | None = None
    latency_mean_ms: float  # positif = multi lebih lambat
    input_tokens: int = 0
    output_tokens: int = 0


class RQ2Comparison(BaseModel):
    multi_agent: AggregateMetrics
    single_agent: AggregateMetrics
    delta: MetricDelta


class PairedCaseDelta(BaseModel):
    case_id: str
    esr: float
    tc: float
    kgr: float | None = None
    season_accuracy: float | None = None
    latency_ms: float
    input_tokens: int
    output_tokens: int


class CaseRecord(BaseModel):
    """Satu entri rq2_trace_logs.json."""

    trace: ExecutionTrace
    metrics: CaseMetrics
    result: FinalAnalysisResult | None = None
    report_file: str | None = None


class RQ2TraceLog(BaseModel):
    """Root dokumen outputs/rq2_trace_logs.json."""

    generated_at: datetime = Field(default_factory=_utcnow)
    llm_model: str
    attribute_source: str = "multi_agent:cnn;single_agent:cnn"
    n_cases: int
    records: list[CaseRecord] = Field(default_factory=list)
    aggregates: list[AggregateMetrics] = Field(default_factory=list)
    paired_case_deltas: list[PairedCaseDelta] = Field(default_factory=list)
    comparison: RQ2Comparison | None = None


# --------------------------------------------------------------------------- #
# 6. Data laporan HTML (RQ3)
# --------------------------------------------------------------------------- #


class ColorSwatch(BaseModel):
    """Swatch laporan; warna dikonfigurasi di backend."""

    label: str = ""
    hex: HexColor


class ReportProduct(BaseModel):
    product_id: str
    product_name: str
    brand: str
    category: str
    shade: str = ""
    image_path: str  # assets/products/<image_file> or HTTP URL
    reason: str = ""
    url: str = ""


class ReportSeason(BaseModel):
    code: str
    name: str
    color: ColorSwatch


class ReportData(BaseModel):
    """Seluruh data yang dibutuhkan html_generator untuk satu laporan."""

    case_id: str
    mode: AgentMode = AgentMode.MULTI_AGENT

    # Input & atribut
    input_image_path: str
    skintone_label: str
    skintone_confidence: float = Field(ge=0.0, le=1.0)
    undertone_label: str
    undertone_swatch: ColorSwatch
    undertone_confidence: float = Field(ge=0.0, le=1.0)
    saturation_level: str
    saturation_image_path: str  # assets/saturations/<level>_saturation.jpg
    saturation_confidence: float = Field(ge=0.0, le=1.0)

    # Season swatch tampilan dikonfigurasi di backend; bukan bagian dari KB.
    season_primary: ReportSeason
    season_secondary: ReportSeason

    # Look & warna makeup (kb_rules.json)
    makeup_look: str
    look_image_path: str  # assets/looks/<look_file_name>
    eye_colors: list[ColorSwatch] = Field(default_factory=list)
    blush_colors: list[ColorSwatch] = Field(default_factory=list)
    lip_colors: list[ColorSwatch] = Field(default_factory=list)
    eye_technique: str = ""
    eye_direction: str = ""
    blush_application: str = ""
    blush_direction: str = ""
    lip_finish: str = ""
    lip_direction: str = ""
    complexion_finish: str = ""

    # Produk & penjelasan
    products: list[ReportProduct] = Field(default_factory=list)
    explanation: NaturalLanguageExplanation

    @property
    def filename(self) -> str:
        return f"report_{self.case_id}.html"


# --------------------------------------------------------------------------- #
# 7. Skema API Analyze & Report (FastAPI)
# --------------------------------------------------------------------------- #

AnalysisRequest = AnalyzeRequest


class ExpertVerificationRequest(BaseModel):
    """Payload saat pakar melakukan verifikasi laporan analisis."""

    expert_name: str
    status: str = "verified"  # "verified", "approved", "revision_needed"
    notes: str = ""
    adjustments: dict[str, Any] = Field(default_factory=dict)


class ExpertVerificationRecord(BaseModel):
    """Data status verifikasi pakar yang tersimpan."""

    is_verified: bool
    status: str
    expert_name: str
    notes: str = ""
    adjustments: dict[str, Any] = Field(default_factory=dict)
    verified_at: str


class AnalysisResponse(BaseModel):
    """Response endpoint /api/v1/analyze dan /api/v1/reports/{report_id}."""

    report_id: str
    case_id: str
    status: str  # "success" | "failed"
    mode: AgentMode = AgentMode.MULTI_AGENT
    created_at: str
    data: dict[str, Any] | None = None
    report_data: ReportData | None = None
    photo_url: str | None = None
    expert_verification: ExpertVerificationRecord | dict[str, Any] | None = None
    error_message: str | None = None


class ReportSummary(BaseModel):
    """Ringkasan laporan untuk daftar riwayat analisis (My Report)."""

    report_id: str
    case_id: str
    client_name: str | None = None
    created_at: str
    season_name: str | None = None
    skintone: str | None = None
    undertone: str | None = None
    is_verified: bool = False
    photo_url: str | None = None