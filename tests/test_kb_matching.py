import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from app.core.agents.reasoning_agent import ReasoningAgent
from app.core.agents.single_agent import SingleAgent
from app.core.agents.orchestrator import OrchestratorAgent
from app.core.eval_metrics import _rule_mismatches
from app.core.kb_matching import matching_seasons, reasoning_violations, rule_matches
from app.core.llm_client import StructuredGeneration
from app.schemas.analysis import (
    AgentName,
    FacialAgentOutput,
    NaturalLanguageExplanation,
    ProductAgentOutput,
    ReasonGroup,
    ReasoningAgentOutput,
    SeasonRule,
    SingleAgentOutput,
    StepStatus,
    VisualAttributes,
)
from app.core.config import Settings


KB_PATH = Path(__file__).resolve().parents[1] / "app" / "core" / "kb_rules.json"


def _kb_rules() -> dict:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def _attributes(undertone: str, skintone: str, saturation: str) -> VisualAttributes:
    return VisualAttributes(
        undertone=undertone,
        skintone=skintone,
        saturation=saturation,
    )


def _reasoning(
    season_code: str = "SPRING_LIGHT",
    secondary_season_code: str = "SPRING_CLEAR",
    *,
    is_ambiguous: bool = False,
) -> ReasoningAgentOutput:
    return ReasoningAgentOutput(
        season_code=season_code,
        season_name="Spring Light",
        secondary_season_code=secondary_season_code,
        makeup_look="Soft & Fresh Glowing Makeup",
        look_file_name="1_face.png",
        eye_hex_colors=["#FED2DB"],
        blush_hex_colors=["#FEA0A2"],
        lip_hex_colors=["#ED6254"],
        eye_technique="Soft shimmer",
        eye_direction="Peach",
        blush_application="Soft",
        blush_direction="Peach",
        lip_finish="Glossy",
        lip_direction="Peach",
        complexion_finish="Natural",
        kb_rules_referenced=["SPRING_LIGHT", "SPRING_CLEAR"],
        is_ambiguous=is_ambiguous,
        ambiguity_note="",
        rationale="Rasional",
        attribute_rationale="Atribut",
        season_rationale="Season",
        palette_rationale="Palet",
        avoidance_rationale="Hindari",
        makeup_style_rationale="Gaya",
        complexion_rationale="Complexion",
        eye_rationale="Eye",
        blush_rationale="Blush",
        lip_rationale="Lip",
        explanation_groups=[
            ReasonGroup(
                title="Dari atribut kulit ke season",
                items=[],
            )
        ],
    )


def test_real_kb_maps_all_30_attribute_combinations_to_one_season() -> None:
    kb = _kb_rules()

    for undertone, skintone, saturation in itertools.product(
        ("Warm", "Cool"),
        ("Fair", "Medium", "Tanned", "Deep", "Dark"),
        ("Low", "Medium", "High"),
    ):
        assert len(matching_seasons(undertone, skintone, saturation, kb)) == 1


def test_real_kb_matching_examples() -> None:
    kb = _kb_rules()
    cases = {
        ("Warm", "Tanned", "Low"): "AUTUMN_MUTED",
        ("Warm", "Tanned", "Medium"): "AUTUMN_DEEP",
        ("Warm", "Tanned", "High"): "SPRING_STRONG",
        ("Warm", "Fair", "Medium"): "SPRING_CLEAR",
        ("Cool", "Medium", "Low"): "SUMMER_MUTE",
        ("Cool", "Medium", "High"): "SUMMER_CLEAR",
        ("Cool", "Dark", "Low"): "WINTER_DARK",
    }

    for attributes, expected in cases.items():
        assert matching_seasons(*attributes, kb) == [expected]


def test_rule_matching_falls_back_to_summary_lists() -> None:
    kb = {
        "TEST": {
            "undertone": " Warm ",
            "target_combinations": [],
            "target_skintone": [" Tanned "],
            "target_saturation": ["LOW"],
        }
    }

    assert matching_seasons("warm", "tanned", "low", kb) == ["TEST"]


def test_rule_matches_accepts_season_rule_objects() -> None:
    rule = SeasonRule.model_validate(_kb_rules()["SPRING_LIGHT"])

    assert rule_matches(rule, " warm ", "FAIR", "low")


def test_reasoning_violations_checks_season_sister_and_ambiguity() -> None:
    kb = _kb_rules()
    attrs = _attributes("Warm", "Fair", "Low")

    assert reasoning_violations(_reasoning(), attrs, kb) == []

    wrong_season = _reasoning(
        season_code="SPRING_CLEAR",
        secondary_season_code="AUTUMN_MUTED",
    )
    assert any(
        "season_code" in violation
        for violation in reasoning_violations(wrong_season, attrs, kb)
    )

    same_sister = SimpleNamespace(
        season_code="SPRING_LIGHT",
        secondary_season_code="SPRING_LIGHT",
        is_ambiguous=False,
    )
    assert any(
        "secondary_season_code" in violation
        for violation in reasoning_violations(same_sister, attrs, kb)
    )

    unknown_sister = SimpleNamespace(
        season_code="SPRING_LIGHT",
        secondary_season_code="UNKNOWN",
        is_ambiguous=False,
    )
    assert "secondary_season_code tidak ada di KB" in reasoning_violations(
        unknown_sister, attrs, kb
    )

    ambiguous = _reasoning(is_ambiguous=True)
    assert "is_ambiguous harus false" in reasoning_violations(ambiguous, attrs, kb)


def test_rule_mismatches_uses_target_combinations() -> None:
    rule = SeasonRule.model_validate(_kb_rules()["SPRING_CLEAR"])
    attrs = _attributes("Warm", "Fair", "High")
    invalid_pair = _attributes("Warm", "Fair", "Low")

    assert _rule_mismatches(rule, attrs) == []
    assert _rule_mismatches(rule, invalid_pair) == [
        "pair Fair/Low not in target_combinations"
    ]


class _SequenceLLM:
    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return StructuredGeneration(
            value=self.values.pop(0),
            latency_ms=10,
            attempts=1,
            llm_rounds=1,
            input_tokens=100,
            output_tokens=20,
        )


def test_both_reasoning_modes_retry_and_count_contract_retry() -> None:
    kb = _kb_rules()
    attrs = _attributes("Warm", "Fair", "Low")
    wrong = _reasoning(
        season_code="SPRING_CLEAR",
        secondary_season_code="SPRING_LIGHT",
    )
    correct = _reasoning()

    reasoning_llm = _SequenceLLM([wrong, correct])
    generation = ReasoningAgent(
        llm_client=reasoning_llm,
        kb_rules=kb,
        products=object(),
    ).analyze(attrs)

    assert generation is not None
    assert generation.value.season_code == "SPRING_LIGHT"
    assert generation.llm_rounds == 2
    assert generation.input_tokens == 200
    assert generation.contract_retries == 1
    assert "contract_violations" in json.loads(reasoning_llm.calls[1]["prompt"])

    bad_single = SingleAgentOutput(
        reasoning=wrong,
        products=ProductAgentOutput(),
        explanation=NaturalLanguageExplanation(summary="", groups=[]),
    )
    good_single = SingleAgentOutput(
        reasoning=correct,
        products=ProductAgentOutput(),
        explanation=NaturalLanguageExplanation(summary="", groups=[]),
    )
    single_llm = _SequenceLLM([bad_single, good_single])
    single_agent = SingleAgent(
        llm_client=single_llm,
        settings=Settings(ATTRIBUTE_SOURCE="ground_truth"),
        kb_rules=kb,
        products=object(),
    )
    trace, result = single_agent.analyze(
        "case_test", "unused.png", attributes_override=attrs
    )

    assert result is not None
    assert result.reasoning.season_code == "SPRING_LIGHT"
    assert trace.llm_calls == 2
    assert trace.steps[0].retries == 1
    assert trace.steps[0].llm_rounds == 2
    assert trace.steps[0].input_tokens == 200
    assert "contract_violations" in json.loads(single_llm.calls[1]["prompt"])


def test_second_contract_violation_fails_and_keeps_retry_metrics_in_both_traces() -> None:
    kb = _kb_rules()
    attrs = _attributes("Warm", "Fair", "Low")
    wrong = _reasoning(
        season_code="SPRING_CLEAR",
        secondary_season_code="SPRING_LIGHT",
    )
    facial = FacialAgentOutput(
        attributes=attrs,
        source="ground_truth",
    )

    multi_agent = OrchestratorAgent(
        reasoning_agent=ReasoningAgent(
            llm_client=_SequenceLLM([wrong, wrong]),
            kb_rules=kb,
            products=object(),
        ),
        product_agent=object(),
        products=object(),
    )
    multi_trace, multi_result = multi_agent.analyze(
        "case_multi", "unused.png", facial_output=facial
    )

    assert multi_result is None
    assert multi_trace.final_status == StepStatus.FAILED
    assert multi_trace.error_message.startswith("Season contract violated:")
    reasoning_step = next(
        step for step in multi_trace.steps if step.agent == AgentName.REASONING
    )
    assert reasoning_step.retries == 1
    assert reasoning_step.llm_rounds == 2
    assert reasoning_step.input_tokens == 200
    assert multi_trace.llm_calls == 2

    bad_single = SingleAgentOutput(
        reasoning=wrong,
        products=ProductAgentOutput(),
        explanation=NaturalLanguageExplanation(summary="", groups=[]),
    )
    single_agent = SingleAgent(
        llm_client=_SequenceLLM([bad_single, bad_single]),
        settings=Settings(ATTRIBUTE_SOURCE="ground_truth"),
        kb_rules=kb,
        products=object(),
    )
    single_trace, single_result = single_agent.analyze(
        "case_single", "unused.png", attributes_override=attrs
    )

    assert single_result is None
    assert single_trace.final_status == StepStatus.FAILED
    assert single_trace.error_message.startswith("Season contract violated:")
    assert single_trace.steps[0].retries == 1
    assert single_trace.steps[0].llm_rounds == 2
    assert single_trace.steps[0].input_tokens == 200
    assert single_trace.llm_calls == 2
