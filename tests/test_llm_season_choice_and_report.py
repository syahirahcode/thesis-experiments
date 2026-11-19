import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.agents.prompts import (
    EXPLANATION_RULES,
    PRODUCT_RULES,
    REASONING_RULES,
    REASONING_SYSTEM,
    SINGLE_AGENT_SYSTEM,
)
from app.core.agents.reasoning_agent import ReasoningAgent
from app.core.html_generator import HTMLGenerator
from app.core.tools import AnalysisTools
from app.schemas.analysis import (
    AgentMode,
    ColorSwatch,
    NaturalLanguageExplanation,
    ReasoningAgentOutput,
    ReportData,
    ReportSeason,
    VisualAttributes,
)


class CapturingLLM:
    def __init__(self) -> None:
        self.call = None

    def generate_structured(self, **kwargs):
        self.call = kwargs
        return None


def test_reasoning_agent_asks_llm_to_choose_seasons_without_score_tool() -> None:
    client = CapturingLLM()
    agent = ReasoningAgent(
        llm_client=client,
        kb_rules={"SPRING_LIGHT": {"season_name": "Spring Light"}},
        products=object(),
    )

    agent.analyze(
        VisualAttributes(skintone="Fair", undertone="Warm", saturation="Low")
    )

    prompt = json.loads(client.call["prompt"])
    assert "knowledge_base" in prompt
    assert client.call.get("tools") is None
    assert client.call["system"] == REASONING_SYSTEM
    assert PRODUCT_RULES not in client.call["system"]
    assert "score_seasons" not in prompt["requirements"]
    assert "season_scores" not in ReasoningAgentOutput.model_fields
    assert "match_percentage" not in ReasoningAgentOutput.model_fields
    assert "secondary_match_percentage" not in ReasoningAgentOutput.model_fields
    assert "never return null for the sister season" in prompt["requirements"]


def test_hex_color_repairs_gemini_bracket_and_rejects_other_invalid_values() -> None:
    color = ColorSwatch(label="Eye", hex="#[FED2DB")

    assert color.hex == "#FED2DB"
    with pytest.raises(ValidationError):
        ColorSwatch(label="Eye", hex="#[FED2DG")


def test_reasoning_schema_requires_a_distinct_sister_season() -> None:
    schema = ReasoningAgentOutput.model_json_schema()

    assert "secondary_season_code" in schema["required"]
    assert "distinct sister" in REASONING_SYSTEM
    assert all(
        rules in SINGLE_AGENT_SYSTEM
        for rules in (REASONING_RULES, PRODUCT_RULES, EXPLANATION_RULES)
    )
    assert PRODUCT_RULES not in REASONING_SYSTEM


def test_analysis_tools_do_not_offer_calculated_season_scores() -> None:
    tools = AnalysisTools(
        kb_rules={"SPRING_LIGHT": {}},
        products=object(),
        attributes=VisualAttributes(
            skintone="Fair", undertone="Warm", saturation="Low"
        ),
        top_k=6,
    )

    assert "score_seasons" not in tools.functions
    assert {item["name"] for item in tools.declarations} == {
        "get_season_rules",
        "search_products",
    }


def test_report_payload_uses_llm_sister_season_and_exposes_palette_hex() -> None:
    report = ReportData(
        case_id="case_test",
        mode=AgentMode.MULTI_AGENT,
        input_image_path="photo.png",
        skintone_label="Fair",
        skintone_confidence=0.9,
        undertone_label="Warm",
        undertone_swatch=ColorSwatch(label="Warm", hex="#D9A066"),
        undertone_confidence=0.8,
        saturation_level="Low",
        saturation_image_path="saturation.png",
        saturation_confidence=0.85,
        season_primary=ReportSeason(
            code="SPRING_LIGHT",
            name="Spring Light",
            color=ColorSwatch(label="Spring Light", hex="#FFD9B0"),
        ),
        season_secondary=ReportSeason(
            code="SPRING_CLEAR",
            name="Spring Clear",
            color=ColorSwatch(label="Spring Clear", hex="#FF8A65"),
        ),
        makeup_look="Fresh",
        look_image_path="look.png",
        eye_colors=[ColorSwatch(label="Eye 1", hex="#FED2DB")],
        blush_colors=[ColorSwatch(label="Blush 1", hex="#FEA0A2")],
        lip_colors=[ColorSwatch(label="Lip 1", hex="#ED6254")],
        explanation=NaturalLanguageExplanation(summary="", groups=[]),
    )

    payload = HTMLGenerator()._payload(report, "25 October 2026", False, None)

    assert payload["season"]["primary"]["name"] == "Spring Light"
    assert payload["season"]["secondary"]["name"] == "Spring Clear"
    assert "pct" not in payload["season"]["primary"]
    assert "pct" not in payload["season"]["secondary"]
    assert payload["colors"]["eye"][0]["hex"] == "#FED2DB"
    assert payload["colors"]["blush"][0]["hex"] == "#FEA0A2"
    assert payload["colors"]["lip"][0]["hex"] == "#ED6254"

    template = (
        Path(__file__).resolve().parents[1] / "makeup-report-template.html"
    ).read_text(encoding="utf-8")
    assert 'class:"season-primary"' in template
    assert 'class:"season-secondary"' in template
    assert ".season-primary{font-size:1.5rem;font-weight:800" in template
    assert ".season-secondary{font-size:.95rem" in template
    assert '$("code",{},c.hex)' in template
    assert "d.season.primary.pct" not in template
    assert "d.season.secondary.pct" not in template
