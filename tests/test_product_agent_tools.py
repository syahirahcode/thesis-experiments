import difflib
import json
from types import SimpleNamespace

import pytest

from app.core.agents.product_agent import ProductAgent
from app.core.agents.orchestrator import OrchestratorAgent
from app.core.agents.prompts import PRODUCT_RULES, SINGLE_AGENT_SYSTEM
from app.core.agents.single_agent import SingleAgent
from app.core.config import Settings
from app.core.llm_client import (
    GoogleGenAIClient,
    StructuredGenerationError,
    _gemini_response_schema,
)
from app.core.eval_metrics import (
    build_case_metrics,
    check_grounding,
    classify_grounding_failures,
)
from app.core.product_repository import JsonProductRepository
from app.core.tools import ProductSearchTool, ToolExecutionError
from app.core.llm_client import StructuredGeneration
from app.schemas.analysis import (
    AgentMode,
    AgentName,
    ExecutionTrace,
    ExplanationItem,
    FinalAnalysisResult,
    FacialAgentOutput,
    NaturalLanguageExplanation,
    ProductAgentOutput,
    ProductCategory,
    ProductPick,
    PipelineStage,
    ReasonGroup,
    ReasoningAgentOutput,
    SeasonRule,
    SingleAgentOutput,
    ToolCallRecord,
    TraceStep,
    VisualAttributes,
    StepStatus,
)


def _reasoning() -> ReasoningAgentOutput:
    return ReasoningAgentOutput(
        season_code="SPRING_LIGHT",
        season_name="Spring Light",
        secondary_season_code="SPRING_CLEAR",
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
        is_ambiguous=False,
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
                items=[
                    ExplanationItem(label="Atribut yang dipakai", text="Fair, Warm, Low"),
                    ExplanationItem(label="Penetapan season", text="Spring Light"),
                ],
            )
        ],
    )


class FakeLLM:
    def __init__(
        self,
        *,
        season_override: str | None = None,
        hallucinate_product_id: bool = False,
    ):
        self.season_override = season_override
        self.hallucinate_product_id = hallucinate_product_id
        self.calls: list[dict] = []

    def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        tools = kwargs["tools"]
        season_code = self.season_override or "SPRING_LIGHT"
        recommendations = []
        tool_calls = []
        for category in ProductCategory:
            args = {
                "season_code": season_code,
                "category": category.value,
                "k": tools.top_k,
            }
            found, latency_ms = tools.invoke("search_products", args)
            ids = [product["product_id"] for product in found]
            tool_calls.append(
                ToolCallRecord(
                    name="search_products",
                    args=args,
                    result_summary=json.dumps(ids),
                    latency_ms=latency_ms,
                )
            )
            for item in found:
                recommendations.append(
                    ProductPick(
                        product_id=(
                            "HALLUCINATED_ID"
                            if self.hallucinate_product_id and not recommendations
                            else item["product_id"]
                        ),
                        category=category,
                        reason="Cocok dengan atribut visual.",
                        matched_criteria=["season", "undertone"],
                    )
                )

        if kwargs["schema"] is SingleAgentOutput:
            value = SingleAgentOutput(
                reasoning=_reasoning(),
                products=ProductAgentOutput(recommendations=recommendations),
                explanation=NaturalLanguageExplanation(
                    summary="Ringkasan",
                    groups=_reasoning().explanation_groups,
                ),
            )
        else:
            value = ProductAgentOutput(recommendations=recommendations)
        return StructuredGeneration(
            value=value,
            latency_ms=10.0,
            attempts=2,
            llm_rounds=2,
            input_tokens=100,
            output_tokens=40,
            tool_calls=tool_calls,
        )


def _product_agent(llm, products=None):
    settings = Settings()
    repository = products or JsonProductRepository(settings.PRODUCTS_PATH)
    kb_rules = json.loads(settings.KB_RULES_PATH.read_text(encoding="utf-8"))
    return ProductAgent(
        settings=settings,
        products=repository,
        llm_client=llm,
        kb_rules=kb_rules,
    )


def _check_product_grounding(output: ProductAgentOutput):
    settings = Settings()
    kb_json = json.loads(settings.KB_RULES_PATH.read_text(encoding="utf-8"))
    catalog = JsonProductRepository(settings.PRODUCTS_PATH)
    result = FinalAnalysisResult(
        case_id="case_test",
        mode=AgentMode.MULTI_AGENT,
        attributes=VisualAttributes(
            skintone="Fair", undertone="Warm", saturation="Low"
        ),
        reasoning=_reasoning(),
        products=output,
        explanation=NaturalLanguageExplanation(summary="", groups=[]),
    )
    return check_grounding(
        result.reasoning,
        result.attributes,
        result.products,
        {key: SeasonRule.model_validate(value) for key, value in kb_json.items()},
        {item.product_id: item for item in catalog.list_products()},
    )


def test_product_agent_uses_tool_results_for_each_category():
    llm = FakeLLM()
    settings = Settings()
    attributes = VisualAttributes(
        skintone="Fair", undertone="Warm", saturation="Low"
    )
    generation = _product_agent(llm).recommend(_reasoning(), attributes)

    assert generation.value.recommendations
    assert len(generation.tool_calls) == len(ProductCategory)
    assert all(call.name == "search_products" for call in generation.tool_calls)
    assert all(
        call.result_summary.startswith("[") for call in generation.tool_calls
    )
    returned_ids = {
        call.args["category"]: json.loads(call.result_summary)
        for call in generation.tool_calls
    }
    recommendation_counts = {
        category.value: sum(
            pick.category.value == category.value
            for pick in generation.value.recommendations
        )
        for category in ProductCategory
    }
    assert all(
        recommendation_counts[category.value] == len(returned_ids[category.value])
        for category in ProductCategory
    )
    assert all(
        {
            pick.product_id
            for pick in generation.value.recommendations
            if pick.category.value == category.value
        }
        == set(returned_ids[category.value])
        for category in ProductCategory
    )
    assert all(
        call.args["k"] == settings.TOP_K_PRODUCTS_PER_CATEGORY
        for call in generation.tool_calls
    )
    k_parameter = llm.calls[0]["tools"].declarations[0]["parameters"]["properties"]["k"]
    assert k_parameter["type"] == "INTEGER"
    assert k_parameter["minimum"] == settings.TOP_K_PRODUCTS_PER_CATEGORY
    assert k_parameter["maximum"] == settings.TOP_K_PRODUCTS_PER_CATEGORY
    assert "enum" not in k_parameter
    assert llm.calls[0]["system"] == PRODUCT_RULES
    prompt = json.loads(llm.calls[0]["prompt"])
    assert "user_attributes" in prompt
    assert f"k exactly {settings.TOP_K_PRODUCTS_PER_CATEGORY}" in prompt["instructions"]


def test_wrong_season_tool_call_is_not_corrected_and_fails_product_grounding():
    llm = FakeLLM(season_override="AUTUMN_MUTED")
    output = _product_agent(llm).recommend(
        _reasoning(),
        VisualAttributes(skintone="Fair", undertone="Warm", saturation="Low"),
    ).value
    grounding = _check_product_grounding(output)
    wrong_season_calls = [
        call
        for call in llm.calls[0]["tools"].tool_calls
        if call.args["season_code"] != _reasoning().season_code
    ]

    assert wrong_season_calls
    assert all(call.args["season_code"] == "AUTUMN_MUTED" for call in wrong_season_calls)
    assert any(
        not check.grounded and check.component_type.value == "product"
        for check in grounding.checks
    )
    assert "invalid_product_mapping" in {
        category.value for category in classify_grounding_failures(grounding)
    }

    settings = Settings()
    repository = JsonProductRepository(settings.PRODUCTS_PATH)
    final_result = FinalAnalysisResult(
        case_id="case_test",
        mode=AgentMode.MULTI_AGENT,
        attributes=VisualAttributes(
            skintone="Fair", undertone="Warm", saturation="Low"
        ),
        reasoning=_reasoning(),
        products=output,
        explanation=NaturalLanguageExplanation(summary="", groups=[]),
    )
    trace = ExecutionTrace(
        trace_id="trace_product_test",
        case_id="case_test",
        mode=AgentMode.MULTI_AGENT,
        steps=[
            TraceStep(
                step_index=3,
                stage=PipelineStage.PRODUCT_RECOMMENDATION,
                agent=AgentName.PRODUCT,
                status=StepStatus.SUCCESS,
                input_summary={"season_code": "SPRING_LIGHT"},
                output_summary={"product_ids": []},
                tool_calls=llm.calls[0]["tools"].tool_calls,
            )
        ],
    )
    kb_json = json.loads(settings.KB_RULES_PATH.read_text(encoding="utf-8"))
    metrics = build_case_metrics(
        trace,
        final_result,
        {key: SeasonRule.model_validate(value) for key, value in kb_json.items()},
        {item.product_id: item for item in repository.list_products()},
    )
    assert metrics.wrong_season_search_calls == len(wrong_season_calls)


def test_hallucinated_product_id_is_pydantic_valid_and_detected_by_grounding():
    llm = FakeLLM(hallucinate_product_id=True)
    output = _product_agent(llm).recommend(
        _reasoning(),
        VisualAttributes(skintone="Fair", undertone="Warm", saturation="Low"),
    ).value

    assert output.recommendations[0].product_id == "HALLUCINATED_ID"
    grounding = _check_product_grounding(output)
    invalid = [
        check
        for check in grounding.checks
        if check.component_type.value == "product" and not check.grounded
    ]
    assert invalid
    assert "invalid_product_mapping" in {
        category.value for category in classify_grounding_failures(grounding)
    }


def test_product_search_errors_are_logged_and_propagated():
    class FailingRepository:
        def search_products(self, season_code, category, k, attributes):
            raise RuntimeError("catalog unavailable")

    settings = Settings()
    kb_rules = json.loads(settings.KB_RULES_PATH.read_text(encoding="utf-8"))
    tool = ProductSearchTool(
        products=FailingRepository(),
        attributes=VisualAttributes(
            skintone="Fair", undertone="Warm", saturation="Low"
        ),
        season_codes=list(kb_rules),
        top_k=settings.TOP_K_PRODUCTS_PER_CATEGORY,
    )
    with pytest.raises(ToolExecutionError) as error:
        tool.invoke(
            "search_products",
            {
                "season_code": "SPRING_LIGHT",
                "category": "eye",
                "k": settings.TOP_K_PRODUCTS_PER_CATEGORY,
            },
        )

    assert error.value.tool_calls[0].name == "search_products"
    assert error.value.tool_calls[0].args["season_code"] == "SPRING_LIGHT"
    assert "ERROR: catalog unavailable" in error.value.tool_calls[0].result_summary
    assert error.value.tool_calls[0].latency_ms >= 0


def test_gemini_developer_schema_omits_unsupported_additional_properties():
    schema = ProductAgentOutput.model_json_schema()
    assert "additionalProperties" in schema["properties"]["category_rationales"]

    compatible_schema = _gemini_response_schema(ProductAgentOutput)

    def contains_additional_properties(value):
        if isinstance(value, dict):
            return "additionalProperties" in value or any(
                contains_additional_properties(nested) for nested in value.values()
            )
        if isinstance(value, list):
            return any(contains_additional_properties(nested) for nested in value)
        return False

    assert not contains_additional_properties(compatible_schema)
    assert "additionalProperties" in schema["properties"]["category_rationales"]


def test_llm_client_preserves_product_tool_error_telemetry():
    class FailingRepository:
        def search_products(self, season_code, category, k, attributes):
            raise RuntimeError("catalog unavailable")

    call = SimpleNamespace(
        name="search_products",
        args={
            "season_code": "SPRING_LIGHT",
            "category": "eye",
            "k": Settings().TOP_K_PRODUCTS_PER_CATEGORY,
        },
    )
    content = SimpleNamespace(parts=[SimpleNamespace(function_call=call)])
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=content)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=12,
            candidates_token_count=3,
        ),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    settings = Settings()
    kb_rules = json.loads(settings.KB_RULES_PATH.read_text(encoding="utf-8"))
    tools = ProductSearchTool(
        products=FailingRepository(),
        attributes=VisualAttributes(
            skintone="Fair", undertone="Warm", saturation="Low"
        ),
        season_codes=list(kb_rules),
        top_k=settings.TOP_K_PRODUCTS_PER_CATEGORY,
    )
    llm = GoogleGenAIClient(
        api_key="test",
        model="test-model",
        temperature=0,
        timeout_seconds=1,
        max_retries=0,
        max_rounds=2,
        client=client,
    )

    with pytest.raises(StructuredGenerationError) as error:
        llm.generate_structured("system", "prompt", ProductAgentOutput, tools)

    assert error.value.llm_rounds == 1
    assert error.value.input_tokens == 12
    assert error.value.output_tokens == 3
    assert error.value.tool_calls[0].args["season_code"] == "SPRING_LIGHT"
    assert "ERROR: catalog unavailable" in error.value.tool_calls[0].result_summary


def test_both_modes_use_the_same_search_tool_and_print_prompt_diff():
    settings = Settings(ATTRIBUTE_SOURCE="ground_truth")
    repository = JsonProductRepository(settings.PRODUCTS_PATH)
    kb_rules = json.loads(settings.KB_RULES_PATH.read_text(encoding="utf-8"))
    attributes = VisualAttributes(
        skintone="Fair", undertone="Warm", saturation="Low"
    )

    multi_llm = FakeLLM()
    multi_product_call = _product_agent(multi_llm, repository).recommend(
        _reasoning(), attributes
    )
    single_llm = FakeLLM()
    single = SingleAgent(
        llm_client=single_llm,
        settings=settings,
        kb_rules=kb_rules,
        products=repository,
    )
    trace, result = single.analyze(
        "case_test",
        "unused.png",
        attributes_override=attributes,
    )

    multi_tools = multi_llm.calls[0]["tools"]
    single_tools = single_llm.calls[0]["tools"]
    assert type(multi_tools) is type(single_tools) is ProductSearchTool
    assert ProductSearchTool.search_products.__qualname__ == (
        type(multi_tools).search_products.__qualname__
    )
    assert multi_tools.declarations == single_tools.declarations
    assert result is not None
    assert trace.llm_calls == 1
    assert trace.steps[0].llm_rounds == 2
    assert len(trace.steps[0].tool_calls) == len(ProductCategory)
    assert single_llm.calls[0]["system"] == SINGLE_AGENT_SYSTEM
    assert multi_llm.calls[0]["system"] == PRODUCT_RULES
    assert multi_product_call.value.recommendations

    multi_prompt = (
        multi_llm.calls[0]["system"] + "\n\n" + multi_llm.calls[0]["prompt"]
    )
    single_prompt = (
        single_llm.calls[0]["system"] + "\n\n" + single_llm.calls[0]["prompt"]
    )
    print("=== MULTI-AGENT PRODUCT PROMPT ===")
    print(multi_prompt)
    print("=== SINGLE-AGENT PROMPT ===")
    print(single_prompt)
    print("=== PROMPT DIFF ===")
    print(
        "".join(
            difflib.unified_diff(
                multi_prompt.splitlines(keepends=True),
                single_prompt.splitlines(keepends=True),
                fromfile="multi_agent_product_prompt",
                tofile="single_agent_prompt",
            )
        )
    )


def test_multi_product_stage_records_llm_and_tool_telemetry():
    class FakeReasoningAgent:
        def analyze(self, attributes, probabilities=None):
            return StructuredGeneration(
                value=_reasoning(),
                latency_ms=5,
                attempts=2,
                llm_rounds=1,
                input_tokens=80,
                output_tokens=30,
            )

    settings = Settings()
    kb_rules = json.loads(settings.KB_RULES_PATH.read_text(encoding="utf-8"))
    repository = JsonProductRepository(settings.PRODUCTS_PATH)
    product_agent = ProductAgent(
        settings=settings,
        products=repository,
        llm_client=FakeLLM(),
        kb_rules=kb_rules,
    )
    orchestrator = OrchestratorAgent(
        reasoning_agent=FakeReasoningAgent(),
        product_agent=product_agent,
        settings=settings,
        products=repository,
    )
    facial = FacialAgentOutput(
        attributes=VisualAttributes(
            skintone="Fair", undertone="Warm", saturation="Low"
        ),
        probabilities={},
    )

    trace, result = orchestrator.analyze(
        "case_test", "unused.png", facial_output=facial
    )

    assert result is not None
    assert trace.llm_calls == 2
    product_step = trace.steps[-1]
    assert product_step.stage == PipelineStage.PRODUCT_RECOMMENDATION
    assert product_step.llm_rounds == 2
    assert product_step.input_tokens == 100
    assert product_step.output_tokens == 40
    assert product_step.retries == 0
    assert len(product_step.tool_calls) == len(ProductCategory)
