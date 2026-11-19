"""Sequential Multi-Agent coordinator; orchestration itself never calls an LLM."""

import json
import time
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.core.agents.facial_agent import FacialAgent
from app.core.agents.product_agent import ProductAgent
from app.core.agents.reasoning_agent import ReasoningAgent, build_analysis_explanation
from app.core.config import Settings, get_settings
from app.core.llm_client import StructuredGenerationError
from app.core.product_repository import JsonProductRepository, ProductRepository
from app.core.product_repository_pg import build_product_repository
from app.schemas.analysis import (
    AgentMode,
    AgentName,
    ExecutionTrace,
    FailureCategory,
    FacialAgentOutput,
    FinalAnalysisResult,
    PipelineStage,
    StepStatus,
    TraceStep,
)


def classify_runtime_error(error: Exception) -> FailureCategory | None:
    error = getattr(error, "original_error", error)
    if isinstance(error, (json.JSONDecodeError, ValidationError)):
        return FailureCategory.JSON_PARSING
    if isinstance(error, (TimeoutError, ConnectionError)):
        return FailureCategory.API_TIMEOUT
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    if status in {408, 429, 500, 502, 503, 504}:
        return FailureCategory.API_TIMEOUT
    text = str(error).casefold()
    if any(word in text for word in ("timeout", "timed out", "connection", "429", "5xx")):
        return FailureCategory.API_TIMEOUT
    return None


class OrchestratorAgent:
    def __init__(
        self,
        facial_agent: Any | None = None,
        reasoning_agent: ReasoningAgent | None = None,
        product_agent: ProductAgent | None = None,
        settings: Settings | None = None,
        products: ProductRepository | None = None,
    ):
        self.settings = settings or get_settings()
        self.facial_agent = facial_agent
        shared_products = products or build_product_repository(self.settings)
        self.reasoning_agent = reasoning_agent or ReasoningAgent(products=shared_products)
        self.product_agent = product_agent or ProductAgent(products=shared_products)

    def analyze(
        self,
        case_id: str,
        image_path: str | Path,
        facial_output: FacialAgentOutput | None = None,
        facial_latency_ms: float | None = None,
    ) -> tuple[ExecutionTrace, FinalAnalysisResult | None]:
        steps: list[TraceStep] = []
        reasoning_output = None
        product_output = None
        reasoning_generation = None
        product_generation = None
        reasoning_error: StructuredGenerationError | None = None
        product_error: StructuredGenerationError | None = None
        error_message: str | None = None
        failure_category: FailureCategory | None = None
        final_status = StepStatus.SUCCESS

        facial_started = time.perf_counter()
        try:
            if facial_output is None:
                if self.facial_agent is None:
                    self.facial_agent = FacialAgent()
                facial_output = self.facial_agent.analyze(str(image_path))
                measured_facial_latency = (time.perf_counter() - facial_started) * 1000
            else:
                measured_facial_latency = facial_latency_ms or 0.0
            steps.append(
                TraceStep(
                    step_index=1,
                    stage=PipelineStage.FACIAL_ANALYSIS,
                    agent=AgentName.FACIAL,
                    status=StepStatus.SUCCESS,
                    latency_ms=round(measured_facial_latency, 2),
                    input_summary={"image_path": str(image_path)},
                    output_summary={
                        "attributes": facial_output.attributes.model_dump(mode="json"),
                        "probabilities": facial_output.probabilities,
                    },
                    evidence=[facial_output.model_name, facial_output.source],
                )
            )
        except Exception as error:
            error_message = str(error)
            failure_category = classify_runtime_error(error)
            final_status = StepStatus.FAILED
            steps.append(
                TraceStep(
                    step_index=1,
                    stage=PipelineStage.FACIAL_ANALYSIS,
                    agent=AgentName.FACIAL,
                    status=StepStatus.FAILED,
                    latency_ms=round((time.perf_counter() - facial_started) * 1000, 2),
                    input_summary={"image_path": str(image_path)},
                    error_message=error_message,
                    failure_category=failure_category,
                )
            )

        if final_status == StepStatus.SUCCESS and facial_output is not None:
            reasoning_started = time.perf_counter()
            try:
                reasoning_generation = self.reasoning_agent.analyze(
                    facial_output.attributes, facial_output.probabilities
                )
                reasoning_output = reasoning_generation.value
                steps.append(
                    TraceStep(
                        step_index=2,
                        stage=PipelineStage.SEASON_REASONING,
                        agent=AgentName.REASONING,
                        status=StepStatus.SUCCESS,
                        latency_ms=round(reasoning_generation.latency_ms, 2),
                        input_summary={
                            "attributes": facial_output.attributes.model_dump(mode="json"),
                            "probabilities": facial_output.probabilities,
                        },
                        output_summary=reasoning_output.model_dump(mode="json"),
                        evidence=reasoning_output.kb_rules_referenced,
                        retries=max(
                            reasoning_generation.attempts - reasoning_generation.llm_rounds,
                            0,
                        ),
                        tool_calls=reasoning_generation.tool_calls,
                        llm_rounds=reasoning_generation.llm_rounds,
                        input_tokens=reasoning_generation.input_tokens,
                        output_tokens=reasoning_generation.output_tokens,
                    )
                )
            except Exception as error:
                if isinstance(error, StructuredGenerationError):
                    reasoning_error = error
                error_message = str(error)
                failure_category = classify_runtime_error(error)
                final_status = StepStatus.FAILED
                steps.append(
                    TraceStep(
                        step_index=2,
                        stage=PipelineStage.SEASON_REASONING,
                        agent=AgentName.REASONING,
                        status=StepStatus.FAILED,
                        latency_ms=round((time.perf_counter() - reasoning_started) * 1000, 2),
                        input_summary={
                            "attributes": facial_output.attributes.model_dump(mode="json")
                        },
                        error_message=error_message,
                        failure_category=failure_category,
                    )
                )

        if final_status == StepStatus.SUCCESS and reasoning_output is not None:
            product_started = time.perf_counter()
            try:
                product_generation = self.product_agent.recommend(
                    reasoning_output, facial_output.attributes
                )
                product_output = product_generation.value
                product_ids = [pick.product_id for pick in product_output.recommendations]
                steps.append(
                    TraceStep(
                        step_index=3,
                        stage=PipelineStage.PRODUCT_RECOMMENDATION,
                        agent=AgentName.PRODUCT,
                        status=StepStatus.SUCCESS,
                        latency_ms=round(product_generation.latency_ms, 2),
                        input_summary={
                            "season_code": reasoning_output.season_code,
                            "attributes": facial_output.attributes.model_dump(mode="json"),
                        },
                        output_summary={
                            "product_ids": product_ids,
                        },
                        evidence=product_ids,
                        retries=max(
                            product_generation.attempts - product_generation.llm_rounds,
                            0,
                        ),
                        tool_calls=product_generation.tool_calls,
                        llm_rounds=product_generation.llm_rounds,
                        input_tokens=product_generation.input_tokens,
                        output_tokens=product_generation.output_tokens,
                    )
                )
            except Exception as error:
                if isinstance(error, StructuredGenerationError):
                    product_error = error
                error_message = str(error)
                failure_category = classify_runtime_error(error)
                final_status = StepStatus.FAILED
                steps.append(
                    TraceStep(
                        step_index=3,
                        stage=PipelineStage.PRODUCT_RECOMMENDATION,
                        agent=AgentName.PRODUCT,
                        status=StepStatus.FAILED,
                        latency_ms=round((time.perf_counter() - product_started) * 1000, 2),
                        input_summary={"season_code": reasoning_output.season_code},
                        error_message=error_message,
                        failure_category=failure_category,
                        retries=(
                            max(product_error.attempts - product_error.llm_rounds, 0)
                            if product_error
                            else 0
                        ),
                        tool_calls=product_error.tool_calls if product_error else [],
                        llm_rounds=product_error.llm_rounds if product_error else 0,
                        input_tokens=product_error.input_tokens if product_error else 0,
                        output_tokens=product_error.output_tokens if product_error else 0,
                    )
                )

        total_latency = sum(step.latency_ms for step in steps)
        trace = ExecutionTrace(
            trace_id=f"trace_{case_id}_{time.time_ns()}",
            case_id=case_id,
            mode=AgentMode.MULTI_AGENT,
            steps=steps,
            llm_calls=int(
                reasoning_generation is not None or reasoning_error is not None
            ) + int(product_generation is not None or product_error is not None),
            total_llm_rounds=sum(step.llm_rounds for step in steps),
            total_tool_calls=sum(len(step.tool_calls) for step in steps),
            total_input_tokens=sum(step.input_tokens for step in steps),
            total_output_tokens=sum(step.output_tokens for step in steps),
            total_latency_ms=round(total_latency, 2),
            final_status=final_status,
            failure_category=failure_category,
            error_message=error_message,
        )
        if (
            final_status != StepStatus.SUCCESS
            or facial_output is None
            or reasoning_output is None
            or product_output is None
        ):
            return trace, None

        explanation = build_analysis_explanation(reasoning_output)
        result = FinalAnalysisResult(
            case_id=case_id,
            mode=AgentMode.MULTI_AGENT,
            attributes=facial_output.attributes,
            attribute_probabilities=facial_output.probabilities,
            reasoning=reasoning_output,
            products=product_output,
            explanation=explanation,
        )
        return trace, result
