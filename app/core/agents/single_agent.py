"""Monolithic baseline with an independent image-to-report inference path."""

import json
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pydantic import ValidationError

from app.core.agents.prompts import SINGLE_AGENT_SYSTEM
from app.core.config import Settings, get_settings
from app.core.image_preprocessing import preprocess_image
from app.core.kb_matching import (
    SeasonContractViolation,
    merge_retry_generation,
    reasoning_violations,
)
from app.core.llm_client import (
    LLMClientProtocol,
    StructuredGeneration,
    StructuredGenerationError,
)
from app.core.llm_client_vikey import build_llm_client
from app.core.product_repository import JsonProductRepository, ProductRepository
from app.core.product_repository_pg import build_product_repository
from app.core.tools import ProductSearchTool
from app.schemas.analysis import (
    AgentMode,
    AgentName,
    ExecutionTrace,
    FacialAgentOutput,
    FailureCategory,
    FinalAnalysisResult,
    PipelineStage,
    Saturation,
    SkinTone,
    SingleAgentOutput,
    StepStatus,
    TraceStep,
    Undertone,
    VisualAttributes,
)


class SingleAgentCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = models.efficientnet_v2_s(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(p=0.3)
        self.head_skintone = nn.Linear(in_features, 5)
        self.head_undertone = nn.Linear(in_features, 2)
        self.head_saturation = nn.Linear(in_features, 3)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(image))
        return {
            "head_skintone": self.head_skintone(features),
            "head_undertone": self.head_undertone(features),
            "head_saturation": self.head_saturation(features),
        }


SINGLE_SKINTONE_LABELS = {0: "Fair", 1: "Medium", 2: "Tanned", 3: "Deep", 4: "Dark"}
SINGLE_UNDERTONE_LABELS = {0: "Cool", 1: "Warm"}
SINGLE_SATURATION_LABELS = {0: "Low", 1: "Medium", 2: "High"}
class SingleAgent:
    def __init__(
        self,
        llm_client: LLMClientProtocol | None = None,
        settings: Settings | None = None,
        kb_rules: dict[str, Any] | None = None,
        products: ProductRepository | None = None,
    ):
        self.settings = settings or get_settings()
        self.kb_rules = kb_rules or self._load_kb_rules()
        self.products = products or build_product_repository(self.settings)
        self.device = torch.device(
            "cuda"
            if self.settings.CNN_DEVICE == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.settings.CNN_DEVICE == "auto"
            else self.settings.CNN_DEVICE
        )
        self.cnn_model: SingleAgentCNN | None = None
        if self.settings.ATTRIBUTE_SOURCE == "cnn":
            self.cnn_model = SingleAgentCNN().to(self.device)
            if not self.settings.CNN_MODEL_PATH.is_file():
                raise FileNotFoundError(
                    f"Model B tidak ditemukan di path: {self.settings.CNN_MODEL_PATH}"
                )
            self.cnn_model.load_state_dict(
                torch.load(self.settings.CNN_MODEL_PATH, map_location=self.device)
            )
            self.cnn_model.eval()
        self.llm_client = llm_client or build_llm_client(self.settings)

    def _load_kb_rules(self) -> dict[str, Any]:
        with self.settings.KB_RULES_PATH.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        return preprocess_image(image_path, self.device)

    def _analyze_image(self, image_path: str) -> FacialAgentOutput:
        if self.cnn_model is None:
            raise RuntimeError("Model CNN Single-Agent tidak diinisialisasi.")

        image_tensor = self._preprocess_image(image_path)
        with torch.no_grad():
            outputs = self.cnn_model(image_tensor)
            probabilities = {
                "skintone": F.softmax(outputs["head_skintone"], dim=1)[0].cpu().numpy(),
                "undertone": F.softmax(outputs["head_undertone"], dim=1)[0].cpu().numpy(),
                "saturation": F.softmax(outputs["head_saturation"], dim=1)[0].cpu().numpy(),
            }

        skintone_index = int(np.argmax(probabilities["skintone"]))
        undertone_index = int(np.argmax(probabilities["undertone"]))
        saturation_index = int(np.argmax(probabilities["saturation"]))
        probability_map = {
            "skintone": {
                SINGLE_SKINTONE_LABELS[index]: float(probabilities["skintone"][index])
                for index in SINGLE_SKINTONE_LABELS
            },
            "undertone": {
                SINGLE_UNDERTONE_LABELS[index]: float(probabilities["undertone"][index])
                for index in SINGLE_UNDERTONE_LABELS
            },
            "saturation": {
                SINGLE_SATURATION_LABELS[index]: float(probabilities["saturation"][index])
                for index in SINGLE_SATURATION_LABELS
            },
        }
        return FacialAgentOutput(
            attributes=VisualAttributes(
                skintone=SkinTone(SINGLE_SKINTONE_LABELS[skintone_index]),
                undertone=Undertone(SINGLE_UNDERTONE_LABELS[undertone_index]),
                saturation=Saturation(SINGLE_SATURATION_LABELS[saturation_index]),
            ),
            probabilities=probability_map,
            model_name=self.settings.CNN_MODEL_PATH.name,
            source="cnn",
        )

    @staticmethod
    def _failure_category(error: Exception) -> FailureCategory | None:
        error = getattr(error, "original_error", error)
        if isinstance(error, (json.JSONDecodeError, ValidationError)):
            return FailureCategory.JSON_PARSING
        if isinstance(error, (TimeoutError, ConnectionError)):
            return FailureCategory.API_TIMEOUT
        status = getattr(error, "status_code", None) or getattr(error, "code", None)
        if status in {408, 429, 500, 502, 503, 504}:
            return FailureCategory.API_TIMEOUT
        text = str(error).casefold()
        if any(
            word in text
            for word in (
                "timeout",
                "timed out",
                "connection",
                "429",
                "5xx",
                "service unavailable",
                "internal server error",
            )
        ):
            return FailureCategory.API_TIMEOUT
        return None

    def analyze(
        self,
        case_id: str,
        image_path: str,
        attributes_override: VisualAttributes | None = None,
    ) -> tuple[ExecutionTrace, FinalAnalysisResult | None]:
        started = time.perf_counter()
        error_message: str | None = None
        failure_category: FailureCategory | None = None
        generation: StructuredGeneration[SingleAgentOutput] | None = None
        generation_error: StructuredGenerationError | None = None
        output: SingleAgentOutput | None = None
        attributes: VisualAttributes | None = None
        probabilities: dict[str, dict[str, float]] = {}
        try:
            if self.settings.ATTRIBUTE_SOURCE == "ground_truth":
                if attributes_override is None:
                    raise ValueError(
                        "attributes_override wajib diisi untuk ATTRIBUTE_SOURCE=ground_truth."
                    )
                attributes = attributes_override
            else:
                facial_output = self._analyze_image(image_path)
                attributes = facial_output.attributes
                probabilities = facial_output.probabilities

            prompt_payload = {
                "visual_attributes": attributes.model_dump(mode="json"),
                "class_probabilities": probabilities or {},
                "knowledge_base": self.kb_rules,
                "output_schema": SingleAgentOutput.model_json_schema(),
                "instructions": (
                    "In one model response, choose as primary the only KB season whose "
                    "undertone and target_combinations match the supplied attributes, "
                    "and a different sister season from the supplied knowledge base. "
                    "Both are required; never return null for the sister season. "
                    "Search for products using the supplied tool and include the "
                    "resulting product picks in the same response. Do not calculate "
                    "or include season scores, rankings, or percentages. Return every "
                    "makeup color as #RRGGBB without brackets. Do not call or delegate "
                    "to agents."
                ),
            }
            prompt = json.dumps(prompt_payload, ensure_ascii=False)
            tools = ProductSearchTool(
                products=self.products,
                attributes=attributes,
                season_codes=list(self.kb_rules),
                top_k=self.settings.TOP_K_PRODUCTS_PER_CATEGORY,
            )
            generation = self.llm_client.generate_structured(
                system=SINGLE_AGENT_SYSTEM,
                prompt=prompt,
                schema=SingleAgentOutput,
                tools=tools,
            )
            if generation is not None:
                violations = reasoning_violations(
                    generation.value.reasoning, attributes, self.kb_rules
                )
                if violations:
                    prompt_payload["contract_violations"] = violations
                    retry_generation = self.llm_client.generate_structured(
                        system=SINGLE_AGENT_SYSTEM,
                        prompt=json.dumps(prompt_payload, ensure_ascii=False),
                        schema=SingleAgentOutput,
                        tools=tools,
                    )
                    if retry_generation is not None:
                        merge_retry_generation(generation, retry_generation)
                        violations = reasoning_violations(
                            generation.value.reasoning, attributes, self.kb_rules
                        )
                        if violations:
                            raise SeasonContractViolation(
                                "Season contract violated: " + "; ".join(violations),
                                generation,
                            )
                    else:
                        generation = None
            output = generation.value
            step = TraceStep(
                step_index=1,
                stage=PipelineStage.MONOLITHIC,
                agent=AgentName.SINGLE,
                status=StepStatus.SUCCESS,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                input_summary={
                    "image_path": image_path,
                    "attributes": attributes.model_dump(mode="json"),
                    "probabilities": probabilities or {},
                },
                output_summary={
                    "attributes": attributes.model_dump(mode="json"),
                    "season_code": output.reasoning.season_code,
                    "product_ids": [pick.product_id for pick in output.products.recommendations],
                },
                evidence=[
                    output.reasoning.season_code,
                    *[pick.product_id for pick in output.products.recommendations],
                ],
                retries=max(generation.attempts - generation.llm_rounds, 0) + getattr(
                    generation, "contract_retries", 0
                ),
                tool_calls=generation.tool_calls,
                llm_rounds=generation.llm_rounds,
                input_tokens=generation.input_tokens,
                output_tokens=generation.output_tokens,
            )
        except Exception as error:
            error_message = str(error)
            failure_category = self._failure_category(error)
            if isinstance(error, SeasonContractViolation):
                generation = error.generation
            if isinstance(error, StructuredGenerationError):
                generation_error = error
            step = TraceStep(
                step_index=1,
                stage=PipelineStage.MONOLITHIC,
                agent=AgentName.SINGLE,
                status=StepStatus.FAILED,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                input_summary={
                    "image_path": image_path,
                    "attributes": (
                        attributes.model_dump(mode="json") if attributes else {}
                    ),
                    "probabilities": probabilities or {},
                },
                output_summary={},
                error_message=error_message,
                failure_category=failure_category,
                llm_rounds=(
                    generation.llm_rounds
                    if generation
                    else generation_error.llm_rounds
                    if generation_error
                    else 0
                ),
                retries=(
                    max(generation.attempts - generation.llm_rounds, 0)
                    + getattr(generation, "contract_retries", 0)
                    if generation
                    else max(
                        generation_error.attempts - generation_error.llm_rounds, 0
                    )
                    if generation_error
                    else 0
                ),
                tool_calls=generation.tool_calls if generation else (
                    generation_error.tool_calls if generation_error else []
                ),
                input_tokens=(
                    generation.input_tokens
                    if generation
                    else generation_error.input_tokens
                    if generation_error
                    else 0
                ),
                output_tokens=(
                    generation.output_tokens
                    if generation
                    else generation_error.output_tokens
                    if generation_error
                    else 0
                ),
            )

        trace = ExecutionTrace(
            trace_id=f"trace_single_{case_id}_{time.time_ns()}",
            case_id=case_id,
            mode=AgentMode.SINGLE_AGENT,
            steps=[step],
            llm_calls=(
                1
                if generation or (generation_error and generation_error.llm_rounds)
                else 0
            ) + (
                getattr(generation, "contract_retries", 0) if generation else 0
            ),
            total_llm_rounds=(
                generation.llm_rounds
                if generation
                else generation_error.llm_rounds
                if generation_error
                else 0
            ),
            total_tool_calls=len(step.tool_calls),
            total_input_tokens=step.input_tokens,
            total_output_tokens=step.output_tokens,
            total_latency_ms=round((time.perf_counter() - started) * 1000, 2),
            final_status=step.status,
            failure_category=failure_category,
            error_message=error_message,
        )
        if generation is None:
            return trace, None
        if output is None or attributes is None:
            return trace, None

        return trace, FinalAnalysisResult(
            case_id=case_id,
            mode=AgentMode.SINGLE_AGENT,
            attributes=attributes,
            attribute_probabilities=probabilities or {},
            reasoning=output.reasoning,
            products=output.products,
            explanation=output.explanation,
        )
