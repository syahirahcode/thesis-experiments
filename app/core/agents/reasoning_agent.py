"""Multi-Agent season reasoning and report explanation generation."""

import json
from pathlib import Path
from typing import Any

from app.core.agents.prompts import REASONING_SYSTEM
from app.core.config import Settings, get_settings
from app.core.llm_client import LLMClientProtocol, StructuredGeneration
from app.core.llm_client_vikey import build_llm_client
from app.core.product_repository import JsonProductRepository, ProductRepository
from app.core.product_repository_pg import build_product_repository
from app.schemas.analysis import (
    NaturalLanguageExplanation,
    ReasoningAgentOutput,
    VisualAttributes,
)


class ReasoningAgent:
    def __init__(
        self,
        llm_client: LLMClientProtocol | None = None,
        settings: Settings | None = None,
        kb_rules: dict[str, Any] | None = None,
        products: ProductRepository | None = None,
    ):
        self.settings = settings or get_settings()
        self.kb_rules = kb_rules or self._load_kb_rules(self.settings.KB_RULES_PATH)
        self.products = products or build_product_repository(self.settings)
        self.llm_client = llm_client or build_llm_client(self.settings)

    @staticmethod
    def _load_kb_rules(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"File KB tidak ditemukan: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def analyze(
        self,
        attributes: VisualAttributes,
        probabilities: dict[str, dict[str, float]] | None = None,
    ) -> StructuredGeneration[ReasoningAgentOutput]:
        prompt = json.dumps(
            {
                "visual_attributes": attributes.model_dump(mode="json"),
                "class_probabilities": probabilities or {},
                "knowledge_base": self.kb_rules,
                "output_schema": ReasoningAgentOutput.model_json_schema(),
                "requirements": (
                    "Use your own reasoning to select two different season codes, primary "
                    "and sister, from the provided knowledge base. Both are required; never "
                    "return null for the sister season. Do not calculate or include season "
                    "scores, rankings, or percentages. Return the complete "
                    "ReasoningAgentOutput JSON. Include the exact three explanation groups "
                    "and preserve all KB-backed fields verbatim. Return every makeup color "
                    "as #RRGGBB without brackets."
                ),
            },
            ensure_ascii=False,
        )
        return self.llm_client.generate_structured(
            system=REASONING_SYSTEM,
            prompt=prompt,
            schema=ReasoningAgentOutput,
        )


def build_analysis_explanation(
    reasoning: ReasoningAgentOutput,
) -> NaturalLanguageExplanation:
    return NaturalLanguageExplanation(
        summary=reasoning.season_rationale,
        season_rationale=reasoning.season_rationale,
        style_rationale=reasoning.makeup_style_rationale,
        product_rationale="",
        evidence_refs=reasoning.kb_rules_referenced,
        groups=reasoning.explanation_groups,
    )
