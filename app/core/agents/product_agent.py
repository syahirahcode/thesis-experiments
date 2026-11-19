"""LLM product-selection stage with deterministic catalog search tools."""

import json

from app.core.agents.prompts import PRODUCT_RULES
from app.core.config import Settings, get_settings
from app.core.llm_client import (
    LLMClientProtocol,
    StructuredGeneration,
)
from app.core.llm_client_vikey import build_llm_client
from app.core.product_repository import JsonProductRepository, ProductRepository
from app.core.product_repository_pg import build_product_repository
from app.core.tools import ProductSearchTool
from app.schemas.analysis import (
    ProductAgentOutput,
    ReasoningAgentOutput,
    VisualAttributes,
)


class ProductAgent:
    def __init__(
        self,
        settings: Settings | None = None,
        products: ProductRepository | None = None,
        llm_client: LLMClientProtocol | None = None,
        kb_rules: dict | None = None,
    ):
        self.settings = settings or get_settings()
        self.products = products or build_product_repository(self.settings)
        if kb_rules is None:
            with self.settings.KB_RULES_PATH.open("r", encoding="utf-8") as file:
                kb_rules = json.load(file)
        self.season_codes = list(kb_rules)
        self.llm_client = llm_client or build_llm_client(self.settings)

    def recommend(
        self,
        reasoning: ReasoningAgentOutput,
        attributes: VisualAttributes,
    ) -> StructuredGeneration[ProductAgentOutput]:
        tools = ProductSearchTool(
            products=self.products,
            attributes=attributes,
            season_codes=self.season_codes,
            top_k=self.settings.TOP_K_PRODUCTS_PER_CATEGORY,
        )
        prompt = json.dumps(
            {
                "reasoning": {
                    "season_code": reasoning.season_code,
                    "season_name": reasoning.season_name,
                    "eye_hex_colors": reasoning.eye_hex_colors,
                    "blush_hex_colors": reasoning.blush_hex_colors,
                    "lip_hex_colors": reasoning.lip_hex_colors,
                },
                "user_attributes": attributes.model_dump(mode="json"),
                "output_schema": ProductAgentOutput.model_json_schema(),
                "instructions": (
                    "Call search_products for each product category, using the chosen "
                    "season_code and k exactly "
                    f"{self.settings.TOP_K_PRODUCTS_PER_CATEGORY}. Include every product "
                    "returned by each tool call exactly once in the output, with a short "
                    "Indonesian reason per product. Use only returned products. Return "
                    "ProductAgentOutput JSON; do not invent product IDs."
                ),
            },
            ensure_ascii=False,
        )
        return self.llm_client.generate_structured(
            system=PRODUCT_RULES,
            prompt=prompt,
            schema=ProductAgentOutput,
            tools=tools,
        )
