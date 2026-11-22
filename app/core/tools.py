"""Shared function tools used by the LLM stages in both architectures."""

import json
import time
from typing import Any, Callable

from app.core.product_repository import ProductRepository
from app.schemas.analysis import ProductCategory, ToolCallRecord, VisualAttributes

PRODUCT_TOOL_FIELDS = {
    "product_id",
    "product_name",
    "brand",
    "category",
    "shade",
    "target_season",
    "target_undertone",
    "target_skintone",
    "target_saturation",
}


class ToolExecutionError(RuntimeError):
    def __init__(self, error: Exception, tool_calls: list[ToolCallRecord]):
        super().__init__(str(error))
        self.original_error = error
        self.tool_calls = tool_calls


class ProductSearchTool:
    """One catalog-search implementation with user attributes bound server-side."""

    def __init__(
        self,
        products: ProductRepository,
        attributes: VisualAttributes,
        season_codes: list[str],
        top_k: int,
    ):
        self.products = products
        self.attributes = attributes
        self.season_codes = season_codes
        self.top_k = top_k
        self.tool_calls: list[ToolCallRecord] = []
        self.searched_categories: set[str] = set()

    def search_products(self, season_code: str, category: str, k: int) -> list[dict[str, Any]]:
        if k != self.top_k:
            raise ValueError(f"k harus sama dengan TOP_K_PRODUCTS_PER_CATEGORY={self.top_k}")
        found = self.products.search_products(
            season_code=season_code,
            category=category,
            k=k,
            attributes={
                "undertone": self.attributes.undertone.value,
                "skintone": self.attributes.skintone.value,
                "saturation": self.attributes.saturation.value,
            },
        )
        self.searched_categories.add(category)
        return [
            product.model_dump(mode="json", include=PRODUCT_TOOL_FIELDS)
            for product in found
        ]

    @property
    def declarations(self) -> list[dict[str, Any]]:
        remaining_categories = [
            category.value
            for category in ProductCategory
            if category.value not in self.searched_categories
        ]
        if not remaining_categories:
            return []
        return [
            {
                "name": "search_products",
                "description": (
                    "Search catalog products for exactly the requested season code and "
                    "category, ranked using the bound user attributes."
                ),
                "parameters": {
                    "type": "OBJECT",
                    "properties": {
                        "season_code": {"type": "STRING", "enum": self.season_codes},
                        "category": {
                            "type": "STRING",
                            "enum": remaining_categories,
                        },
                        "k": {
                            "type": "INTEGER",
                            "description": (
                                "Must equal TOP_K_PRODUCTS_PER_CATEGORY="
                                f"{self.top_k}; include all returned products."
                            ),
                            "minimum": self.top_k,
                            "maximum": self.top_k,
                        },
                    },
                    "required": ["season_code", "category", "k"],
                },
            }
        ]

    def invoke(self, name: str, args: dict[str, Any]) -> tuple[Any, float]:
        if name != "search_products":
            raise ValueError(f"Tool tidak dikenal: {name}")
        started = time.perf_counter()
        try:
            result = self.search_products(**args)
        except Exception as error:
            if isinstance(error, ToolExecutionError):
                raise
            latency_ms = (time.perf_counter() - started) * 1000
            failed_call = ToolCallRecord(
                name=name,
                args=args,
                result_summary=f"ERROR: {error}",
                latency_ms=round(latency_ms, 3),
            )
            self.tool_calls.append(failed_call)
            raise ToolExecutionError(error, list(self.tool_calls)) from error
        latency_ms = (time.perf_counter() - started) * 1000
        self.tool_calls.append(
            ToolCallRecord(
                name=name,
                args=args,
                result_summary=json.dumps(
                    [product["product_id"] for product in result], ensure_ascii=False
                ),
                latency_ms=round(latency_ms, 3),
            )
        )
        return result, latency_ms


class AnalysisTools:
    """Compatibility adapter for KB lookup and the shared product search tool."""

    def __init__(
        self,
        kb_rules: dict[str, Any],
        products: ProductRepository,
        attributes: VisualAttributes,
        top_k: int,
        enabled_tools: set[str] | None = None,
    ):
        self.kb_rules = kb_rules
        self.product_search = ProductSearchTool(
            products, attributes, list(kb_rules), top_k
        )
        self.enabled_tools = (
            {"get_season_rules", "search_products"}
            if enabled_tools is None
            else enabled_tools
        )

    def get_season_rules(self, season_code: str | None = None) -> dict[str, Any]:
        if season_code is None:
            return self.kb_rules
        if season_code not in self.kb_rules:
            return {}
        return {season_code: self.kb_rules[season_code]}

    def search_products(
        self, season_code: str, category: str, k: int
    ) -> list[dict[str, Any]]:
        return self.product_search.search_products(season_code, category, k)

    @property
    def functions(self) -> dict[str, Callable[..., Any]]:
        return {
            "get_season_rules": self.get_season_rules,
            "search_products": self.search_products,
        }

    @property
    def declarations(self) -> list[dict[str, Any]]:
        season_declaration = {
            "name": "get_season_rules",
            "description": "Return KB season rules, optionally restricted to one season code.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "season_code": {
                        "type": "STRING",
                        "enum": list(self.kb_rules),
                    },
                },
            },
        }
        declarations = [season_declaration, *self.product_search.declarations]
        return [
            declaration for declaration in declarations
            if declaration["name"] in self.enabled_tools
        ]

    def invoke(self, name: str, args: dict[str, Any]) -> tuple[Any, float]:
        if name == "search_products" and name in self.enabled_tools:
            return self.product_search.invoke(name, args)
        function = self.functions.get(name)
        if function is None or name not in self.enabled_tools:
            raise ValueError(f"Tool tidak dikenal: {name}")
        started = time.perf_counter()
        result = function(**args)
        return result, (time.perf_counter() - started) * 1000
