"""Product catalog repositories used by deterministic tool implementations."""

import json
from pathlib import Path
from typing import Protocol

from app.schemas.analysis import Product, TemplateProductCategory, season_to_code


PRODUCT_CATEGORY_ALIASES: dict[str, TemplateProductCategory] = {
    "complexion": TemplateProductCategory.COMPLEXION,
    "eye": TemplateProductCategory.EYE,
    "blush": TemplateProductCategory.BLUSH,
    "lip": TemplateProductCategory.LIP,
}


class ProductRepository(Protocol):
    def list_products(self) -> list[Product]: ...

    def search_products(
        self, season_code: str, category: str, k: int, attributes: dict[str, str]
    ) -> list[Product]: ...


class JsonProductRepository:
    def __init__(self, path: Path):
        self.path = path
        if not path.is_file():
            raise FileNotFoundError(f"Katalog produk tidak ditemukan: {path}")
        with path.open("r", encoding="utf-8") as file:
            raw = json.load(file)
        records = list(raw.values()) if isinstance(raw, dict) else raw
        self.products = [Product.model_validate(item) for item in records]

    def list_products(self) -> list[Product]:
        return list(self.products)

    def search_products(
        self, season_code: str, category: str, k: int, attributes: dict[str, str]
    ) -> list[Product]:
        if k < 1:
            raise ValueError("k harus minimal 1")

        requested_season = season_to_code(season_code)
        requested_category = PRODUCT_CATEGORY_ALIASES.get(category.strip().lower())
        if requested_category is None:
            raise ValueError(f"Kategori produk tidak dikenal: {category!r}")

        matches = [
            product
            for product in self.products
            if product.target_season.strip()
            and season_to_code(product.target_season) == requested_season
            and PRODUCT_CATEGORY_ALIASES.get(product.category.strip().lower())
            == requested_category
        ]

        def rank(product: Product) -> tuple[int, str]:
            score = sum(
                (
                    product.target_undertone.strip().casefold()
                    == attributes.get("undertone", "").strip().casefold(),
                    product.target_skintone.strip().casefold()
                    == attributes.get("skintone", "").strip().casefold(),
                    product.target_saturation.strip().casefold()
                    == attributes.get("saturation", "").strip().casefold(),
                )
            )
            return (-score, product.product_id)

        return sorted(matches, key=rank)[:k]