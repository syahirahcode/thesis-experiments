"""Portable standalone HTML reports rendered from the supplied makeup template."""

import json
import os
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

from app.core.config import get_settings
from app.schemas.analysis import (
    ColorSwatch,
    FinalAnalysisResult,
    ReportData,
    ReportProduct,
    ReportSeason,
)


class HTMLGenerator:
    def __init__(self):
        self.settings = get_settings()
        self.template_path = self.settings.project_root / "makeup-report-template.html"

    def _image_reference(self, path: Path | str) -> str:
        if isinstance(path, str) and path.startswith(("http://", "https://")):
            return path
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Aset gambar report tidak ditemukan: {path}")
        relative_path = os.path.relpath(
            path.resolve(), start=self.settings.reports_dir.resolve()
        )
        return quote(Path(relative_path).as_posix(), safe="/")

    @staticmethod
    def _confidence(result: FinalAnalysisResult, attribute: str) -> float:
        label = getattr(result.attributes, attribute).value
        probabilities = result.attribute_probabilities.get(attribute, {})
        return float(probabilities.get(label, 0.0))

    def _report_data(
        self,
        result: FinalAnalysisResult,
        input_image_path: str | Path | None,
        kb_rules: dict[str, Any],
        catalog: list[dict[str, Any]],
    ) -> ReportData:
        reasoning = result.reasoning
        season_code = reasoning.season_code
        rule = kb_rules.get(season_code)
        if rule is None:
            raise ValueError(f"Season {season_code!r} tidak ditemukan di KB untuk laporan.")

        if input_image_path is None:
            raise ValueError("Path foto input wajib diberikan untuk membuat report.")
        image_path = Path(input_image_path)
        if not image_path.is_absolute():
            image_path = self.settings.project_root / image_path
        photo_data = self._image_reference(image_path)

        primary_season = ReportSeason(
            code=season_code,
            name=rule["season_name"],
            color=ColorSwatch(
                label=rule["season_name"],
                hex=self.settings.season_display_hex(season_code),
            ),
        )
        secondary_code = reasoning.secondary_season_code
        if secondary_code == season_code:
            raise ValueError("Primary season and sister season must be different.")
        secondary_rule = kb_rules.get(secondary_code)
        if secondary_rule is None:
            raise ValueError(
                f"Sister season {secondary_code!r} tidak ditemukan di KB untuk laporan."
            )
        secondary_season = ReportSeason(
            code=secondary_code,
            name=secondary_rule["season_name"],
            color=ColorSwatch(
                label=secondary_rule["season_name"],
                hex=self.settings.season_display_hex(secondary_code),
            ),
        )

        catalog_by_id = {str(item.get("product_id", "")): item for item in catalog}
        report_products: list[ReportProduct] = []
        for pick in result.products.recommendations:
            item = catalog_by_id.get(pick.product_id)
            if item is None:
                raise ValueError(
                    f"Produk {pick.product_id!r} tidak ditemukan di katalog untuk report."
                )
            product_url = str(
                item.get("image_url")
                or item.get("url")
                or item.get("link")
                or ""
            ).strip()
            image_val = str(item.get("image_file") or "").strip()

            if image_val.startswith(("http://", "https://")):
                image_path = image_val
            elif image_val:
                local_path = self.settings.product_image_path(image_val)
                if local_path.is_file():
                    image_path = self._image_reference(local_path)
                elif product_url.startswith(("http://", "https://")):
                    image_path = product_url
                else:
                    image_path = ""
            elif product_url.startswith(("http://", "https://")):
                image_path = product_url
            else:
                image_path = ""

            report_products.append(
                ReportProduct(
                    product_id=pick.product_id,
                    product_name=pick.product_name or str(item.get("product_name", "")),
                    brand=str(item.get("brand", "")),
                    category=pick.category,
                    shade=pick.shade or str(item.get("shade", "")),
                    image_path=image_path,
                    url=product_url,
                    reason=pick.reason,
                )
            )

        def makeup_swatches(category: str, hexes: list[str]) -> list[ColorSwatch]:
            return [
                ColorSwatch(label=f"{category.title()} {index}", hex=value)
                for index, value in enumerate(hexes, start=1)
            ]

        return ReportData(
            case_id=result.case_id,
            mode=result.mode,
            input_image_path=photo_data,
            skintone_label=result.attributes.skintone.value,
            skintone_confidence=self._confidence(result, "skintone"),
            undertone_label=result.attributes.undertone.value,
            undertone_swatch=ColorSwatch(
                label=result.attributes.undertone.value,
                hex=self.settings.undertone_display_hex(result.attributes.undertone.value),
            ),
            undertone_confidence=self._confidence(result, "undertone"),
            saturation_level=result.attributes.saturation.value,
            saturation_image_path=self._image_reference(
                self.settings.saturation_image_path(result.attributes.saturation.value)
            ),
            saturation_confidence=self._confidence(result, "saturation"),
            season_primary=primary_season,
            season_secondary=secondary_season,
            makeup_look=rule["makeup_look"],
            look_image_path=self._image_reference(
                self.settings.look_image_path(rule["look_file_name"])
            ),
            eye_colors=makeup_swatches("Eye", reasoning.eye_hex_colors),
            blush_colors=makeup_swatches("Blush", reasoning.blush_hex_colors),
            lip_colors=makeup_swatches("Lip", reasoning.lip_hex_colors),
            eye_technique=rule["eye_makeup"]["technique"],
            eye_direction=rule["eye_makeup"]["direction"],
            blush_application=rule["blush_makeup"]["application"],
            blush_direction=rule["blush_makeup"]["direction"],
            lip_finish=rule["lip_makeup"]["finish"],
            lip_direction=rule["lip_makeup"]["direction"],
            complexion_finish=rule["complexion_finish"],
            products=report_products,
            explanation=result.explanation,
        )

    def _payload(
        self,
        report: ReportData,
        analysis_date: str,
        ambiguity: bool,
        client_name: str | None,
    ) -> dict[str, Any]:
        products: dict[str, list[dict[str, str]]] = {
            "complexion": [],
            "lip": [],
            "eye": [],
            "blush": [],
        }
        for product in report.products:
            if product.category in products:
                products[product.category].append(
                    {
                        "brand": product.brand,
                        "name": product.product_name,
                        "shade": product.shade,
                        "hex": "#EADFE0",
                        "image": product.image_path,
                        "url": getattr(product, "url", "") or "",
                    }
                )

        return {
            "client": {"name": client_name or report.case_id, "date": analysis_date},
            "photo": report.input_image_path,
            "saturation": {
                "label": report.saturation_level,
                "confidence": report.saturation_confidence,
                "sample": report.saturation_image_path,
            },
            "skinTone": {
                "label": report.skintone_label,
                "confidence": report.skintone_confidence,
                "swatches": [
                    {"label": label, "colors": list(colors)}
                    for label, colors in self.settings.skintone_display_swatches().items()
                ],
            },
            "undertone": {
                "label": report.undertone_label,
                "confidence": report.undertone_confidence,
                "color": report.undertone_swatch.hex,
            },
            "season": {
                "primary": {
                    "code": report.season_primary.code,
                    "name": report.season_primary.name,
                    "color": report.season_primary.color.hex,
                },
                "secondary": {
                    "code": report.season_secondary.code,
                    "name": report.season_secondary.name,
                    "color": report.season_secondary.color.hex,
                },
                "ambiguous": ambiguity,
            },
            "look": {
                "name": report.makeup_look,
                "image": report.look_image_path,
                "desc": report.complexion_finish,
            },
            "colors": {
                "eye": [
                    {"name": item.label, "hex": item.hex}
                    for item in report.eye_colors
                ],
                "blush": [
                    {"name": item.label, "hex": item.hex}
                    for item in report.blush_colors
                ],
                "lip": [
                    {"name": item.label, "hex": item.hex}
                    for item in report.lip_colors
                ],
            },
            "products": products,
            "reasons": [
                {
                    "title": group.title,
                    "items": [
                        {"label": item.label, "text": item.text}
                        for item in group.items
                        if item.label != "Warna yang dihindari" or item.text.strip()
                    ],
                }
                for group in report.explanation.groups
            ],
        }

    def generate_report_html(
        self,
        result: FinalAnalysisResult,
        client_name: str | None = None,
        analysis_date: str | None = None,
        input_image_url: str = "",
        saturation_sample_url: str = "",
        input_image_path: str | Path | None = None,
        kb_rules_data: dict | None = None,
        catalog_data: list[dict] | None = None,
    ) -> str:
        if input_image_url or saturation_sample_url:
            raise ValueError("Report assets are embedded from local paths; URL overrides are unsupported.")
        if kb_rules_data is None:
            with self.settings.KB_RULES_PATH.open("r", encoding="utf-8") as file:
                kb_rules_data = json.load(file)
        if catalog_data is None:
            from app.core.product_repository_pg import build_product_repository

            repo = build_product_repository(self.settings)
            catalog_data = [p.model_dump(mode="json") for p in repo.list_products()]

        report = self._report_data(
            result,
            input_image_path,
            kb_rules_data,
            catalog_data,
        )
        payload = self._payload(
            report,
            analysis_date or date.today().strftime("%d %B %Y"),
            result.reasoning.is_ambiguous,
            client_name,
        )
        if not self.template_path.is_file():
            raise FileNotFoundError(f"Template laporan tidak ditemukan: {self.template_path}")
        html_template = self.template_path.read_text(encoding="utf-8")
        start_marker = "let DATA={"
        end_marker = ";\n/* Opsional"
        start = html_template.find(start_marker)
        end = html_template.find(end_marker, start)
        if start < 0 or end < 0:
            raise ValueError("Penanda DATA pada makeup-report-template.html tidak ditemukan.")
        encoded_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace(
            "<", "\\u003c"
        )
        return html_template[:start] + f"let DATA={encoded_payload}" + html_template[end:]

    def save_report(
        self,
        result: FinalAnalysisResult,
        client_name: str | None = None,
        input_image_path: str | Path | None = None,
        report_filename: str | None = None,
        catalog_data: list[dict] | None = None,
    ) -> Path:
        self.settings.reports_dir.mkdir(parents=True, exist_ok=True)
        html = self.generate_report_html(
            result,
            client_name=client_name,
            input_image_path=input_image_path,
            catalog_data=catalog_data,
        )
        filename = report_filename or f"report_{result.case_id}.html"
        if Path(filename).name != filename or not filename.endswith(".html"):
            raise ValueError("Nama file report harus berupa nama file .html tanpa path.")
        output_path = self.settings.reports_dir / filename
        output_path.write_text(html, encoding="utf-8")
        return output_path
