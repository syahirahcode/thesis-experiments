"""Convert the product CSV to JSON and upsert the products into Supabase."""

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.config import PROJECT_ROOT, get_settings
from app.core.product_repository_pg import connection_kwargs, pooler_kwargs

CSV_INPUT_PATH = PROJECT_ROOT / "Products - Products.csv"
OUTPUT_JSON_PATH = PROJECT_ROOT / "app" / "core" / "products.json"
PRODUCTS_TABLE = "products"


def season_to_code(season_name: str) -> str:
    """Convert season labels such as 'Spring Light' to 'SPRING_LIGHT'."""
    if not isinstance(season_name, str) or not season_name.strip():
        return ""
    value = season_name.strip().upper().replace("-", " ").replace("_", " ")
    return "_".join(value.split())


def normalize_category(cat_str: str) -> str:
    """Group makeup categories into lip, blush, eye, or complexion."""
    if not isinstance(cat_str, str):
        return "other"
    value = cat_str.strip().lower()
    if "lip" in value:
        return "lip"
    if "blush" in value:
        return "blush"
    if "eye" in value:
        return "eye"
    if any(
        term in value
        for term in ("foundation", "complexion", "cushion", "powder", "concealer")
    ):
        return "complexion"
    return value


def _cell(row: Any, *columns: str) -> str:
    for column in columns:
        value = row.get(column)
        if pd.notna(value):
            text = str(value).strip()
            if text and text.casefold() != "nan":
                return text
    return ""


def _as_code_season(value: str) -> int | None:
    if not value:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def convert_csv_to_json(
    csv_path: Path = CSV_INPUT_PATH,
    output_path: Path = OUTPUT_JSON_PATH,
) -> list[dict[str, Any]]:
    """Convert the source product CSV and write the resulting catalog JSON."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"File CSV produk tidak ditemukan: {csv_path}")

    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"File CSV produk kosong: {csv_path}")

    image_column = next(
        (
            column
            for column in frame.columns
            if any(token in str(column).casefold() for token in ("unnamed", "file", "png"))
            and frame[column]
            .astype(str)
            .str.contains(r"\.(?:png|jpg|jpeg)", case=False, regex=True)
            .any()
        ),
        None,
    )
    if image_column is None and len(frame.columns) >= 4:
        image_column = frame.columns[3]

    products: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        number = _cell(row, "No")
        try:
            product_number = int(float(number))
        except ValueError:
            product_number = index + 1
        product_id = f"PROD_{product_number:03d}"

        image_file = _cell(row, image_column) if image_column is not None else ""
        if not image_file:
            image_file = _cell(row, "Nama File", "File Name")
            if image_file and not re.search(r"\.(png|jpg|jpeg)$", image_file, re.I):
                image_file = f"{image_file}.png"

        season = season_to_code(_cell(row, "Season"))
        brand = _cell(row, "Brand")
        name = _cell(row, "Product Name")
        shade = _cell(row, "Shade")
        full_name = " ".join(part for part in (brand, name) if part)
        if shade:
            full_name = f"{full_name} - {shade}" if full_name else shade

        products.append(
            {
                "product_id": product_id,
                "product_name": full_name or f"Product {product_id}",
                "brand": brand,
                "category": normalize_category(_cell(row, "Category")),
                "raw_category": _cell(row, "Category"),
                "shade": shade,
                "code_season": _as_code_season(_cell(row, "Code Season")),
                "target_season": season,
                "target_undertone": _cell(row, "Target Undertone"),
                "target_skintone": _cell(row, "Target Skintone"),
                "target_saturation": _cell(row, "Target Saturation"),
                "image_file": image_file,
                "image_url": _cell(row, "Link foto", "Link", "URL"),
                "notes": _cell(row, "Notes"),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(products, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return products


def _list_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _database_product_id(product_id: str, data_type: str) -> str | int:
    if data_type in {"smallint", "integer", "bigint"}:
        match = re.fullmatch(r"PROD_(\d+)", product_id, re.IGNORECASE)
        if not match:
            raise ValueError(
                f"ID produk {product_id!r} tidak bisa dipetakan ke kolom integer."
            )
        return int(match.group(1))
    return product_id


def _column_value(column: dict[str, Any], product: dict[str, Any]) -> Any:
    name = column["column_name"]
    key = name.casefold()
    data_type = column["data_type"]
    is_array = data_type == "ARRAY"

    if key in {"product_id", "no", "id"}:
        return _database_product_id(str(product["product_id"]), data_type)
    if key == "brand":
        return product["brand"]
    if key in {"product_name", "name"}:
        return product["product_name"]
    if key == "shade":
        return product["shade"]
    if key in {"category", "raw_category"}:
        return product["raw_category"] or product["category"]
    if key == "code_season":
        return product["code_season"]
    if key in {"target_color_season", "target_season", "season"}:
        values = _list_values(product["target_season"])
        return values if is_array else (values[0] if values else "")
    if key in {"target_undertone", "target_skintone", "target_saturation"}:
        values = _list_values(product[key])
        return values if is_array else ", ".join(values)
    if key in {"image_url", "link", "url", "link foto"}:
        return product["image_url"]
    if key in {"image_file", "file_name", "nama file"}:
        return product["image_file"]
    if key == "notes":
        return product["notes"]
    return None


def upsert_supabase_products(products: list[dict[str, Any]]) -> int:
    """Insert new products and update existing rows by their unchanged product_id."""
    try:
        import psycopg2
        from psycopg2 import sql
    except ImportError as error:
        raise RuntimeError(
            "Driver PostgreSQL belum terpasang. Install psycopg2-binary "
            "di environment backend."
        ) from error

    settings = get_settings()
    connection_string = settings.SUPABASE_CONNECTION_STRING.get_secret_value().strip()
    if not connection_string:
        raise RuntimeError(
            "SUPABASE_CONNECTION_STRING belum diisi di backend/.env."
        )

    kwargs = connection_kwargs(connection_string)
    if "dsn" not in kwargs:
        kwargs.setdefault("connect_timeout", 15)
        host = str(kwargs.get("host") or "")
        if host.startswith("db.") and host.endswith(".supabase.co"):
            kwargs = pooler_kwargs(kwargs, settings.SUPABASE_POOLER_REGION)
    connection = (
        psycopg2.connect(kwargs["dsn"])
        if "dsn" in kwargs
        else psycopg2.connect(**kwargs)
    )

    try:
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    "WHERE table_schema = %s AND table_name = %s "
                    "ORDER BY ordinal_position",
                    ("public", PRODUCTS_TABLE),
                )
                columns = [
                    {
                        "column_name": row[0],
                        "data_type": row[1],
                        "is_nullable": row[2],
                        "column_default": row[3],
                    }
                    for row in cursor.fetchall()
                ]
                if not columns:
                    raise RuntimeError(
                        "Tabel public.products tidak ditemukan di Supabase."
                    )

                id_column = next(
                    (
                        column
                        for column in columns
                        if column["column_name"] == "product_id"
                    ),
                    None,
                )
                if id_column is None:
                    raise RuntimeError(
                        "Kolom product_id tidak ditemukan; upsert dibatalkan "
                        "agar ID produk tidak berubah."
                    )

                supported = {
                    "product_id", "no", "id", "brand", "product_name", "name",
                    "shade", "category", "raw_category", "code_season",
                    "target_color_season", "target_season", "season",
                    "target_undertone", "target_skintone", "target_saturation",
                    "image_url", "link", "url", "link foto", "image_file",
                    "file_name", "nama file", "notes",
                }
                mapped_columns = [
                    column
                    for column in columns
                    if column["column_name"].casefold() in supported
                    and column["column_name"] != "created_at"
                ]
                mapped_names = {column["column_name"] for column in mapped_columns}
                missing_required = [
                    column["column_name"]
                    for column in columns
                    if column["is_nullable"] == "NO"
                    and column["column_default"] is None
                    and column["column_name"] not in mapped_names
                ]
                if missing_required:
                    raise RuntimeError(
                        "Kolom wajib tabel belum dipetakan oleh script: "
                        + ", ".join(missing_required)
                    )

                update_columns = [
                    column["column_name"]
                    for column in mapped_columns
                    if column["column_name"] != "product_id"
                ]
                if not update_columns:
                    raise RuntimeError(
                        "Tidak ada kolom produk yang dapat diperbarui."
                    )

                statement = sql.SQL(
                    "INSERT INTO {}.{} ({}) VALUES ({}) "
                    "ON CONFLICT ({}) DO UPDATE SET {}"
                ).format(
                    sql.Identifier("public"),
                    sql.Identifier(PRODUCTS_TABLE),
                    sql.SQL(", ").join(
                        sql.Identifier(column["column_name"])
                        for column in mapped_columns
                    ),
                    sql.SQL(", ").join(
                        sql.Placeholder() for _ in mapped_columns
                    ),
                    sql.Identifier("product_id"),
                    sql.SQL(", ").join(
                        sql.SQL("{} = EXCLUDED.{}").format(
                            sql.Identifier(name), sql.Identifier(name)
                        )
                        for name in update_columns
                    ),
                )

                for product in products:
                    values = [
                        _column_value(column, product)
                        for column in mapped_columns
                    ]
                    cursor.execute(statement, values)

                if id_column["data_type"] in {"smallint", "integer", "bigint"}:
                    cursor.execute(
                        "SELECT pg_get_serial_sequence(%s, %s)",
                        ("public.products", "product_id"),
                    )
                    sequence = cursor.fetchone()[0]
                    if sequence:
                        cursor.execute(
                            "SELECT setval("
                            "%s::regclass, "
                            "COALESCE((SELECT MAX(product_id) FROM public.products), 1), "
                            "EXISTS (SELECT 1 FROM public.products)"
                            ")",
                            (sequence,),
                        )
        return len(products)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Products CSV to JSON and upsert rows into Supabase."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert CSV to JSON only; do not connect to or modify Supabase.",
    )
    args = parser.parse_args()

    products = convert_csv_to_json()
    print(f"JSON diperbarui: {len(products)} produk -> {OUTPUT_JSON_PATH}")
    if args.dry_run:
        print("Dry run: tabel Supabase tidak diubah.")
        return

    count = upsert_supabase_products(products)
    print(f"Supabase diperbarui: {count} produk di public.products.")


if __name__ == "__main__":
    main()
