"""
PostgreSQL-backed product catalog (`public.products` on Supabase).

This is a drop-in replacement for `JsonProductRepository`:
`ProductSearchTool` and `ProductAgent` only talk to the `ProductRepository`
protocol, so swapping the repository is enough — no tool or agent rewrite.

Row mapping supports the Supabase schema:
    CREATE TABLE public.products (
        product_id BIGSERIAL PRIMARY KEY,
        brand VARCHAR(100) NOT NULL,
        product_name VARCHAR(255) NOT NULL,
        shade VARCHAR(100) NOT NULL,
        category VARCHAR(50),
        code_season INTEGER,
        target_color_season VARCHAR(50)[] NOT NULL,
        target_undertone VARCHAR(20)[] NOT NULL,
        target_skintone VARCHAR(20)[] NOT NULL,
        target_saturation VARCHAR(20)[] NOT NULL,
        image_url TEXT NOT NULL,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

Environment (.env at the project root):
    SUPABASE_CONNECTION_STRING   postgres://...  (required if PRODUCT_SOURCE=postgres)
    SUPABASE_POOLER_REGION       default: eu-west-1
    PRODUCT_SOURCE               postgres | json (default)
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

from app.core.config import PROJECT_ROOT, Settings, get_settings
from app.core.product_repository import JsonProductRepository, ProductRepository
from app.schemas.analysis import Product, season_to_code

ENV_PATH = PROJECT_ROOT / ".env"


def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    """Minimal .env reader; fallback when settings not fully populated."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def normalize_category(raw: str) -> str:
    """Standardize category strings to complexion, eye, blush, lip."""
    if not isinstance(raw, str):
        return "other"
    value = raw.strip().lower()
    if "lip" in value:
        return "lip"
    if "blush" in value:
        return "blush"
    if "eye" in value:
        return "eye"
    if any(
        term in value
        for term in ["foundation", "complexion", "cushion", "powder", "concealer"]
    ):
        return "complexion"
    return value


def full_product_name(brand: str, name: str, shade: str) -> str:
    """Format Brand + Name + ' - ' + Shade."""
    parts = [part for part in (brand, name) if part]
    full = " ".join(parts)
    if shade:
        full = f"{full} - {shade}" if full else shade
    return full


def connection_kwargs(connection: str) -> dict[str, Any]:
    """Decode a postgres URI into psycopg2 kwargs.

    Supabase passwords can start with '@', which libpq's URI parser splits
    incorrectly, so the password is decoded here instead of passed as a URI.
    """
    if not connection.startswith(("postgresql://", "postgres://")):
        return {"dsn": connection}
    parsed = urlparse(connection)
    kwargs: dict[str, Any] = {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": (parsed.path or "/postgres").lstrip("/") or "postgres",
        "user": unquote(parsed.username or "postgres"),
    }
    if parsed.password:
        kwargs["password"] = unquote(parsed.password)
    return kwargs


def pooler_kwargs(kwargs: dict[str, Any], region: str) -> dict[str, Any]:
    """Rewrite the (often unreachable) direct db host to the Supabase pooler."""
    host = str(kwargs.get("host") or "")
    if not host.startswith("db.") or not host.endswith(".supabase.co"):
        return kwargs
    ref = host[len("db."):-len(".supabase.co")]
    return {
        **kwargs,
        "host": f"aws-0-{region}.pooler.supabase.com",
        "port": 5432,
        "user": f"postgres.{ref}",
    }


class PostgresProductRepository:
    """Reads the makeup catalog from PostgreSQL (`public.products` on Supabase)."""

    def __init__(
        self,
        connection: str | None = None,
        pooler_region: str | None = None,
        settings: Settings | None = None,
        table: str = "public.products",
    ):
        self.settings = settings or get_settings()
        env = load_env()
        conn_from_settings = ""
        if hasattr(self.settings, "SUPABASE_CONNECTION_STRING"):
            conn_from_settings = self.settings.SUPABASE_CONNECTION_STRING.get_secret_value().strip()

        self.connection_string = (
            connection
            or conn_from_settings
            or os.environ.get("SUPABASE_CONNECTION_STRING")
            or env.get("SUPABASE_CONNECTION_STRING", "")
        )
        if not self.connection_string:
            raise RuntimeError(
                "SUPABASE_CONNECTION_STRING belum diisi di .env — "
                "PostgresProductRepository tidak bisa terhubung."
            )
        self.pooler_region = (
            pooler_region
            or getattr(self.settings, "SUPABASE_POOLER_REGION", None)
            or os.environ.get("SUPABASE_POOLER_REGION")
            or env.get("SUPABASE_POOLER_REGION")
            or "eu-west-1"
        )
        self.table = table
        self._conn: Any | None = None
        self._col_types: dict[str, str] | None = None

    # --- koneksi ---------------------------------------------------------- #
    def _driver(self):
        try:
            import psycopg2
            import psycopg2.extras

            return psycopg2
        except ImportError as error:
            raise RuntimeError(
                "psycopg2 belum terpasang. Jalankan: pip install psycopg2-binary"
            ) from error

    def _connect(self):
        driver = self._driver()
        kwargs = connection_kwargs(self.connection_string)
        if "dsn" in kwargs:
            return driver.connect(kwargs["dsn"])
        kwargs.setdefault("connect_timeout", 15)
        host = str(kwargs.get("host") or "")
        if host.startswith("db.") and host.endswith(".supabase.co"):
            kwargs = pooler_kwargs(kwargs, self.pooler_region)
        return driver.connect(**kwargs)

    @contextmanager
    def _cursor(self) -> Iterator[Any]:
        """Yield a dict cursor, reconnecting once if the pooled socket dropped."""
        driver = self._driver()
        if self._conn is None or getattr(self._conn, "closed", 0):
            self._conn = self._connect()
        try:
            with self._conn.cursor(cursor_factory=driver.extras.RealDictCursor) as cur:
                yield cur
            self._conn.commit()
        except driver.OperationalError:
            self._conn = self._connect()
            with self._conn.cursor(cursor_factory=driver.extras.RealDictCursor) as cur:
                yield cur
            self._conn.commit()

    def close(self) -> None:
        if self._conn is not None and not getattr(self._conn, "closed", 1):
            self._conn.close()
        self._conn = None

    def __enter__(self) -> "PostgresProductRepository":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # --- schema inspection ------------------------------------------------ #
    def _inspect_columns(self) -> dict[str, str]:
        if self._col_types is not None:
            return self._col_types
        schema, _, name = self.table.partition(".")
        with self._cursor() as cur:
            cur.execute(
                "select column_name, data_type from information_schema.columns "
                "where table_schema = %s and table_name = %s "
                "order by ordinal_position",
                (schema or "public", name),
            )
            self._col_types = {row["column_name"]: row["data_type"] for row in cur.fetchall()}
        return self._col_types

    # --- mapping ---------------------------------------------------------- #
    @staticmethod
    def _to_product(row: dict[str, Any]) -> Product:
        raw_id = row.get("product_id") or row.get("No") or row.get("id") or 0
        if isinstance(raw_id, int) or (isinstance(raw_id, str) and str(raw_id).isdigit()):
            prod_id = f"PROD_{int(raw_id):03d}"
        else:
            prod_id = str(raw_id).strip()

        brand = str(row.get("brand") or row.get("Brand") or "").strip()
        name = str(row.get("product_name") or row.get("Product Name") or "").strip()
        shade = str(row.get("shade") or row.get("Shade") or "").strip()
        raw_category = str(row.get("category") or row.get("Category") or "").strip()

        # Season
        raw_season = (
            row.get("target_color_season")
            or row.get("Season")
            or row.get("season")
            or ""
        )
        if isinstance(raw_season, (list, tuple)):
            target_season = season_to_code(str(raw_season[0])) if raw_season else ""
        else:
            target_season = season_to_code(str(raw_season or ""))

        # Attributes
        def _fmt(val: Any) -> str:
            if isinstance(val, (list, tuple)):
                return ", ".join(str(v).strip() for v in val if v)
            return str(val or "").strip()

        target_undertone = _fmt(row.get("target_undertone") or row.get("Target Undertone"))
        target_skintone = _fmt(row.get("target_skintone") or row.get("Target Skintone"))
        target_saturation = _fmt(row.get("target_saturation") or row.get("Target Saturation"))

        # Link from Supabase Postgres (image_url)
        image_url = str(
            row.get("image_url")
            or row.get("Link Bucket")
            or row.get("link_bucket")
            or row.get("Link foto")
            or row.get("link")
            or row.get("url")
            or ""
        ).strip()

        # File name
        image_file = str(
            row.get("image_file")
            or row.get("File Name")
            or row.get("file_name")
            or row.get("Nama File")
            or row.get("nama_file")
            or ""
        ).strip()
        if not image_file and image_url:
            image_file = image_url

        full_name = full_product_name(brand, name, shade) or f"Product {prod_id}"

        return Product(
            product_id=prod_id,
            product_name=full_name,
            brand=brand,
            category=normalize_category(raw_category),
            raw_category=raw_category,
            shade=shade,
            target_season=target_season,
            target_undertone=target_undertone,
            target_skintone=target_skintone,
            target_saturation=target_saturation,
            image_file=image_file,
            image_url=image_url,
            url=image_url,
            link=image_url,
            notes=str(row.get("notes") or row.get("Notes") or "").strip(),
        )

    # --- ProductRepository protocol --------------------------------------- #
    def list_products(self) -> list[Product]:
        cols = self._inspect_columns()
        id_col = '"product_id"' if "product_id" in cols else ('"No"' if "No" in cols else '"id"')
        with self._cursor() as cur:
            cur.execute(f"select * from {self.table} order by {id_col} asc")
            return [self._to_product(row) for row in cur.fetchall()]

    def search_products(
        self,
        season_code: str,
        category: str,
        k: int,
        attributes: dict[str, str],
    ) -> list[Product]:
        if k < 1:
            raise ValueError("k harus minimal 1")
        requested_category = normalize_category(category)
        if requested_category not in {"complexion", "eye", "blush", "lip"}:
            raise ValueError(f"Kategori produk tidak dikenal: {category!r}")
        requested_season = season_to_code(season_code)

        cols = self._inspect_columns()
        id_col = '"product_id"' if "product_id" in cols else ('"No"' if "No" in cols else '"id"')
        cat_col = '"category"' if "category" in cols else '"Category"'

        # Category condition in SQL
        category_sql = f"""
            case
                when strpos(lower({cat_col}), 'lip') > 0 then 'lip'
                when strpos(lower({cat_col}), 'blush') > 0 then 'blush'
                when strpos(lower({cat_col}), 'eye') > 0 then 'eye'
                when strpos(lower({cat_col}), 'foundation') > 0
                  or strpos(lower({cat_col}), 'complexion') > 0
                  or strpos(lower({cat_col}), 'cushion') > 0
                  or strpos(lower({cat_col}), 'powder') > 0
                  or strpos(lower({cat_col}), 'concealer') > 0 then 'complexion'
                else lower({cat_col})
            end = %(category)s
        """

        # Season condition
        if "target_color_season" in cols:
            season_col = '"target_color_season"'
            if cols["target_color_season"] == "ARRAY":
                season_sql = f"""
                    EXISTS (
                        SELECT 1 FROM unnest({season_col}) _s
                        WHERE upper(regexp_replace(btrim(_s), '[\\s-]+', '_', 'g')) = %(season)s
                    )
                """
            else:
                season_sql = f"upper(regexp_replace(btrim({season_col}), '[\\s-]+', '_', 'g')) = %(season)s"
        else:
            season_col = '"Season"' if "Season" in cols else '"season"'
            season_sql = f"upper(regexp_replace(btrim({season_col}), '[\\s-]+', '_', 'g')) = %(season)s"

        # Attribute score condition
        def _score_expr(col_name: str, param_name: str) -> str:
            quoted = f'"{col_name}"'
            if cols.get(col_name) == "ARRAY":
                return f"(EXISTS (SELECT 1 FROM unnest({quoted}) _v WHERE lower(btrim(_v)) = lower(btrim(%({param_name})s))))::int"
            return f"(lower(btrim({quoted})) = lower(btrim(%({param_name})s)))::int"

        undertone_col = "target_undertone" if "target_undertone" in cols else "Target Undertone"
        skintone_col = "target_skintone" if "target_skintone" in cols else "Target Skintone"
        saturation_col = "target_saturation" if "target_saturation" in cols else "Target Saturation"

        score_sql = f"""
            (
                {_score_expr(undertone_col, "undertone")}
              + {_score_expr(skintone_col, "skintone")}
              + {_score_expr(saturation_col, "saturation")}
            ) as match_score
        """

        query = f"""
            select *,
                   {score_sql}
              from {self.table}
             where {season_sql}
               and {category_sql}
             order by match_score desc, {id_col} asc
             limit %(limit)s
        """
        params = {
            "season": requested_season,
            "category": requested_category,
            "undertone": attributes.get("undertone", ""),
            "skintone": attributes.get("skintone", ""),
            "saturation": attributes.get("saturation", ""),
            "limit": k,
        }
        with self._cursor() as cur:
            cur.execute(query, params)
            return [self._to_product(row) for row in cur.fetchall()]


def build_product_repository(
    settings: Settings | None = None,
) -> ProductRepository:
    """Pick the catalog backend based on PRODUCT_SOURCE in settings/.env."""
    settings = settings or get_settings()
    env = load_env()
    source = (
        getattr(settings, "PRODUCT_SOURCE", None)
        or os.environ.get("PRODUCT_SOURCE")
        or env.get("PRODUCT_SOURCE")
        or "json"
    )
    source = str(source).strip().lower()
    if source == "postgres":
        conn = None
        if hasattr(settings, "has_supabase_connection") and settings.has_supabase_connection:
            conn = settings.SUPABASE_CONNECTION_STRING.get_secret_value().strip()
        conn = conn or os.environ.get("SUPABASE_CONNECTION_STRING") or env.get("SUPABASE_CONNECTION_STRING")
        region = (
            getattr(settings, "SUPABASE_POOLER_REGION", None)
            or os.environ.get("SUPABASE_POOLER_REGION")
            or env.get("SUPABASE_POOLER_REGION")
        )
        return PostgresProductRepository(
            connection=conn,
            pooler_region=region,
            settings=settings,
        )
    return JsonProductRepository(settings.PRODUCTS_PATH)
