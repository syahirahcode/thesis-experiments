"""
app/core/config.py

Konfigurasi lokal (path + environment) untuk eksperimen RQ2 & RQ3.
Membaca nilai dari file `.env` di root proyek. Semua path relatif otomatis
di-resolve terhadap root proyek, sehingga skrip bisa dijalankan dari mana saja:

    python experiments/run_eval_rq2_rq3.py

Pemakaian:
    from app.core.config import get_settings
    settings = get_settings()
    settings.ensure_output_dirs()
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> parents[0]=core, [1]=app, [2]=root proyek
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

SATURATION_LEVELS: tuple[str, ...] = ("low", "medium", "high")

# --- Warna tampilan laporan HTML. Ubah nilai di backend ini untuk menyesuaikan swatch;
# --- kode warna tidak ditampilkan sebagai teks atau kontrol edit di laporan.
DEFAULT_SKINTONE_SWATCHES: dict[str, tuple[str, str]] = {
    "Fair": ("#F6EDE4", "#F3E7DB"),
    "Medium": ("#F7EAD0", "#EADABA"),
    "Tanned": ("#D7BD96", "#A07E56"),
    "Deep": ("#825C43", "#604134"),
    "Dark": ("#3A312A", "#292420"),
}
DEFAULT_UNDERTONE_HEX: dict[str, str] = {"warm": "#E8B26A", "cool": "#C7A3D6"}
DEFAULT_SEASON_HEX: dict[str, str] = {
    "SPRING_LIGHT": "#FFD9B0", "SPRING_CLEAR": "#FF8A65", "SPRING_STRONG": "#FF6F3C",
    "SUMMER_LIGHT": "#D9C6E8", "SUMMER_CLEAR": "#9FC5F8", "SUMMER_MUTE": "#B8A9C9", "SUMMER_MUTED": "#B8A9C9",
    "AUTUMN_MUTED": "#C9A66B", "AUTUMN_MUTE": "#C9A66B", "AUTUMN_DEEP": "#A0522D", "AUTUMN_DARK": "#6B3E26",
    "WINTER_STRONG": "#C2185B", "WINTER_DEEP": "#4A148C", "WINTER_DARK": "#1A1A40",
}
FALLBACK_SWATCH_HEX = "#CCCCCC"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM (Google AI Studio) -------------------------------------------
    GEMINI_API_KEY: SecretStr = SecretStr("")
    LLM_MODEL: str = "gemini-3.1-flash-lite"
    LLM_TEMPERATURE: float = 0.0  # 0 agar komparasi multi vs single adil & reproducible
    LLM_TIMEOUT_SECONDS: float = 60.0
    LLM_MAX_RETRIES: int = 2
    LLM_MAX_ROUNDS: int = Field(default=8, ge=1)

    # --- LLM (VIKEY, OpenAI-compatible endpoint) ---------------------------
    LLM_OPTION: str = "gemini"  # gemini | vikey
    VIKEY_URL: str = "https://api.vikey.ai/v1"
    VIKEY_KEY: SecretStr = SecretStr("")
    VIKEY_MODEL: str = "deepseek-v4.1-flash"

    # --- Path model, KB, katalog, aset ------------------------------------
    PRODUCT_SOURCE: str = "json"  # postgres | json
    SUPABASE_CONNECTION_STRING: SecretStr = SecretStr("")
    SUPABASE_POOLER_REGION: str = "eu-west-1"
    CNN_MODEL_PATH: Path = Path("app/models/best_model_b_rgb.pth")
    CNN_DEVICE: str = "auto"  # auto | cpu | cuda
    KB_RULES_PATH: Path = Path("app/core/kb_rules.json")
    PRODUCTS_PATH: Path = Path("app/core/products.json")
    ASSETS_DIR: Path = Path("assets")

    # --- Eksperimen --------------------------------------------------------
    TEST_DATA_DIR: Path = Path("experiments/test_data")
    GROUND_TRUTH_CSV: Path = Path("experiments/Final Dataset - Data.csv")
    GROUND_TRUTH_ENCODING: str = "utf-8-sig"  # aman untuk CSV export Google Sheets (BOM)
    # "cnn" (default): each architecture independently runs its own CNN inference.
    # "ground_truth": optional expert-label ablation.
    ATTRIBUTE_SOURCE: str = "cnn"
    OUTPUTS_DIR: Path = Path("outputs")
    EXPECTED_N_CASES: int = 24
    TOP_K_PRODUCTS_PER_CATEGORY: int = Field(default=6, ge=1)
    EXPERIMENT_SEED: int = 42
    IMAGE_EXTENSIONS: tuple[str, ...] = (".png", ".jpg", ".jpeg")

    # --- Normalisasi: path relatif -> absolut terhadap PROJECT_ROOT --------
    @field_validator(
        "CNN_MODEL_PATH",
        "KB_RULES_PATH",
        "PRODUCTS_PATH",
        "ASSETS_DIR",
        "TEST_DATA_DIR",
        "GROUND_TRUTH_CSV",
        "OUTPUTS_DIR",
        mode="after",
    )
    @classmethod
    def _resolve_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @field_validator("ATTRIBUTE_SOURCE")
    @classmethod
    def _check_attr_source(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"cnn", "ground_truth"}:
            raise ValueError("ATTRIBUTE_SOURCE harus 'cnn' atau 'ground_truth'")
        return value

    @field_validator("CNN_DEVICE")
    @classmethod
    def _check_device(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"auto", "cpu", "cuda"}:
            raise ValueError("CNN_DEVICE harus 'auto', 'cpu', atau 'cuda'")
        return value

    @field_validator("LLM_OPTION")
    @classmethod
    def _check_llm_option(cls, value: str) -> str:
        value = value.lower().strip()
        if value not in {"gemini", "vikey"}:
            raise ValueError("LLM_OPTION harus 'gemini' atau 'vikey'")
        return value

    @field_validator("PRODUCT_SOURCE")
    @classmethod
    def _check_product_source(cls, value: str) -> str:
        value = value.lower().strip()
        if value in {"postgres", "postgre", "postgresql"}:
            return "postgres"
        if value == "json":
            return "json"
        raise ValueError("PRODUCT_SOURCE harus 'postgres' atau 'json'")

    # --- Path turunan ------------------------------------------------------
    @property
    def project_root(self) -> Path:
        return PROJECT_ROOT

    @property
    def looks_dir(self) -> Path:
        return self.ASSETS_DIR / "looks"

    @property
    def products_dir(self) -> Path:
        return self.ASSETS_DIR / "products"

    @property
    def saturations_dir(self) -> Path:
        return self.ASSETS_DIR / "saturations"

    @property
    def reports_dir(self) -> Path:
        return self.OUTPUTS_DIR / "reports"

    @property
    def trace_log_path(self) -> Path:
        return self.OUTPUTS_DIR / "rq2_trace_logs.json"

    @property
    def summary_csv_path(self) -> Path:
        return self.OUTPUTS_DIR / "rq2_summary.csv"

    @property
    def checkpoint_path(self) -> Path:
        return self.OUTPUTS_DIR / "rq2_checkpoint.json"

    # --- Helper path aset --------------------------------------------------
    def look_image_path(self, look_file_name: str) -> Path:
        """assets/looks/<look_file_name> (mis. '1_face.png')."""
        return self.looks_dir / look_file_name

    def product_image_path(self, image_file: str) -> Path:
        """assets/products/<image_file> (mis. 'Product 1.png')."""
        return self.products_dir / image_file

    def saturation_image_path(self, level: str) -> Path:
        """assets/saturations/<level>_saturation.jpg; level: Low/Medium/High (case-insensitive)."""
        key = level.strip().lower()
        if key not in SATURATION_LEVELS:
            raise ValueError(f"Level saturasi tidak dikenal: {level!r} (pilih {SATURATION_LEVELS})")
        return self.saturations_dir / f"{key}_saturation.jpg"

    # --- Warna tampilan laporan -------------------------------------------
    @staticmethod
    def skintone_display_swatches() -> dict[str, tuple[str, str]]:
        return DEFAULT_SKINTONE_SWATCHES.copy()

    @staticmethod
    def undertone_display_hex(label: str) -> str:
        return DEFAULT_UNDERTONE_HEX.get(label.strip().lower(), FALLBACK_SWATCH_HEX)

    @staticmethod
    def season_display_hex(season_code: str) -> str:
        return DEFAULT_SEASON_HEX.get(season_code.strip().upper(), FALLBACK_SWATCH_HEX)

    # --- Utilitas ----------------------------------------------------------
    @property
    def has_api_key(self) -> bool:
        return bool(self.GEMINI_API_KEY.get_secret_value().strip()) and (
            self.GEMINI_API_KEY.get_secret_value().strip() != "xxxxxxxxx"
        )

    def require_api_key(self) -> str:
        if not self.has_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY belum diisi. Isi di file .env (root proyek) lalu jalankan ulang."
            )
        return self.GEMINI_API_KEY.get_secret_value().strip()

    @property
    def has_vikey_key(self) -> bool:
        return bool(self.VIKEY_KEY.get_secret_value().strip())

    def require_vikey_key(self) -> str:
        if not self.has_vikey_key:
            raise RuntimeError(
                "VIKEY_KEY belum diisi. Isi di file .env (root proyek) lalu jalankan ulang."
            )
        return self.VIKEY_KEY.get_secret_value().strip()

    @property
    def has_supabase_connection(self) -> bool:
        return bool(self.SUPABASE_CONNECTION_STRING.get_secret_value().strip())

    def require_supabase_connection(self) -> str:
        if not self.has_supabase_connection:
            raise RuntimeError(
                "SUPABASE_CONNECTION_STRING belum diisi di .env untuk PRODUCT_SOURCE=postgres."
            )
        return self.SUPABASE_CONNECTION_STRING.get_secret_value().strip()

    def ensure_output_dirs(self) -> None:
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def list_test_images(self) -> list[Path]:
        """Foto uji terurut (nama file), mengabaikan .gitkeep dan berkas non-gambar."""
        if not self.TEST_DATA_DIR.exists():
            return []
        return sorted(
            p for p in self.TEST_DATA_DIR.iterdir() if p.suffix.lower() in self.IMAGE_EXTENSIONS
        )

    def validate_runtime(self) -> list[str]:
        """Kembalikan daftar masalah (kosong = siap jalan). Dipanggil di awal runner."""
        problems: list[str] = []
        required = {
            "KB_RULES_PATH": self.KB_RULES_PATH,
            "ASSETS_DIR": self.ASSETS_DIR,
            "looks_dir": self.looks_dir,
            "products_dir": self.products_dir,
            "saturations_dir": self.saturations_dir,
            "TEST_DATA_DIR": self.TEST_DATA_DIR,
            "GROUND_TRUTH_CSV": self.GROUND_TRUTH_CSV,
        }
        if self.PRODUCT_SOURCE == "json":
            required["PRODUCTS_PATH"] = self.PRODUCTS_PATH
        if self.ATTRIBUTE_SOURCE == "cnn":
            required["CNN_MODEL_PATH"] = self.CNN_MODEL_PATH
        for name, path in required.items():
            if not path.exists():
                problems.append(f"{name} tidak ditemukan: {path}")
        n_images = len(self.list_test_images())
        if n_images != self.EXPECTED_N_CASES:
            problems.append(
                f"Jumlah foto uji = {n_images}, diharapkan {self.EXPECTED_N_CASES} "
                f"(cek EXPECTED_N_CASES di .env)"
            )
        if self.LLM_OPTION == "vikey":
            if not self.has_vikey_key:
                problems.append("VIKEY_KEY kosong (LLM_OPTION=vikey)")
        elif not self.has_api_key:
            problems.append("GEMINI_API_KEY kosong / masih placeholder")
        if self.PRODUCT_SOURCE == "postgres":
            if not self.has_supabase_connection:
                problems.append("SUPABASE_CONNECTION_STRING kosong (PRODUCT_SOURCE=postgres)")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton settings (di-cache); panggil get_settings.cache_clear() di test bila perlu."""
    return Settings()
