"""
backend/app/api/v1/analyze.py

API endpoint untuk analisis rekomendasi makeup menggunakan Multi-Agent architecture
dan pengelolaan laporan sesi (In-Memory / Temporary JSON Session Storage).
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from PIL import Image

from app.core.agents.facial_agent import FacialAgent
from app.core.agents.orchestrator import OrchestratorAgent
from app.core.config import Settings, get_settings
from app.core.html_generator import HTMLGenerator
from app.core.product_repository import ProductRepository
from app.core.product_repository_pg import build_product_repository
from app.schemas.analysis import (
    AgentMode,
    AnalysisRequest,
    AnalysisResponse,
    StepStatus,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Max allowed file size: 10MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGE_WIDTH = 200
MIN_IMAGE_HEIGHT = 200

# Singleton orchestrator agent
_orchestrator: OrchestratorAgent | None = None
_product_repo: ProductRepository | None = None


def get_product_repo(settings: Settings) -> ProductRepository:
    global _product_repo
    if _product_repo is None:
        _product_repo = build_product_repository(settings)
    return _product_repo


def get_orchestrator(settings: Settings) -> OrchestratorAgent:
    global _orchestrator
    if _orchestrator is None:
        repo = get_product_repo(settings)
        facial_agent = FacialAgent() if settings.ATTRIBUTE_SOURCE == "cnn" else None
        _orchestrator = OrchestratorAgent(
            facial_agent=facial_agent,
            reasoning_agent=None,
            product_agent=None,
            settings=settings,
            products=repo,
        )
    return _orchestrator


def _normalize_asset_url(path_str: str) -> str:
    """Mengubah relative path seperti ../../assets/looks/1.png menjadi URL web /assets/looks/1.png."""
    if not path_str:
        return ""
    cleaned = path_str.replace("\\", "/")
    if cleaned.startswith("http://") or cleaned.startswith("https://") or cleaned.startswith("data:"):
        return cleaned
    if "../../assets/" in cleaned:
        return "/assets/" + cleaned.split("../../assets/")[1]
    if "../../outputs/" in cleaned:
        return "/outputs/" + cleaned.split("../../outputs/")[1]
    if "../../experiments/test_data/" in cleaned:
        return "/experiments/test_data/" + cleaned.split("../../experiments/test_data/")[1]
    if cleaned.startswith("assets/"):
        return "/" + cleaned
    if cleaned.startswith("outputs/"):
        return "/" + cleaned
    return cleaned


def _normalize_payload_urls(payload: dict[str, Any], photo_url: str | None = None) -> dict[str, Any]:
    """Menyelaraskan semua URL gambar di dalam payload report agar dapat di-render langsung di browser."""
    if not isinstance(payload, dict):
        return payload

    if photo_url:
        payload["photo"] = photo_url
    elif "photo" in payload and isinstance(payload["photo"], str):
        payload["photo"] = _normalize_asset_url(payload["photo"])

    if "saturation" in payload and isinstance(payload["saturation"], dict):
        if "sample" in payload["saturation"]:
            payload["saturation"]["sample"] = _normalize_asset_url(payload["saturation"]["sample"])

    if "look" in payload and isinstance(payload["look"], dict):
        if "image" in payload["look"]:
            payload["look"]["image"] = _normalize_asset_url(payload["look"]["image"])

    if "products" in payload and isinstance(payload["products"], dict):
        for category, items in payload["products"].items():
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict) and "image" in item:
                        item["image"] = _normalize_asset_url(item["image"])

    return payload


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_makeup(
    request: Request,
    file: UploadFile | None = File(None),
    client_name: str | None = Form(None),
    mode: str = Form(AgentMode.MULTI_AGENT.value),
) -> AnalysisResponse:
    """
    Endpoint utama analisis rekomendasi makeup:
    1. Validasi gambar: format JPEG/JPG/PNG, ukuran maks 10MB, resolusi minimal 200x200 piksel.
    2. Menjalankan pipeline Multi-Agent (Facial Agent -> Reasoning Agent -> Product Agent).
    3. Mengubah hasil ke payload laporan tampilan HTML (schemas/analysis.py & html_generator.py).
    4. Menyimpan hasil ke In-Memory Storage / Temporary JSON Session Storage.
    """
    settings = get_settings()
    settings.ensure_output_dirs()

    uploads_dir = settings.OUTPUTS_DIR / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    # 1. Penanganan Input File atau JSON Body
    image_bytes: bytes | None = None
    original_filename = "upload.jpg"

    if file is not None:
        original_filename = file.filename or "upload.jpg"
        ext = Path(original_filename).suffix.lower()

        # Validasi ekstensi
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Upload gagal: Format file tidak didukung. Harap upload gambar dengan format JPEG, JPG, atau PNG.",
            )

        image_bytes = await file.read()
    else:
        # Cek apakah request berupa JSON (AnalyzeRequest)
        try:
            body = await request.json()
            if "image_path" in body:
                req_obj = AnalysisRequest.model_validate(body)
                local_path = Path(req_obj.image_path)
                if not local_path.is_file():
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"File gambar lokal tidak ditemukan: {req_obj.image_path}",
                    )
                image_bytes = local_path.read_bytes()
                original_filename = local_path.name
                if not client_name:
                    client_name = req_obj.case_id
                mode = req_obj.mode.value
        except Exception:
            pass

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Upload gagal: File gambar wajah wajib disertakan.",
        )

    # 2. Validasi Ukuran File (Maksimal 10MB)
    if len(image_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload gagal: Ukuran file gambar lebih dari 10MB ({len(image_bytes) / (1024 * 1024):.1f}MB).",
        )

    # 3. Validasi Resolusi Gambar (Minimal 200x200 piksel)
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload gagal: File rusak atau bukan file gambar yang valid ({exc}).",
        )

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Upload gagal karena resolusi gambar kekecilan ({width}x{height}px, minimal {MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT}px).",
        )

    # 4. Simpan Gambar ke Direktori Uploads
    report_uuid = uuid.uuid4().hex[:10]
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_id = f"rep_{timestamp_str}_{report_uuid}"
    case_id = f"case_{report_uuid}"

    clean_ext = Path(original_filename).suffix.lower() or ".jpg"
    saved_filename = f"{report_id}{clean_ext}"
    saved_path = uploads_dir / saved_filename
    saved_path.write_bytes(image_bytes)

    photo_web_url = f"/outputs/uploads/{saved_filename}"

    # 5. Jalankan Multi-Agent Pipeline
    agent_mode = AgentMode.MULTI_AGENT if mode != AgentMode.SINGLE_AGENT.value else AgentMode.SINGLE_AGENT
    orchestrator = get_orchestrator(settings)

    try:
        trace, result = orchestrator.analyze(
            case_id=case_id,
            image_path=str(saved_path),
        )
    except Exception as exc:
        logger.error(f"Error during orchestrator execution: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analisis agen gagal dieksekusi: {exc}",
        )

    if result is None or trace.final_status != StepStatus.SUCCESS:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analisis tidak menghasilkan rekomendasi: {trace.error_message or 'Eksekusi gagal'}",
        )

    # 6. Format Hasil Analisis Menjadi Payload Laporan
    html_gen = HTMLGenerator()
    with settings.KB_RULES_PATH.open("r", encoding="utf-8") as f:
        kb_rules_data = json.load(f)

    catalog_data = [p.model_dump(mode="json") for p in get_product_repo(settings).list_products()]

    report_data = html_gen._report_data(
        result=result,
        input_image_path=saved_path,
        kb_rules=kb_rules_data,
        catalog=catalog_data,
    )

    display_name = client_name or f"Pengguna #{report_uuid[:4].upper()}"
    analysis_date_str = datetime.now().strftime("%d %B %Y")

    raw_payload = html_gen._payload(
        report=report_data,
        analysis_date=analysis_date_str,
        ambiguity=result.reasoning.is_ambiguous,
        client_name=display_name,
    )

    web_payload = _normalize_payload_urls(raw_payload, photo_url=photo_web_url)

    created_at = datetime.now(timezone.utc).isoformat()

    return AnalysisResponse(
        report_id=report_id,
        case_id=case_id,
        status="success",
        mode=agent_mode,
        created_at=created_at,
        data=web_payload,
        photo_url=photo_web_url,
        expert_verification=None,
    )
