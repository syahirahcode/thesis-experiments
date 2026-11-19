"""FastAPI application entry point."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.v1.analyze import router as analyze_router
from app.core.config import get_settings

settings = get_settings()
settings.ensure_output_dirs()

# Ensure uploads and sessions directories exist
uploads_dir = settings.OUTPUTS_DIR / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)
sessions_dir = settings.OUTPUTS_DIR / "sessions"
sessions_dir.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="GlamAI - Image Consulting & Makeup Recommendation API",
    description="Main API untuk analisis fitur wajah, penetapan color season, rekomendasi makeup, dan verifikasi pakar.",
    version="1.0.0",
)

# Enable CORS for frontend development and production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directories so the frontend can load images
if settings.ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=settings.ASSETS_DIR), name="assets")

if settings.OUTPUTS_DIR.exists():
    app.mount("/outputs", StaticFiles(directory=settings.OUTPUTS_DIR), name="outputs")

if settings.TEST_DATA_DIR.exists():
    app.mount("/experiments/test_data", StaticFiles(directory=settings.TEST_DATA_DIR), name="test_data")

# Register API v1 endpoints
app.include_router(analyze_router, prefix="/api/v1", tags=["analysis"])


@app.get("/health", tags=["system"])
async def health_check():
    return {"status": "ok", "app": "GlamAI API", "version": "1.0.0"}


@app.get("/", tags=["system"])
async def root():
    return {
        "message": "GlamAI API is active.",
        "docs": "/docs",
        "endpoints": {
            "analyze": "/api/v1/analyze"        },
    }
