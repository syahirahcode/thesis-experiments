"""
backend/app/core/report_store.py

In-Memory Storage dan Temporary JSON Session Storage untuk menyimpan
hasil analisis makeup report tanpa PostgreSQL.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ReportSessionStore:
    """
    Menyimpan laporan analisis makeup di In-Memory Cache dan
    menyelaraskannya ke file JSON sementara di folder outputs/sessions.
    """

    def __init__(self, session_dir: Path | None = None) -> None:
        self.settings = get_settings()
        self.session_dir = session_dir or (self.settings.OUTPUTS_DIR / "sessions")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._in_memory: dict[str, dict[str, Any]] = {}
        self._load_existing_from_disk()
        self._seed_sample_if_empty()

    def _load_existing_from_disk(self) -> None:
        """Memuat session JSON yang sudah ada di disk ke dalam in-memory cache saat inisialisasi."""
        try:
            for file_path in self.session_dir.glob("*.json"):
                try:
                    with file_path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    report_id = data.get("report_id") or file_path.stem
                    self._in_memory[report_id] = data
                except Exception as e:
                    logger.warning(f"Gagal memuat session file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Gagal membaca folder session: {e}")

    def _seed_sample_if_empty(self) -> None:
        """Bila session masih kosong, muat satu laporan contoh dari file output yang ada."""
        if self._in_memory:
            return
        reports_dir = self.settings.OUTPUTS_DIR / "reports"
        if not reports_dir.is_dir():
            return
        html_files = [f for f in reports_dir.glob("*.html") if f.name != ".gitkeep"]
        if not html_files:
            return
        try:
            content = html_files[0].read_text(encoding="utf-8")
            start = content.find("let DATA=")
            end = content.find(";\n/* Opsional", start)
            if start != -1 and end != -1:
                raw_json = content[start + len("let DATA=") : end]
                data = json.loads(raw_json)
                report_id = "sample_case_994"
                sample_record = {
                    "report_id": report_id,
                    "case_id": "case_994",
                    "client_name": data.get("client", {}).get("name", "Demo User"),
                    "mode": "multi_agent",
                    "status": "success",
                    "created_at": _utcnow_iso(),
                    "photo_url": "/experiments/test_data/24729.png",
                    "data": data,
                    "expert_verification": {
                        "is_verified": True,
                        "status": "verified",
                        "expert_name": "Rina Sp.KK / MUA Specialist",
                        "notes": "Analisis warna kulit dan saturasi sangat akurat. Palet soft & glowing makeup sangat sesuai untuk Spring Light.",
                        "adjustments": {},
                        "verified_at": _utcnow_iso(),
                    },
                }
                self.save(report_id, sample_record)
        except Exception as e:
            logger.warning(f"Gagal seeding sample report: {e}")

    def save(self, report_id: str, report_data: dict[str, Any]) -> dict[str, Any]:
        """
        Menyimpan laporan ke in-memory dan file JSON sementara.
        """
        if "created_at" not in report_data:
            report_data["created_at"] = _utcnow_iso()
        report_data["updated_at"] = _utcnow_iso()
        report_data["report_id"] = report_id

        # In-Memory
        self._in_memory[report_id] = report_data

        # Temporary JSON Session Storage di Disk
        disk_file = self.session_dir / f"{report_id}.json"
        try:
            with disk_file.open("w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Gagal menulis session ke {disk_file}: {e}")

        return report_data

    def get(self, report_id: str) -> dict[str, Any] | None:
        """
        Mengambil laporan berdasarkan report_id (pertama dari in-memory, fallback ke JSON disk).
        """
        if report_id in self._in_memory:
            return self._in_memory[report_id]

        disk_file = self.session_dir / f"{report_id}.json"
        if disk_file.is_file():
            try:
                with disk_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                self._in_memory[report_id] = data
                return data
            except Exception as e:
                logger.error(f"Gagal membaca session dari {disk_file}: {e}")
        return None

    def list_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Mengembalikan daftar ringkasan laporan terbaru.
        """
        reports = list(self._in_memory.values())
        # Urutkan berdasarkan created_at descending
        reports.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        return reports[:limit]

    def update_expert_verification(
        self,
        report_id: str,
        expert_name: str,
        status: str,
        notes: str = "",
        adjustments: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Menambahkan atau memperbarui status verifikasi pakar pada laporan.
        """
        record = self.get(report_id)
        if not record:
            return None

        verification_entry = {
            "is_verified": status.lower() in ("verified", "approved", "valid"),
            "status": status,
            "expert_name": expert_name,
            "notes": notes,
            "adjustments": adjustments or {},
            "verified_at": _utcnow_iso(),
        }

        record["expert_verification"] = verification_entry
        record["updated_at"] = _utcnow_iso()

        # Update both memory & disk
        self._in_memory[report_id] = record
        disk_file = self.session_dir / f"{report_id}.json"
        try:
            with disk_file.open("w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Gagal memperbarui session di {disk_file}: {e}")

        return record


# Global singleton instance
_session_store_instance: ReportSessionStore | None = None


def get_report_store() -> ReportSessionStore:
    global _session_store_instance
    if _session_store_instance is None:
        _session_store_instance = ReportSessionStore()
    return _session_store_instance
