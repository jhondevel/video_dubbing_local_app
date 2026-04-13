from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import JOBS_DIR


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "job.json"


def create_job(filename: str, target_language: str, voice_name: str) -> dict[str, Any]:
    job_id = uuid4().hex[:12]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "status": "queued",
        "message": "En cola",
        "stage": "queued",
        "progress": 0,
        "filename": filename,
        "target_language": target_language,
        "voice_name": voice_name,
        "output_file": None,
        "created_at": _utc_now_iso(),
        "updated_at": _utc_now_iso(),
    }
    _job_path(job_id).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


def get_job(job_id: str) -> dict[str, Any] | None:
    path = _job_path(job_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def update_job(job_id: str, **changes: Any) -> dict[str, Any]:
    job = get_job(job_id)
    if job is None:
        raise FileNotFoundError(f"No existe el trabajo {job_id}")

    if "progress" in changes:
        try:
            progress = int(changes["progress"])
        except Exception:
            progress = 0
        changes["progress"] = max(0, min(100, progress))

    job.update(changes)
    job["updated_at"] = _utc_now_iso()
    _job_path(job_id).write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job