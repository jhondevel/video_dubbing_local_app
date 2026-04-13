from __future__ import annotations

import re
import shutil
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import DEFAULT_VOICE, OUTPUT_DIR, UPLOAD_DIR, VOICES_DIR
from .db import create_job, get_job
from .lang import SUGGESTED_VOICES, normalize_language

app = FastAPI(title="Doblaje local de video")

BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

LANGUAGE_OPTIONS = [
    {"code": "es", "label": "Español"},
    {"code": "en", "label": "English"},
    {"code": "fr", "label": "Français"},
    {"code": "de", "label": "Deutsch"},
    {"code": "it", "label": "Italiano"},
    {"code": "pt", "label": "Português"},
    {"code": "ru", "label": "Русский"},
    {"code": "ja", "label": "日本語"},
    {"code": "zh", "label": "中文"},
    {"code": "ko", "label": "한국어"},
    {"code": "ar", "label": "العربية"},
    {"code": "hi", "label": "हिन्दी"},
    {"code": "tr", "label": "Türkçe"},
    {"code": "pl", "label": "Polski"},
    {"code": "nl", "label": "Nederlands"},
    {"code": "ca", "label": "Català"},
]


def get_available_voices() -> list[str]:
    voices = sorted({p.stem for p in VOICES_DIR.glob("*.onnx")})
    if not voices:
        voices = [DEFAULT_VOICE]
    return voices


def build_index_context(request: Request) -> dict:
    available_voices = get_available_voices()
    default_voice = DEFAULT_VOICE if DEFAULT_VOICE in available_voices else available_voices[0]

    return {
        "request": request,
        "default_voice": default_voice,
        "voices": SUGGESTED_VOICES,
        "voice_options": available_voices,
        "language_options": LANGUAGE_OPTIONS,
    }


def save_upload(video: UploadFile) -> str:
    safe_name = Path(video.filename or "video.mp4").name
    destination = UPLOAD_DIR / safe_name
    with destination.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    return safe_name


def create_processing_job(
    background_tasks: BackgroundTasks,
    filename: str,
    target_language: str,
    voice_name: str,
) -> dict:
    from .pipeline import run_job

    job = create_job(filename, target_language, voice_name)
    background_tasks.add_task(run_job, job["id"], filename, target_language, voice_name)
    return job


def sanitize_download_name(raw_name: str | None, output_path: Path) -> str:
    if not raw_name:
        return output_path.name

    value = Path(raw_name.strip()).name
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" .")
    if not value:
        return output_path.name

    ext = output_path.suffix or ".mp4"
    if not value.lower().endswith(ext.lower()):
        value += ext
    return value


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "index.html", build_index_context(request))


@app.post("/jobs")
async def create_new_job(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form(...),
    voice_name: str = Form(""),
):
    try:
        target_code = normalize_language(target_language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    selected_voice = voice_name.strip() or SUGGESTED_VOICES.get(target_code, DEFAULT_VOICE)
    safe_name = save_upload(video)
    job = create_processing_job(background_tasks, safe_name, target_code, selected_voice)
    return RedirectResponse(url=f"/jobs/{job['id']}", status_code=303)


@app.post("/api/jobs")
async def create_new_job_api(
    request: Request,
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    target_language: str = Form(...),
    voice_name: str = Form(""),
):
    try:
        target_code = normalize_language(target_language)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    selected_voice = voice_name.strip() or SUGGESTED_VOICES.get(target_code, DEFAULT_VOICE)
    safe_name = save_upload(video)
    job = create_processing_job(background_tasks, safe_name, target_code, selected_voice)

    return {
        "ok": True,
        "job_id": job["id"],
        "redirect_url": f"/jobs/{job['id']}",
    }


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def job_status_page(request: Request, job_id: str) -> HTMLResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return templates.TemplateResponse(request, "job.html", {"request": request, "job": job})


@app.get("/api/jobs/{job_id}")
async def job_status_api(job_id: str):
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Trabajo no encontrado")
    return job


@app.get("/download/{filename}")
async def download_file(
    filename: str,
    download_name: str | None = Query(default=None),
):
    path = OUTPUT_DIR / Path(filename).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Archivo no encontrado")

    final_name = sanitize_download_name(download_name, path)
    return FileResponse(path, filename=final_name)