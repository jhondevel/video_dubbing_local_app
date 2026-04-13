from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
JOBS_DIR = DATA_DIR / "jobs"
OUTPUT_DIR = DATA_DIR / "outputs"
VOICES_DIR = DATA_DIR / "voices"

for directory in (DATA_DIR, UPLOAD_DIR, JOBS_DIR, OUTPUT_DIR, VOICES_DIR):
    directory.mkdir(parents=True, exist_ok=True)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "22050"))
TARGET_FORMAT = os.getenv("TARGET_FORMAT", "mp4")
DEFAULT_VOICE = os.getenv("DEFAULT_VOICE", "es_ES-sharvard-medium")
