from __future__ import annotations

import importlib
import shutil
from pathlib import Path

modules = [
    "fastapi",
    "faster_whisper",
    "transformers",
    "sentencepiece",
    "piper",
    "imageio_ffmpeg",
    "soundfile",
    "numpy",
]

print("Verificando módulos...")
for name in modules:
    importlib.import_module(name)
    print(f"OK  {name}")

print("\nVerificando FFmpeg embebido...")
import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print("OK  ffmpeg:", ffmpeg_exe)
print("Existe:", Path(ffmpeg_exe).exists())

print("\nVerificando archivo principal...")
from main import app
print("OK  main:app cargó correctamente")

print("\nTodo parece instalado.")
