from __future__ import annotations

import logging
import math
import shutil
import subprocess
import traceback
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from imageio_ffmpeg import get_ffmpeg_exe
from huggingface_hub.errors import RepositoryNotFoundError
from piper import PiperVoice
from transformers import MarianMTModel, MarianTokenizer

from .config import (
    DEFAULT_VOICE,
    JOBS_DIR,
    OUTPUT_DIR,
    SAMPLE_RATE,
    UPLOAD_DIR,
    VOICES_DIR,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)
from .db import update_job
from .lang import SUGGESTED_VOICES


logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


ProgressCallback = Callable[[int, str, str | None], None]
_WHISPER_MODEL: WhisperModel | None = None
TRANSCRIBE_CHUNK_SECONDS = 90


def set_progress(job_id: str, progress: int, message: str, stage: str | None = None, status: str = "processing") -> None:
    payload: dict[str, Any] = {
        "status": status,
        "progress": progress,
        "message": message,
    }
    if stage is not None:
        payload["stage"] = stage
    update_job(job_id, **payload)


class TranslatorCache:
    def __init__(self) -> None:
        self._cache: dict[str, tuple[MarianTokenizer, MarianMTModel]] = {}

    def _get(self, model_name: str) -> tuple[MarianTokenizer, MarianMTModel]:
        if model_name not in self._cache:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            self._cache[model_name] = (tokenizer, model)
        return self._cache[model_name]

    def translate(self, text: str, src_code: str, dst_code: str) -> str:
        if src_code == dst_code:
            return text

        direct = f"Helsinki-NLP/opus-mt-{src_code}-{dst_code}"
        try:
            return self._translate_with_model(direct, text)
        except Exception:
            if src_code != "en" and dst_code != "en":
                first = self._translate_with_model(f"Helsinki-NLP/opus-mt-{src_code}-en", text)
                return self._translate_with_model(f"Helsinki-NLP/opus-mt-en-{dst_code}", first)
            raise PipelineError(
                f"No encontré un modelo local de traducción para {src_code} → {dst_code}."
            )

    def _translate_with_model(self, model_name: str, text: str) -> str:
        try:
            tokenizer, model = self._get(model_name)
        except RepositoryNotFoundError as exc:
            raise PipelineError(f"No existe el modelo {model_name} en Hugging Face.") from exc

        chunks = split_text_for_translation(text, 400)
        outputs: list[str] = []
        for chunk in chunks:
            encoded = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            generated = model.generate(**encoded, max_new_tokens=512)
            outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
        return " ".join(x.strip() for x in outputs if x.strip())


TRANSLATOR_CACHE = TranslatorCache()


def run_job(job_id: str, input_filename: str, target_language: str, voice_name: str) -> None:
    work_dir = JOBS_DIR / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        input_path = UPLOAD_DIR / input_filename
        if not input_path.exists():
            raise PipelineError("No se encontró el video subido.")

        set_progress(job_id, 2, "Inicializando trabajo", "init")
        segments, source_language = transcribe_video(
            input_path,
            work_dir,
            lambda progress, message, stage=None: set_progress(job_id, progress, message, stage),
        )
        if not segments:
            raise PipelineError("No pude detectar voz útil en el video.")

        set_progress(job_id, 35, f"Traduciendo de {source_language} a {target_language}", "translate")
        translated_segments = translate_segments(
            segments,
            source_language,
            target_language,
            lambda progress, message, stage=None: set_progress(job_id, progress, message, stage),
        )

        effective_voice = voice_name.strip() or SUGGESTED_VOICES.get(target_language, DEFAULT_VOICE)
        set_progress(job_id, 60, f"Generando voz local con Piper: {effective_voice}", "tts")
        fitted_segments = synthesize_segments(
            translated_segments,
            effective_voice,
            work_dir,
            lambda progress, message, stage=None: set_progress(job_id, progress, message, stage),
        )

        set_progress(job_id, 92, "Construyendo pista doblada", "mix")
        silence_path = work_dir / "base_silence.wav"
        create_silence_from_video(input_path, silence_path)
        dubbed_audio_path = work_dir / "dubbed_track.wav"
        overlay_segments(silence_path, fitted_segments, dubbed_audio_path)

        set_progress(job_id, 97, "Uniendo audio y video", "mux")
        output_filename = f"{job_id}_doblado.mp4"
        output_path = OUTPUT_DIR / output_filename
        mux_video_with_audio(input_path, dubbed_audio_path, output_path)

        update_job(
            job_id,
            status="done",
            stage="done",
            progress=100,
            message="Listo para descargar",
            output_file=output_filename,
        )
    except Exception as exc:
        error_text = f"{exc}\n\n{traceback.format_exc()}"
        (work_dir / "error.log").write_text(error_text, encoding="utf-8")
        update_job(job_id, status="failed", stage="failed", progress=100, message=str(exc))


def resolve_ffmpeg() -> str:
    env_value = shutil.which("ffmpeg")
    if env_value:
        return env_value
    return get_ffmpeg_exe()


def run_cmd(command: list[str], timeout: int | None = None) -> None:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Error ejecutando FFmpeg").strip()
        raise PipelineError(detail)


def get_whisper_model() -> WhisperModel:
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        _WHISPER_MODEL = WhisperModel(
            WHISPER_MODEL,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
    return _WHISPER_MODEL


def extract_audio_for_whisper(video_path: Path, wav_path: Path) -> None:
    run_cmd([
        resolve_ffmpeg(),
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ], timeout=1800)


def get_audio_duration(audio_path: Path) -> float:
    try:
        return float(sf.info(str(audio_path)).duration)
    except Exception as exc:
        raise PipelineError(f"No pude medir la duración de {audio_path.name}") from exc


def create_audio_chunk(source_audio: Path, chunk_path: Path, start_seconds: float, duration_seconds: float) -> None:
    run_cmd([
        resolve_ffmpeg(),
        "-y",
        "-ss",
        f"{start_seconds:.3f}",
        "-t",
        f"{duration_seconds:.3f}",
        "-i",
        str(source_audio),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(chunk_path),
    ], timeout=1800)


def transcribe_video(
    video_path: Path,
    work_dir: Path,
    progress: ProgressCallback | None = None,
) -> tuple[list[dict[str, Any]], str]:
    audio_path = work_dir / "whisper_input.wav"
    chunks_dir = work_dir / "whisper_chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if progress:
        progress(3, "Extrayendo audio para transcripción", "transcribe")
    extract_audio_for_whisper(video_path, audio_path)

    total_duration = get_audio_duration(audio_path)
    if total_duration <= 0:
        raise PipelineError("No pude medir la duración del audio para transcripción.")

    if progress:
        progress(5, "Cargando modelo Whisper", "transcribe")
    model = get_whisper_model()

    segments: list[dict[str, Any]] = []
    detected_language = ""
    chunk_count = max(1, math.ceil(total_duration / TRANSCRIBE_CHUNK_SECONDS))

    for chunk_index in range(chunk_count):
        chunk_start = chunk_index * TRANSCRIBE_CHUNK_SECONDS
        chunk_duration = min(TRANSCRIBE_CHUNK_SECONDS, total_duration - chunk_start)
        if chunk_duration <= 0:
            continue

        chunk_path = chunks_dir / f"chunk_{chunk_index:04d}.wav"
        create_audio_chunk(audio_path, chunk_path, chunk_start, chunk_duration)

        if progress:
            base_ratio = min(1.0, max(0.0, chunk_start / total_duration))
            percent = 5 + int(base_ratio * 30)
            progress(percent, f"Transcribiendo audio por bloques ({chunk_index + 1}/{chunk_count})", "transcribe")

        kwargs: dict[str, Any] = {
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0,
            "vad_filter": False,
            "condition_on_previous_text": False,
        }
        if detected_language:
            kwargs["language"] = detected_language

        logger.info(
            "Transcribiendo bloque %s/%s desde %.2fs durante %.2fs",
            chunk_index + 1,
            chunk_count,
            chunk_start,
            chunk_duration,
        )

        segments_iter, info = model.transcribe(str(chunk_path), **kwargs)

        if not detected_language:
            detected_language = str(getattr(info, "language", "") or "")

        for seg in segments_iter:
            text = seg.text.strip()
            if not text:
                continue

            absolute_start = float(seg.start) + float(chunk_start)
            absolute_end = min(total_duration, float(seg.end) + float(chunk_start))
            segments.append({
                "start": absolute_start,
                "end": absolute_end,
                "source_text": text,
            })

            if progress:
                ratio = min(1.0, max(0.0, absolute_end / total_duration))
                percent = 5 + int(ratio * 30)
                progress(percent, f"Transcribiendo el audio ({int(ratio * 100)}%)", "transcribe")

    if progress:
        progress(35, "Transcripción completada", "transcribe")

    return segments, (detected_language or "auto")


def translate_segments(
    segments: list[dict[str, Any]],
    src_code: str,
    dst_code: str,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    translated: list[dict[str, Any]] = []
    total = max(1, len(segments))
    for index, seg in enumerate(segments, start=1):
        out = seg.copy()
        out["translated_text"] = TRANSLATOR_CACHE.translate(seg["source_text"], src_code, dst_code)
        translated.append(out)

        if progress:
            ratio = index / total
            percent = 35 + int(ratio * 25)
            progress(
                percent,
                f"Traduciendo segmentos ({index}/{total})",
                "translate",
            )
    return translated


def voice_model_paths(voice_name: str) -> tuple[Path, Path]:
    model_path = VOICES_DIR / f"{voice_name}.onnx"
    config_path = VOICES_DIR / f"{voice_name}.onnx.json"
    if not model_path.exists() or not config_path.exists():
        raise PipelineError(
            "No encontré la voz de Piper. Descargue la voz con este comando: "
            f"python -m piper.download_voices {voice_name} --data-dir data/voices"
        )
    return model_path, config_path


def synthesize_segments(
    segments: list[dict[str, Any]],
    voice_name: str,
    work_dir: Path,
    progress: ProgressCallback | None = None,
) -> list[dict[str, Any]]:
    model_path, _ = voice_model_paths(voice_name)
    voice = PiperVoice.load(str(model_path))
    raw_dir = work_dir / "tts_raw"
    fit_dir = work_dir / "tts_fit"
    raw_dir.mkdir(parents=True, exist_ok=True)
    fit_dir.mkdir(parents=True, exist_ok=True)

    synthesized: list[dict[str, Any]] = []
    total = max(1, len(segments))
    for index, seg in enumerate(segments, start=1):
        raw_wav = raw_dir / f"seg_{index - 1:04d}.wav"
        with wave.open(str(raw_wav), "wb") as wav_file:
            voice.synthesize_wav(seg["translated_text"], wav_file)

        fitted_wav = fit_dir / f"seg_{index - 1:04d}.wav"
        target_duration = max(0.8, float(seg["end"] - seg["start"]))
        fit_audio_duration(raw_wav, fitted_wav, target_duration)
        synthesized.append({"start": seg["start"], "end": seg["end"], "path": fitted_wav})

        if progress:
            ratio = index / total
            percent = 60 + int(ratio * 30)
            progress(
                percent,
                f"Generando voz ({index}/{total})",
                "tts",
            )
    return synthesized


def fit_audio_duration(input_wav: Path, output_wav: Path, target_duration: float) -> None:
    data, sr = sf.read(str(input_wav), dtype="float32")
    current_duration = len(data) / float(sr)
    if current_duration <= 0:
        raise PipelineError(f"No pude medir la duración de {input_wav.name}")

    tempo_factor = current_duration / target_duration
    filters = build_atempo_chain(tempo_factor)
    filters.append(f"aresample={SAMPLE_RATE}")
    filter_str = ",".join(filters)

    run_cmd([
        resolve_ffmpeg(),
        "-y",
        "-i",
        str(input_wav),
        "-af",
        filter_str,
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-t",
        f"{target_duration:.3f}",
        str(output_wav),
    ], timeout=1800)


def build_atempo_chain(tempo_factor: float) -> list[str]:
    if tempo_factor <= 0:
        return ["atempo=1.0"]

    filters: list[str] = []
    remaining = tempo_factor
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5

    if math.isclose(remaining, 1.0, rel_tol=1e-3):
        if not filters:
            filters.append("atempo=1.0")
    else:
        filters.append(f"atempo={remaining:.6f}")
    return filters


def create_silence_from_video(video_path: Path, silence_path: Path) -> None:
    run_cmd([
        resolve_ffmpeg(),
        "-y",
        "-i",
        str(video_path),
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r={SAMPLE_RATE}:cl=mono",
        "-map",
        "1:a:0",
        "-c:a",
        "pcm_s16le",
        "-t",
        f"{get_video_duration(video_path)}",
        str(silence_path),
    ], timeout=1800)


def get_video_duration(video_path: Path) -> float:
    result = subprocess.run(
        [resolve_ffmpeg(), "-i", str(video_path)],
        stderr=subprocess.PIPE,
        text=True
    )
    duration = None
    for line in result.stderr.split("\n"):
        if "Duration" in line:
            parts = line.split(",")[0]
            duration_str = parts.split("Duration:")[1].strip()
            h, m, s = duration_str.split(":")
            duration = int(h) * 3600 + int(m) * 60 + float(s)
            break
    return duration or 0


def overlay_segments(base_wav: Path, segments: list[dict[str, Any]], output_wav: Path) -> None:
    base, sr = sf.read(str(base_wav), dtype="float32")
    if base.ndim > 1:
        base = base[:, 0]
    mixed = base.copy()

    for seg in segments:
        clip, clip_sr = sf.read(str(seg["path"]), dtype="float32")
        if clip.ndim > 1:
            clip = clip[:, 0]
        if clip_sr != sr:
            raise PipelineError("Las frecuencias de muestreo no coinciden en la mezcla final.")
        start_index = max(0, int(float(seg["start"]) * sr))
        end_index = min(len(mixed), start_index + len(clip))
        mixed[start_index:end_index] += clip[: end_index - start_index]

    mixed = np.clip(mixed, -1.0, 1.0)
    sf.write(str(output_wav), mixed, sr, subtype="PCM_16")


def mux_video_with_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
    run_cmd([
        resolve_ffmpeg(),
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(output_path),
    ], timeout=1800)


def split_text_for_translation(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    words = text.split()
    pieces: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        if current and current_len + 1 + len(word) > max_chars:
            pieces.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += len(word) + (1 if current_len else 0)
    if current:
        pieces.append(" ".join(current))
    return pieces


# from __future__ import annotations

# import math
# import shutil
# import subprocess
# import traceback
# import wave
# from pathlib import Path
# from typing import Any, Callable

# import numpy as np
# import soundfile as sf
# from faster_whisper import WhisperModel
# from imageio_ffmpeg import get_ffmpeg_exe
# from huggingface_hub.errors import RepositoryNotFoundError
# from piper import PiperVoice
# from transformers import MarianMTModel, MarianTokenizer

# from .config import (
#     DEFAULT_VOICE,
#     JOBS_DIR,
#     OUTPUT_DIR,
#     SAMPLE_RATE,
#     UPLOAD_DIR,
#     VOICES_DIR,
#     WHISPER_COMPUTE_TYPE,
#     WHISPER_DEVICE,
#     WHISPER_MODEL,
# )
# from .db import update_job
# from .lang import SUGGESTED_VOICES


# class PipelineError(Exception):
#     pass


# ProgressCallback = Callable[[int, str, str | None], None]


# def set_progress(job_id: str, progress: int, message: str, stage: str | None = None, status: str = "processing") -> None:
#     payload: dict[str, Any] = {
#         "status": status,
#         "progress": progress,
#         "message": message,
#     }
#     if stage is not None:
#         payload["stage"] = stage
#     update_job(job_id, **payload)


# class TranslatorCache:
#     def __init__(self) -> None:
#         self._cache: dict[str, tuple[MarianTokenizer, MarianMTModel]] = {}

#     def _get(self, model_name: str) -> tuple[MarianTokenizer, MarianMTModel]:
#         if model_name not in self._cache:
#             tokenizer = MarianTokenizer.from_pretrained(model_name)
#             model = MarianMTModel.from_pretrained(model_name)
#             self._cache[model_name] = (tokenizer, model)
#         return self._cache[model_name]

#     def translate(self, text: str, src_code: str, dst_code: str) -> str:
#         if src_code == dst_code:
#             return text

#         direct = f"Helsinki-NLP/opus-mt-{src_code}-{dst_code}"
#         try:
#             return self._translate_with_model(direct, text)
#         except Exception:
#             if src_code != "en" and dst_code != "en":
#                 first = self._translate_with_model(f"Helsinki-NLP/opus-mt-{src_code}-en", text)
#                 return self._translate_with_model(f"Helsinki-NLP/opus-mt-en-{dst_code}", first)
#             raise PipelineError(
#                 f"No encontré un modelo local de traducción para {src_code} → {dst_code}."
#             )

#     def _translate_with_model(self, model_name: str, text: str) -> str:
#         try:
#             tokenizer, model = self._get(model_name)
#         except RepositoryNotFoundError as exc:
#             raise PipelineError(f"No existe el modelo {model_name} en Hugging Face.") from exc

#         chunks = split_text_for_translation(text, 400)
#         outputs: list[str] = []
#         for chunk in chunks:
#             encoded = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
#             generated = model.generate(**encoded, max_new_tokens=512)
#             outputs.append(tokenizer.decode(generated[0], skip_special_tokens=True))
#         return " ".join(x.strip() for x in outputs if x.strip())


# TRANSLATOR_CACHE = TranslatorCache()


# def run_job(job_id: str, input_filename: str, target_language: str, voice_name: str) -> None:
#     work_dir = JOBS_DIR / job_id
#     work_dir.mkdir(parents=True, exist_ok=True)
#     try:
#         input_path = UPLOAD_DIR / input_filename
#         if not input_path.exists():
#             raise PipelineError("No se encontró el video subido.")

#         set_progress(job_id, 2, "Inicializando trabajo", "init")
#         segments, source_language = transcribe_video(
#             input_path,
#             lambda progress, message, stage=None: set_progress(job_id, progress, message, stage),
#         )
#         if not segments:
#             raise PipelineError("No pude detectar voz útil en el video.")

#         set_progress(job_id, 35, f"Traduciendo de {source_language} a {target_language}", "translate")
#         translated_segments = translate_segments(
#             segments,
#             source_language,
#             target_language,
#             lambda progress, message, stage=None: set_progress(job_id, progress, message, stage),
#         )

#         effective_voice = voice_name.strip() or SUGGESTED_VOICES.get(target_language, DEFAULT_VOICE)
#         set_progress(job_id, 60, f"Generando voz local con Piper: {effective_voice}", "tts")
#         fitted_segments = synthesize_segments(
#             translated_segments,
#             effective_voice,
#             work_dir,
#             lambda progress, message, stage=None: set_progress(job_id, progress, message, stage),
#         )

#         set_progress(job_id, 92, "Construyendo pista doblada", "mix")
#         silence_path = work_dir / "base_silence.wav"
#         create_silence_from_video(input_path, silence_path)
#         dubbed_audio_path = work_dir / "dubbed_track.wav"
#         overlay_segments(silence_path, fitted_segments, dubbed_audio_path)

#         set_progress(job_id, 97, "Uniendo audio y video", "mux")
#         output_filename = f"{job_id}_doblado.mp4"
#         output_path = OUTPUT_DIR / output_filename
#         mux_video_with_audio(input_path, dubbed_audio_path, output_path)

#         update_job(
#             job_id,
#             status="done",
#             stage="done",
#             progress=100,
#             message="Listo para descargar",
#             output_file=output_filename,
#         )
#     except Exception as exc:
#         error_text = f"{exc}\n\n{traceback.format_exc()}"
#         (work_dir / "error.log").write_text(error_text, encoding="utf-8")
#         update_job(job_id, status="failed", stage="failed", progress=100, message=str(exc))


# def resolve_ffmpeg() -> str:
#     env_value = shutil.which("ffmpeg")
#     if env_value:
#         return env_value
#     return get_ffmpeg_exe()


# def run_cmd(command: list[str]) -> None:
#     result = subprocess.run(command, capture_output=True, text=True)
#     if result.returncode != 0:
#         detail = (result.stderr or result.stdout or "Error ejecutando FFmpeg").strip()
#         raise PipelineError(detail)


# def transcribe_video(video_path: Path, progress: ProgressCallback | None = None) -> tuple[list[dict[str, Any]], str]:
#     if progress:
#         progress(3, "Cargando modelo Whisper", "transcribe")
#     model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE_TYPE)

#     if progress:
#         progress(5, "Transcribiendo el video", "transcribe")
#     segments_iter, info = model.transcribe(str(video_path), vad_filter=True, beam_size=5)

#     total_duration = float(getattr(info, "duration", 0.0) or 0.0)
#     segments: list[dict[str, Any]] = []
#     for seg in segments_iter:
#         text = seg.text.strip()
#         if not text:
#             continue
#         segments.append({
#             "start": float(seg.start),
#             "end": float(seg.end),
#             "source_text": text,
#         })

#         if progress and total_duration > 0:
#             ratio = min(1.0, max(0.0, float(seg.end) / total_duration))
#             percent = 5 + int(ratio * 30)
#             progress(percent, f"Transcribiendo el video ({int(ratio * 100)}%)", "transcribe")

#     if progress:
#         progress(35, "Transcripción completada", "transcribe")
#     return segments, str(info.language)


# def translate_segments(
#     segments: list[dict[str, Any]],
#     src_code: str,
#     dst_code: str,
#     progress: ProgressCallback | None = None,
# ) -> list[dict[str, Any]]:
#     translated: list[dict[str, Any]] = []
#     total = max(1, len(segments))
#     for index, seg in enumerate(segments, start=1):
#         out = seg.copy()
#         out["translated_text"] = TRANSLATOR_CACHE.translate(seg["source_text"], src_code, dst_code)
#         translated.append(out)

#         if progress:
#             ratio = index / total
#             percent = 35 + int(ratio * 25)
#             progress(
#                 percent,
#                 f"Traduciendo segmentos ({index}/{total})",
#                 "translate",
#             )
#     return translated


# def voice_model_paths(voice_name: str) -> tuple[Path, Path]:
#     model_path = VOICES_DIR / f"{voice_name}.onnx"
#     config_path = VOICES_DIR / f"{voice_name}.onnx.json"
#     if not model_path.exists() or not config_path.exists():
#         raise PipelineError(
#             "No encontré la voz de Piper. Descargue la voz con este comando: "
#             f"python -m piper.download_voices {voice_name} --data-dir data/voices"
#         )
#     return model_path, config_path


# def synthesize_segments(
#     segments: list[dict[str, Any]],
#     voice_name: str,
#     work_dir: Path,
#     progress: ProgressCallback | None = None,
# ) -> list[dict[str, Any]]:
#     model_path, _ = voice_model_paths(voice_name)
#     voice = PiperVoice.load(str(model_path))
#     raw_dir = work_dir / "tts_raw"
#     fit_dir = work_dir / "tts_fit"
#     raw_dir.mkdir(parents=True, exist_ok=True)
#     fit_dir.mkdir(parents=True, exist_ok=True)

#     synthesized: list[dict[str, Any]] = []
#     total = max(1, len(segments))
#     for index, seg in enumerate(segments, start=1):
#         raw_wav = raw_dir / f"seg_{index - 1:04d}.wav"
#         with wave.open(str(raw_wav), "wb") as wav_file:
#             voice.synthesize_wav(seg["translated_text"], wav_file)

#         fitted_wav = fit_dir / f"seg_{index - 1:04d}.wav"
#         target_duration = max(0.8, float(seg["end"] - seg["start"]))
#         fit_audio_duration(raw_wav, fitted_wav, target_duration)
#         synthesized.append({"start": seg["start"], "end": seg["end"], "path": fitted_wav})

#         if progress:
#             ratio = index / total
#             percent = 60 + int(ratio * 30)
#             progress(
#                 percent,
#                 f"Generando voz ({index}/{total})",
#                 "tts",
#             )
#     return synthesized


# def fit_audio_duration(input_wav: Path, output_wav: Path, target_duration: float) -> None:
#     data, sr = sf.read(str(input_wav), dtype="float32")
#     current_duration = len(data) / float(sr)
#     if current_duration <= 0:
#         raise PipelineError(f"No pude medir la duración de {input_wav.name}")

#     tempo_factor = current_duration / target_duration
#     filters = build_atempo_chain(tempo_factor)
#     filters.append(f"aresample={SAMPLE_RATE}")
#     filter_str = ",".join(filters)

#     run_cmd([
#         resolve_ffmpeg(),
#         "-y",
#         "-i",
#         str(input_wav),
#         "-af",
#         filter_str,
#         "-ac",
#         "1",
#         "-ar",
#         str(SAMPLE_RATE),
#         "-t",
#         f"{target_duration:.3f}",
#         str(output_wav),
#     ])


# def build_atempo_chain(tempo_factor: float) -> list[str]:
#     if tempo_factor <= 0:
#         return ["atempo=1.0"]

#     filters: list[str] = []
#     remaining = tempo_factor
#     while remaining > 2.0:
#         filters.append("atempo=2.0")
#         remaining /= 2.0
#     while remaining < 0.5:
#         filters.append("atempo=0.5")
#         remaining /= 0.5

#     if math.isclose(remaining, 1.0, rel_tol=1e-3):
#         if not filters:
#             filters.append("atempo=1.0")
#     else:
#         filters.append(f"atempo={remaining:.6f}")
#     return filters


# def create_silence_from_video(video_path: Path, silence_path: Path) -> None:
#     run_cmd([
#         resolve_ffmpeg(),
#         "-y",
#         "-i",
#         str(video_path),
#         "-f",
#         "lavfi",
#         "-i",
#         f"anullsrc=r={SAMPLE_RATE}:cl=mono",
#         "-map",
#         "1:a:0",
#         "-c:a",
#         "pcm_s16le",
#         "-t",
#         f"{get_video_duration(video_path)}",
#         str(silence_path),
#     ])


# def get_video_duration(video_path: Path) -> float:
#     result = subprocess.run(
#         [resolve_ffmpeg(), "-i", str(video_path)],
#         stderr=subprocess.PIPE,
#         text=True
#     )
#     duration = None
#     for line in result.stderr.split("\n"):
#         if "Duration" in line:
#             parts = line.split(",")[0]
#             duration_str = parts.split("Duration:")[1].strip()
#             h, m, s = duration_str.split(":")
#             duration = int(h) * 3600 + int(m) * 60 + float(s)
#             break
#     return duration or 0


# def overlay_segments(base_wav: Path, segments: list[dict[str, Any]], output_wav: Path) -> None:
#     base, sr = sf.read(str(base_wav), dtype="float32")
#     if base.ndim > 1:
#         base = base[:, 0]
#     mixed = base.copy()

#     for seg in segments:
#         clip, clip_sr = sf.read(str(seg["path"]), dtype="float32")
#         if clip.ndim > 1:
#             clip = clip[:, 0]
#         if clip_sr != sr:
#             raise PipelineError("Las frecuencias de muestreo no coinciden en la mezcla final.")
#         start_index = max(0, int(float(seg["start"]) * sr))
#         end_index = min(len(mixed), start_index + len(clip))
#         mixed[start_index:end_index] += clip[: end_index - start_index]

#     mixed = np.clip(mixed, -1.0, 1.0)
#     sf.write(str(output_wav), mixed, sr, subtype="PCM_16")


# def mux_video_with_audio(video_path: Path, audio_path: Path, output_path: Path) -> None:
#     run_cmd([
#         resolve_ffmpeg(),
#         "-y",
#         "-i",
#         str(video_path),
#         "-i",
#         str(audio_path),
#         "-map",
#         "0:v:0",
#         "-map",
#         "1:a:0",
#         "-c:v",
#         "copy",
#         "-c:a",
#         "aac",
#         "-shortest",
#         str(output_path),
#     ])


# def split_text_for_translation(text: str, max_chars: int) -> list[str]:
#     if len(text) <= max_chars:
#         return [text]
#     words = text.split()
#     pieces: list[str] = []
#     current: list[str] = []
#     current_len = 0
#     for word in words:
#         if current and current_len + 1 + len(word) > max_chars:
#             pieces.append(" ".join(current))
#             current = [word]
#             current_len = len(word)
#         else:
#             current.append(word)
#             current_len += len(word) + (1 if current_len else 0)
#     if current:
#         pieces.append(" ".join(current))
#     return pieces