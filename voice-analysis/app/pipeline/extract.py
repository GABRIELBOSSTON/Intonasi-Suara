"""Decode original channels before producing a 16 kHz mono copy."""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import soundfile as sf

from .common import AnalysisError
from .quality import inspect_original


def _binary(name, label):
    found = shutil.which(name)
    if not found:
        raise AnalysisError(f"{label}_not_found", f"{label} tidak ditemukan; periksa PATH atau config.")
    return found


def _execute(args, cfg):
    try:
        return subprocess.run(args, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=cfg.FFMPEG_TIMEOUT_SECONDS, check=True,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    except subprocess.TimeoutExpired as exc:
        raise AnalysisError("decode_timeout", "Probe/decode audio melewati timeout.") from exc
    except subprocess.CalledProcessError as exc:
        raise AnalysisError("decode_failed", "ffmpeg/ffprobe gagal membaca audio; periksa format dan integritas file.") from exc


def extract_audio(source, temp_dir, cfg):
    path = Path(source).resolve()
    if not path.is_file():
        raise AnalysisError("file_not_found", "File input tidak ditemukan.")
    if path.stat().st_size > cfg.MAX_INPUT_MB * 1024 ** 2:
        raise AnalysisError("file_too_large", "Ukuran input melebihi batas konfigurasi.")
    ffmpeg = _binary(cfg.FFMPEG, "ffmpeg")
    ffprobe = _binary(cfg.FFPROBE, "ffprobe")
    metadata = json.loads(_execute([ffprobe, "-v", "error", "-select_streams", "a:0",
        "-show_streams", "-show_format", "-of", "json", str(path)], cfg))
    streams = metadata.get("streams", [])
    if not streams:
        raise AnalysisError("no_audio_stream", "File tidak memiliki stream audio.")
    stream = streams[0]
    declared = stream.get("duration", metadata.get("format", {}).get("duration"))
    try:
        declared = float(declared)
    except (TypeError, ValueError):
        declared = None
    if declared is not None and declared > cfg.MAX_DURATION_SECONDS:
        raise AnalysisError("audio_too_long", "Durasi audio melebihi batas konfigurasi.")
    try:
        original_sr = int(stream["sample_rate"])
        channels = int(stream["channels"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AnalysisError("invalid_audio_metadata", "Metadata sample rate/channel tidak valid.") from exc
    if original_sr <= 0 or channels <= 0:
        raise AnalysisError("invalid_audio_metadata", "Sample rate/channel tidak valid.")
    decode_limit = min(cfg.MAX_DURATION_SECONDS + 1,
                       cfg.MAX_DECODED_SAMPLES / (original_sr * channels) + 1)
    original = Path(temp_dir) / "original.wav"
    base = [ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    _execute(base + ["-i", str(path), "-map", "0:a:0", "-vn", "-t", str(decode_limit),
                     "-c:a", "pcm_f32le", str(original)], cfg)
    header = sf.info(original)
    if header.frames * header.channels > cfg.MAX_DECODED_SAMPLES:
        raise AnalysisError("decoded_audio_too_large", "PCM audio melebihi batas memori konfigurasi.")
    if header.duration > cfg.MAX_DURATION_SECONDS:
        raise AnalysisError("audio_too_long", "Durasi hasil decode melebihi batas.")
    if header.duration < cfg.MIN_AUDIO_SECONDS:
        raise AnalysisError("audio_too_short", "Audio terlalu pendek untuk analisis.")
    raw, original_sr = sf.read(original, dtype="float32", always_2d=True)
    original_qc = inspect_original(raw, cfg)
    selected = original_qc["selected_channel"]
    mono = np.mean(raw, axis=1, dtype=np.float64) if selected is None else raw[:, selected]
    mono_native = Path(temp_dir) / "mono_native.wav"
    sf.write(mono_native, mono, original_sr, subtype="FLOAT")
    target = Path(temp_dir) / "mono_16k.wav"
    _execute(base + ["-i", str(mono_native), "-ac", "1", "-ar", str(cfg.SAMPLE_RATE),
                     "-c:a", "pcm_f32le", str(target)], cfg)
    audio, sr = sf.read(target, dtype="float32")
    if sr != cfg.SAMPLE_RATE or audio.ndim != 1 or not np.isfinite(audio).all():
        raise AnalysisError("invalid_decoded_audio", "Hasil decode audio tidak valid.")
    info = {"original_channels": channels, "original_sample_rate": original_sr,
            "original_channel_layout": stream.get("channel_layout"),
            "original_codec": stream.get("codec_name"), "duration_seconds": len(audio) / sr,
            "analysis_sample_rate": sr, "ffmpeg_version": _execute([ffmpeg, "-version"], cfg).splitlines()[0],
            **original_qc}
    return {"audio": audio, "sample_rate": sr, "wav_path": str(target), "audio_info": info}
