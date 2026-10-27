"""CPU Whisper inference with revision-pinned weights and local caching."""

from functools import lru_cache
import math
from pathlib import Path
import re
import numpy as np

from .common import AnalysisError, file_sha256


def _load_model(repo, revision, local_path, cache_dir, offline, threads, compute_type):
    if local_path:
        path = Path(local_path).resolve()
        if not path.is_dir():
            raise AnalysisError("asr_model_not_found", "Direktori model ASR lokal tidak ditemukan.")
        local_path = str(path)
        # Replacing local weights must invalidate the cached model/provenance.
        identity = tuple((item.name, item.stat().st_size, item.stat().st_mtime_ns)
                         for item in sorted(path.iterdir()) if item.is_file())
    else:
        identity = ()
    return _cached_model(repo, revision, local_path, cache_dir, offline, threads, compute_type, identity)


@lru_cache(maxsize=2)
def _cached_model(repo, revision, local_path, cache_dir, offline, threads, compute_type, identity):
    from faster_whisper import WhisperModel
    from huggingface_hub import snapshot_download
    if local_path:
        path = Path(local_path)
    else:
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise AnalysisError("asr_revision_unpinned", "ASR_REVISION harus commit SHA 40 karakter.")
        path = Path(snapshot_download(repo_id=repo, revision=revision, cache_dir=cache_dir,
            local_files_only=offline, allow_patterns=["config.json", "preprocessor_config.json",
            "model.bin", "tokenizer.json", "vocabulary.*"]))
    hashes = {item.name: file_sha256(item) for item in sorted(path.iterdir()) if item.is_file()}
    model = WhisperModel(str(path), device="cpu", compute_type=compute_type, cpu_threads=threads,
                         num_workers=1, local_files_only=True)
    return model, {"repo": repo if not local_path else None,
                   "revision": revision if not local_path else None,
                   "source": "local" if local_path else "huggingface_snapshot",
                   "file_sha256": hashes, "device": "cpu", "compute_type": compute_type}


def transcribe_audio(audio, sample_rate, vad, cfg):
    if not vad["segments"]:
        raise AnalysisError("asr_no_speech", "Tidak ada speech untuk transkripsi.")
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000 or audio.ndim != 1 or not audio.size or not np.isfinite(audio).all():
        raise AnalysisError("asr_invalid_audio", "ASR memerlukan audio mono 16 kHz yang finite dan tidak kosong.")
    duration = len(audio) / sample_rate
    bounds = vad["segments"]
    previous_end = 0.0
    for start, end in bounds:
        if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start < end <= duration or start < previous_end:
            raise AnalysisError("asr_invalid_speech_bounds", "Batas speech tidak valid untuk audio ASR.")
        previous_end = end
    if not cfg.ASR_LOCAL_PATH and cfg.ASR_REPO != f"Systran/faster-whisper-{cfg.ASR_MODEL_SIZE}":
        raise AnalysisError("asr_config_mismatch", "Ukuran model dan ASR_REPO tidak cocok.")
    model, provenance = _load_model(cfg.ASR_REPO, cfg.ASR_REVISION, cfg.ASR_LOCAL_PATH,
        cfg.ASR_CACHE_DIR, cfg.ASR_LOCAL_FILES_ONLY, cfg.ASR_CPU_THREADS, cfg.ASR_COMPUTE_TYPE)
    begin = round(vad["segments"][0][0] * sample_rate)
    finish = round(vad["segments"][-1][1] * sample_rate)
    offset = begin / sample_rate
    segments, info = model.transcribe(audio[begin:finish], language="id", word_timestamps=True,
        beam_size=cfg.ASR_BEAM_SIZE, vad_filter=False,
        condition_on_previous_text=cfg.ASR_CONDITION_ON_PREVIOUS_TEXT)
    # Decode runs lazily; consume inside the protected pipeline stage.
    segments = list(segments)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    if not text:
        raise AnalysisError("asr_empty_transcript", "ASR tidak menghasilkan transkrip.")
    words, warnings = [], []
    previous_start = 0.0
    cropped_duration = (finish - begin) / sample_rate
    for segment in segments:
        for word in segment.words or []:
            start, end = word.start, word.end
            valid = all(isinstance(value, (int, float)) and math.isfinite(value) for value in (start, end))
            valid = valid and 0 <= start <= end <= cropped_duration and start >= previous_start
            if valid:
                previous_start = start
            else:
                warnings.append("asr_word_timestamps_invalid")
            probability = word.probability
            if not isinstance(probability, (int, float)) or not math.isfinite(probability) or not 0 <= probability <= 1:
                probability = None
                warnings.append("asr_word_probability_invalid")
            words.append({"word": word.word.strip(), "start": start + offset if valid else None,
                          "end": end + offset if valid else None, "probability": probability})
    if not words:
        warnings.append("asr_word_timestamps_unavailable")
    return {"text": text, "words": words, "language": info.language,
            "span_start": offset, "span_end": finish / sample_rate,
            "warnings": sorted(set(warnings)),
            "model": {**provenance, "size": cfg.ASR_MODEL_SIZE}}
