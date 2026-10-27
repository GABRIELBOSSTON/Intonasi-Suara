"""Speech segmentation and internal pause statistics."""

from functools import lru_cache
from importlib import resources
import numpy as np

from .common import AnalysisError, file_sha256


@lru_cache(maxsize=1)
def _model():
    from silero_vad import load_silero_vad
    return load_silero_vad(onnx=True)


def summarize_segments(segments, duration, cfg):
    if not np.isfinite(duration) or duration < 0:
        raise AnalysisError("invalid_vad_duration", "Durasi audio VAD harus finite dan tidak negatif.")
    normalized = []
    try:
        for start, end in segments:
            start, end = float(start), float(end)
            if not np.isfinite(start) or not np.isfinite(end) or end < start:
                raise ValueError("invalid interval")
            normalized.append((start, end))
    except (TypeError, ValueError) as exc:
        raise AnalysisError("invalid_vad_segments", "Interval speech VAD tidak valid.") from exc
    cleaned = []
    for start, end in sorted(normalized):
        start, end = max(0.0, start), min(duration, end)
        if end <= start:
            continue
        if cleaned and start <= cleaned[-1][1]:
            cleaned[-1][1] = max(end, cleaned[-1][1])
        else:
            cleaned.append([start, end])
    if not cleaned:
        return {"segments": [], "phrases": [], "pauses": [], "valid_speech_seconds": 0.0,
                "speech_span_seconds": None, "articulation_seconds": None, "pause_ratio": None,
                "pause_median_seconds": None, "pause_p90_seconds": None, "long_pause_per_min": None}
    span = cleaned[-1][1] - cleaned[0][0]
    pauses, phrases = [], [cleaned[0].copy()]
    for previous, current in zip(cleaned, cleaned[1:]):
        gap = current[0] - previous[1]
        if gap + 1e-9 >= cfg.PAUSE_MIN_SECONDS:
            pauses.append([previous[1], current[0]])
            phrases.append(current.copy())
        else:
            phrases[-1][1] = current[1]
    lengths = [end - start for start, end in pauses]
    total_pause = sum(lengths)
    return {"segments": cleaned, "phrases": phrases, "pauses": pauses,
            "valid_speech_seconds": sum(end - start for start, end in cleaned),
            "speech_span_seconds": span, "articulation_seconds": span - total_pause,
            "pause_ratio": total_pause / span,
            "pause_median_seconds": float(np.median(lengths)) if lengths else None,
            "pause_p90_seconds": float(np.percentile(lengths, 90)) if lengths else None,
            "long_pause_per_min": sum(length > cfg.LONG_PAUSE_SECONDS for length in lengths) * 60 / span}


def analyze_vad(audio, sample_rate, cfg):
    import torch
    from silero_vad import get_speech_timestamps
    model = _model()
    model.reset_states()
    try:
        timestamps = get_speech_timestamps(torch.from_numpy(np.asarray(audio, dtype=np.float32)),
            model, sampling_rate=sample_rate, threshold=cfg.VAD_THRESHOLD,
            min_speech_duration_ms=cfg.VAD_MIN_SPEECH_MS,
            min_silence_duration_ms=cfg.VAD_MIN_SILENCE_MS,
            speech_pad_ms=cfg.VAD_SPEECH_PAD_MS, return_seconds=False)
    finally:
        model.reset_states()
    segments = [[item["start"] / sample_rate, item["end"] / sample_rate] for item in timestamps]
    result = summarize_segments(segments, len(audio) / sample_rate, cfg)
    model_path = resources.files("silero_vad.data").joinpath("silero_vad.onnx")
    with resources.as_file(model_path) as path:
        result["model_sha256"] = file_sha256(path)
    result["model_id"] = "silero-vad/packaged-onnx"
    return result
