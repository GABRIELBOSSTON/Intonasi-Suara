"""Stage isolation and the versioned JSON analysis contract."""

from importlib import import_module, metadata
from functools import lru_cache
import logging
import math
from pathlib import Path
import platform
import tempfile
import time

from app.config import Config
from .common import AnalysisError
from .scoring import score

LOG = logging.getLogger("voice_analysis")
STAGES = ("extract", "vad", "quality", "pitch", "transcribe", "syllable", "tempo")
NUMERIC_FIELDS = ("valid_speech_seconds", "f0_valid_coverage", "pitch_range_p10_p90_st",
    "pitch_sd_st_phrase_median", "f0_median_hz", "f0_valid_seconds", "octave_jump_count",
    "octave_jump_per_min", "articulation_rate_syl_per_s", "speech_rate_syl_per_s",
    "articulation_rate_wpm", "speech_rate_wpm", "pause_ratio", "pause_median_seconds",
    "pause_p90_seconds", "long_pause_per_min", "speech_span_seconds", "articulation_seconds",
    "word_count", "syllable_count", "energy_end_vs_start_db")


@lru_cache(maxsize=1)
def _versions():
    names = ("numpy", "soundfile", "praat-parselmouth", "silero-vad", "torch", "torchaudio",
             "onnxruntime", "faster-whisper", "ctranslate2", "huggingface-hub")
    result = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _stage(result, name, function, *args):
    start = time.perf_counter()
    LOG.info("stage=%s start", name)
    try:
        module = import_module(f"app.pipeline.{name}")
        output = getattr(module, function)(*args)
        if not isinstance(output, dict):
            raise AnalysisError(f"{name}_invalid_output", f"Modul {name} tidak menghasilkan objek hasil yang valid.")
        result["module_status"][name] = "ok"
        return output
    except Exception as exc:
        code = exc.code if isinstance(exc, AnalysisError) else f"{name}_failed"
        message = str(exc) if isinstance(exc, AnalysisError) else f"Modul {name} gagal ({type(exc).__name__}); periksa instalasi, konfigurasi, dan input."
        result["errors"].append({"module": name, "code": code, "message": message,
                                 "exception_type": type(exc).__name__})
        result["module_status"][name] = "failed"
        LOG.error("stage=%s code=%s type=%s", name, code, type(exc).__name__)
        return None
    finally:
        elapsed = time.perf_counter() - start
        result["timings_seconds"][name] = round(elapsed, 4)
        LOG.info("stage=%s elapsed_seconds=%.3f", name, elapsed)


def _strict_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _strict_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strict_values(item) for item in value]
    return value


def analyze_file(path, cfg=None):
    cfg = cfg or Config()
    started = time.perf_counter()
    result = {field: None for field in NUMERIC_FIELDS}
    result.update({"analysis_version": cfg.ANALYSIS_VERSION, "mode": cfg.MODE,
        "shadow_mode": True, "input_file": str(Path(path)), "status": "perlu_peninjauan",
        "abstain_reason": None, "intonation_item_passed": None, "descriptors": [],
        "audio_quality": {"status": None, "snr_est_db": None, "clipping_pct": None,
                          "valid_speech_seconds": None, "snr_unavailable_reason": None, "warnings": []},
        "audio_info": None, "evidence_timestamps": [], "vad": None, "pitch": None,
        "transcription": None, "syllables": None, "errors": [], "warnings": [],
        "module_status": {name: "skipped" for name in STAGES}, "timings_seconds": {},
        "provenance": {"python": platform.python_version(), "platform": platform.platform(),
                       "packages": _versions(), "config": cfg.snapshot(),
                       "config_sha256": cfg.fingerprint()},
        "limitations": ["Ambang pilot belum tervalidasi; hasil tidak memengaruhi GII.",
            "Speech valid adalah proksi VAD, belum diverifikasi bebas kontaminasi.",
            "Deteksi musik, overlap, gema, dropout dan energi belum tersedia.",
            "Silabifikasi dan transkrip ASR adalah estimasi yang perlu audit."]})
    try:
        with tempfile.TemporaryDirectory(prefix="gasgawe_voice_") as temp_dir:
            extracted = _stage(result, "extract", "extract_audio", path, temp_dir, cfg)
            if extracted is not None:
                audio, sr = extracted["audio"], extracted["sample_rate"]
                info = extracted["audio_info"]
                result["audio_info"] = info
                result["audio_quality"]["clipping_pct"] = info["clipping_pct"]
                result["warnings"].extend(info.get("warnings", []))
                vad = _stage(result, "vad", "analyze_vad", audio, sr, cfg)
                if vad is not None:
                    result["vad"] = vad
                    for key in NUMERIC_FIELDS:
                        if key in vad:
                            result[key] = vad[key]
                    quality = _stage(result, "quality", "analyze_quality", audio, sr, vad["segments"], info, cfg)
                    if quality is not None:
                        result["audio_quality"] = quality
                        result["warnings"].extend(quality["warnings"])
                    pitch = _stage(result, "pitch", "analyze_pitch", audio, sr, vad["segments"], vad["phrases"], cfg)
                    if pitch is not None:
                        result["pitch"] = pitch
                        for key in NUMERIC_FIELDS:
                            if key in pitch:
                                result[key] = pitch[key]
                        result["evidence_timestamps"] = pitch["evidence_timestamps"]
                    if not cfg.ASR_ENABLED:
                        result["warnings"].append("asr_disabled")
                    elif vad["segments"]:
                        transcript = _stage(result, "transcribe", "transcribe_audio", audio, sr, vad, cfg)
                        if transcript is not None:
                            result["transcription"] = transcript
                            result["warnings"].extend(transcript.get("warnings", []))
                            counts = _stage(result, "syllable", "count_text", transcript["text"])
                            if counts is not None:
                                result["syllables"] = counts
                                result["warnings"].extend(counts["warnings"])
                                tempo = _stage(result, "tempo", "analyze_tempo", counts, vad, cfg.MIN_SPEECH_SECONDS)
                                if tempo is not None:
                                    result.update(tempo)
    except Exception as exc:
        result["errors"].append({"module": "run", "code": "pipeline_failed",
                                 "message": "Orkestrasi/cleanup file temporer gagal.",
                                 "exception_type": type(exc).__name__})
        LOG.error("stage=run code=pipeline_failed type=%s", type(exc).__name__)
    result["warnings"] = list(dict.fromkeys(result["warnings"]))
    result = _strict_values(result)
    try:
        result.update(score(result, cfg))
    except Exception as exc:
        # A malformed partial result must never become a decision or crash the CLI.
        result["errors"].append({"module": "scoring", "code": "scoring_failed",
                                 "message": "Penggabungan hasil gagal; keputusan memerlukan peninjauan.",
                                 "exception_type": type(exc).__name__})
        result.update(status="perlu_peninjauan", abstain_reason="scoring_failed",
                      intonation_item_passed=None, descriptors=["Hasil memerlukan peninjauan pengukuran."])
        LOG.error("stage=scoring code=scoring_failed type=%s", type(exc).__name__)
    result["timings_seconds"]["total"] = round(time.perf_counter() - started, 4)
    return result
