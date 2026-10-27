"""Invalid configuration and incomplete measurements must fail closed."""

import json

import pytest

from app.config import Config, load_config
from app.pipeline.scoring import score
from app.pipeline import run


@pytest.mark.parametrize("overrides", [
    {"ASR_CPU_THREADS": True}, {"ASR_CPU_THREADS": 1.5},
    {"ASR_ENABLED": "false"}, {"MIN_PITCH_RANGE_ST": "4"},
    {"F0_SILENCE_THRESHOLD": 3}, {"CHANNEL_ANTIPHASE_CORRELATION": -2},
    {"CHANNEL_WEAK_RATIO": 2}, {"CLIP_REVIEW_PCT": 120},
    {"MAX_DECODED_SAMPLES": 0}, {"MIN_PHRASE_VOICED_SECONDS": 0},
    {"MIN_NOISE_SECONDS": 0}, {"F0_MIN_ADAPT_FRAMES": 0},
    {"SNR_LOW_DB": 25, "SNR_WARN_DB": 20}, {"F0_STEP_SECONDS": float("nan")},
    {"MIN_AUDIO_SECONDS": 601}, {"ASR_COMPUTE_TYPE": "float16"},
    {"ASR_REVISION": "main"}, {"ASR_MODEL_SIZE": "tiny"},
    {"VIDEO_EXTENSIONS": ()}, {"VIDEO_EXTENSIONS": ("mp4",)},
    {"FFMPEG": " "}, {"FFPROBE": None},
])
def test_invalid_direct_config_rejected(overrides):
    with pytest.raises(ValueError):
        Config(**overrides)


def test_config_extensions_canonicalized():
    config = Config(VIDEO_EXTENSIONS=(".MP4", ".mp4", ".MOV"))
    assert config.VIDEO_EXTENSIONS == (".mp4", ".mov")
    assert config.fingerprint() == Config(VIDEO_EXTENSIONS=(".mp4", ".mov")).fingerprint()


def test_json_duplicate_keys_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"MODE":"intonasi","MODE":"ekspresivitas"}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplikat"):
        load_config(path)


def test_local_model_does_not_require_remote_revision():
    assert Config(ASR_LOCAL_PATH="local-model", ASR_REVISION="").ASR_LOCAL_PATH == "local-model"
    assert Config(ASR_ENABLED=False, ASR_REVISION="").ASR_ENABLED is False


def valid_features():
    return {"valid_speech_seconds": 40.0, "f0_valid_coverage": 0.8,
            "pitch_range_p10_p90_st": 7.0, "pitch_sd_st_phrase_median": 3.0,
            "octave_jump_count": 0, "octave_jump_per_min": 0.0,
            "audio_quality": {"status": "ok", "clipping_pct": 0.0},
            "articulation_rate_syl_per_s": 6.5, "errors": [], "warnings": []}


@pytest.mark.parametrize("field", ["valid_speech_seconds", "f0_valid_coverage",
    "pitch_range_p10_p90_st", "pitch_sd_st_phrase_median", "octave_jump_count", "octave_jump_per_min"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True, "1"])
def test_invalid_essential_feature_abstains(field, value):
    features = valid_features()
    features[field] = value
    result = score(features, Config())
    assert result["intonation_item_passed"] is None
    assert result["abstain_reason"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("quality", [
    {"status": "insufficient", "clipping_pct": 0},
    {"status": "unknown", "clipping_pct": 0},
    {"status": "ok", "clipping_pct": None},
    {"status": "ok", "clipping_pct": float("nan")},
    {"status": "ok", "clipping_pct": 101},
])
def test_invalid_or_insufficient_quality_cannot_pass(quality):
    features = valid_features()
    features["audio_quality"] = quality
    result = score(features, Config())
    assert result["intonation_item_passed"] is None
    if quality["status"] == "insufficient":
        assert result["status"] == "bukti_belum_cukup"


def test_invalid_supporting_metrics_not_printed_as_numbers():
    features = valid_features()
    features.update(articulation_rate_syl_per_s=float("nan"), pause_ratio=2)
    result = score(features, Config(MODE="intonasi"))
    assert result["intonation_item_passed"] is True
    assert all("nan" not in line and "200.0%" not in line for line in result["descriptors"])


def test_stage_returning_none_is_recorded_as_failure(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(run, "import_module", lambda name: SimpleNamespace(analyze=lambda: None))
    result = {"module_status": {}, "errors": [], "timings_seconds": {}}
    assert run._stage(result, "transcribe", "analyze") is None
    assert result["errors"][0]["code"] == "transcribe_invalid_output"
    assert result["module_status"]["transcribe"] == "failed"


def test_scoring_exception_preserves_partial_json(monkeypatch, tmp_path):
    def broken_score(*args):
        raise TypeError("malformed result")
    monkeypatch.setattr(run, "score", broken_score)
    result = run.analyze_file(tmp_path / "missing.mp4")
    assert result["intonation_item_passed"] is None
    assert result["abstain_reason"] == "scoring_failed"
    assert result["errors"][-1]["code"] == "scoring_failed"
    json.dumps(result, allow_nan=False)
