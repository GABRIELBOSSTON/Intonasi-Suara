"""Regressions for recording artifacts and malformed measurement inputs."""

import json

import numpy as np
import pytest

from app.config import Config
from app.pipeline import extract
from app.pipeline.common import AnalysisError
from app.pipeline.pitch import analyze_pitch
from app.pipeline.quality import inspect_original
from app.pipeline.vad import summarize_segments


def test_nonspeech_peak_does_not_suppress_quiet_speech_pitch():
    sr = 16000
    times = np.arange(sr * 3) / sr
    audio = np.concatenate([np.zeros(sr), 0.01 * np.sin(2 * np.pi * 200 * times)])
    contaminated = audio.copy()
    contaminated[100:300] = 0.99
    speech = [[1, 4]]
    clean = analyze_pitch(audio, sr, speech, speech, Config())
    result = analyze_pitch(contaminated, sr, speech, speech, Config())
    assert result["f0_valid_coverage"] == pytest.approx(clean["f0_valid_coverage"])
    assert result["f0_valid_coverage"] > 0.9
    assert result["f0_median_hz"] == pytest.approx(200, abs=1)
    assert result["pitch_range_p10_p90_st"] == pytest.approx(clean["pitch_range_p10_p90_st"], abs=0.01)
    assert result["tracking"]["non_speech_masked"] is True
    assert contaminated[100] == 0.99


def test_dead_channel_with_dc_offset_does_not_win_channel_selection():
    times = np.arange(16000) / 16000
    speech = 0.01 * np.sin(2 * np.pi * 200 * times)
    dead = np.full(16000, 0.3)
    result = inspect_original(np.column_stack([speech, dead]), Config())
    assert result["selected_channel"] == 0
    assert "channel_weak" in result["warnings"]
    assert result["rms_per_channel"][1] > result["rms_per_channel"][0]
    assert result["ac_rms_per_channel"][1] < result["ac_rms_per_channel"][0]
    assert result["ac_rms_per_channel"][1] < 1e-12


def test_block_qc_agrees_with_direct_statistics():
    rng = np.random.default_rng(42)
    audio = rng.normal(0, 0.3, (600_000, 2)).astype(np.float32)
    audio[0] = [-1.0, 1.0]
    result = inspect_original(audio, Config())
    reference = audio.astype(np.float64)
    np.testing.assert_allclose(result["rms_per_channel"], np.sqrt(np.mean(reference ** 2, axis=0)))
    np.testing.assert_allclose(result["ac_rms_per_channel"], np.std(reference, axis=0))
    np.testing.assert_allclose(result["clipping_pct_per_channel"], 100 * np.mean(np.abs(reference) >= 0.999, axis=0))
    assert result["stereo_correlation"] == pytest.approx(np.corrcoef(reference.T)[0, 1], abs=1e-10)


@pytest.mark.parametrize("audio", [np.zeros((0, 2)), np.zeros((3, 0)), np.zeros((2, 2, 2)), np.array([np.nan, 0])])
def test_qc_rejects_invalid_shapes_and_samples(audio):
    with pytest.raises(ValueError):
        inspect_original(audio, Config())


@pytest.mark.parametrize("segments", [[[float("nan"), 2]], [[0, float("inf")]], [[3, 1]], [[0]]])
def test_vad_rejects_invalid_intervals(segments):
    with pytest.raises(AnalysisError, match="Interval speech"):
        summarize_segments(segments, 4, Config())


def test_vad_normalizes_before_sorting_and_unions_overlap():
    result = summarize_segments([["10", "11"], ["2", "3"], [2.5, 4]], 12, Config())
    assert result["segments"] == [[2.0, 4.0], [10.0, 11.0]]
    assert result["valid_speech_seconds"] == 3


@pytest.mark.parametrize("duration", [-1, float("nan"), float("inf")])
def test_vad_rejects_invalid_duration(duration):
    with pytest.raises(AnalysisError, match="Durasi audio"):
        summarize_segments([], duration, Config())


@pytest.mark.parametrize("stream", [
    {"sample_rate": "N/A", "channels": 1},
    {"sample_rate": "16000", "channels": "unknown"},
    {"channels": 1},
])
def test_decoder_metadata_error_has_explicit_code(monkeypatch, tmp_path, stream):
    source = tmp_path / "malformed.mp4"
    source.write_bytes(b"placeholder")
    monkeypatch.setattr(extract, "_binary", lambda name, label: name)
    monkeypatch.setattr(extract, "_execute", lambda *args: json.dumps({"streams": [stream]}))
    with pytest.raises(AnalysisError) as failure:
        extract.extract_audio(source, tmp_path, Config())
    assert failure.value.code == "invalid_audio_metadata"
