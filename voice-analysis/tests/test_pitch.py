"""Generated waveforms test F0 independently of speech detection."""

import numpy as np
import pytest
from app.config import Config
from app.pipeline.pitch import analyze_pitch, count_octave_jumps, hz_to_semitones


def signal(frequencies, sample_rate=16000):
    return (0.25 * np.sin(2 * np.pi * np.cumsum(frequencies) / sample_rate)).astype(np.float32)


def test_semitones_octaves_and_invalid_values():
    np.testing.assert_allclose(hz_to_semitones([100, 200, 400], 200), [-12, 0, 12])
    assert np.isnan(hz_to_semitones([0, -2, np.nan])).all()
    with pytest.raises(ValueError):
        hz_to_semitones([100], 0)


def test_flat_sine_has_near_zero_range():
    audio = signal(np.full(16000 * 4, 200.0))
    result = analyze_pitch(audio, 16000, [[0, 4]], [[0, 4]], Config())
    # CREPE quantizes to 20-cent bins plus dither, so a flat tone spreads ~0.2 st;
    # still far below MIN_PITCH_RANGE_ST.
    assert result["pitch_range_p10_p90_st"] < 0.5
    assert result["pitch_sd_st_phrase_median"] < 0.2
    assert result["f0_median_hz"] == pytest.approx(200, abs=1)
    assert result["f0_valid_coverage"] > 0.95
    assert result["octave_jump_count"] == 0


def test_modulated_sine_has_wide_range():
    times = np.arange(16000 * 8) / 16000
    audio = signal(180 * 2 ** (5 * np.sin(2 * np.pi * 0.5 * times) / 12))
    result = analyze_pitch(audio, 16000, [[0, 8]], [[0, 4], [4, 8]], Config())
    assert result["pitch_range_p10_p90_st"] > 8
    assert result["pitch_sd_st_phrase_median"] > 3
    assert result["octave_jump_count"] == 0


def test_silence_and_outside_speech_are_not_pitch():
    audio = np.concatenate([np.zeros(16000), signal(np.full(16000 * 3, 180.0)), np.zeros(16000 * 2)])
    result = analyze_pitch(audio, 16000, [[1, 4]], [[1, 4]], Config())
    assert result["f0_valid_coverage"] > 0.9
    silent = analyze_pitch(np.zeros(32000), 16000, [[0, 2]], [[0, 2]], Config())
    assert silent["pitch_range_p10_p90_st"] is None
    assert silent["f0_valid_coverage"] == 0


def test_gain_invariance():
    times = np.arange(64000) / 16000
    audio = signal(180 * 2 ** (4 * np.sin(2 * np.pi * times) / 12))
    first = analyze_pitch(audio, 16000, [[0, 4]], [[0, 4]], Config())
    quieter = analyze_pitch(audio * 0.1, 16000, [[0, 4]], [[0, 4]], Config())
    assert first["pitch_range_p10_p90_st"] == pytest.approx(quieter["pitch_range_p10_p90_st"], abs=0.02)


def test_jump_counter_does_not_bridge_unvoiced_or_segments():
    st = np.array([0, 12, np.nan, -12, 0], dtype=float)
    valid = np.isfinite(st)
    regions = np.array([0, 0, 0, 1, 2])
    assert count_octave_jumps(st, valid, regions, 9) == 1



def test_praat_tracker_still_available():
    audio = signal(np.full(16000 * 4, 200.0))
    result = analyze_pitch(audio, 16000, [[0, 4]], [[0, 4]], Config(PITCH_TRACKER="praat"))
    assert result["tracking"]["tracker"] == "praat_ac"
    assert result["f0_median_hz"] == pytest.approx(200, abs=1)
    assert "audit_praat" not in result


def test_crepe_result_has_praat_audit_and_is_reproducible():
    times = np.arange(16000 * 4) / 16000
    audio = signal(180 * 2 ** (4 * np.sin(2 * np.pi * times) / 12))
    first = analyze_pitch(audio, 16000, [[0, 4]], [[0, 4]], Config())
    second = analyze_pitch(audio, 16000, [[0, 4]], [[0, 4]], Config())
    assert first["tracking"]["tracker"] == "torchcrepe_tiny"
    assert first["pitch_range_p10_p90_st"] == second["pitch_range_p10_p90_st"]
    assert first["audit_praat"]["tracking"]["tracker"] == "praat_ac"


def test_crepe_ignores_low_level_hum_between_words():
    # Speech-like tone bursts with a steady 60 Hz hum underneath: the hum must not
    # become pitch (the failure seen with Praat on real phone recordings).
    sr = 16000
    times = np.arange(sr * 4) / sr
    hum = 0.002 * np.sin(2 * np.pi * 60 * times)
    bursts = ((times % 1.0) < 0.6).astype(float)
    audio = (hum + bursts * signal(np.full(len(times), 220.0))).astype(np.float32)
    result = analyze_pitch(audio, sr, [[0, 4]], [[0, 4]], Config())
    assert result["f0_median_hz"] == pytest.approx(220, abs=3)
    assert result["pitch_range_p10_p90_st"] < 1.0
    assert result["octave_jump_count"] == 0


def test_contour_only_when_requested():
    audio = signal(np.full(16000 * 2, 200.0))
    assert "contour" not in analyze_pitch(audio, 16000, [[0, 2]], [[0, 2]], Config())
    result = analyze_pitch(audio, 16000, [[0, 2]], [[0, 2]], Config(PITCH_INCLUDE_CONTOUR=True))
    assert len(result["contour"]["t"]) == len(result["contour"]["hz"]) > 100
    assert all(190 < hz < 210 for hz in result["contour"]["hz"])
