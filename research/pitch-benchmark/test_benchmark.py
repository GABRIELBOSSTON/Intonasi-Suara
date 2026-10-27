"""Regression tests for evaluation correctness, not model quality claims."""
import numpy as np
import pytest
from metrics import accuracy, align, summarize, semitones
from trackers import reject_outliers, Trackers


def test_exact_reference():
    ref = np.array([100., 200., np.nan, np.nan])
    result = accuracy(ref, ref, np.ones(4, bool))
    assert result["gpe_pct"] == result["voicing_error_pct"] == 0
    assert result["voiced_recall_pct"] == 100


def test_octave_and_voicing_denominators():
    ref = np.array([100., 100., 100., np.nan])
    result = accuracy(np.array([200., 100., np.nan, 150.]), ref, np.ones(4, bool))
    assert result["gpe_pct"] == result["octave_error_pct"] == 50
    assert result["voicing_error_pct"] == 50
    assert result["f0_frame_error_pct"] == 75
    assert result["voiced_recall_pct"] == pytest.approx(200 / 3)


def test_all_rejected_is_not_perfect():
    result = accuracy(np.full(5, np.nan), np.ones(5) * 100, np.ones(5, bool))
    assert result["gpe_pct"] is None
    assert result["voiced_recall_pct"] == 0
    assert result["voicing_error_pct"] == 100


def test_guard_excludes_ambiguous_error():
    result = accuracy(np.array([100., 50.]), np.array([100., 100.]), np.array([True, False]))
    assert result["gpe_pct"] == 0
    assert result["eligible_frames"] == 1


def test_align_does_not_bridge_gap():
    actual = align(np.array([.01, .02, .03]), np.array([100., np.nan, 110.]), np.array([0., .02, .04]))
    assert np.isnan(actual).all()


def test_outlier_rejection_preserves_gaps_and_sustained_register():
    values = np.array([100.] * 10 + [200.] + [100.] * 10 + [np.nan] + [200.] * 20)
    cleaned, rejected = reject_outliers(values)
    assert rejected.sum() == 1
    assert np.isnan(cleaned[10]) and np.isnan(cleaned[21])
    assert cleaned[-1] == 200


def test_no_jumps_across_segments():
    t = np.arange(100) * .01
    result = summarize(t, np.r_[np.ones(50) * 100, np.ones(50) * 200], [[0, .5], [.5, 1]], [[0, .5], [.5, 1]])
    assert result["octave_jump_count"] == 0
    assert result["phrase_sd_st"] == pytest.approx(0)


def test_semitone_octave():
    assert semitones([200])[0] - semitones([100])[0] == pytest.approx(12)


def test_current_adapter_matches_application():
    import sys
    from pathlib import Path
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice-analysis"))
    from app.config import Config
    from app.pipeline.pitch import analyze_pitch
    cfg = Config()
    t = np.arange(48000) / 16000
    f0 = 180 * 2 ** (.2 * np.sin(2 * np.pi * t))
    audio = .4 * np.sin(2 * np.pi * np.cumsum(f0) / 16000)
    segments = [[.2, 1.3], [1.6, 2.8]]
    expected = analyze_pitch(audio, 16000, segments, segments, cfg)
    times, pitch, _ = Trackers(cfg).run("praat_current", audio, segments)
    actual = summarize(times, pitch, segments, segments)
    assert actual["range_st"] == pytest.approx(expected["pitch_range_p10_p90_st"])
    assert actual["phrase_sd_st"] == pytest.approx(expected["pitch_sd_st_phrase_median"])
    assert actual["octave_jump_count"] == expected["octave_jump_count"]


def test_crepe_dither_reproducible():
    import sys
    from pathlib import Path
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "voice-analysis"))
    from app.config import Config
    import torch
    torch.set_num_threads(4)
    tracker = Trackers(Config())
    audio = (.3 * np.sin(2 * np.pi * 180 * np.arange(8000) / 16000)).astype(np.float32)
    first = tracker.run("torchcrepe_tiny", audio, [[0, .5]])[1]
    second = tracker.run("torchcrepe_tiny", audio, [[0, .5]])[1]
    assert np.isfinite(first).any()
    np.testing.assert_equal(first, second)
