"""Pitch accuracy and prosody metrics with explicit denominators."""
import numpy as np
from generate import in_regions


def semitones(f0):
    f0 = np.asarray(f0, float)
    return 12 * np.log2(np.where(np.isfinite(f0) & (f0 > 0), f0, np.nan))


def align(times, f0, grid):
    times, f0 = np.asarray(times), np.asarray(f0)
    if len(times) < 2:
        return np.full(len(grid), np.nan)
    right = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    left = np.maximum(right - 1, 0)
    indices = np.where(np.abs(times[left] - grid) < np.abs(times[right] - grid), left, right)
    values = f0[indices].copy()
    values[np.abs(times[indices] - grid) > np.median(np.diff(times)) * .51] = np.nan
    return values


def accuracy(estimate, reference, eligible):
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    rv = np.isfinite(reference) & (reference > 0)
    ev = np.isfinite(estimate) & (estimate > 0)
    both = rv & ev & eligible
    err_st = np.abs(semitones(estimate[both]) - semitones(reference[both]))
    gross = np.abs(estimate[both] / reference[both] - 1) > .2
    def pct(mask, denominator):
        return float(100 * np.sum(mask) / denominator) if denominator else None
    return {
        "eligible_frames": int(eligible.sum()), "joint_voiced_frames": int(both.sum()),
        "gpe_pct": pct(gross, both.sum()),
        "octave_error_pct": pct(np.abs(err_st - 12) <= 1, both.sum()),
        "voicing_error_pct": pct((rv != ev) & eligible, eligible.sum()),
        "voiced_recall_pct": pct(both, (rv & eligible).sum()),
        "false_alarm_pct": pct(ev & ~rv & eligible, (~rv & eligible).sum()),
        "f0_frame_error_pct": pct(((rv != ev) & eligible).sum() + gross.sum(), eligible.sum()),
        "median_abs_error_st": float(np.median(err_st)) if len(err_st) else None,
    }


def summarize(times, f0, segments, phrases, min_phrase=.3):
    times, f0 = np.asarray(times), np.asarray(f0)
    step = float(np.median(np.diff(times))) if len(times) > 1 else .01
    speech = in_regions(times, segments)
    valid = speech & np.isfinite(f0) & (f0 > 0)
    st = semitones(np.where(valid, f0, np.nan))
    regions = np.full(len(times), -1)
    for i, (start, end) in enumerate(segments):
        regions[(times >= start) & (times < end)] = i
    adjacent = valid[1:] & valid[:-1] & (regions[1:] == regions[:-1])
    jumps = int(np.sum(adjacent & (np.abs(np.diff(st)) >= 9)))
    phrase_sd, phrase_range = [], []
    for start, end in phrases:
        mask = valid & (times >= start) & (times < end)
        if mask.sum() * step >= min_phrase:
            phrase_sd.append(np.std(st[mask]))
            phrase_range.append(np.diff(np.percentile(st[mask], [10, 90]))[0])
    duration = sum(end - start for start, end in segments)
    return {"f0_median_hz": float(np.median(f0[valid])) if valid.any() else None,
            "range_st": float(np.diff(np.percentile(st[valid], [10, 90]))[0]) if valid.any() else None,
            "phrase_sd_st": float(np.median(phrase_sd)) if phrase_sd else None,
            "phrase_range_st": float(np.median(phrase_range)) if phrase_range else None,
            "mad_st": float(np.median(np.abs(st[valid] - np.median(st[valid])))) if valid.any() else None,
            "coverage_pct": float(100 * valid.sum() / speech.sum()) if speech.any() else None,
            "octave_jump_count": jumps,
            "octave_jump_per_min": jumps * 60 / duration if duration else None}
