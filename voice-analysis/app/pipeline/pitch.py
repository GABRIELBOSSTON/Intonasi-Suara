"""Speech-masked pitch tracking (CREPE primary, Praat audit) and statistics."""

import numpy as np
import parselmouth


def hz_to_semitones(frequencies, reference_hz=None):
    values = np.asarray(frequencies, dtype=np.float64)
    valid = np.isfinite(values) & (values > 0)
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if reference_hz is None:
        if not valid.any():
            return result
        reference_hz = float(np.median(values[valid]))
    if not np.isfinite(reference_hz) or reference_hz <= 0:
        raise ValueError("Referensi F0 harus positif dan finite")
    result[valid] = 12 * np.log2(values[valid] / reference_hz)
    return result


def _region_ids(times, segments):
    ids = np.full(len(times), -1, dtype=int)
    for index, (start, end) in enumerate(segments):
        ids[(times >= start) & (times < end)] = index
    return ids


def count_octave_jumps(st, valid, regions, threshold):
    adjacent = valid[:-1] & valid[1:] & (regions[:-1] == regions[1:]) & (regions[:-1] >= 0)
    return int(np.sum(adjacent & (np.abs(np.diff(st)) >= threshold)))


def _speech_only(audio, sample_rate, segments):
    # Exclude non-speech transients and hum outside VAD segments without
    # moving or rescaling speech.
    samples = np.asarray(audio, dtype=np.float64)
    masked = np.zeros_like(samples)
    for start, end in segments:
        first_sample = max(0, int(np.ceil(start * sample_rate)))
        last_sample = min(len(samples), int(np.ceil(end * sample_rate)))
        masked[first_sample:last_sample] = samples[first_sample:last_sample]
    return masked


def track_praat(audio, sample_rate, segments, cfg):
    """Two-pass Praat raw autocorrelation; unvoiced frames are NaN."""
    sound = parselmouth.Sound(audio, sampling_frequency=sample_rate)

    def track(floor, ceiling):
        return sound.to_pitch_ac(time_step=cfg.F0_STEP_SECONDS, pitch_floor=floor,
            pitch_ceiling=ceiling, voicing_threshold=cfg.F0_VOICING_THRESHOLD,
            silence_threshold=cfg.F0_SILENCE_THRESHOLD)

    coarse = track(cfg.F0_FLOOR_HZ, cfg.F0_CEILING_HZ)
    first = coarse.selected_array["frequency"]
    first_valid = np.isfinite(first) & (first > 0) & (_region_ids(coarse.xs(), segments) >= 0)
    floor, ceiling = cfg.F0_FLOOR_HZ, cfg.F0_CEILING_HZ
    if first_valid.sum() >= cfg.F0_MIN_ADAPT_FRAMES:
        low, high = np.percentile(first[first_valid], [cfg.F0_ADAPT_LOW_PERCENTILE, cfg.F0_ADAPT_HIGH_PERCENTILE])
        floor = max(floor, float(low / cfg.F0_ADAPT_MARGIN))
        ceiling = min(ceiling, float(high * cfg.F0_ADAPT_MARGIN))
    fine = track(floor, ceiling)
    f0 = fine.selected_array["frequency"].astype(np.float64)
    f0[(f0 < floor) | (f0 > ceiling)] = np.nan
    info = {"tracker": "praat_ac", "pass1_floor_hz": cfg.F0_FLOOR_HZ,
            "pass1_ceiling_hz": cfg.F0_CEILING_HZ, "pass2_floor_hz": floor,
            "pass2_ceiling_hz": ceiling, "praat_version": parselmouth.PRAAT_VERSION}
    return fine.xs(), f0, info


def _relative_frame_db(audio, hop, window):
    """Frame energy in dB relative to the loudest speech frames (gain-invariant).

    torchcrepe's own Silence gate uses absolute, spectrum-averaged loudness, so a
    quiet phone recording can be gated out entirely; this gate cannot.
    """
    samples = np.asarray(audio, dtype=np.float64)
    padded = np.pad(samples, window // 2)
    frames = 1 + len(samples) // hop
    windows = np.lib.stride_tricks.sliding_window_view(padded, window)[::hop][:frames]
    energy = np.mean(windows ** 2, axis=1)
    db = np.full(len(energy), -np.inf)
    positive = energy > 0
    db[positive] = 10 * np.log10(energy[positive])
    if not positive.any():
        return db
    return db - np.percentile(db[positive], 90)


def track_crepe(audio, sample_rate, cfg):
    """torchcrepe with Viterbi decoding, periodicity and relative-energy gating; unvoiced frames are NaN."""
    import torch
    import torchcrepe

    hop = int(round(cfg.F0_STEP_SECONDS * sample_rate))
    # torchcrepe dithers bins through the global NumPy RNG; seed it locally so
    # results are reproducible, then restore the caller's RNG state.
    rng_state = np.random.get_state()
    np.random.seed(cfg.CREPE_SEED)
    try:
        with torch.inference_mode():
            tensor = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)).unsqueeze(0)
            pitch, periodicity = torchcrepe.predict(tensor, sample_rate, hop, cfg.F0_FLOOR_HZ,
                cfg.F0_CEILING_HZ, cfg.CREPE_MODEL, batch_size=cfg.CREPE_BATCH_SIZE,
                device="cpu", return_periodicity=True)
            periodicity = torchcrepe.filter.median(periodicity, 3)
    finally:
        np.random.set_state(rng_state)
    f0 = pitch[0].numpy().astype(np.float64)
    periodicity = periodicity[0].numpy().astype(np.float64)
    relative_db = _relative_frame_db(audio, hop, torchcrepe.WINDOW_SIZE)[:len(f0)]
    voiced = (periodicity >= cfg.CREPE_PERIODICITY_THRESHOLD) & (relative_db >= cfg.CREPE_RELATIVE_SILENCE_DB)
    f0[~voiced] = np.nan
    info = {"tracker": f"torchcrepe_{cfg.CREPE_MODEL}", "decoder": "viterbi",
            "periodicity_threshold": cfg.CREPE_PERIODICITY_THRESHOLD,
            "relative_silence_db": cfg.CREPE_RELATIVE_SILENCE_DB,
            "torchcrepe_version": _version("torchcrepe")}
    return np.arange(len(f0)) * hop / sample_rate, f0, info


def _version(package):
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def summarize_pitch(times, f0, segments, phrases, cfg):
    times = np.asarray(times, dtype=np.float64)
    f0 = np.asarray(f0, dtype=np.float64)
    result = {"pitch_range_p10_p90_st": None, "pitch_sd_st_phrase_median": None,
              "f0_valid_coverage": None, "octave_jump_count": None,
              "octave_jump_per_min": None, "f0_median_hz": None,
              "f0_valid_seconds": 0.0, "phrase_statistics": [], "evidence_timestamps": []}
    regions = _region_ids(times, segments)
    in_speech = regions >= 0
    valid = in_speech & np.isfinite(f0) & (f0 > 0)
    result["speech_frame_count"] = int(in_speech.sum())
    result["valid_frame_count"] = int(valid.sum())
    result["f0_valid_coverage"] = float(valid.sum() / in_speech.sum()) if in_speech.any() else None
    result["f0_valid_seconds"] = float(valid.sum() * cfg.F0_STEP_SECONDS)
    if not valid.any():
        return result
    median = float(np.median(f0[valid]))
    st = hz_to_semitones(np.where(valid, f0, np.nan), median)
    percentiles = [cfg.F0_RANGE_LOW_PERCENTILE, cfg.F0_RANGE_HIGH_PERCENTILE]
    low, high = np.percentile(st[valid], percentiles)
    result["f0_median_hz"] = median
    result["pitch_range_p10_p90_st"] = float(high - low)
    result["octave_jump_count"] = count_octave_jumps(st, valid, regions, cfg.OCTAVE_JUMP_ST)
    speech_seconds = sum(end - start for start, end in segments)
    result["octave_jump_per_min"] = result["octave_jump_count"] * 60 / speech_seconds
    for start, end in phrases:
        mask = valid & (times >= start) & (times < end)
        voiced_seconds = float(mask.sum() * cfg.F0_STEP_SECONDS)
        if voiced_seconds < cfg.MIN_PHRASE_VOICED_SECONDS:
            continue
        p10, p90 = np.percentile(st[mask], percentiles)
        result["phrase_statistics"].append({
            "start": start, "end": end, "valid_voiced_seconds": voiced_seconds,
            "sd_st": float(np.std(st[mask], ddof=0)), "range_st": float(p90 - p10)})
    stats = result["phrase_statistics"]
    if stats:
        result["pitch_sd_st_phrase_median"] = float(np.median([item["sd_st"] for item in stats]))
        representative = sorted(stats, key=lambda item: (abs(item["sd_st"] - result["pitch_sd_st_phrase_median"]), item["start"]))
        result["evidence_timestamps"] = sorted([[item["start"], item["end"]] for item in representative[:cfg.EVIDENCE_LIMIT]])
    return result


def analyze_pitch(audio, sample_rate, segments, phrases, cfg):
    if not segments:
        result = summarize_pitch([], [], segments, phrases, cfg)
        result["tracking"] = None
        return result
    masked = _speech_only(audio, sample_rate, segments)
    if cfg.PITCH_TRACKER == "crepe":
        times, f0, info = track_crepe(masked, sample_rate, cfg)
    else:
        times, f0, info = track_praat(masked, sample_rate, segments, cfg)
    result = summarize_pitch(times, f0, segments, phrases, cfg)
    info.update({"speech_frame_count": result.pop("speech_frame_count"),
                 "valid_frame_count": result.pop("valid_frame_count"),
                 "step_seconds": cfg.F0_STEP_SECONDS, "non_speech_masked": True})
    result["tracking"] = info
    if cfg.PITCH_INCLUDE_CONTOUR:
        keep = np.isfinite(f0) & (f0 > 0) & (_region_ids(np.asarray(times), segments) >= 0)
        result["contour"] = {"t": [round(float(x), 3) for x in np.asarray(times)[keep]],
                             "hz": [round(float(x), 1) for x in f0[keep]]}
    if cfg.PITCH_TRACKER == "crepe" and cfg.PITCH_AUDIT_PRAAT:
        # Audit only: never used for scoring, kept to explain tracker disagreement.
        a_times, a_f0, a_info = track_praat(masked, sample_rate, segments, cfg)
        audit = summarize_pitch(a_times, a_f0, segments, phrases, cfg)
        result["audit_praat"] = {key: audit[key] for key in (
            "pitch_range_p10_p90_st", "pitch_sd_st_phrase_median", "f0_median_hz",
            "f0_valid_coverage", "octave_jump_count")}
        result["audit_praat"]["tracking"] = a_info
    return result
