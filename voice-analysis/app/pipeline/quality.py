"""Recording QC; SNR is only a stationary-noise estimate."""

import numpy as np


def inspect_original(audio, cfg):
    data = np.asarray(audio)
    if data.ndim == 1:
        data = data[:, None]
    if data.ndim != 2 or not data.shape[0] or not data.shape[1]:
        raise ValueError("Audio harus berupa sampel mono atau multichannel yang tidak kosong")
    channels = data.shape[1]
    mean = np.zeros(channels, dtype=np.float64)
    centered_squared = np.zeros(channels, dtype=np.float64)
    total_squared = np.zeros(channels, dtype=np.float64)
    clipped = np.zeros(channels, dtype=np.int64)
    peaks = np.zeros(channels, dtype=np.float64)
    centered_cross_product = 0.0
    frames_seen = 0
    # Bound temporary arrays even when decoded PCM approaches the input cap.
    block_frames = max(1, 1_048_576 // channels)
    for start in range(0, len(data), block_frames):
        block = np.asarray(data[start:start + block_frames], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("Audio memiliki sampel non-finite")
        absolute = np.abs(block)
        block_mean = np.mean(block, axis=0)
        centered = block - block_mean
        delta = block_mean - mean
        combined_count = frames_seen + len(block)
        merge_weight = frames_seen * len(block) / combined_count
        centered_squared += np.sum(centered ** 2, axis=0) + delta ** 2 * merge_weight
        if channels == 2:
            centered_cross_product += float(np.sum(centered[:, 0] * centered[:, 1])) + delta[0] * delta[1] * merge_weight
        mean += delta * len(block) / combined_count
        frames_seen = combined_count
        total_squared += np.sum(block ** 2, axis=0)
        clipped += np.sum(absolute >= cfg.CLIP_AMPLITUDE, axis=0)
        peaks = np.maximum(peaks, np.max(absolute, axis=0))
    rms = np.sqrt(total_squared / len(data))
    centered_rms = np.sqrt(centered_squared / len(data))
    clipping = 100 * clipped / len(data)
    correlation = None
    warnings = []
    selected = None
    if data.shape[1] == 2:
        if np.all(centered_rms > cfg.POWER_EPSILON):
            covariance = centered_cross_product / len(data)
            correlation = float(np.clip(covariance / np.prod(centered_rms), -1, 1))
        weak = min(centered_rms) < max(centered_rms) * cfg.CHANNEL_WEAK_RATIO
        anti = correlation is not None and correlation < cfg.CHANNEL_ANTIPHASE_CORRELATION
        if weak or anti:
            selected = int(np.argmax(centered_rms))
            if weak:
                warnings.append("channel_weak")
            if anti:
                warnings.append("channel_antiphase")
    elif data.shape[1] > 2:
        selected = int(np.argmax(centered_rms))
        warnings.append("multichannel_selected_by_rms")
    return {
        "clipping_pct": float(np.max(clipping)),
        "clipping_pct_per_channel": clipping.tolist(),
        "rms_per_channel": rms.tolist(),
        "ac_rms_per_channel": centered_rms.tolist(),
        "dc_offset_per_channel": mean.tolist(),
        "peak_per_channel": peaks.tolist(),
        "stereo_correlation": correlation,
        "selected_channel": selected,
        "channel_strategy": "select" if selected is not None else "mean",
        "warnings": warnings,
    }


def estimate_snr(audio, sample_rate, segments, cfg):
    width = max(1, round(cfg.QC_FRAME_SECONDS * sample_rate))
    count = len(audio) // width
    if not count:
        return None, "snr_insufficient_frames"
    frames = np.asarray(audio[:count * width]).reshape(count, width)
    powers = np.mean(frames.astype(np.float64) ** 2, axis=1)
    starts = np.arange(count) * width / sample_rate
    ends = starts + width / sample_rate
    speech = np.zeros(count, dtype=bool)
    noise = np.ones(count, dtype=bool)
    for start, end in segments:
        speech |= (starts >= start) & (ends <= end)
        noise &= (ends <= start - cfg.NOISE_MARGIN_SECONDS) | (starts >= end + cfg.NOISE_MARGIN_SECONDS)
    if noise.sum() * width / sample_rate < cfg.MIN_NOISE_SECONDS:
        return None, "snr_no_noise_reference"
    if speech.sum() * width / sample_rate < cfg.MIN_SNR_SPEECH_SECONDS:
        return None, "snr_insufficient_speech"
    noise_power = float(np.median(powers[noise]))
    total_power = float(np.mean(powers[speech]))
    if noise_power <= cfg.POWER_EPSILON:
        return None, "snr_noise_below_floor"
    if total_power <= noise_power:
        return None, "snr_nonpositive_signal_estimate"
    return float(10 * np.log10((total_power - noise_power) / noise_power)), None


def analyze_quality(audio, sample_rate, segments, original_info, cfg):
    seconds = float(sum(end - start for start, end in segments))
    snr, snr_reason = estimate_snr(audio, sample_rate, segments, cfg)
    warnings = list(original_info.get("warnings", []))
    if snr_reason:
        warnings.append(snr_reason)
    elif snr < cfg.SNR_LOW_DB:
        warnings.append("snr_low")
    elif snr < cfg.SNR_WARN_DB:
        warnings.append("snr_warning")
    clipping = original_info["clipping_pct"]
    if clipping > cfg.CLIP_REVIEW_PCT:
        warnings.append("clipping_high")
    elif clipping > cfg.CLIP_WARN_PCT:
        warnings.append("clipping_warning")
    status = "insufficient" if seconds < cfg.MIN_SPEECH_SECONDS else ("warning" if warnings else "ok")
    return {"status": status, "snr_est_db": snr, "snr_unavailable_reason": snr_reason,
            "clipping_pct": clipping, "valid_speech_seconds": seconds, "warnings": warnings}
