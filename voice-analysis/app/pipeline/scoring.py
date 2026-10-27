"""Conservative pilot decisions (pitch zone + speaking tempo) with neutral descriptions."""

import math
from numbers import Real


def _finite_number(value, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        return None
    if value < minimum or (maximum is not None and value > maximum):
        return None
    return float(value)


def pitch_zone(pitch_range, phrase_sd, cfg):
    """datar / wajar / ekstrem; ekstrem usually means a measurement problem, not expressiveness."""
    if pitch_range is None or phrase_sd is None:
        return None
    if pitch_range > cfg.MAX_PITCH_RANGE_ST or phrase_sd > cfg.MAX_PITCH_SD_ST:
        return "ekstrem"
    if pitch_range >= cfg.MIN_PITCH_RANGE_ST and phrase_sd >= cfg.MIN_PITCH_SD_ST:
        return "wajar"
    return "datar"


def tempo_zone(rate, cfg):
    if rate is None:
        return None
    if rate < cfg.MIN_ARTICULATION_SYL_S:
        return "lambat"
    if rate > cfg.FAST_ARTICULATION_SYL_S:
        return "cepat"
    return "wajar"


def score(features, cfg):
    reasons = [error["code"] for error in features.get("errors", [])]
    descriptors = []
    speech = _finite_number(features.get("valid_speech_seconds"))
    coverage = _finite_number(features.get("f0_valid_coverage"), maximum=1)
    pitch_range = _finite_number(features.get("pitch_range_p10_p90_st"))
    phrase_sd = _finite_number(features.get("pitch_sd_st_phrase_median"))
    rate = _finite_number(features.get("articulation_rate_syl_per_s"))
    quality = features.get("audio_quality") or {}
    expressivity = cfg.MODE == "ekspresivitas"
    zones = {"pitch_zone": pitch_zone(pitch_range, phrase_sd, cfg),
             "tempo_zone": tempo_zone(rate, cfg)}
    if speech is None:
        reasons.append("speech_duration_unavailable")
    elif speech < cfg.MIN_SPEECH_SECONDS:
        reasons.append("insufficient_speech_duration")
    if coverage is None or coverage < cfg.MIN_F0_COVERAGE:
        reasons.append("f0_coverage_insufficient")
    if pitch_range is None or phrase_sd is None:
        reasons.append("pitch_features_unavailable")
    jumps = _finite_number(features.get("octave_jump_count"))
    jump_rate = _finite_number(features.get("octave_jump_per_min"))
    if jumps is None or jump_rate is None or not jumps.is_integer():
        reasons.append("pitch_reliability_unavailable")
    elif jumps >= cfg.OCTAVE_REVIEW_MIN_COUNT and jump_rate > cfg.MAX_OCTAVE_JUMPS_PER_MIN:
        reasons.append("pitch_octave_jumps_high")
    clipping = _finite_number(quality.get("clipping_pct"), maximum=100)
    if clipping is None:
        reasons.append("clipping_measurement_unavailable")
    elif clipping > cfg.CLIP_REVIEW_PCT:
        reasons.append("clipping_high")
    for warning in features.get("warnings", []):
        if warning in ("channel_weak", "channel_antiphase", "multichannel_selected_by_rms", "asr_disabled"):
            reasons.append(warning)
    if quality.get("status") not in ("ok", "warning", "insufficient"):
        reasons.append("quality_unavailable")
    elif quality["status"] == "insufficient":
        reasons.append("insufficient_audio_quality")
    if pitch_range is not None and phrase_sd is not None:
        if zones["pitch_zone"] == "ekstrem":
            # Very wide pitch is far more often a tracking error than real expressiveness.
            reasons.append("pitch_above_plausible_zone")
        elif (abs(pitch_range - cfg.MIN_PITCH_RANGE_ST) <= cfg.PITCH_RANGE_BORDER_ST or
                abs(phrase_sd - cfg.MIN_PITCH_SD_ST) <= cfg.PITCH_SD_BORDER_ST):
            reasons.append("pitch_near_pilot_threshold")
        descriptors.append(f"Rentang nada P10–P90 {pitch_range:.2f} semitone (zona {zones['pitch_zone']}).")
        descriptors.append(f"Median variasi nada per proksi frasa {phrase_sd:.2f} semitone.")
    if rate is not None:
        descriptors.append(f"Kecepatan artikulasi terestimasi {rate:.2f} suku kata/detik (tempo {zones['tempo_zone']}).")
    if expressivity:
        if rate is None:
            reasons.append("tempo_unavailable")
        elif abs(rate - cfg.MIN_ARTICULATION_SYL_S) <= cfg.ARTICULATION_BORDER_SYL_S:
            reasons.append("tempo_near_pilot_threshold")
    pause_ratio = _finite_number(features.get("pause_ratio"), maximum=1)
    if pause_ratio is not None:
        descriptors.append(f"Jeda internal mencakup {pause_ratio * 100:.1f}% rentang tuturan.")
    if reasons:
        insufficient = {"audio_too_short", "no_audio_stream", "insufficient_speech_duration", "insufficient_audio_quality"}
        status = "bukti_belum_cukup" if insufficient.intersection(reasons) else "perlu_peninjauan"
        descriptors.append("Bukti belum cukup untuk keputusan biner." if status == "bukti_belum_cukup" else "Hasil memerlukan peninjauan manual.")
        return {"status": status, "abstain_reason": ";".join(dict.fromkeys(reasons)),
                "intonation_item_passed": None, "descriptors": descriptors, **zones}
    pitch_ok = zones["pitch_zone"] == "wajar"
    tempo_ok = zones["tempo_zone"] != "lambat"
    passed = pitch_ok and (tempo_ok or not expressivity)
    if passed:
        descriptors.append("Nada bervariasi wajar" + (" dan tempo tidak lambat." if expressivity else "."))
    elif not pitch_ok:
        descriptors.append("Variasi nada di bawah ambang uji pilot (cenderung datar).")
    else:
        descriptors.append("Nada bervariasi, tetapi tempo bicara di bawah ambang uji pilot (lambat).")
    return {"status": "dapat_dinilai", "abstain_reason": None,
            "intonation_item_passed": bool(passed), "descriptors": descriptors, **zones}
