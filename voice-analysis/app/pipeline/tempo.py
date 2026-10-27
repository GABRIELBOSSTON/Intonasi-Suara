"""Global rates use matching transcript and VAD speech spans."""

from .common import AnalysisError
import math


def analyze_tempo(counts, vad, min_speech_seconds=0.0):
    span, articulation = vad["speech_span_seconds"], vad["articulation_seconds"]
    syllables, words = counts.get("syllable_count"), counts.get("word_count")
    durations_valid = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                          and math.isfinite(value) and value > 0 for value in (span, articulation))
    counts_valid = all(type(value) is int and value > 0 for value in (syllables, words))
    if not durations_valid or not counts_valid:
        raise AnalysisError("tempo_insufficient_data", "Durasi atau jumlah suku kata tidak cukup.")
    if articulation < min_speech_seconds:
        # Rates over a fraction of a second are noise, not speaking tempo.
        raise AnalysisError("tempo_insufficient_data", "Durasi bicara terlalu pendek untuk tempo.")
    if articulation > span:
        raise AnalysisError("tempo_invalid_duration", "Durasi artikulasi tidak boleh melebihi rentang tuturan.")
    return {"speech_rate_syl_per_s": counts["syllable_count"] / span,
            "articulation_rate_syl_per_s": counts["syllable_count"] / articulation,
            "speech_rate_wpm": counts["word_count"] * 60 / span,
            "articulation_rate_wpm": counts["word_count"] * 60 / articulation,
            "word_count": counts["word_count"], "syllable_count": counts["syllable_count"]}
