"""Regression cases for transcript normalization and inference boundaries."""

from types import SimpleNamespace
import numpy as np
import pytest

from app.config import Config
from app.pipeline.common import AnalysisError
from app.pipeline.syllable import count_text, normalize_text, syllable
from app.pipeline.tempo import analyze_tempo
from app.pipeline import transcribe


@pytest.mark.parametrize("text", ["saya makan dan bekerja", "SAYA MAKAN DAN BEKERJA"])
def test_normal_words_are_not_spelled_by_case(text):
    result = count_text(text)
    assert result["normalized_text"] == "saya makan dan bekerja"
    assert result["syllable_count"] == 8
    assert result["word_count"] == 4


@pytest.mark.parametrize("text, normalized", [
    ("2-3 tahun", "dua sampai tiga tahun"),
    ("2020–2024", "dua ribu dua puluh sampai dua ribu dua puluh empat"),
    ("−2", "minus dua"), ("-2--1", "minus dua sampai minus satu"),
    ("café résumé", "cafe resume"), ("２ tahun", "dua tahun"),
    ("2,05", "dua koma nol lima"),
])
def test_ranges_unicode_and_decimals(text, normalized):
    assert normalize_text(text)[0] == normalized


@pytest.mark.parametrize("text", ["B2B", "b2b", "abc123", "saya 東京", "12.34.56", "hmm", "99abc"])
def test_unsupported_tokens_cannot_create_partial_tempo_counts(text):
    with pytest.raises(AnalysisError) as failure:
        count_text(text)
    assert failure.value.code == "syllable_unsupported_text"


@pytest.mark.parametrize("word, count", [
    ("pantainya", 3), ("pulaunya", 3), ("surveiku", 3),
    ("baiknya", 3), ("mempertanggungjawabkannya", 9),
])
def test_diphthongs_before_clitics(word, count):
    assert syllable(word) == count


@pytest.mark.parametrize("span, articulation", [(-2, 1), (2, -1), (float("nan"), 1), (2, float("inf")), (2, 3)])
def test_tempo_rejects_invalid_durations(span, articulation):
    with pytest.raises(AnalysisError):
        analyze_tempo({"word_count": 2, "syllable_count": 4},
                      {"speech_span_seconds": span, "articulation_seconds": articulation})


@pytest.mark.parametrize("words, syllables", [(-1, 2), (2, -1), (2.5, 3), (True, 3), (2, None)])
def test_tempo_rejects_invalid_counts(words, syllables):
    with pytest.raises(AnalysisError):
        analyze_tempo({"word_count": words, "syllable_count": syllables},
                      {"speech_span_seconds": 2, "articulation_seconds": 1})


def test_invalid_timestamp_is_null_without_losing_global_transcript(monkeypatch):
    class Model:
        def transcribe(self, audio, **kwargs):
            words = [SimpleNamespace(word=" saya", start=0.1, end=0.5, probability=0.9),
                     SimpleNamespace(word=" bekerja", start=0.5, end=20, probability=float("nan"))]
            return iter([SimpleNamespace(text=" saya bekerja", words=words)]), SimpleNamespace(language="id")
    monkeypatch.setattr(transcribe, "_load_model", lambda *args: (Model(), {}))
    result = transcribe.transcribe_audio(np.zeros(80000, dtype=np.float32), 16000, {"segments": [[1, 4]]}, Config())
    assert result["text"] == "saya bekerja"
    assert result["words"][0]["start"] == pytest.approx(1.1)
    assert result["words"][1]["start"] is None
    assert result["words"][1]["end"] is None
    assert result["words"][1]["probability"] is None
    assert "asr_word_timestamps_invalid" in result["warnings"]


@pytest.mark.parametrize("audio, sample_rate, segments", [
    (np.zeros(16000), 8000, [[0, 1]]),
    (np.zeros((16000, 2)), 16000, [[0, 1]]),
    (np.array([np.nan] * 16000), 16000, [[0, 1]]),
    (np.zeros(16000), 16000, [[-1, 1]]),
    (np.zeros(16000), 16000, [[0, 2]]),
    (np.zeros(16000), 16000, [[0.4, 0.8], [0, 0.4]]),
])
def test_invalid_asr_input_rejected_before_model_download(monkeypatch, audio, sample_rate, segments):
    monkeypatch.setattr(transcribe, "_load_model", lambda *args: pytest.fail("model must not load"))
    with pytest.raises(AnalysisError):
        transcribe.transcribe_audio(audio, sample_rate, {"segments": segments}, Config())


def test_local_model_cache_identity_changes_with_file_replacement(monkeypatch, tmp_path):
    model = tmp_path / "model.bin"
    model.write_bytes(b"original")
    calls = []
    monkeypatch.setattr(transcribe, "_cached_model", lambda *args: calls.append(args) or (None, {}))
    args = ("repo", "revision", str(tmp_path), "cache", True, 1, "int8")
    transcribe._load_model(*args)
    model.write_bytes(b"replacement with different size")
    transcribe._load_model(*args)
    assert calls[0][-1] != calls[1][-1]
