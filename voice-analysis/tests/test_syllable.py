"""Pilot word cases and normalization conventions."""

import pytest
from app.pipeline.common import AnalysisError
from app.pipeline.syllable import count_text, integer_to_words, normalize_text, syllable


@pytest.mark.parametrize("word, expected", [
    ("mempertanggungjawabkan", 8), ("pantai", 2), ("saya", 2),
    ("makan", 2), ("minum", 2), ("bekerja", 3), ("pelamar", 3),
    ("pengalaman", 4), ("wawancara", 4), ("Indonesia", 5),
    ("pulau", 2), ("pandai", 2), ("amboi", 2), ("survei", 2),
    ("air", 2), ("baik", 2), ("naik", 2), ("main", 2),
    ("daun", 2), ("laut", 2), ("taat", 2), ("koordinasi", 5),
    ("keahlian", 4), ("karyawan", 3), ("komunikasi", 5),
    ("anak-anak", 4), ("sekolah", 3), ("sains", 1),
])
def test_words(word, expected):
    assert syllable(word) == expected


@pytest.mark.parametrize("value, expected", [
    (0, "nol"), (11, "sebelas"), (25, "dua puluh lima"),
    (105, "seratus lima"), (2026, "dua ribu dua puluh enam"),
    (-3, "minus tiga"), (1000000, "satu juta"),
])
def test_integer_normalization(value, expected):
    assert integer_to_words(value) == expected


def test_numbers_abbreviations_and_punctuation():
    normalized, warnings = normalize_text("Saya lulusan S1, punya 2,5 tahun pengalaman di PT. ABC.")
    assert normalized == "saya lulusan es satu punya dua koma lima tahun pengalaman di pe te a be se"
    assert "number_pronunciation_assumed" in warnings


def test_thousands_and_leading_zero():
    assert normalize_text("1.500 012")[0] == "seribu lima ratus nol satu dua"


def test_count_and_override_disclosure():
    assert count_text("saya makan")['syllable_count'] == 4
    assert "pilot_override_mempertanggungjawabkan_8" in count_text("mempertanggungjawabkan")["warnings"]
    assert count_text("")["word_count"] == 0



def test_tech_terms_and_vowelless_tokens_do_not_reject_transcript():
    result = count_text("saya memakai MySQL dan NodeJS serta XYZ")
    assert result["syllable_count"] > 0
    assert count_text("MySQL")["syllable_count"] == 4
    assert "vowelless_token_spelled" in count_text("saya pakai xyz")["warnings"]


def test_many_vowelless_tokens_still_rejected():
    with pytest.raises(AnalysisError):
        count_text("xyz qwr pst")
