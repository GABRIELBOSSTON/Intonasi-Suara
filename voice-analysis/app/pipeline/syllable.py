"""Approximate Indonesian syllable counts with explicit pilot exceptions."""

import re
import unicodedata

from .common import AnalysisError

RULE_VERSION = "id-rule-0.3.0-pilot-override"
VOWELS = "aiueo"
DIPHTHONGS = {"ai", "au", "oi", "ei"}
# The first override is required by the pilot acceptance specification.
OVERRIDES = {"mempertanggungjawabkan": 8, "air": 2, "baik": 2, "main": 2,
             "naik": 2, "lain": 2, "kain": 2, "pahit": 2, "daun": 2,
             "laut": 2, "jauh": 2, "taat": 2, "sains": 1, "survei": 2,
             "survei-survei": 4, "boikot": 2}
ABBREVIATIONS = {"sd": "es de", "smp": "es em pe", "sma": "es em a",
    "smk": "es em ka", "s1": "es satu", "s2": "es dua", "s3": "es tiga",
    "d3": "de tiga", "d4": "de empat", "pt": "pe te", "cv": "se ve",
    "ktp": "ka te pe", "sim": "sim", "hrd": "ha er de", "it": "ai ti",
    "bumn": "be u em en", "bpjs": "be pe je es", "dll": "dan lain lain",
    "dsb": "dan sebagainya", "dr": "dokter", "rp": "rupiah",
    "abc": "a be se", "ai": "a i",
    # Satuan umum di video profil (tinggi/berat badan, jarak, waktu).
    "cm": "senti", "kg": "kilo", "m": "meter", "km": "kilometer",
    "gr": "gram", "mm": "milimeter", "th": "tahun", "thn": "tahun",
    # Istilah teknis yang sering muncul di video pelamar IT.
    "mysql": "mai es kyu el", "sql": "es kyu el", "nosql": "no es kyu el",
    "js": "je es", "html": "ha te em el", "css": "se es es", "php": "pe ha pe",
    "aws": "a we es", "gcp": "ge se pe", "ui": "yu ai", "ux": "yu eks"}
LETTER_NAMES = dict(zip("abcdefghijklmnopqrstuvwxyz", ["a", "be", "se", "de", "e", "ef", "ge", "ha", "i", "je", "ka", "el", "em", "en", "o", "pe", "ki", "er", "es", "te", "u", "ve", "we", "eks", "ye", "zet"]))
DIGITS = ["nol", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
TOKEN = re.compile(r"[+-]?\d+(?:[.,]\d+)*(?!\w)|[^\W_]+", re.UNICODE)
NUMBER_RANGE = re.compile(r"(?<![\w.,])([+-]?\d+(?:[.,]\d+)*)\s*[-–—]\s*([+-]?\d+(?:[.,]\d+)*)(?![\w.,])")
CLITICS = ("nya", "lah", "kah", "pun", "ku", "mu")


def integer_to_words(value):
    if type(value) is not int:
        raise TypeError("Bilangan harus integer")
    if value < 0:
        return "minus " + integer_to_words(-value)
    if value < 10:
        return DIGITS[value]
    if value == 10:
        return "sepuluh"
    if value == 11:
        return "sebelas"
    if value < 20:
        return integer_to_words(value - 10) + " belas"
    if value < 100:
        quotient, remainder = divmod(value, 10)
        return integer_to_words(quotient) + " puluh" + (" " + integer_to_words(remainder) if remainder else "")
    for scale, name in ((10**12, "triliun"), (10**9, "miliar"), (10**6, "juta"), (1000, "ribu"), (100, "ratus")):
        if value >= scale:
            quotient, remainder = divmod(value, scale)
            prefix = "se" + name if quotient == 1 and scale in (100, 1000) else integer_to_words(quotient) + " " + name
            return prefix + (" " + integer_to_words(remainder) if remainder else "")
    raise ValueError("Bilangan tidak didukung")


def number_to_words(token):
    sign = "minus " if token.startswith("-") else ""
    token = token.lstrip("+-")
    # Indonesian convention: dots group thousands; comma marks decimals.
    if re.fullmatch(r"\d{1,3}(?:\.\d{3})+(?:,\d+)?", token):
        token = token.replace(".", "")
    if re.fullmatch(r"\d+[.,]\d+", token):
        whole, fraction = re.split(r"[.,]", token)
        return sign + number_to_words(whole) + " koma " + " ".join(DIGITS[int(d)] for d in fraction)
    if not token.isdigit():
        raise ValueError("Format bilangan ambigu/tidak didukung")
    if (len(token) > 1 and token.startswith("0")) or len(token) > 15:
        return sign + " ".join(DIGITS[int(d)] for d in token)
    return sign + integer_to_words(int(token))


def normalize_text(text):
    if not isinstance(text, str):
        raise TypeError("Transkrip harus berupa string")
    parts, warnings = [], []
    original = text
    text = unicodedata.normalize("NFKC", text).replace("−", "-")
    # Strip Latin diacritics without dropping letters or splitting a word.
    folded = []
    for char in text:
        if "LATIN" in unicodedata.name(char, ""):
            folded.extend(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c))
        elif char.isdecimal():
            folded.append(str(unicodedata.decimal(char)))
        else:
            folded.append(char)
    text = "".join(folded)
    if original != text:
        warnings.append("unicode_normalization_applied")

    def expand_range(match):
        try:
            words = number_to_words(match[1]) + " sampai " + number_to_words(match[2])
        except ValueError:
            return match[0]
        warnings.append("number_range_pronunciation_assumed")
        return " " + words + " "

    text = NUMBER_RANGE.sub(expand_range, text)
    for token in TOKEN.findall(text):
        lower = token.lower()
        if lower in ABBREVIATIONS:
            parts.append(ABBREVIATIONS[lower])
            warnings.append("abbreviation_pronunciation_assumed")
        elif re.fullmatch(r"[+-]?\d+(?:[.,]\d+)*", token):
            try:
                parts.append(number_to_words(token))
                warnings.append("number_pronunciation_assumed")
            except ValueError:
                parts.append(lower)
                warnings.append("number_format_unsupported")
        elif token.isupper() and token.isascii() and token.isalpha() and not any(char in VOWELS for char in lower):
            parts.append(" ".join(LETTER_NAMES.get(char.lower(), char.lower()) for char in token))
            warnings.append("uppercase_acronym_spelled")
        else:
            parts.append(lower)
            if any(char.isdigit() for char in token):
                warnings.append("unknown_alphanumeric_token")
            elif not token.isascii():
                warnings.append("unsupported_script_token")
            elif token.isupper() and len(token) > 1:
                warnings.append("uppercase_word_preserved")
    return " ".join(parts), sorted(set(warnings))


def syllable(word):
    word = word.lower()
    if word in OVERRIDES:
        return OVERRIDES[word]
    if "-" in word:
        return sum(syllable(part) for part in word.split("-"))
    for suffix in CLITICS:
        if word.endswith(suffix) and len(word) > len(suffix):
            stem = word[:-len(suffix)]
            if stem[-2:] in DIPHTHONGS or stem in OVERRIDES:
                return syllable(stem) + syllable(suffix)
    count = sum(char in VOWELS for char in word)
    # Most productive Indonesian diphthongs occur word-finally.
    if word[-2:] in DIPHTHONGS:
        count -= 1
    return max(0, count)


def count_text(text):
    normalized, warnings = normalize_text(text)
    words = normalized.split()
    counts = [syllable(word) for word in words]
    # A vowelless Latin token (e.g. an unknown acronym) is spelled letter by
    # letter instead of rejecting the whole transcript; too many still fails.
    fallback = 0
    for index, word in enumerate(words):
        # Fillers such as "hmm" are not acronyms and keep the old rejection.
        if counts[index] == 0 and word.isascii() and word.isalpha() and not re.fullmatch(r"[hmn]+", word):
            counts[index] = sum(syllable(LETTER_NAMES[char]) for char in word)
            fallback += 1
    if fallback:
        warnings.append("vowelless_token_spelled")
    unsupported = {"unknown_alphanumeric_token", "number_format_unsupported", "unsupported_script_token"}
    if (unsupported.intersection(warnings) or any(count == 0 for count in counts)
            or fallback > max(1, 0.05 * len(words))):
        raise AnalysisError("syllable_unsupported_text", "Transkrip berisi token yang pelafalannya tidak dapat diperkirakan; hitungan suku kata dan tempo perlu audit manual.")
    if any(word.startswith("mempertanggungjawabkan") for word in words):
        warnings.append("pilot_override_mempertanggungjawabkan_8")
    return {"normalized_text": normalized, "word_count": len(words),
            "syllable_count": sum(counts), "rule_version": RULE_VERSION,
            "warnings": sorted(set(warnings))}
