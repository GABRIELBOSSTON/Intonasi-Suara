"""QC, timing, abstention, stage failures and real decoder smoke tests."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

from app.config import Config, load_config
from app.pipeline import extract, pitch, transcribe, vad
from app.pipeline.common import AnalysisError
from app.pipeline.quality import analyze_quality, inspect_original
from app.pipeline.run import analyze_file
from app.pipeline.scoring import score
from app.pipeline.tempo import analyze_tempo
from scripts.batch_pilot import run_batch


def good_features():
    return {"valid_speech_seconds": 40, "f0_valid_coverage": 0.8,
        "pitch_range_p10_p90_st": 7, "pitch_sd_st_phrase_median": 3,
        "octave_jump_count": 0, "octave_jump_per_min": 0,
        "audio_quality": {"status": "ok", "clipping_pct": 0},
        "articulation_rate_syl_per_s": 6.5, "pause_ratio": 0.1, "errors": [], "warnings": []}


def test_pause_threshold_and_matching_rate_denominators():
    stats = vad.summarize_segments([[1, 2], [2.1, 3], [3.25, 5], [8, 10]], 12, Config())
    assert stats["speech_span_seconds"] == 9
    assert stats["articulation_seconds"] == 5.75
    assert stats["pause_ratio"] == pytest.approx(3.25 / 9)
    assert stats["long_pause_per_min"] == pytest.approx(60 / 9)
    assert stats["phrases"][0] == [1, 3]
    rates = analyze_tempo({"syllable_count": 18, "word_count": 9}, stats)
    assert rates["speech_rate_syl_per_s"] == 2
    assert rates["articulation_rate_syl_per_s"] == pytest.approx(18 / 5.75)


def test_no_pause_vs_no_speech():
    continuous = vad.summarize_segments([[2, 8]], 10, Config())
    assert continuous["pause_ratio"] == 0
    assert continuous["pause_median_seconds"] is None
    assert vad.summarize_segments([], 10, Config())["pause_ratio"] is None


def test_original_clipping_and_antiphase():
    wave = np.sin(2 * np.pi * np.arange(16000) * 200 / 16000)
    info = inspect_original(np.column_stack([wave, -wave]), Config())
    assert info["selected_channel"] is not None
    assert "channel_antiphase" in info["warnings"]
    assert info["clipping_pct"] > 0


def test_snr_and_no_reference():
    rng = np.random.default_rng(12)
    audio = rng.normal(0, 0.01, 16000 * 4)
    audio[16000:48000] += 0.2 * np.sin(2 * np.pi * np.arange(32000) * 180 / 16000)
    info = inspect_original(audio, Config())
    quality = analyze_quality(audio, 16000, [[1, 3]], info, Config())
    assert 20 < quality["snr_est_db"] < 26
    assert quality["status"] == "insufficient"
    missing = analyze_quality(audio, 16000, [[0, 4]], info, Config())
    assert missing["snr_est_db"] is None
    assert missing["snr_unavailable_reason"] == "snr_no_noise_reference"


def test_low_snr_alone_does_not_fail_pitch():
    features = good_features()
    features["audio_quality"]["status"] = "warning"
    features["warnings"] = ["snr_low"]
    assert score(features, Config())["intonation_item_passed"] is True


@pytest.mark.parametrize("change, reason", [
    ({"f0_valid_coverage": 0.1}, "f0_coverage_insufficient"),
    ({"pitch_range_p10_p90_st": 4.1}, "pitch_near_pilot_threshold"),
    ({"octave_jump_count": 8, "octave_jump_per_min": 12}, "pitch_octave_jumps_high"),
    ({"valid_speech_seconds": 2}, "insufficient_speech_duration"),
])
def test_abstain_is_null(change, reason):
    result = score({**good_features(), **change}, Config())
    assert result["intonation_item_passed"] is None
    assert reason in result["abstain_reason"]


def test_pitch_only_and_expressivity_modes():
    features = good_features()
    features["articulation_rate_syl_per_s"] = 3.0
    # Mode intonasi: tempo hanya deskripsi.
    assert score(features, Config(MODE="intonasi"))["intonation_item_passed"] is True
    # Mode ekspresivitas (default): nada wajar tetapi lambat tidak terpenuhi.
    slow = score(features, Config())
    assert slow["intonation_item_passed"] is False
    assert slow["tempo_zone"] == "lambat"
    features["pitch_range_p10_p90_st"] = 0.3
    features["pitch_sd_st_phrase_median"] = 0.1
    flat = score(features, Config(MODE="intonasi"))
    assert flat["intonation_item_passed"] is False
    assert flat["pitch_zone"] == "datar"
    assert score(good_features(), Config())["intonation_item_passed"] is True


@pytest.mark.parametrize("change, reason", [
    ({"pitch_range_p10_p90_st": 25.0, "pitch_sd_st_phrase_median": 9.0}, "pitch_above_plausible_zone"),
    ({"articulation_rate_syl_per_s": 4.35}, "tempo_near_pilot_threshold"),
    ({"articulation_rate_syl_per_s": None}, "tempo_unavailable"),
])
def test_extreme_pitch_and_borderline_tempo_go_to_review(change, reason):
    result = score({**good_features(), **change}, Config())
    assert result["status"] == "perlu_peninjauan"
    assert result["intonation_item_passed"] is None
    assert reason in result["abstain_reason"]


def test_extreme_pitch_never_passes_even_in_pitch_only_mode():
    features = {**good_features(), "pitch_range_p10_p90_st": 25.0}
    result = score(features, Config(MODE="intonasi"))
    assert result["intonation_item_passed"] is None
    assert result["pitch_zone"] == "ekstrem"


def test_config_validation(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"MODE":"bad"}')
    with pytest.raises(ValueError):
        load_config(path)
    path.write_text('{"PAUSE_MIN_SECONDS":"0.25"}')
    with pytest.raises(ValueError):
        load_config(path)
    with pytest.raises(ValueError):
        Config(PAUSE_MIN_SECONDS=0.05)


@pytest.fixture
def isolated_pipeline(monkeypatch):
    cfg = Config()
    audio = np.zeros(40 * 16000, dtype=np.float32)
    info = {"clipping_pct": 0, "warnings": []}
    monkeypatch.setattr(extract, "extract_audio", lambda *args: {"audio": audio, "sample_rate": 16000, "audio_info": info})
    monkeypatch.setattr(vad, "analyze_vad", lambda *args: vad.summarize_segments([[0, 40]], 40, cfg))
    monkeypatch.setattr(pitch, "analyze_pitch", lambda *args: {
        "pitch_range_p10_p90_st": 7, "pitch_sd_st_phrase_median": 3,
        "f0_valid_coverage": 0.8, "octave_jump_count": 0, "octave_jump_per_min": 0,
        "evidence_timestamps": [[0, 40]]})
    return cfg


def test_asr_failure_preserves_pitch_and_null_tempo(monkeypatch, isolated_pipeline):
    def fail(*args):
        raise RuntimeError("simulated offline model")
    monkeypatch.setattr(transcribe, "transcribe_audio", fail)
    result = analyze_file("fake.mp4", isolated_pipeline)
    assert result["pitch_range_p10_p90_st"] == 7
    assert result["transcription"] is None
    assert result["articulation_rate_syl_per_s"] is None
    assert result["intonation_item_passed"] is None
    assert "transcribe_failed" in result["abstain_reason"]
    json.dumps(result, allow_nan=False)


def test_lazy_asr_failure_occurs_inside_module(monkeypatch):
    class LazyFailure:
        def transcribe(self, *args, **kwargs):
            def segments():
                raise RuntimeError("decoder failed during iteration")
                yield
            return segments(), None
    monkeypatch.setattr(transcribe, "_load_model", lambda *args: (LazyFailure(), {}))
    cfg = Config()
    stats = vad.summarize_segments([[0, 3]], 3, cfg)
    with pytest.raises(RuntimeError, match="during iteration"):
        transcribe.transcribe_audio(np.zeros(48000, dtype=np.float32), 16000, stats, cfg)


def test_successful_orchestration(monkeypatch, isolated_pipeline):
    monkeypatch.setattr(transcribe, "transcribe_audio", lambda *args: {"text": "saya bekerja", "words": []})
    result = analyze_file("fake.mp4", isolated_pipeline)
    # Nada wajar, tetapi 5 suku kata untuk rekaman panjang = tempo lambat; mode
    # ekspresivitas (default) harus meneruskan tempo sampai keputusan akhir.
    assert result["status"] == "dapat_dinilai"
    assert result["pitch_zone"] == "wajar"
    assert result["tempo_zone"] == "lambat"
    assert result["intonation_item_passed"] is False
    assert result["syllable_count"] == 5
    assert result["shadow_mode"] is True
    assert result["energy_end_vs_start_db"] is None
    for prohibited in ("tidak niat", "malas", "gugup"):
        assert prohibited not in " ".join(result["descriptors"]).lower()


def test_vad_failure_preserves_original_qc(monkeypatch, isolated_pipeline):
    def fail(*args):
        raise RuntimeError("vad")
    monkeypatch.setattr(vad, "analyze_vad", fail)
    result = analyze_file("fake.mp4", isolated_pipeline)
    assert result["audio_quality"]["clipping_pct"] == 0
    assert result["valid_speech_seconds"] is None
    assert result["module_status"]["pitch"] == "skipped"
    assert result["intonation_item_passed"] is None


def test_missing_ffmpeg_returns_json(tmp_path):
    path = tmp_path / "input.wav"
    sf.write(path, np.zeros(32000), 16000)
    result = analyze_file(path, Config(FFMPEG="ffmpeg_intentionally_missing_123.exe"))
    assert result["errors"][0]["code"] == "ffmpeg_not_found"
    assert result["intonation_item_passed"] is None


def test_cli_missing_file_is_json(tmp_path):
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run([sys.executable, str(root / "scripts/analyze_file.py"), str(tmp_path / "missing.mp4")], capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 1
    assert json.loads(completed.stdout)["errors"][0]["code"] == "file_not_found"
    assert "stage=extract" in completed.stderr


def test_batch_keeps_failed_rows(tmp_path):
    for name in ("b.mp4", "a.mp4"):
        (tmp_path / name).write_bytes(b"broken")
    output = tmp_path / "results.csv"
    summary = run_batch(tmp_path, output, Config(FFMPEG="nonexistent_ffmpeg_123"))
    assert summary["files"] == 2
    assert summary["files_with_errors"] == 2
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert Path(rows[0]["input_file"]).name == "a.mp4"
    assert "audio_quality.snr_est_db" in rows[0]


@pytest.fixture
def decoder():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    return shutil.which("ffmpeg")


def test_real_decoder_no_audio(decoder, tmp_path):
    video = tmp_path / "no_audio.mp4"
    subprocess.run([decoder, "-nostdin", "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=64x64:d=1", "-an", "-c:v", "mpeg4", str(video)], check=True)
    result = analyze_file(video)
    assert result["status"] == "bukti_belum_cukup"
    assert result["errors"][0]["code"] == "no_audio_stream"


def test_real_decoder_short_audio(decoder, tmp_path):
    source = tmp_path / "short.wav"
    sf.write(source, np.zeros(1600), 16000)
    result = analyze_file(source)
    assert result["errors"][0]["code"] == "audio_too_short"


def test_real_decoder_stereo_antiphase(decoder, tmp_path):
    t = np.arange(48000 * 2) / 48000
    wave = 0.2 * np.sin(2 * np.pi * 200 * t)
    source = tmp_path / "stereo.wav"
    sf.write(source, np.column_stack([wave, -wave]), 48000, subtype="FLOAT")
    work = tmp_path / "work"
    work.mkdir()
    result = extract.extract_audio(source, work, Config())
    assert result["sample_rate"] == 16000
    assert result["audio_info"]["original_channels"] == 2
    assert np.sqrt(np.mean(result["audio"] ** 2)) > 0.1


def test_real_silero_silence_offline():
    result = vad.analyze_vad(np.zeros(32000, dtype=np.float32), 16000, Config())
    assert result["segments"] == []
    assert len(result["model_sha256"]) == 64


def test_tempo_not_reported_for_fraction_of_second_speech():
    from app.pipeline.common import AnalysisError
    from app.pipeline.tempo import analyze_tempo
    counts = {"syllable_count": 11, "word_count": 4}
    vad = {"speech_span_seconds": 0.16, "articulation_seconds": 0.16}
    with pytest.raises(AnalysisError):
        analyze_tempo(counts, vad, 3.0)
    assert analyze_tempo(counts, vad)["syllable_count"] == 11



def test_fast_speech_with_good_pitch_passes():
    features = {**good_features(), "articulation_rate_syl_per_s": 10.0}
    result = score(features, Config())
    assert result["tempo_zone"] == "cepat"
    assert result["intonation_item_passed"] is True
