"""Output safety, partial batch failures and UTF-8 CLI regressions."""

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from app.config import Config
from app.output import atomic_text_file, safe_csv_cell, validate_output_path
from scripts import batch_pilot


def _result(path, **extra):
    return {"input_file": str(path), "analysis_version": Config().ANALYSIS_VERSION,
            "status": "dapat_dinilai", "abstain_reason": None,
            "intonation_item_passed": True, "errors": [], **extra}


@pytest.mark.parametrize("value", ["=1+1", "+SUM(A1)", "-SUM(A1)", "@SUM(A1)",
                                  "  =1+1", "\tformula", "\rformula", "\nformula"])
def test_csv_formula_text_is_escaped(value):
    assert safe_csv_cell(value) == "'" + value


@pytest.mark.parametrize("value", [-12.4, 0, True, None, "saya bekerja", "[{\"key\": 1}]"])
def test_csv_numbers_and_ordinary_text_preserved(value):
    assert safe_csv_cell(value) == value


def test_atomic_file_preserves_previous_output_on_write_failure(tmp_path):
    output = tmp_path / "results.csv"
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(OSError, match="disk write"):
        with atomic_text_file(output) as handle:
            handle.write("partial")
            raise OSError("disk write failed")
    assert output.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_file_preserves_previous_output_on_replace_failure(tmp_path, monkeypatch):
    from app import output as output_module
    destination = tmp_path / "results.json"
    destination.write_text("previous", encoding="utf-8")
    def fail(*args):
        raise PermissionError("file open in another app")
    monkeypatch.setattr(output_module.os, "replace", fail)
    with pytest.raises(PermissionError):
        with atomic_text_file(destination) as handle:
            handle.write("replacement")
    assert destination.read_text(encoding="utf-8") == "previous"
    assert not list(tmp_path.glob("*.tmp"))


def test_output_protects_hardlink_alias(tmp_path):
    source = tmp_path / "input.json"
    alias = tmp_path / "alias.json"
    source.write_text("important", encoding="utf-8")
    try:
        alias.hardlink_to(source)
    except OSError as exc:
        pytest.skip(f"Hard links unavailable: {type(exc).__name__}")
    with pytest.raises(ValueError, match="menimpa"):
        validate_output_path(alias, ".json", (source,))


def test_batch_does_not_overwrite_excluded_audio_file(tmp_path, monkeypatch):
    (tmp_path / "input.mp4").write_bytes(b"video")
    audio = tmp_path / "original.wav"
    audio.write_bytes(b"original audio")
    def forbidden(*args):
        raise AssertionError("analysis must not start")
    monkeypatch.setattr(batch_pilot, "analyze_file", forbidden)
    with pytest.raises(ValueError, match=".csv"):
        batch_pilot.run_batch(tmp_path, audio, Config())
    assert audio.read_bytes() == b"original audio"


def test_batch_protects_config_path_even_if_named_csv(tmp_path):
    (tmp_path / "input.mp4").write_bytes(b"video")
    config = tmp_path / "config.csv"
    config.write_text("{}")
    with pytest.raises(ValueError, match="menimpa"):
        batch_pilot.run_batch(tmp_path, config, Config(), protected_paths=(config,))
    assert config.read_text() == "{}"


def test_batch_unexpected_file_exception_does_not_stop_other_files(tmp_path, monkeypatch):
    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).write_bytes(b"video")
    def analyze(path, cfg):
        if path.name == "a.mp4":
            raise RuntimeError("unexpected top-level failure")
        return _result(path)
    monkeypatch.setattr(batch_pilot, "analyze_file", analyze)
    output = tmp_path / "results.csv"
    summary = batch_pilot.run_batch(tmp_path, output, Config())
    with output.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert summary["files_with_errors"] == 1
    assert len(rows) == 2
    assert rows[0]["intonation_item_passed"] == ""
    assert rows[0]["abstain_reason"] == "unexpected_pipeline_failure"
    assert rows[0]["pitch_range_p10_p90_st"] == ""
    assert rows[1]["intonation_item_passed"] == "True"


def test_batch_csv_escapes_transcript_and_preserves_negative_numbers(tmp_path, monkeypatch):
    (tmp_path / "input.mp4").write_bytes(b"video")
    monkeypatch.setattr(batch_pilot, "analyze_file", lambda path, cfg:
        _result(path, transcription={"text": "=1+1"}, energy_end_vs_start_db=-2.0))
    output = tmp_path / "results.csv"
    batch_pilot.run_batch(tmp_path, output, Config())
    with output.open(encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["transcription.text"] == "'=1+1"
    assert row["energy_end_vs_start_db"] == "-2.0"


def test_batch_existing_csv_survives_failed_serialized_row_write(tmp_path, monkeypatch):
    (tmp_path / "input.mp4").write_bytes(b"video")
    monkeypatch.setattr(batch_pilot, "analyze_file", lambda path, cfg: _result(path))
    original_writerow = csv.DictWriter.writerow
    def fail_row(writer, row):
        if row.get("input_file") != "input_file":
            raise OSError("disk write failed")
        return original_writerow(writer, row)
    monkeypatch.setattr(csv.DictWriter, "writerow", fail_row)
    output = tmp_path / "results.csv"
    output.write_text("previous", encoding="utf-8")
    with pytest.raises(OSError, match="disk write"):
        batch_pilot.run_batch(tmp_path, output, Config())
    assert output.read_text(encoding="utf-8") == "previous"


def test_single_cli_output_writes_utf8_and_keeps_stdout(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_file.py"
    output = tmp_path / "nested" / "hasil.json"
    completed = subprocess.run([sys.executable, str(script), str(tmp_path / "profil-é.mp4"),
                                "--output", str(output)], capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 1
    assert json.loads(completed.stdout) == json.loads(output.read_text(encoding="utf-8"))
    assert b"\xc3\xa9" in output.read_bytes()
    assert not output.read_bytes().startswith(b"\xef\xbb\xbf")


def test_single_cli_protects_config_from_output(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_file.py"
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    completed = subprocess.run([sys.executable, str(script), str(tmp_path / "missing.mp4"),
        "--config", str(config), "--output", str(config)], capture_output=True, text=True, encoding="utf-8")
    assert completed.returncode == 2
    assert "menimpa" in completed.stderr
    assert config.read_text(encoding="utf-8") == "{}"
