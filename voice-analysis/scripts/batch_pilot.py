"""Analyze videos sequentially and flatten all results to one CSV."""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import load_config
from app.output import atomic_text_file, safe_csv_cell, validate_output_path
from app.pipeline.run import NUMERIC_FIELDS, STAGES, analyze_file

LOG = logging.getLogger("voice_analysis.batch")


def flatten(value, prefix=""):
    output = {}
    if isinstance(value, dict) and value:
        for key, item in value.items():
            output.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    else:
        output[prefix] = json.dumps(value, ensure_ascii=False, allow_nan=False) if isinstance(value, (dict, list)) else value
    return output


def _unexpected_failure(path, cfg, exc):
    result = {name: None for name in NUMERIC_FIELDS}
    result.update({"input_file": str(path), "analysis_version": cfg.ANALYSIS_VERSION,
        "mode": cfg.MODE, "shadow_mode": True, "status": "perlu_peninjauan",
        "abstain_reason": "unexpected_pipeline_failure", "intonation_item_passed": None,
        "descriptors": ["Analisis file gagal; hasil memerlukan peninjauan."],
        "audio_quality": {"status": None, "snr_est_db": None, "clipping_pct": None},
        "errors": [{"module": "run", "code": "unexpected_pipeline_failure",
                    "message": "Analisis file gagal di luar tahap modul; batch dilanjutkan.",
                    "exception_type": type(exc).__name__}],
        "warnings": [], "evidence_timestamps": [],
        "module_status": {name: "skipped" for name in STAGES}})
    return result


def run_batch(folder, output, cfg, protected_paths=()):
    folder, output = Path(folder), Path(output)
    if not folder.is_dir():
        raise ValueError("Folder input tidak ditemukan")
    files = sorted((path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in cfg.VIDEO_EXTENSIONS), key=lambda path: (path.name.casefold(), path.name))
    if not files:
        raise ValueError("Tidak ada video dengan ekstensi yang didukung di folder input")
    output = validate_output_path(output, ".csv", (*files, *protected_paths))
    output.parent.mkdir(parents=True, exist_ok=True)
    columns, failed = set(), 0
    # Spool rows to avoid keeping all transcripts in RAM.
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as spool:
        for path in files:
            try:
                result = analyze_file(path, cfg)
                row = flatten(result)
                encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
                has_errors = bool(result["errors"])
            except Exception as exc:
                LOG.error("unexpected_pipeline_failure type=%s", type(exc).__name__)
                result = _unexpected_failure(path, cfg, exc)
                row = flatten(result)
                encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
                has_errors = True
            failed += has_errors
            columns.update(row)
            spool.write(encoded + "\n")
        leading = ["input_file", "analysis_version", "status", "abstain_reason", "intonation_item_passed"]
        names = leading + sorted(columns.difference(leading))
        spool.seek(0)
        with atomic_text_file(output, newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=names)
            writer.writeheader()
            for line in spool:
                writer.writerow({key: safe_csv_cell(value) for key, value in json.loads(line).items()})
    return {"files": len(files), "files_with_errors": failed, "output_csv": str(output)}


def main():
    parser = argparse.ArgumentParser(description="Batch video nonrekursif → CSV; error tidak menghentikan batch.")
    parser.add_argument("folder")
    parser.add_argument("output_csv")
    parser.add_argument("--config")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        cfg = load_config(args.config)
        result = run_batch(args.folder, args.output_csv, cfg, protected_paths=(args.config,))
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False))
    return 1 if result["files_with_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
