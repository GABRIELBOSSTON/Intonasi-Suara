"""Print one strict JSON result; diagnostic logs go to stderr."""

import argparse
import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import load_config
from app.output import atomic_text_file, validate_output_path
from app.pipeline.run import analyze_file


def main():
    parser = argparse.ArgumentParser(description="Analisis prosodi lokal dalam shadow mode.")
    parser.add_argument("video", help="Path video/audio lokal")
    parser.add_argument("--config", help="JSON override konfigurasi")
    parser.add_argument("--output", help="Simpan salinan JSON UTF-8 secara atomik; stdout tetap JSON")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        cfg = load_config(args.config)
        destination = validate_output_path(args.output, ".json", (args.video, args.config)) if args.output else None
    except (ValueError, OSError, TypeError) as exc:
        parser.error(str(exc))
    result = analyze_file(args.video, cfg)
    serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(serialized)
    if destination is not None:
        try:
            with atomic_text_file(destination) as handle:
                handle.write(serialized + "\n")
        except OSError as exc:
            logging.error("output_write_failed type=%s; JSON tersedia di stdout", type(exc).__name__)
            return 1
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
