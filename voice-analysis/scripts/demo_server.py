"""Local test page: upload a video, play it, watch the real pipeline's result live.

Runs only on 127.0.0.1. The uploaded file is written to a temporary folder,
analysed with the same pipeline as the CLI, and deleted right after.
"""

import argparse
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import re
import shutil
import socket
import sys
import tempfile
import threading
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.config import load_config
from app.pipeline.common import AnalysisError
from app.pipeline.run import analyze_file
from app.pipeline.syllable import count_text

PAGE = ROOT / "app" / "demo" / "index.html"
# Models (Silero state, Whisper, CREPE) are cached per process; one analysis at a time.
ANALYSIS_LOCK = threading.Lock()


def with_local_ffmpeg(cfg):
    """Use the portable ffmpeg in .runtime when none is on PATH."""
    if shutil.which(cfg.FFMPEG) and shutil.which(cfg.FFPROBE):
        return cfg
    found = next((ROOT / ".runtime").glob("ffmpeg/**/bin/ffmpeg.exe"), None) if (ROOT / ".runtime").exists() else None
    if found is None:
        return cfg
    return replace(cfg, FFMPEG=str(found), FFPROBE=str(found.with_name("ffprobe.exe")))


def word_syllables(words):
    """Per-word syllable estimate so the page can show tempo building up while the video plays."""
    output = []
    for word in words or []:
        try:
            syllables = count_text(word.get("word", ""))["syllable_count"]
        except (AnalysisError, TypeError, ValueError):
            syllables = None
        output.append({"word": word.get("word", "").strip(), "start": word.get("start"),
                       "end": word.get("end"), "syllables": syllables})
    return output


class ExclusiveServer(ThreadingHTTPServer):
    """Refuse a port another program already uses.

    The stdlib server sets SO_REUSEADDR, which on Windows lets two servers share
    one port silently; the browser then reaches whichever answers first.
    """
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def open_server(handler, port, attempts=20):
    """Bind the first free port from `port` upward."""
    for candidate in range(port, port + attempts):
        try:
            return ExclusiveServer(("127.0.0.1", candidate), handler)
        except OSError:
            logging.info("port %s sedang dipakai program lain, coba berikutnya", candidate)
    raise SystemExit(f"Tidak ada port kosong antara {port} dan {port + attempts - 1}. Jalankan dengan --port lain.")


def make_handler(cfg):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logging.info("http " + fmt, *args)

        def _send(self, status, body, content_type):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-App", "gasgawe-tes-intonasi")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, status, payload):
            self._send(status, json.dumps(payload, ensure_ascii=False, allow_nan=False), "application/json; charset=utf-8")

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/config":
                self._json(200, {key: getattr(cfg, key) for key in (
                    "MODE", "ANALYSIS_VERSION", "MIN_PITCH_RANGE_ST", "MAX_PITCH_RANGE_ST",
                    "MIN_PITCH_SD_ST", "MAX_PITCH_SD_ST", "MIN_ARTICULATION_SYL_S",
                    "FAST_ARTICULATION_SYL_S", "F0_FLOOR_HZ", "F0_CEILING_HZ")})
            else:
                self._send(404, "Tidak ditemukan", "text/plain; charset=utf-8")

        def do_POST(self):
            if self.path != "/analyze":
                return self._send(404, "Tidak ditemukan", "text/plain; charset=utf-8")
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return self._json(400, {"message": "File kosong. Pilih video lalu coba lagi."})
            if length > cfg.MAX_INPUT_MB * 1024 * 1024:
                return self._json(413, {"message": f"Video lebih besar dari {cfg.MAX_INPUT_MB:.0f} MB."})
            name = self.headers.get("X-Filename") or "video.mp4"
            suffix = Path(name).suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,5}", suffix or ""):
                suffix = ".mp4"
            with tempfile.TemporaryDirectory(prefix="gasgawe_demo_") as temp_dir:
                path = Path(temp_dir) / f"upload{suffix}"
                remaining = length
                with path.open("wb") as handle:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        handle.write(chunk)
                        remaining -= len(chunk)
                if remaining:
                    return self._json(400, {"message": "Upload terputus. Coba lagi."})
                with ANALYSIS_LOCK:
                    result = analyze_file(path, cfg)
            result["input_file"] = name
            result["demo_words"] = word_syllables((result.get("transcription") or {}).get("words"))
            self._json(200, result)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Halaman tes lokal analisis intonasi.")
    parser.add_argument("--port", type=int, default=8791, help="Port awal; jika dipakai program lain, port berikutnya dicoba")
    parser.add_argument("--config", help="JSON override konfigurasi")
    parser.add_argument("--no-browser", action="store_true", help="Jangan buka browser otomatis")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)
    cfg = with_local_ffmpeg(replace(load_config(args.config), PITCH_INCLUDE_CONTOUR=True))
    server = open_server(make_handler(cfg), args.port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Halaman tes berjalan di {url}  (Ctrl+C untuk berhenti)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
