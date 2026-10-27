"""Local test page helpers; the HTTP server itself is exercised manually."""

import json
import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import demo_server  # noqa: E402
from app.config import Config


def test_word_syllables_tolerates_unknown_words():
    words = [{"word": " saya", "start": 0.0, "end": 0.3}, {"word": "hmm", "start": 0.3, "end": 0.5}]
    out = demo_server.word_syllables(words)
    assert out[0] == {"word": "saya", "start": 0.0, "end": 0.3, "syllables": 2}
    assert out[1]["syllables"] is None
    assert demo_server.word_syllables(None) == []


def test_server_serves_page_and_config_and_rejects_empty_upload():
    server = ThreadingHTTPServer(("127.0.0.1", 0), demo_server.make_handler(Config(PITCH_INCLUDE_CONTOUR=True)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"Tes Intonasi" in urlopen(base + "/").read()
        config = json.loads(urlopen(base + "/config").read())
        assert config["MODE"] == "ekspresivitas" and config["MIN_ARTICULATION_SYL_S"] == 4.5
        try:
            urlopen(Request(base + "/analyze", data=b"", method="POST"))
            assert False, "upload kosong harus ditolak"
        except Exception as exc:  # urllib raises HTTPError for 400
            assert getattr(exc, "code", None) == 400
    finally:
        server.shutdown()
        server.server_close()


def test_busy_port_is_not_shared():
    handler = demo_server.make_handler(Config())
    first = demo_server.open_server(handler, 18791)
    try:
        port = first.server_address[1]
        second = demo_server.open_server(handler, port)
        try:
            assert second.server_address[1] != port
        finally:
            second.server_close()
    finally:
        first.server_close()
