"""Shared error and serialization helpers."""

import hashlib
from pathlib import Path


class AnalysisError(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
