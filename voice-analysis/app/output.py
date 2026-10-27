"""Safe local output paths and atomic text file replacement."""

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile


def validate_output_path(path, suffix, protected=()):
    destination = Path(path)
    if destination.suffix.lower() != suffix:
        raise ValueError(f"File output harus berekstensi {suffix}")
    if destination.exists() and not destination.is_file():
        raise ValueError("Path output harus berupa file")
    for source in protected:
        if source is None:
            continue
        source = Path(source)
        identical = destination.resolve() == source.resolve()
        if not identical and destination.exists() and source.exists():
            identical = destination.samefile(source)
        if identical:
            raise ValueError("Output tidak boleh menimpa input atau konfigurasi")
    return destination


@contextmanager
def atomic_text_file(path, encoding="utf-8", newline=None):
    """Keep any existing destination intact until writing completes."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, newline=newline,
                prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
                delete=False) as handle:
            temporary = Path(handle.name)
            yield handle
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def safe_csv_cell(value):
    """Escape text interpreted as spreadsheet formulas; preserve numbers."""
    if isinstance(value, str):
        stripped = value.lstrip()
        if (stripped.startswith(("=", "+", "-", "@")) or
                value.startswith(("\t", "\r", "\n"))):
            return "'" + value
    return value
