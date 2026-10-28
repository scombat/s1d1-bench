"""Prompt normalization and key, identical to the research pipeline (pipeline/01_interim.py)."""

import hashlib
import unicodedata
from pathlib import Path

KEY_HEX = 16


def normalize(text: str) -> str:
    """NFC, `\\r\\n` -> `\\n`, leading/trailing whitespace removed. Nothing else."""
    return unicodedata.normalize("NFC", text).replace("\r\n", "\n").strip()


def prompt_key(normalized: str) -> str:
    """First 16 hex chars of the sha256 of the UTF-8 encoded, already normalized prompt."""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:KEY_HEX]


def load_exclusions(path: Path | str) -> frozenset[str]:
    """Prompt keys listed one per line; surrounding whitespace is stripped, blank lines are ignored."""
    with open(path, encoding="utf-8") as f:
        return frozenset(line for raw in f if (line := raw.strip()))
