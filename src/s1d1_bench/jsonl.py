"""Minimal JSON Lines helpers: UTF-8, one JSON object per line."""

import json
from collections.abc import Iterable
from pathlib import Path


def read_jsonl(path: Path | str) -> list[dict]:
    """Read every record. Only truly empty lines are skipped; anything else must parse."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:  # file iteration, not str.splitlines(): U+2028 etc. may sit inside a string
            if line.rstrip("\r\n") == "":
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path | str, rows: Iterable[dict]) -> int:
    """Write one JSON object per line (ensure_ascii=False, trailing newline); return the row count."""
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
