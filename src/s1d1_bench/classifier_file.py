"""The classifier directory (`classifiers/<name>/{meta.json,labels.jsonl}`): written and read like a judge's."""

import shutil
from pathlib import Path

from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import ABSTAINED, LABEL_KEYS, OPTIONAL_LABEL_KEYS, SYSTEM, write_meta

SYSTEM_META = {"provider": SYSTEM, "instructions": "none", "thinking": "none"}  # forced, whatever `meta` says


def _row_fault(row: dict) -> str | None:
    missing = [k for k in LABEL_KEYS if k not in row]
    unknown = [k for k in row if k not in LABEL_KEYS and k not in OPTIONAL_LABEL_KEYS]
    if missing or unknown:
        return ", ".join([*(f"missing {k!r}" for k in missing), *(f"unknown {k!r}" for k in unknown)])
    return None


def write_classifier_dir(out_dir: Path | str, rows: list[dict], meta: dict) -> None:
    """Write `meta.json` then `labels.jsonl` into the new folder `out_dir`; a classifier folder is immutable.

    `rows` hold `id`, `complexity`, `task` and `needs_tools` (the last two may be None), and optionally `score` and
    `needs_tools_probability`. `meta` has the keys of `judge_file.write_meta`; its provider, instructions and thinking
    are replaced by the system ones. The folder is validated afterwards by `bench validate`, not here.
    """
    out_dir = Path(out_dir)
    for number, row in enumerate(rows, start=1):
        fault = _row_fault(row)
        if fault:
            raise ValueError(f"row {number}: {fault}")
    if out_dir.exists():
        raise FileExistsError(f"{out_dir} already exists; classifier files are immutable")
    out_dir.mkdir(parents=True)
    try:
        write_meta(out_dir / "meta.json", **{**meta, **SYSTEM_META})
        ordered = [{k: row[k] for k in [*LABEL_KEYS, *OPTIONAL_LABEL_KEYS] if k in row} for row in rows]
        write_jsonl(out_dir / "labels.jsonl", ordered)
    except Exception:
        shutil.rmtree(out_dir)  # nothing half-written is left behind
        raise


def read_classifier_labels(path: Path | str) -> dict[str, dict]:
    """`id` -> complexity, task, needs_tools, score and needs_tools_probability (None when absent). A row with
    `abstained` is refused: a classifier does not abstain (`bench validate` says so too)."""
    labels: dict[str, dict] = {}
    for row in read_jsonl(path):
        if ABSTAINED in row:
            raise ValueError(f"{path}: id {row['id']!r} has {ABSTAINED!r}: a classifier does not abstain")
        if row["id"] in labels:
            raise ValueError(f"{path}: duplicate id {row['id']!r}")
        labels[row["id"]] = {k: row.get(k) for k in LABEL_KEYS + OPTIONAL_LABEL_KEYS if k != "id"}
    return labels
