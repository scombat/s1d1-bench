"""The judge directory (`judges/<name>/{meta.json,labels.jsonl}`): format, acceptance rule, `bench validate`."""

import json
import sys
from pathlib import Path

from s1d1_bench.jsonl import read_jsonl
from s1d1_bench.rubric import LEVELS, TASKS

META_KEYS = [
    "judge",
    "model",
    "provider",
    "revision",
    "date",
    "temperature",
    "thinking",
    "instructions",
    "prompts_sha256",
    "author",
    "replayable",
    "licence",
    "notes",
]
OPTIONAL_META_KEYS = ["quantization", "mlx_lm_version"]  # added by `bench judge`, allowed but never required
PROVIDERS = ["local", "chat", "api", "human"]
INSTRUCTIONS = ["v7-single", "v7-batch"]
LICENCE = "CC-BY-4.0"
STRING_KEYS = ["judge", "model", "revision", "date", "author", "notes"]  # temperature and thinking: anything
LABEL_KEYS = ["id", "complexity", "task", "needs_tools"]
MISSING_SHOWN = 5  # ids named in the "missing" summary


def _prompts_files(prompts_dir: Path) -> dict[str, Path]:
    """Digest (first token of each `*.sha256`) -> the matching `<name>.jsonl`, whether or not that file exists."""
    found: dict[str, Path] = {}
    for sha_file in sorted(prompts_dir.glob("*.sha256")):
        tokens = sha_file.read_text(encoding="ascii", errors="replace").split()
        if tokens:
            found.setdefault(tokens[0], sha_file.with_suffix(".jsonl"))
    return found


def _read_json_object(path: Path, label: str, errors: list[str]) -> dict | None:
    if not path.is_file():
        errors.append(f"{label}: file not found in {path.parent}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        errors.append(f"{label}: not valid JSON ({exc})")
        return None
    if not isinstance(data, dict):
        errors.append(f"{label}: must be a JSON object")
        return None
    return data


def _check_meta(meta: dict, prompts_dir: Path, errors: list[str]) -> Path | None:
    """Append the faults of `meta`; return the prompts file it points to, if it points to one."""
    for key in META_KEYS:
        if key not in meta:
            errors.append(f"meta.json: missing '{key}'")
    for key in meta:
        if key not in META_KEYS and key not in OPTIONAL_META_KEYS:
            errors.append(f"meta.json: unknown key '{key}'")
    for key in STRING_KEYS:
        if key in meta and not isinstance(meta[key], str):
            errors.append(f"meta.json: '{key}' must be a string, got {meta[key]!r}")
    if "provider" in meta and meta["provider"] not in PROVIDERS:
        errors.append(f"meta.json: provider {meta['provider']!r} not in {PROVIDERS}")
    if "instructions" in meta and meta["instructions"] not in INSTRUCTIONS:
        errors.append(f"meta.json: instructions {meta['instructions']!r} not in {INSTRUCTIONS}")
    if "replayable" in meta and type(meta["replayable"]) is not bool:
        errors.append(f"meta.json: replayable must be true or false (a JSON boolean), got {meta['replayable']!r}")
    if "licence" in meta and meta["licence"] != LICENCE:
        errors.append(
            f"meta.json: licence {meta['licence']!r} must be {LICENCE!r}: judge files are published under it, "
            "and a file under another licence cannot be merged"
        )
    if "prompts_sha256" not in meta:
        return None
    known = _prompts_files(prompts_dir) if prompts_dir.is_dir() else {}
    sha = meta["prompts_sha256"]
    if not isinstance(sha, str) or sha not in known:
        errors.append(f"meta.json: prompts_sha256 {sha!r} matches no prompts file in {prompts_dir}")
        return None
    return known[sha]


def _label_faults(row: object) -> list[str]:
    """The faults of one parsed line, each naming its field (the `id` checks are made by the caller)."""
    if not isinstance(row, dict):
        return ["not a JSON object"]
    faults = []
    missing = [k for k in LABEL_KEYS if k not in row]
    extra = [k for k in row if k not in LABEL_KEYS]
    if missing:
        faults.append("missing field " + ", ".join(repr(k) for k in missing))
    if extra:
        faults.append("unexpected field " + ", ".join(repr(k) for k in extra))
    if "id" in row and not isinstance(row["id"], str):
        faults.append(f"id {row['id']!r} is not a string")
    if "complexity" in row and row["complexity"] is not None:
        if not (isinstance(row["complexity"], str) and row["complexity"] in LEVELS):
            faults.append(f"complexity {row['complexity']!r} not in {LEVELS} or null")
    if "task" in row and not (isinstance(row["task"], str) and row["task"] in TASKS):
        faults.append(f"task {row['task']!r} not in the {len(TASKS)} tasks of the rubric")
    if "needs_tools" in row and type(row["needs_tools"]) is not bool:  # bool is an int: 1 and 0 must not pass
        faults.append(f"needs_tools {row['needs_tools']!r} is not true or false (a JSON boolean)")
    return faults


def _check_labels(path: Path, expected: list[str] | None, errors: list[str]) -> None:
    """Append one error per faulty line, then one summary for the ids never seen. `expected` is None when the
    prompts file is unknown: the ids are then not checked."""
    if not path.is_file():
        errors.append(f"labels.jsonl: file not found in {path.parent}")
        return
    wanted = set(expected) if expected is not None else None
    first_line: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for number, raw in enumerate(f, start=1):  # physical line numbers; truly empty lines are skipped
                if raw.rstrip("\r\n") == "":
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as exc:
                    errors.append(f"labels.jsonl:{number}: not valid JSON ({exc.msg})")
                    continue
                faults = _label_faults(row)
                row_id = row.get("id") if isinstance(row, dict) else None
                if isinstance(row_id, str):
                    if row_id in first_line:
                        faults.append(f"duplicate id {row_id!r} (first at line {first_line[row_id]})")
                    else:
                        first_line[row_id] = number
                    if wanted is not None and row_id not in wanted:
                        faults.append(f"unknown id {row_id!r}")
                if faults:
                    errors.append(f"labels.jsonl:{number}: " + "; ".join(faults))
    except UnicodeDecodeError as exc:
        errors.append(f"labels.jsonl: not valid UTF-8 ({exc.reason})")
        return
    if expected is not None:
        missing = [i for i in expected if i not in first_line]  # a line with a bad value still counts as seen
        if missing:
            shown = ", ".join(missing[:MISSING_SHOWN]) + (", ..." if len(missing) > MISSING_SHOWN else "")
            errors.append(f"labels.jsonl: missing {len(missing)} id(s): {shown}")


def validate_judge_dir(judge_dir: Path | str, prompts_dir: Path | str) -> list[str]:
    """The faults of a judge directory against the prompts files of `prompts_dir`; an empty list means accepted.

    Each message starts with the file (`meta.json: ...` or `labels.jsonl:LINE: ...`) and names the field at fault.
    """
    judge_dir, prompts_dir = Path(judge_dir), Path(prompts_dir)
    if not judge_dir.is_dir():
        return [f"{judge_dir}: not a directory"]
    errors: list[str] = []
    meta = _read_json_object(judge_dir / "meta.json", "meta.json", errors)
    prompts_file = _check_meta(meta, prompts_dir, errors) if meta is not None else None
    expected = None
    if prompts_file is not None:
        if prompts_file.is_file():
            expected = [row["id"] for row in read_jsonl(prompts_file)]
        else:
            errors.append(f"meta.json: the prompts file {prompts_file.name} is missing from {prompts_dir}")
    _check_labels(judge_dir / "labels.jsonl", expected, errors)
    return errors


def read_labels(path: Path | str) -> dict[str, dict]:
    """`id` -> its three label fields, from a labels file already accepted by `validate_judge_dir`."""
    labels: dict[str, dict] = {}
    for row in read_jsonl(path):
        if row["id"] in labels:
            raise ValueError(f"{path}: duplicate id {row['id']!r}")
        labels[row["id"]] = {k: row[k] for k in LABEL_KEYS if k != "id"}
    return labels


def write_meta(path: Path | str, **fields) -> None:
    """Write `meta.json`: the 13 keys in their fixed order, then the optional keys that were given."""
    missing = [k for k in META_KEYS if k not in fields]
    unknown = [k for k in fields if k not in META_KEYS and k not in OPTIONAL_META_KEYS]
    if missing or unknown:
        problems = []
        if missing:
            problems.append("missing " + ", ".join(repr(k) for k in missing))
        if unknown:
            problems.append("unknown " + ", ".join(repr(k) for k in unknown))
        raise ValueError("meta.json: " + "; ".join(problems))
    ordered = {k: fields[k] for k in [*META_KEYS, *OPTIONAL_META_KEYS] if k in fields}
    text = json.dumps(ordered, indent=2, ensure_ascii=False) + "\n"
    Path(path).write_text(text, encoding="utf-8", newline="\n")


def run_validate(args) -> int:
    errors = validate_judge_dir(args.judge_dir, args.prompts)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("ok")
    return 0
