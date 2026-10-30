"""The judge directory (`judges/<name>/{meta.json,labels.jsonl}`): format, acceptance rule, `bench validate`.

A classifier directory (`classifiers/<name>/`) has the same format: `provider` is then "system", and a label may carry
a `score` and a `needs_tools_probability`, or leave `task` and `needs_tools` null when the system does not predict them.

An abstention row records a judge's non-answer (its final status after the retry was not `ok`): the three fields null
and `abstained` set to that status, which the last line of the id in `log.jsonl` must carry. A classifier never
abstains.
"""

import hashlib
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
OPTIONAL_META_KEYS = ["quantization", "mlx_lm_version", "max_tokens"]  # by `bench judge`, never required
PROVIDERS = ["local", "chat", "api", "human", "system"]
INSTRUCTIONS = ["v7-single", "v7-batch", "none"]
SYSTEM = "system"  # a classifier, not a judge: it reads no instructions ("none"), and may predict fewer fields
LICENCE = "CC-BY-4.0"
STRING_KEYS = ["judge", "model", "revision", "date", "author", "notes"]  # temperature and thinking: anything
LABEL_KEYS = ["id", "complexity", "task", "needs_tools"]
OPTIONAL_LABEL_KEYS = ["score", "needs_tools_probability"]  # absent or a number in [0, 1], in any directory
MISSING_SHOWN = 5  # ids named in the "missing" summary
ABSTAINED = "abstained"  # the key of an abstention row, whose value is the final status of the id
ABSTENTION_STATUSES = ["json_error", "truncated", "error"]  # every final status of `bench judge` but "ok"


def _prompts_files(prompts_dir: Path, errors: list[str]) -> dict[str, Path]:
    """Real SHA-256 of each `*.jsonl` of `prompts_dir` -> that file. A `<name>.sha256` sidecar is only a claim: it is
    checked against the file itself, and a missing file or a disagreeing digest is appended to `errors`."""
    found: dict[str, Path] = {}
    for jsonl in sorted(prompts_dir.glob("*.jsonl")):
        found.setdefault(hashlib.sha256(jsonl.read_bytes()).hexdigest(), jsonl)
    for sidecar in sorted(prompts_dir.glob("*.sha256")):
        jsonl = sidecar.with_suffix(".jsonl")
        tokens = sidecar.read_text(encoding="ascii", errors="replace").split()
        if not jsonl.is_file():
            errors.append(f"prompts: {sidecar.name} has no {jsonl.name} in {prompts_dir}")
        elif not tokens or tokens[0] != hashlib.sha256(jsonl.read_bytes()).hexdigest():
            errors.append(f"prompts: {sidecar.name} does not match the SHA-256 of {jsonl.name}")
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
    if meta.get("provider") in PROVIDERS and meta.get("instructions") in INSTRUCTIONS:
        if (meta["provider"] == SYSTEM) != (meta["instructions"] == "none"):
            errors.append(
                f"meta.json: provider {meta['provider']!r} and instructions {meta['instructions']!r} do not go "
                f"together: only provider {SYSTEM!r} has instructions 'none'"
            )
    if "replayable" in meta and type(meta["replayable"]) is not bool:
        errors.append(f"meta.json: replayable must be true or false (a JSON boolean), got {meta['replayable']!r}")
    if "max_tokens" in meta and not (type(meta["max_tokens"]) is int and meta["max_tokens"] > 0):
        errors.append(f"meta.json: max_tokens must be a positive integer, got {meta['max_tokens']!r}")
    if "licence" in meta and meta["licence"] != LICENCE:
        errors.append(
            f"meta.json: licence {meta['licence']!r} must be {LICENCE!r}: judge files are published under it, "
            "and a file under another licence cannot be merged"
        )
    if "prompts_sha256" not in meta:
        return None
    known = _prompts_files(prompts_dir, errors) if prompts_dir.is_dir() else {}
    sha = meta["prompts_sha256"]
    if not isinstance(sha, str) or sha not in known:
        errors.append(f"meta.json: prompts_sha256 {sha!r} matches no prompts file in {prompts_dir}")
        return None
    return known[sha]


def _is_unit_number(value: object) -> bool:
    return type(value) in (int, float) and 0 <= value <= 1  # bool is an int and NaN fails the comparison


def _abstention_faults(row: dict, system: bool) -> list[str]:
    """The faults of a line that carries `abstained`: the three fields null, a known status, no other key."""
    faults = []
    missing = [k for k in LABEL_KEYS if k not in row]
    extra = [k for k in row if k not in LABEL_KEYS and k != ABSTAINED]
    if missing:
        faults.append("missing field " + ", ".join(repr(k) for k in missing))
    if extra:
        faults.append("unexpected field " + ", ".join(repr(k) for k in extra) + " on an abstention")
    if "id" in row and not isinstance(row["id"], str):
        faults.append(f"id {row['id']!r} is not a string")
    if system:
        faults.append(f"{ABSTAINED} {row[ABSTAINED]!r}: a classifier (provider 'system') does not abstain")
    elif not (isinstance(row[ABSTAINED], str) and row[ABSTAINED] in ABSTENTION_STATUSES):
        faults.append(f"{ABSTAINED} {row[ABSTAINED]!r} not in {ABSTENTION_STATUSES}")
    for key in LABEL_KEYS[1:]:
        if key in row and row[key] is not None:
            faults.append(f"{key} {row[key]!r} on an abstention: the three fields must be null")
    return faults


def _label_faults(row: object, system: bool) -> list[str]:
    """The faults of one parsed line, each naming its field (the `id` checks are made by the caller). `system` lets
    `task` and `needs_tools` be null."""
    if not isinstance(row, dict):
        return ["not a JSON object"]
    if ABSTAINED in row:
        return _abstention_faults(row, system)
    faults = []
    missing = [k for k in LABEL_KEYS if k not in row]
    extra = [k for k in row if k not in LABEL_KEYS and k not in OPTIONAL_LABEL_KEYS]
    if missing:
        faults.append("missing field " + ", ".join(repr(k) for k in missing))
    if extra:
        faults.append("unexpected field " + ", ".join(repr(k) for k in extra))
    if "id" in row and not isinstance(row["id"], str):
        faults.append(f"id {row['id']!r} is not a string")
    if "complexity" in row and row["complexity"] is not None:
        if not (isinstance(row["complexity"], str) and row["complexity"] in LEVELS):
            faults.append(f"complexity {row['complexity']!r} not in {LEVELS} or null")
    or_null = " or null" if system else ""
    if "task" in row and not (isinstance(row["task"], str) and row["task"] in TASKS or system and row["task"] is None):
        faults.append(f"task {row['task']!r} not in the {len(TASKS)} tasks of the rubric{or_null}")
    # bool is an int: 1 and 0 must not pass
    if "needs_tools" in row and not (type(row["needs_tools"]) is bool or system and row["needs_tools"] is None):
        faults.append(f"needs_tools {row['needs_tools']!r} is not true or false (a JSON boolean){or_null}")
    for key in OPTIONAL_LABEL_KEYS:
        if key in row and not _is_unit_number(row[key]):
            faults.append(f"{key} {row[key]!r} is not a number between 0 and 1")
    return faults


def _last_log_statuses(path: Path, errors: list[str]) -> dict[str, object] | None:
    """id -> the `status` of its last line in `log.jsonl` (a resumed id is logged again); None, with an error, when
    a line does not parse."""
    statuses: dict[str, object] = {}
    try:
        with open(path, encoding="utf-8") as f:
            for number, raw in enumerate(f, start=1):
                if raw.rstrip("\r\n") == "":
                    continue
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    errors.append(f"log.jsonl:{number}: not valid JSON, cannot check the abstentions")
                    return None
                if isinstance(row, dict) and isinstance(row.get("id"), str):
                    statuses[row["id"]] = row.get("status")
    except UnicodeDecodeError as exc:
        errors.append(f"log.jsonl: not valid UTF-8 ({exc.reason}), cannot check the abstentions")
        return None
    return statuses


def _check_abstentions_against_log(log: Path, rows: dict[str, tuple[int, str]], errors: list[str]) -> None:
    """Each abstention row (id -> line number, status) must be what the last log line of its id says."""
    if not rows or not log.is_file():
        return
    statuses = _last_log_statuses(log, errors)
    if statuses is None:
        return
    for row_id, (number, status) in rows.items():
        if row_id not in statuses:
            errors.append(
                f"labels.jsonl:{number}: abstention {status!r} for {row_id!r}, which has no line in log.jsonl"
            )
        elif statuses[row_id] != status:
            errors.append(
                f"labels.jsonl:{number}: abstention {status!r} for {row_id!r}, but its last line in log.jsonl says "
                f"{statuses[row_id]!r}"
            )


def _check_labels(path: Path, expected: list[str] | None, system: bool, errors: list[str]) -> None:
    """Append one error per faulty line, then one summary for the ids never seen, then the abstentions that the
    `log.jsonl` next to the labels contradicts. `expected` is None when the prompts file is unknown: the ids are then
    not checked."""
    if not path.is_file():
        errors.append(f"labels.jsonl: file not found in {path.parent}")
        return
    wanted = set(expected) if expected is not None else None
    first_line: dict[str, int] = {}
    abstained: dict[str, tuple[int, str]] = {}  # well-formed abstention rows: id -> (line, status)
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
                faults = _label_faults(row, system)
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
                elif ABSTAINED in row:
                    abstained[row_id] = (number, row[ABSTAINED])
    except UnicodeDecodeError as exc:
        errors.append(f"labels.jsonl: not valid UTF-8 ({exc.reason})")
        return
    if expected is not None:
        missing = [i for i in expected if i not in first_line]  # a line with a bad value still counts as seen
        if missing:
            shown = ", ".join(missing[:MISSING_SHOWN]) + (", ..." if len(missing) > MISSING_SHOWN else "")
            errors.append(f"labels.jsonl: missing {len(missing)} id(s): {shown}")
    _check_abstentions_against_log(path.with_name("log.jsonl"), abstained, errors)


def validate_judge_dir(judge_dir: Path | str, prompts_dir: Path | str) -> list[str]:
    """The faults of a judge (or classifier) directory against the prompts files of `prompts_dir`; an empty list means
    accepted.

    Each message starts with the file (`meta.json: ...`, `labels.jsonl:LINE: ...` or `prompts: ...`) and names the
    field at fault.
    """
    judge_dir, prompts_dir = Path(judge_dir), Path(prompts_dir)
    if not judge_dir.is_dir():
        return [f"{judge_dir}: not a directory"]
    errors: list[str] = []
    meta = _read_json_object(judge_dir / "meta.json", "meta.json", errors)
    prompts_file = _check_meta(meta, prompts_dir, errors) if meta is not None else None
    expected = None
    if prompts_file is not None:
        expected = [row["id"] for row in read_jsonl(prompts_file)]
    _check_labels(judge_dir / "labels.jsonl", expected, meta is not None and meta.get("provider") == SYSTEM, errors)
    return errors


def read_labels(path: Path | str) -> dict[str, dict]:
    """`id` -> its three label fields, from a labels file already accepted by `validate_judge_dir`. An abstention
    keeps its row: the three fields None, plus `abstained` (its status); an answer has no `abstained` key."""
    labels: dict[str, dict] = {}
    for row in read_jsonl(path):
        if row["id"] in labels:
            raise ValueError(f"{path}: duplicate id {row['id']!r}")
        labels[row["id"]] = {k: row[k] for k in LABEL_KEYS if k != "id"}
        if ABSTAINED in row:
            if any(v is not None for v in labels[row["id"]].values()):
                raise ValueError(f"{path}: abstention for id {row['id']!r} with a non-null label field")
            labels[row["id"]][ABSTAINED] = row[ABSTAINED]
    return labels


def abstentions(labels: dict[str, dict]) -> dict[str, str]:
    """id -> status of the abstentions among `labels` (as `read_labels` gives them), in their order."""
    return {i: row[ABSTAINED] for i, row in labels.items() if row.get(ABSTAINED)}


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
    n = sum(ABSTAINED in row for row in read_jsonl(Path(args.judge_dir) / "labels.jsonl"))
    print(f"ok ({n} abstention{'s' if n > 1 else ''})" if n else "ok")
    return 0
