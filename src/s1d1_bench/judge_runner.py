"""The core of `bench judge`: build the judge messages, parse its answers, run a prompt list with an injectable
generation function (no model here), keep `labels.jsonl` and `log.jsonl` up to date line by line, resume."""

import json
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from s1d1_bench.rubric import LEVELS, TASKS, load_rubric

INSTRUCTION = "Label the prompt between the <prompt> tags. Return the JSON label now."
RETRY_SUFFIX = "\nOutput only the JSON object."
LABEL_KEYS = ["complexity", "task", "needs_tools"]
RAW_MAX_CHARS = 2000

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_HARMONY_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>.*?<\|end\|>", re.DOTALL)
_HARMONY_ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
_HARMONY_FINAL_HEADER = re.compile(r"(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>")
_HARMONY_END = re.compile(r"<\|(?:return|end)\|>")
_GEMMA_THOUGHT = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)  # Gemma 4: <|channel>thought ... <channel|>
_GEMMA_THOUGHT_OPEN = "<|channel>"
_FENCE = re.compile(r"```[A-Za-z0-9_-]*")


@dataclass(frozen=True)
class GenerateResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    wall_s: float


GenerateFn = Callable[[list[dict]], GenerateResult]


def build_messages(prompt: str, retry: bool = False) -> list[dict]:
    """System message = the rubric, byte-exact; user message = the prompt as data between <prompt> tags."""
    user = "<prompt>\n" + prompt + "\n</prompt>\n\n" + INSTRUCTION
    if retry:
        user += RETRY_SUFFIX
    return [{"role": "system", "content": load_rubric()}, {"role": "user", "content": user}]


def _strip_reasoning(text: str) -> str:
    text = _THINK_BLOCK.sub(" ", text)
    if "<think>" in text:  # never closed: everything after it is thinking
        text = text.split("<think>", 1)[0]
    elif "</think>" in text:  # the opening tag was in the chat template: the answer follows the last closing tag
        text = text.rsplit("</think>", 1)[1]
    text = _HARMONY_ANALYSIS.sub(" ", text)
    if _HARMONY_ANALYSIS_OPEN in text:
        text = text.split(_HARMONY_ANALYSIS_OPEN, 1)[0]
    text = _GEMMA_THOUGHT.sub(" ", text)
    if _GEMMA_THOUGHT_OPEN in text:
        text = text.split(_GEMMA_THOUGHT_OPEN, 1)[0]
    text = _HARMONY_FINAL_HEADER.sub(" ", text)
    text = _HARMONY_END.sub(" ", text)
    return _FENCE.sub(" ", text)


def _valid_label(obj: dict) -> bool:
    if list(sorted(obj)) != sorted(LABEL_KEYS):
        return False
    complexity, task = obj["complexity"], obj["task"]
    if complexity is not None and not (isinstance(complexity, str) and complexity in LEVELS):
        return False
    if not (isinstance(task, str) and task in TASKS):
        return False
    return type(obj["needs_tools"]) is bool  # bool is an int: 1 and 0 must not pass


def parse_label(text: str) -> dict | None:
    """The label in a judge answer, or None. Reasoning blocks and code fences are ignored; the answer must hold
    exactly one JSON object, with exactly the three keys and in-vocabulary values."""
    text = _strip_reasoning(text)
    decoder = json.JSONDecoder()
    found: list[dict] = []
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError:
            pos = text.find("{", pos + 1)
            continue
        found.append(obj)
        pos = text.find("{", end)
    if len(found) != 1 or not _valid_label(found[0]):
        return None
    obj = found[0]
    return {key: obj[key] for key in LABEL_KEYS}


def _write_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_done(path: Path) -> dict[str, dict]:
    """The labels already written (id -> line). A last line that does not parse is a write cut short: it is
    dropped. An unreadable line anywhere else is a damaged file and raises ValueError."""
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if line.rstrip("\r\n") != ""]
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError("not a label line")
        except ValueError as exc:  # JSONDecodeError is a ValueError
            if number == len(lines):
                break
            raise ValueError(f"{path}:{number}: unreadable line in the middle of the file ({exc})") from exc
        done.setdefault(row["id"], row)
    return done


def _repair_log(path: Path) -> None:
    """Drop a cut-short last line of the log (one written without its newline that does not parse), so the next
    record is not glued onto it. A complete last line missing only its newline gets the newline."""
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if text == "" or text.endswith("\n"):
        return
    head, _, tail = text.rpartition("\n")
    try:
        json.loads(tail)
        fixed = text + "\n"
    except ValueError:
        fixed = head + "\n" if head else ""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(fixed, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def _judge_one(prompt: str, generate: GenerateFn) -> tuple[str, dict | None, dict]:
    """(status, label or None, log fields) for one prompt: at most two generations, none after an exception."""
    label = None
    totals = {"retries": 0, "wall_s": 0.0, "prompt_tokens": 0, "completion_tokens": 0}
    raw = ""
    status = "json_error"
    for retry in (False, True):
        started = time.perf_counter()
        try:
            res = generate(build_messages(prompt, retry=retry))
        except Exception as exc:  # the run goes on: this prompt is logged as an error
            totals["wall_s"] = round(totals["wall_s"] + (time.perf_counter() - started), 3)
            totals["retries"] = int(retry)
            return "error", None, {**totals, "raw": repr(exc)[:RAW_MAX_CHARS]}
        totals["wall_s"] += res.wall_s
        totals["prompt_tokens"] += res.prompt_tokens
        totals["completion_tokens"] += res.completion_tokens
        totals["retries"] = int(retry)
        raw = res.text
        label = parse_label(res.text)
        if label is not None:
            status = "ok"
            break
        if res.finish_reason == "length":
            status = "truncated"
            break
    totals["wall_s"] = round(totals["wall_s"], 3)
    return status, label, totals if status == "ok" else {**totals, "raw": raw[:RAW_MAX_CHARS]}


def run_judge(prompts: list[dict], generate: GenerateFn, out_dir: Path, resume: bool = False) -> dict[str, int]:
    """Judge `prompts` ({"id", "prompt"}) into `out_dir/labels.jsonl` (ok only, prompt order at the end) and
    `out_dir/log.jsonl` (every prompt tried). Returns the count of each status that occurred in this call.

    A judge folder is immutable: without `resume`, existing files raise FileExistsError. With `resume`, ids
    already labelled are skipped and the new lines are appended.
    """
    ids = [p["id"] for p in prompts]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicate prompt id(s): {', '.join(map(repr, duplicates))}")
    out_dir = Path(out_dir)
    labels_path, log_path = out_dir / "labels.jsonl", out_dir / "log.jsonl"
    if not resume:
        existing = [p.name for p in (labels_path, log_path) if p.exists()]
        if existing:
            raise FileExistsError(
                f"{out_dir}: {', '.join(existing)} already exist; judge files are immutable. "
                "Use a new folder, or resume the run."
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    done = _read_done(labels_path) if resume else {}
    foreign = [i for i in done if i not in set(ids)]
    if foreign:  # the wrong prompts file: refuse before touching anything, never drop labels silently
        raise ValueError(f"{labels_path}: labelled id(s) not in the prompts: {', '.join(map(repr, foreign))}")
    if resume:
        _repair_log(log_path)
    if done or labels_path.exists():
        _write_atomic(labels_path, list(done.values()))  # drops a cut-short last line, restores the newline
    counts: dict[str, int] = {}
    with (
        open(labels_path, "a", encoding="utf-8", newline="\n") as labels_f,
        open(log_path, "a", encoding="utf-8", newline="\n") as log_f,
    ):
        for item in prompts:
            if item["id"] in done:
                continue
            status, label, fields = _judge_one(item["prompt"], generate)
            counts[status] = counts.get(status, 0) + 1
            log_row = {"id": item["id"], "status": status, **fields}
            log_f.write(json.dumps(log_row, ensure_ascii=False) + "\n")  # log first: a crash before the label
            log_f.flush()  # line only costs a regeneration on resume
            if label is not None:
                row = {"id": item["id"], **label}
                labels_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                labels_f.flush()
                done[item["id"]] = row
    _write_atomic(labels_path, [done[i] for i in ids if i in done])
    return counts


# meta.json keys a resumed run must share with the run it resumes
RESUME_KEYS = ["model", "revision", "thinking", "prompts_sha256", "quantization", "mlx_lm_version", "max_tokens"]
_COMMIT = re.compile(r"[0-9a-f]{40}")
# an absolute local path (or one under the home directory) inside a meta.json value
_LOCAL_PATH = re.compile(r"(?<![\w.])(?:~/|/(?:Users|home|private|tmp|var|Volumes|mnt|opt)/)")


def _resume_conflicts(old: dict, new: dict) -> list[str]:
    return [f"{key} {old.get(key)!r} (meta.json) != {new[key]!r}" for key in RESUME_KEYS if key in new
            and old.get(key) != new[key]]


def _local_path_keys(meta: dict) -> list[str]:
    home = str(Path.home())
    return [k for k, v in meta.items() if isinstance(v, str) and (home in v or _LOCAL_PATH.search(v))]


def _prompt_fault(prompts: list) -> str | None:
    """Why `prompts` cannot be judged (a row without a string `id` or `prompt`, a duplicate id), else None."""
    seen: set[str] = set()
    for number, row in enumerate(prompts, start=1):
        if not isinstance(row, dict):
            return f"prompt {number}: not a JSON object"
        for key in ("id", "prompt"):
            if not isinstance(row.get(key), str):
                return f"prompt {number}: '{key}' must be a string, got {row.get(key)!r}"
        if row["id"] in seen:
            return f"duplicate prompt id {row['id']!r} (prompt {number})"
        seen.add(row["id"])
    return None


def run_judge_cmd(args) -> int:
    """`bench judge`: judge a prompts file with a local MLX model into `--out` (meta.json first, then labels.jsonl
    and log.jsonl); `--resume` goes on with a run that has the same model, revision, thinking and prompts."""
    import datetime
    import hashlib
    import statistics
    import sys

    from s1d1_bench import mlx_backend  # imported here: importing judge_runner never imports mlx
    from s1d1_bench.jsonl import read_jsonl
    from s1d1_bench.judge_file import LICENCE, write_meta

    def fail(message: str) -> int:
        print(f"bench judge: {message}", file=sys.stderr)
        return 1

    out = Path(args.out)
    meta_path = out / "meta.json"
    existing = [p.name for p in (meta_path, out / "labels.jsonl", out / "log.jsonl") if p.exists()]
    if existing and not args.resume:
        return fail(f"{out}: {', '.join(existing)} already exist; judge files are immutable. "
                    "Use a new --out, or --resume to go on with that run.")
    digest = hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest()
    prompts = read_jsonl(args.prompts)
    problem = _prompt_fault(prompts)
    if problem:  # before the model loads and before anything is written
        return fail(f"{args.prompts}: {problem}")
    old = json.loads(meta_path.read_text(encoding="utf-8")) if args.resume and meta_path.is_file() else None
    if old is not None:  # what can be checked before loading the model
        early = {"thinking": args.thinking, "prompts_sha256": digest, "max_tokens": args.max_tokens}
        conflicts = _resume_conflicts(old, early)
        if conflicts:
            return fail(f"{meta_path}: cannot resume a run with other settings: {'; '.join(conflicts)}")

    started = time.perf_counter()
    try:
        generate, backend = mlx_backend.make_mlx_generate(args.model, args.thinking, max_tokens=args.max_tokens)
    except (FileNotFoundError, ValueError, ModuleNotFoundError) as exc:
        return fail(str(exc))
    print(f"model loaded in {time.perf_counter() - started:.1f} s")
    replayable = bool(_COMMIT.fullmatch(backend["revision"]))
    notes = [backend["behaviour"]]
    if not replayable:
        notes.append(f"replayable false: revision {backend['revision']} (no Hub commit recorded for the weights).")
    if args.notes:
        notes.append(args.notes)
    meta = {
        "judge": args.judge or out.name,
        "model": backend["model"],
        "provider": "local",
        "revision": backend["revision"],
        "date": datetime.date.today().isoformat(),
        "temperature": backend["temperature"],
        "thinking": backend["thinking"],
        "instructions": "v7-single",
        "prompts_sha256": digest,
        "author": args.author,
        "replayable": replayable,
        "licence": LICENCE,
        "notes": " ".join(notes),
        "quantization": backend["quantization"],
        "mlx_lm_version": backend["mlx_lm_version"],
        "max_tokens": args.max_tokens,
    }
    leaks = _local_path_keys(meta)
    if leaks:
        return fail(f"meta.json would hold a local path in {', '.join(leaks)}: meta.json is published, remove it")
    if old is not None:
        conflicts = _resume_conflicts(old, meta)
        if conflicts:
            return fail(f"{meta_path}: cannot resume a run with other settings: {'; '.join(conflicts)}")
    else:  # written before the first generation, so that a resumed run keeps it
        out.mkdir(parents=True, exist_ok=True)
        write_meta(meta_path, **meta)

    rates: list[float] = []

    def timed(messages: list[dict]) -> GenerateResult:
        res = generate(messages)
        if res.wall_s > 0:
            rates.append(res.completion_tokens / res.wall_s)
        return res

    try:
        counts = run_judge(prompts, timed, out, resume=args.resume)
    except (FileExistsError, ValueError) as exc:
        return fail(str(exc))
    print(", ".join(f"{status} {n}" for status, n in sorted(counts.items())) or "nothing left to judge")
    if rates:
        median = statistics.median(rates)
        print(f"median {median:.1f} tokens/s (completion tokens / wall time, {len(rates)} generations)")
    return 0
