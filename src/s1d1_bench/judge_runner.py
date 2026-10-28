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
