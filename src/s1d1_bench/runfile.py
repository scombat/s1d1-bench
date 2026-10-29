"""Helpers shared by the commands that write a model's output line by line into a run file: atomic rewrite,
resume (read what is done, repair a cut-short log), and the reading of a model's answer."""

import json
import os
import re
from pathlib import Path

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_HARMONY_ANALYSIS = re.compile(r"<\|channel\|>analysis<\|message\|>.*?<\|end\|>", re.DOTALL)
_HARMONY_ANALYSIS_OPEN = "<|channel|>analysis<|message|>"
_HARMONY_FINAL_HEADER = re.compile(r"(?:<\|start\|>assistant)?<\|channel\|>final<\|message\|>")
_HARMONY_END = re.compile(r"<\|(?:return|end)\|>")
_GEMMA_THOUGHT = re.compile(r"<\|channel>.*?<channel\|>", re.DOTALL)  # Gemma 4: <|channel>thought ... <channel|>
_GEMMA_THOUGHT_OPEN = "<|channel>"


def strip_reasoning(text: str) -> str:
    """The answer without the reasoning blocks (<think>, harmony, Gemma), closed or not. Code fences stay."""
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
    return _HARMONY_END.sub(" ", text)


def find_json_objects(text: str) -> list[dict]:
    """Every JSON object that decodes in `text`, in order; a brace that does not start one is skipped."""
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
    return found


def write_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_done(path: Path) -> dict[str, dict]:
    """The rows already written (id -> line). A last line that does not parse is a write cut short: it is
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


def repair_log(path: Path) -> None:
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
