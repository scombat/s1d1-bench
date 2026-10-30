import hashlib
import json

import pytest

from s1d1_bench.classifier_file import read_classifier_labels, write_classifier_dir
from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.judge_file import validate_judge_dir

PROMPTS = [
    {"id": "a-eng", "prompt": "What is the capital of France?"},
    {"id": "b-eng", "prompt": "Write a function that reverses a string."},
    {"id": "c-eng", "prompt": "Design a migration plan from a monolith to services."},
]
ROWS = [
    {"id": "a-eng", "complexity": "trivial", "task": None, "needs_tools": None, "score": 0.0},
    {"id": "b-eng", "complexity": "low", "task": None, "needs_tools": None, "score": 0.3},
    {"id": "c-eng", "complexity": "high", "task": None, "needs_tools": None, "score": 0.8},
]


@pytest.fixture
def prompts_dir(tmp_path):
    out = tmp_path / "prompts"
    out.mkdir()
    write_jsonl(out / "mini.jsonl", PROMPTS)
    return out


@pytest.fixture
def prompts_sha(prompts_dir):
    return hashlib.sha256((prompts_dir / "mini.jsonl").read_bytes()).hexdigest()


def classifier_meta(prompts_sha256: str, **changes) -> dict:
    meta = {
        "judge": "fake-classifier",
        "model": "fake/model-1",
        "provider": "local",  # provider, instructions and thinking are forced by write_classifier_dir
        "revision": "abc123",
        "date": "2026-10-30",
        "temperature": 0,
        "thinking": "x",
        "instructions": "v7-single",
        "prompts_sha256": prompts_sha256,
        "author": "tester",
        "replayable": True,
        "licence": "CC-BY-4.0",
        "notes": "",
    }
    return {**meta, **changes}


def test_write_classifier_dir_refuses_existing(tmp_path, prompts_sha):
    out = tmp_path / "classifiers" / "fake"
    out.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="fake"):
        write_classifier_dir(out, ROWS, classifier_meta(prompts_sha))
    assert list(out.iterdir()) == []  # nothing written into it


def test_write_classifier_dir_writes_a_folder_that_validates(tmp_path, prompts_dir, prompts_sha):
    out = tmp_path / "classifiers" / "fake"
    write_classifier_dir(out, ROWS, classifier_meta(prompts_sha))
    assert validate_judge_dir(out, prompts_dir) == []
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert (meta["provider"], meta["instructions"], meta["thinking"]) == ("system", "none", "none")


def test_read_classifier_labels_fills_the_optional_keys_with_none(tmp_path, prompts_sha):
    out = tmp_path / "fake"
    full = {"complexity": "low", "task": "chat", "needs_tools": False}
    rows = [ROWS[0], {"id": "b-eng", **full, "needs_tools_probability": 0.25}]
    write_classifier_dir(out, rows, classifier_meta(prompts_sha))
    assert read_classifier_labels(out / "labels.jsonl") == {
        "a-eng": {"complexity": "trivial", "task": None, "needs_tools": None, "score": 0.0,
                  "needs_tools_probability": None},
        "b-eng": {**full, "score": None, "needs_tools_probability": 0.25},
    }


def test_write_classifier_dir_refuses_bad_rows_and_writes_nothing(tmp_path, prompts_sha):
    out = tmp_path / "fake"
    with pytest.raises(ValueError, match="rank"):
        write_classifier_dir(out, [{**ROWS[0], "rank": 1}], classifier_meta(prompts_sha))
    with pytest.raises(ValueError, match="needs_tools"):
        write_classifier_dir(out, [{"id": "a-eng", "complexity": "low", "task": None}], classifier_meta(prompts_sha))
    assert not out.exists()


def test_read_classifier_labels_refuses_duplicates(tmp_path, prompts_sha):
    out = tmp_path / "fake"
    write_classifier_dir(out, [ROWS[0], ROWS[0]], classifier_meta(prompts_sha))
    with pytest.raises(ValueError, match="a-eng"):
        read_classifier_labels(out / "labels.jsonl")

