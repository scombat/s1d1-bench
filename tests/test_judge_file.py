import hashlib
import json
from pathlib import Path

import pytest

from s1d1_bench.cli import main
from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.judge_file import META_KEYS, read_labels, validate_judge_dir, write_meta

REPO_ROOT = Path(__file__).resolve().parents[1]

PROMPTS = [
    {"id": "a-eng", "prompt": "What is the capital of France?"},
    {"id": "b-eng", "prompt": "Write a function that reverses a string."},
    {"id": "c-eng", "prompt": "Design a migration plan from a monolith to services."},
]
LABELS = {
    "a-eng": {"complexity": "trivial", "task": "information_seeking", "needs_tools": False},
    "b-eng": {"complexity": "low", "task": "coding_debugging", "needs_tools": False},
    "c-eng": {"complexity": "high", "task": "planning", "needs_tools": True},
}


def label_line(id_: str) -> str:
    return json.dumps({"id": id_, **LABELS[id_]})


VALID_LINES = {id_: label_line(id_) for id_ in LABELS}


def valid_meta(prompts_sha256: str) -> dict:
    return {
        "judge": "fake-judge",
        "model": "fake/model-1",
        "provider": "local",
        "revision": "abc123",
        "date": "2026-10-28",
        "temperature": 0,
        "thinking": "unknown",
        "instructions": "v7-single",
        "prompts_sha256": prompts_sha256,
        "author": "tester",
        "replayable": True,
        "licence": "CC-BY-4.0",
        "notes": "",
    }


@pytest.fixture
def prompts_dir(tmp_path):
    out = tmp_path / "prompts"
    out.mkdir()
    write_jsonl(out / "mini.jsonl", PROMPTS)
    sha = hashlib.sha256((out / "mini.jsonl").read_bytes()).hexdigest()
    (out / "mini.sha256").write_text(sha + "\n", encoding="ascii")
    return out


@pytest.fixture
def prompts_sha(prompts_dir):
    return (prompts_dir / "mini.sha256").read_text().split()[0]


@pytest.fixture
def valid_judge_dir(tmp_path, prompts_sha):
    out = tmp_path / "judges" / "fake"
    out.mkdir(parents=True)
    (out / "meta.json").write_text(json.dumps(valid_meta(prompts_sha)), encoding="utf-8")
    (out / "labels.jsonl").write_text("\n".join(VALID_LINES.values()) + "\n", encoding="utf-8")
    return out


@pytest.fixture
def dir_with_line(valid_judge_dir):
    """A judge directory whose labels are `line`, then the valid lines of the other ids."""

    def make(line: str) -> Path:
        try:
            faulty_id = json.loads(line).get("id")
        except (json.JSONDecodeError, AttributeError):  # not an object: it replaces no id
            faulty_id = None
        rest = [text for id_, text in VALID_LINES.items() if id_ != faulty_id]
        (valid_judge_dir / "labels.jsonl").write_text("\n".join([line, *rest]) + "\n", encoding="utf-8")
        return valid_judge_dir

    return make


def edit_meta(judge_dir: Path, **changes) -> None:
    meta = json.loads((judge_dir / "meta.json").read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is KeyError:
            del meta[key]
        else:
            meta[key] = value
    (judge_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_valid_dir_is_accepted(valid_judge_dir, prompts_dir):
    assert validate_judge_dir(valid_judge_dir, prompts_dir) == []


@pytest.mark.parametrize(
    "line,fault",
    [
        ('{"id": "a-eng", "complexity": "Medium", "task": "chat", "needs_tools": false}', "complexity"),
        ('{"id": "a-eng", "complexity": "null", "task": "chat", "needs_tools": false}', "complexity"),
        ('{"id": "a-eng", "complexity": "low", "task": "chat", "needs_tools": "true"}', "needs_tools"),
        ('{"id": "a-eng", "complexity": "low", "task": "chat", "needs_tools": 1}', "needs_tools"),
        ('{"id": "a-eng", "complexity": "low", "task": "unknown_task", "needs_tools": true}', "task"),
        ('{"id": "zzz-eng", "complexity": "low", "task": "chat", "needs_tools": true}', "unknown id"),
        ('{"id": "a-eng", "complexity": "low", "task": "chat"}', "needs_tools"),
        ('{"id": "a-eng", "complexity": "low", "task": "chat", "needs_tools": true, "rank": 0.2}', "rank"),
        ('["a-eng"]', "object"),
        ("{not json", "JSON"),
    ],
)
def test_bad_label_lines_are_rejected_with_the_fault_named(dir_with_line, prompts_dir, line, fault):
    judge_dir = dir_with_line(line)
    errors = validate_judge_dir(judge_dir, prompts_dir)
    assert len(errors) == 1, errors
    assert "labels.jsonl:1" in errors[0] and fault in errors[0]


def test_an_id_that_is_not_a_string_is_rejected(dir_with_line, prompts_dir):
    errors = validate_judge_dir(
        dir_with_line('{"id": 7, "complexity": "low", "task": "chat", "needs_tools": true}'), prompts_dir
    )
    assert len(errors) == 1 and errors[0].startswith("labels.jsonl:1") and "id 7" in errors[0]


def test_a_line_with_two_faults_gives_one_error(dir_with_line, prompts_dir):
    errors = validate_judge_dir(
        dir_with_line('{"id": "a-eng", "complexity": "Medium", "task": "nope", "needs_tools": "x"}'), prompts_dir
    )
    assert len(errors) == 1
    assert all(field in errors[0] for field in ("complexity", "task", "needs_tools"))


def test_null_complexity_is_allowed(dir_with_line, prompts_dir):
    judge_dir = dir_with_line('{"id": "a-eng", "complexity": null, "task": "chat", "needs_tools": false}')
    assert validate_judge_dir(judge_dir, prompts_dir) == []


def test_missing_and_duplicate_ids_are_rejected(valid_judge_dir, prompts_dir):
    lines = (valid_judge_dir / "labels.jsonl").read_text().splitlines()
    (valid_judge_dir / "labels.jsonl").write_text("\n".join([lines[0], lines[0]]) + "\n")  # 3 ids expected
    errors = validate_judge_dir(valid_judge_dir, prompts_dir)
    assert any("labels.jsonl:2" in e and "duplicate id 'a-eng' (first at line 1)" in e for e in errors)
    assert any("missing 2 id" in e and "b-eng" in e and "c-eng" in e for e in errors)
    assert len(errors) == 2


def test_blank_lines_are_skipped_but_line_numbers_stay_physical(dir_with_line, prompts_dir):
    judge_dir = dir_with_line('{"id": "a-eng", "complexity": "low", "task": "chat", "needs_tools": 2}')
    text = (judge_dir / "labels.jsonl").read_text()
    (judge_dir / "labels.jsonl").write_text("\n" + text)
    errors = validate_judge_dir(judge_dir, prompts_dir)
    assert len(errors) == 1 and "labels.jsonl:2" in errors[0]


def test_meta_with_missing_key_or_unknown_sha_is_rejected(valid_judge_dir, prompts_dir):
    meta = json.loads((valid_judge_dir / "meta.json").read_text())
    del meta["author"]
    meta["prompts_sha256"] = "0" * 64
    (valid_judge_dir / "meta.json").write_text(json.dumps(meta))
    errors = validate_judge_dir(valid_judge_dir, prompts_dir)
    assert any("missing 'author'" in e for e in errors) and any("matches no prompts file" in e for e in errors)
    assert not any(e.startswith("labels.jsonl") for e in errors)  # ids are not checked without a prompts file


def test_a_tampered_prompts_file_is_rejected_even_with_its_sidecar_intact(valid_judge_dir, prompts_dir):
    with open(prompts_dir / "mini.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"id": "d-eng", "prompt": "An extra prompt."}) + "\n")
    errors = validate_judge_dir(valid_judge_dir, prompts_dir)
    assert any("mini.sha256" in e and "mini.jsonl" in e for e in errors)
    assert any("matches no prompts file" in e for e in errors)  # the real digest is no longer the one in meta.json


def test_a_sidecar_without_its_prompts_file_is_rejected(valid_judge_dir, prompts_dir):
    (prompts_dir / "ghost.sha256").write_text("f" * 64 + "\n", encoding="ascii")
    errors = validate_judge_dir(valid_judge_dir, prompts_dir)
    assert any("ghost.sha256" in e and "ghost.jsonl" in e for e in errors)


def test_a_prompts_file_without_sidecar_is_accepted_by_its_real_digest(valid_judge_dir, prompts_dir):
    (prompts_dir / "mini.sha256").unlink()
    assert validate_judge_dir(valid_judge_dir, prompts_dir) == []


def test_an_empty_sidecar_is_rejected(valid_judge_dir, prompts_dir):
    (prompts_dir / "mini.sha256").write_text("\n", encoding="ascii")
    assert any("mini.sha256" in e for e in validate_judge_dir(valid_judge_dir, prompts_dir))


@pytest.mark.parametrize(
    "change,fragment",
    [
        ({"provider": "cloud"}, "provider"),
        ({"instructions": "v6"}, "instructions"),
        ({"replayable": "false"}, "replayable"),
        ({"replayable": 0}, "replayable"),
        ({"licence": "MIT"}, "CC-BY-4.0"),
        ({"nickname": "x"}, "nickname"),
        ({"author": 3}, "author"),
    ],
)
def test_meta_values_are_checked(valid_judge_dir, prompts_dir, change, fragment):
    edit_meta(valid_judge_dir, **change)
    errors = validate_judge_dir(valid_judge_dir, prompts_dir)
    assert len(errors) == 1 and errors[0].startswith("meta.json") and fragment in errors[0]


def test_every_meta_key_is_required(valid_judge_dir, prompts_dir):
    for key in META_KEYS:
        edit_meta(valid_judge_dir, **{key: KeyError})
        errors = validate_judge_dir(valid_judge_dir, prompts_dir)
        assert any(f"missing '{key}'" in e for e in errors), key
        edit_meta(valid_judge_dir, **{key: "x"})  # put something back for the next round


def test_temperature_and_thinking_accept_anything(valid_judge_dir, prompts_dir):
    edit_meta(valid_judge_dir, temperature="unknown", thinking={"budget": 1024})
    assert validate_judge_dir(valid_judge_dir, prompts_dir) == []


def test_runner_keys_are_allowed(valid_judge_dir, prompts_dir):
    edit_meta(valid_judge_dir, quantization="4bit", mlx_lm_version="0.28.0", max_tokens=1024)
    assert validate_judge_dir(valid_judge_dir, prompts_dir) == []


@pytest.mark.parametrize("bad", [0, -1, "1024", True, 1.5, None])
def test_max_tokens_must_be_a_positive_integer(valid_judge_dir, prompts_dir, bad):
    edit_meta(valid_judge_dir, max_tokens=bad)
    assert any("max_tokens" in e for e in validate_judge_dir(valid_judge_dir, prompts_dir))


def test_missing_files_give_clear_errors(tmp_path, prompts_dir):
    empty = tmp_path / "empty"
    empty.mkdir()
    errors = validate_judge_dir(empty, prompts_dir)
    assert any(e.startswith("meta.json") and "not found" in e for e in errors)
    assert any(e.startswith("labels.jsonl") and "not found" in e for e in errors)
    assert validate_judge_dir(tmp_path / "nope", prompts_dir)


def test_meta_that_is_not_json_or_not_an_object(valid_judge_dir, prompts_dir):
    (valid_judge_dir / "meta.json").write_text("{oops")
    assert any(e.startswith("meta.json") and "JSON" in e for e in validate_judge_dir(valid_judge_dir, prompts_dir))
    (valid_judge_dir / "meta.json").write_text("[1]")
    assert any(e.startswith("meta.json") and "object" in e for e in validate_judge_dir(valid_judge_dir, prompts_dir))


def test_read_labels_keys_by_id(valid_judge_dir):
    assert read_labels(valid_judge_dir / "labels.jsonl") == LABELS


def test_read_labels_refuses_duplicates(valid_judge_dir):
    path = valid_judge_dir / "labels.jsonl"
    path.write_text(path.read_text() + VALID_LINES["a-eng"] + "\n")
    with pytest.raises(ValueError, match="a-eng"):
        read_labels(path)


def test_write_meta_orders_keys_and_round_trips(tmp_path, prompts_sha):
    fields = valid_meta(prompts_sha)
    path = tmp_path / "meta.json"
    write_meta(path, **dict(reversed(list(fields.items()))), mlx_lm_version="0.28.0", quantization="4bit")
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n") and '\n  "judge"' in text
    data = json.loads(text)
    assert list(data) == [*META_KEYS, "quantization", "mlx_lm_version"]
    assert {k: data[k] for k in META_KEYS} == fields


def test_write_meta_keeps_unicode(tmp_path, prompts_sha):
    path = tmp_path / "meta.json"
    write_meta(path, **{**valid_meta(prompts_sha), "notes": "é 中文"})
    assert "é 中文" in path.read_text(encoding="utf-8")


def test_write_meta_refuses_missing_and_unknown_keys(tmp_path, prompts_sha):
    fields = valid_meta(prompts_sha)
    with pytest.raises(ValueError, match="author"):
        write_meta(tmp_path / "m.json", **{k: v for k, v in fields.items() if k != "author"})
    with pytest.raises(ValueError, match="nickname"):
        write_meta(tmp_path / "m.json", **fields, nickname="x")
    assert not (tmp_path / "m.json").exists()


def test_cli_validate_accepts(valid_judge_dir, prompts_dir, capsys):
    assert main(["validate", str(valid_judge_dir), "--prompts", str(prompts_dir)]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "ok" and captured.err == ""


def test_cli_validate_refuses_with_one_error_per_line_on_stderr(dir_with_line, prompts_dir, capsys):
    judge_dir = dir_with_line('{"id": "a-eng", "complexity": "Medium", "task": "chat", "needs_tools": false}')
    assert main(["validate", str(judge_dir), "--prompts", str(prompts_dir)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = captured.err.splitlines()
    assert len(lines) == 1 and lines[0].startswith("labels.jsonl:1: ")


def test_cli_validate_default_prompts_dir_is_data_prompts():
    from s1d1_bench.cli import build_parser

    args = build_parser().parse_args(["validate", "judges/x"])
    assert args.prompts == Path("data/prompts") and args.judge_dir == Path("judges/x")


def test_the_shipped_example_is_accepted():
    judge_dir = REPO_ROOT / "judges" / "example-chat"
    assert validate_judge_dir(judge_dir, REPO_ROOT / "data" / "prompts") == []
    assert main(["validate", str(judge_dir), "--prompts", str(REPO_ROOT / "data" / "prompts")]) == 0


def system_dir(judge_dir: Path, lines: dict[str, dict]) -> Path:
    """`judge_dir` turned into a classifier folder: system meta, and `lines` (id -> label fields) as its labels."""
    edit_meta(judge_dir, provider="system", instructions="none", thinking="none")
    text = "\n".join(json.dumps({"id": id_, **fields}) for id_, fields in lines.items()) + "\n"
    (judge_dir / "labels.jsonl").write_text(text, encoding="utf-8")
    return judge_dir


def scored(id_: str, **extra) -> dict:
    return {**LABELS[id_], **extra}


def test_validate_accepts_a_system_folder_with_scores(valid_judge_dir, prompts_dir):
    lines = {
        "a-eng": scored("a-eng", score=0.0, needs_tools_probability=0.5),
        "b-eng": scored("b-eng", score=1, needs_tools_probability=1),
        "c-eng": scored("c-eng", score=0.73),  # both keys are optional, one by one
    }
    assert validate_judge_dir(system_dir(valid_judge_dir, lines), prompts_dir) == []


@pytest.mark.parametrize(
    "field,bad",
    [
        ("score", 1.2),
        ("score", -0.01),
        ("score", "0.5"),
        ("score", True),
        ("score", None),
        ("score", float("nan")),
        ("needs_tools_probability", 2),
        ("needs_tools_probability", -1),
        ("needs_tools_probability", False),
        ("needs_tools_probability", None),
    ],
)
def test_validate_refuses_score_out_of_range(valid_judge_dir, prompts_dir, field, bad):
    lines = {id_: scored(id_) for id_ in LABELS}
    lines["a-eng"] = scored("a-eng", **{field: bad})
    errors = validate_judge_dir(system_dir(valid_judge_dir, lines), prompts_dir)
    assert len(errors) == 1 and errors[0].startswith("labels.jsonl:1") and field in errors[0], errors


def test_scores_are_checked_in_a_judge_folder_too(valid_judge_dir, prompts_dir):
    assert validate_judge_dir(valid_judge_dir, prompts_dir) == []
    lines = {id_: scored(id_) for id_ in LABELS}
    lines["a-eng"] = scored("a-eng", score=3)
    (valid_judge_dir / "labels.jsonl").write_text("\n".join(json.dumps({"id": i, **f}) for i, f in lines.items()))
    errors = validate_judge_dir(valid_judge_dir, prompts_dir)
    assert len(errors) == 1 and "score" in errors[0]


@pytest.mark.parametrize("field", ["task", "needs_tools"])
def test_validate_refuses_null_task_for_a_local_judge(dir_with_line, prompts_dir, field):
    row = {"id": "a-eng", **LABELS["a-eng"], field: None}
    errors = validate_judge_dir(dir_with_line(json.dumps(row)), prompts_dir)
    assert len(errors) == 1 and errors[0].startswith("labels.jsonl:1") and field in errors[0], errors


def test_validate_accepts_null_task_for_a_system_folder(valid_judge_dir, prompts_dir):
    lines = {id_: {"complexity": LABELS[id_]["complexity"], "task": None, "needs_tools": None} for id_ in LABELS}
    lines["a-eng"]["score"] = 0.01
    assert validate_judge_dir(system_dir(valid_judge_dir, lines), prompts_dir) == []


def test_a_system_folder_still_refuses_a_bad_task_and_a_bad_complexity(valid_judge_dir, prompts_dir):
    lines = {id_: scored(id_) for id_ in LABELS}
    lines["a-eng"] = {"complexity": "Medium", "task": "nope", "needs_tools": "x"}
    errors = validate_judge_dir(system_dir(valid_judge_dir, lines), prompts_dir)
    assert len(errors) == 1 and all(field in errors[0] for field in ("complexity", "task", "needs_tools"))


def test_a_system_folder_may_not_hide_a_label_key(valid_judge_dir, prompts_dir):
    lines = {id_: scored(id_) for id_ in LABELS}
    lines["a-eng"] = {"complexity": "trivial", "needs_tools": None}  # null is allowed, absent is not
    errors = validate_judge_dir(system_dir(valid_judge_dir, lines), prompts_dir)
    assert len(errors) == 1 and "missing field 'task'" in errors[0]


@pytest.mark.parametrize(
    "change",
    [
        {"provider": "system"},  # a system folder needs instructions "none"
        {"instructions": "none"},  # and a judge cannot have it
        {"provider": "human", "instructions": "none"},
    ],
)
def test_provider_system_and_instructions_none_go_together(valid_judge_dir, prompts_dir, change):
    edit_meta(valid_judge_dir, **change)
    errors = validate_judge_dir(valid_judge_dir, prompts_dir)
    assert len(errors) == 1 and errors[0].startswith("meta.json") and "'none'" in errors[0], errors


def test_cli_validate_accepts_a_system_folder(valid_judge_dir, prompts_dir, capsys):
    lines = {id_: {"complexity": "low", "task": None, "needs_tools": None, "score": 0.25} for id_ in LABELS}
    assert main(["validate", str(system_dir(valid_judge_dir, lines)), "--prompts", str(prompts_dir)]) == 0
    assert capsys.readouterr().out.strip() == "ok"


# --- abstentions: a judge's non-answer (its final status after the retry was not ok)


def abstention_line(id_: str, status: str = "json_error", **change) -> str:
    row = {"id": id_, "complexity": None, "task": None, "needs_tools": None, "abstained": status, **change}
    return json.dumps(row)


def write_log(judge_dir: Path, rows: list[tuple[str, str]]) -> None:
    text = "".join(json.dumps({"id": i, "status": s, "retries": 1}) + "\n" for i, s in rows)
    (judge_dir / "log.jsonl").write_text(text, encoding="utf-8")


@pytest.mark.parametrize("status", ["json_error", "truncated"])
def test_an_abstention_row_is_accepted(dir_with_line, prompts_dir, status):
    assert validate_judge_dir(dir_with_line(abstention_line("a-eng", status)), prompts_dir) == []


@pytest.mark.parametrize(
    "change,fault",
    [
        ({"complexity": "low"}, "complexity"),
        ({"task": "chat"}, "task"),
        ({"needs_tools": False}, "needs_tools"),
        ({"abstained": "ok"}, "abstained"),
        ({"abstained": "refused"}, "abstained"),
        ({"abstained": "error"}, "abstained"),  # ruling R7: a backend error is no abstention
        ({"abstained": None}, "abstained"),
        ({"abstained": True}, "abstained"),
        ({"score": 0.5}, "score"),
    ],
)
def test_a_bad_abstention_row_is_rejected(dir_with_line, prompts_dir, change, fault):
    errors = validate_judge_dir(dir_with_line(abstention_line("a-eng", **change)), prompts_dir)
    assert len(errors) == 1 and errors[0].startswith("labels.jsonl:1") and fault in errors[0], errors


def test_an_abstention_row_still_needs_its_three_fields(dir_with_line, prompts_dir):
    line = json.dumps({"id": "a-eng", "complexity": None, "abstained": "json_error"})
    errors = validate_judge_dir(dir_with_line(line), prompts_dir)
    assert len(errors) == 1 and "missing field 'task', 'needs_tools'" in errors[0], errors


def test_a_system_folder_does_not_abstain(valid_judge_dir, prompts_dir):
    lines = {id_: LABELS[id_] for id_ in LABELS}
    lines["a-eng"] = {"complexity": None, "task": None, "needs_tools": None, "abstained": "truncated"}
    errors = validate_judge_dir(system_dir(valid_judge_dir, lines), prompts_dir)
    assert len(errors) == 1 and errors[0].startswith("labels.jsonl:1") and "abstained" in errors[0], errors


def test_an_abstention_matches_the_last_log_line_of_its_id(dir_with_line, prompts_dir):
    judge_dir = dir_with_line(abstention_line("a-eng", "json_error"))
    write_log(judge_dir, [("a-eng", "error"), ("b-eng", "ok"), ("c-eng", "ok"), ("a-eng", "json_error")])
    assert validate_judge_dir(judge_dir, prompts_dir) == []  # an earlier line is history


@pytest.mark.parametrize(
    "log,fault",
    [
        ([("a-eng", "json_error"), ("a-eng", "truncated")], "'truncated'"),  # the last line decides
        ([("a-eng", "ok")], "'ok'"),
        ([("b-eng", "ok")], "no line"),
    ],
)
def test_an_abstention_the_log_contradicts_is_rejected(dir_with_line, prompts_dir, log, fault):
    judge_dir = dir_with_line(abstention_line("a-eng", "json_error"))
    write_log(judge_dir, log)
    errors = validate_judge_dir(judge_dir, prompts_dir)
    assert len(errors) == 1 and errors[0].startswith("labels.jsonl:1") and "log.jsonl" in errors[0], errors
    assert fault in errors[0]


def test_a_log_line_that_does_not_parse_is_named(dir_with_line, prompts_dir):
    judge_dir = dir_with_line(abstention_line("a-eng", "json_error"))
    write_log(judge_dir, [("a-eng", "json_error")])
    with open(judge_dir / "log.jsonl", "a", encoding="utf-8") as f:
        f.write('{"id": "b-eng", "sta')
    errors = validate_judge_dir(judge_dir, prompts_dir)
    assert errors == ["log.jsonl:2: not valid JSON, cannot check the abstentions"], errors


def test_the_log_is_not_read_without_an_abstention(valid_judge_dir, prompts_dir):
    (valid_judge_dir / "log.jsonl").write_text("not json\n", encoding="utf-8")
    assert validate_judge_dir(valid_judge_dir, prompts_dir) == []


def test_read_labels_keeps_an_abstention_with_null_fields(dir_with_line):
    judge_dir = dir_with_line(abstention_line("a-eng", "truncated"))
    labels = read_labels(judge_dir / "labels.jsonl")
    assert labels["a-eng"] == {"complexity": None, "task": None, "needs_tools": None, "abstained": "truncated"}
    assert labels["b-eng"] == LABELS["b-eng"]  # no `abstained` key on an answer


def test_read_labels_refuses_an_abstention_with_a_label(dir_with_line):
    judge_dir = dir_with_line(abstention_line("a-eng", complexity="low"))
    with pytest.raises(ValueError, match="a-eng"):
        read_labels(judge_dir / "labels.jsonl")


@pytest.mark.parametrize("n,summary", [(0, "ok"), (1, "ok (1 abstention)"), (2, "ok (2 abstentions)")])
def test_cli_validate_counts_the_abstentions(valid_judge_dir, prompts_dir, capsys, n, summary):
    lines = [abstention_line(i) if k < n else VALID_LINES[i] for k, i in enumerate(LABELS)]
    (valid_judge_dir / "labels.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert main(["validate", str(valid_judge_dir), "--prompts", str(prompts_dir)]) == 0
    assert capsys.readouterr().out.strip() == summary
