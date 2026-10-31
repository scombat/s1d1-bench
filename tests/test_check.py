import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1d1_bench.check import (
    REQUEST_INSTRUCTION,
    build_request_messages,
    label_faults,
    parse_request,
    review_sheet,
    run_check,
    run_request,
    structural_faults,
)
from s1d1_bench.cli import build_parser, main
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_runner import GenerateResult

REPO_ROOT = Path(__file__).resolve().parents[1]
FLAG_KEYS = ["id", "base_key", "lang", "check", "rule", "detail", "lifted"]


def test_structure_is_silent_on_a_faithful_translation():
    english, french = "Fix `x`: see https://a.b/c, version 2.1", "Corrige `x` : voir https://a.b/c, version 2.1"
    assert structural_faults(english, french, "fra") == []


@pytest.mark.parametrize(
    "en,tr,rule",
    [
        ("```py\na\n```", "a", "fences"),
        ("see https://a.b", "voir https://a.c", "urls"),
        ("in 1992", "en 1993", "numbers"),
        ('say "hello"', 'dis "bonjour"', "quotes"),
        ("a\nb\nc\nd\ne", "a b c d e", "lines"),
        ("hello", "x" * 20, "length"),
        ("hi", "salut <prompt>", "tag"),
        ("hi", "salut\n</prompt>", "envelope"),
    ],
)
def test_structure_rules(en, tr, rule):
    assert rule in [f["rule"] for f in structural_faults(en, tr, "fra")]


def test_structure_rules_come_in_the_fixed_order_and_say_both_values():
    faults = structural_faults("```\nin 1992\n```", "en l'année 1993", "fra")
    assert [f["rule"] for f in faults] == ["fences", "numbers", "lines"]
    assert faults[0] == {"rule": "fences", "detail": "eng=2 tr=0"}
    assert "1992" in faults[1]["detail"] and "1993" in faults[1]["detail"]


def test_lines_allow_one_line_of_difference_on_a_short_prompt():
    assert structural_faults("one line here", "une\nligne ici", "fra") == []
    assert [f["rule"] for f in structural_faults("one line here", "une\nligne\nici", "fra")] == ["lines"]


def test_numbers_compare_full_width_digits_as_ascii():
    assert structural_faults("about 1992 things", "关于１９９２件事情", "zho") == []


def test_tag_already_in_the_english_is_not_a_fault():
    en = "Write the <prompt> element of my XML schema."
    assert "tag" not in [f["rule"] for f in structural_faults(en, "Écris l'élément <prompt> de mon schéma XML.", "fra")]


def test_chinese_bounds_are_narrower():
    english = "a" * 60
    assert structural_faults(english, "中" * 40, "zho") == []
    assert [f["rule"] for f in structural_faults(english, "中" * 10, "zho")] == ["length"]
    assert [f["rule"] for f in structural_faults(english, "é" * 40, "fra")] == ["length"]  # 0.67 < 0.7 in French


def test_labels_flag_task_and_tools_never_complexity():
    eng = {"complexity": "low", "task": "editing", "needs_tools": False}
    tr = {"complexity": "high", "task": "translation", "needs_tools": True}
    faults = label_faults(eng, tr, None)
    assert [f["rule"] for f in faults] == ["task", "needs_tools"]
    assert faults[0] == {"rule": "task", "detail": "eng=editing tr=translation", "witness": None, "lifted": False}
    assert faults[1]["detail"] == "eng=false tr=true"
    assert label_faults(eng, {**eng, "complexity": "expert"}, None) == []


def test_witness_agreeing_with_the_translation_lifts_the_flag():
    eng = {"complexity": "low", "task": "editing", "needs_tools": False}
    tr = {"complexity": "low", "task": "translation", "needs_tools": False}
    witness = {"complexity": "medium", "task": "translation", "needs_tools": True}
    assert label_faults(eng, tr, witness) == [
        {"rule": "task", "detail": "eng=editing tr=translation", "witness": "agrees_with_translation", "lifted": True}
    ]


def test_witness_agreeing_with_english_keeps_the_flag():
    eng = {"complexity": "low", "task": "editing", "needs_tools": False}
    tr = {"complexity": "low", "task": "translation", "needs_tools": True}
    witness = {"complexity": "low", "task": "editing", "needs_tools": False}
    faults = label_faults(eng, tr, witness)
    assert [(f["rule"], f["witness"], f["lifted"]) for f in faults] == [
        ("task", "agrees_with_english", False),
        ("needs_tools", "agrees_with_english", False),
    ]


def flag(key, rule, check="structure", detail="eng=1 tr=0", lifted=False, lang="fra"):
    return {
        "id": f"{key}-{lang}",
        "base_key": key,
        "lang": lang,
        "check": check,
        "rule": rule,
        "detail": detail,
        "lifted": lifted,
    }


def test_review_sheet_lists_each_flagged_id_once_with_both_texts():
    flags = [
        flag("kb", "numbers"),
        flag("ka", "task", check="labels", detail="eng=editing tr=translation", lifted=True),
        flag("ka", "fences"),
    ]
    english = {"ka": "English A", "kb": "English B", "kc": "English C"}
    translations = {"ka": "Français A", "kb": "Français B", "kc": "Français C"}
    sheet = review_sheet(flags, english, translations, {"ka": "Témoin A"})
    assert sheet.count("## ka-fra") == 1 and sheet.count("## kb-fra") == 1
    assert "kc" not in sheet and "English C" not in sheet  # an id without flags has no section
    assert sheet.index("## ka-fra") < sheet.index("## kb-fra")
    for text in ("English A", "Français A", "English B", "Français B", "Témoin A"):
        assert text in sheet
    section_a = sheet[sheet.index("## ka-fra") : sheet.index("## kb-fra")]
    assert section_a.index("fences") < section_a.index("task")  # structure rules before label rules
    assert "task : eng=editing tr=translation [levé]" in section_a
    assert "Témoin" not in sheet[sheet.index("## kb-fra") :]  # no witness for kb
    assert sheet.count("\navis :\n") == 2 and sheet.count("\ndécision :\n") == 2
    head = sheet[: sheet.index("## ka-fra")]
    assert "fra" in head and "2 ids" in head and "3 signalements" in head and "1 levé" in head


def test_review_sheet_fences_a_text_that_holds_a_fence():
    english = {"ka": "Fix this:\n```py\nx = 1\n```"}
    sheet = review_sheet([flag("ka", "numbers")], english, {"ka": "Corrige"}, {})
    assert "````\nFix this:\n```py\nx = 1\n```\n````" in sheet


ENGLISH = [
    {"id": "ka-eng", "prompt": "Fix `x`: see https://a.b/c, version 2.1"},
    {"id": "kb-eng", "prompt": 'In 1992, say "hello".\nThen stop.'},
    {"id": "kc-eng", "prompt": "Write a poem."},
]
FRENCH = [
    {"id": "kb-fra", "base_key": "kb", "prompt": "En 1993, dis « bonjour ».\nPuis arrête."},
    {"id": "ka-fra", "base_key": "ka", "prompt": "Corrige `x` : voir https://a.b/c, version 2.1"},
    {"id": "kc-fra", "base_key": "kc", "prompt": "Écris un poème."},
]


def write_set(tmp_path):
    """An English prompts file, a French translation folder and a French witness folder (kb only)."""
    english = tmp_path / "base-eng.jsonl"
    write_jsonl(english, ENGLISH)
    translations = tmp_path / "translations"
    translations.mkdir()
    write_jsonl(translations / "fra.jsonl", FRENCH)
    (translations / "fra.meta.json").write_text(json.dumps({"translator": "qwen@abc"}), encoding="utf-8")
    witness = translations / "witness-arena"
    witness.mkdir()
    write_jsonl(witness / "fra.jsonl", [{"id": "kb-fra", "base_key": "kb", "prompt": 'En 1992, dis "bonjour".'}])
    (witness / "fra.meta.json").write_text(json.dumps({"translator": "Google"}), encoding="utf-8")
    return english, translations, witness


def write_labels(folder, rows):
    folder.mkdir(parents=True)
    write_jsonl(folder / "labels.jsonl", rows)
    return folder


def args_for(tmp_path, kind, english, translations, witness=None, **labels):
    return SimpleNamespace(
        kind=kind,
        set="base-v1",
        lang="fra",
        english=english,
        translations=translations,
        witness=witness,
        labels_english=labels.get("labels_english"),
        labels=labels.get("labels"),
        labels_witness=labels.get("labels_witness"),
        out=tmp_path / "checks",
    )


def test_run_check_structure_writes_one_line_per_fault(tmp_path, capsys):
    english, translations, _ = write_set(tmp_path)
    assert run_check(args_for(tmp_path, "structure", english, translations)) == 0
    path = tmp_path / "checks" / "base-v1" / "fra.structure.jsonl"
    rows = read_jsonl(path)
    assert [(r["id"], r["rule"]) for r in rows] == [("kb-fra", "numbers"), ("kb-fra", "quotes")]
    assert all(list(r) == FLAG_KEYS for r in rows)
    assert rows[0] == {
        "id": "kb-fra",
        "base_key": "kb",
        "lang": "fra",
        "check": "structure",
        "rule": "numbers",
        "detail": 'eng=["1992"] tr=["1993"]',
        "lifted": False,
    }
    out = capsys.readouterr().out
    assert "numbers: 1" in out and "quotes: 1" in out
    before = path.read_bytes()
    assert run_check(args_for(tmp_path, "structure", english, translations)) == 0  # a derivation: rewritten
    assert path.read_bytes() == before


def test_run_check_labels_lifts_with_the_witness_and_reports_a_missing_label(tmp_path):
    english, translations, _ = write_set(tmp_path)
    eng = write_labels(
        tmp_path / "judges" / "eng",
        [
            {"id": "ka-eng", "complexity": "low", "task": "coding_debugging", "needs_tools": False},
            {"id": "kb-eng", "complexity": "low", "task": "editing", "needs_tools": False},
            {"id": "kc-eng", "complexity": "low", "task": "creative_writing", "needs_tools": False},
        ],
    )
    fra = write_labels(
        tmp_path / "judges" / "fra",
        [
            {"id": "kb-fra", "complexity": "high", "task": "translation", "needs_tools": False},
            {"id": "ka-fra", "complexity": "low", "task": "coding_debugging", "needs_tools": True},
        ],  # kc-fra was not labelled ok
    )
    witness = [{"id": "kb-fra", "complexity": "low", "task": "translation", "needs_tools": False}]
    wit = write_labels(tmp_path / "judges" / "wit", witness)
    args = args_for(tmp_path, "labels", english, translations, labels_english=eng, labels=fra, labels_witness=wit)
    assert run_check(args) == 0
    rows = read_jsonl(tmp_path / "checks" / "base-v1" / "fra.labels.jsonl")
    assert [(r["id"], r["check"], r["rule"], r["lifted"]) for r in rows] == [
        ("ka-fra", "labels", "needs_tools", False),
        ("kb-fra", "labels", "task", True),
        ("kc-fra", "labels", "label_missing", False),
    ]
    assert all(list(r) == FLAG_KEYS for r in rows)
    assert rows[1]["detail"] == "eng=editing tr=translation witness=translation"
    assert str(fra) in rows[2]["detail"]


def test_run_check_labels_needs_both_label_folders(tmp_path, capsys):
    english, translations, _ = write_set(tmp_path)
    assert run_check(args_for(tmp_path, "labels", english, translations)) == 1
    assert "--labels-english" in capsys.readouterr().err


def test_run_check_review_reads_request_false_as_a_flag(tmp_path):
    english, translations, witness = write_set(tmp_path)
    assert run_check(args_for(tmp_path, "structure", english, translations)) == 0
    checks = tmp_path / "checks" / "base-v1"
    write_jsonl(
        checks / "fra.request.jsonl",
        [
            {"id": "ka-fra", "same_request": True, "reason": "same"},
            {"id": "kc-fra", "same_request": False, "reason": "the poem became a song"},
        ],
    )
    assert run_check(args_for(tmp_path, "review", english, translations, witness)) == 0
    sheet = (checks / "fra.review.md").read_text(encoding="utf-8")
    assert "## kc-fra" in sheet and "request : the poem became a song" in sheet
    assert "## ka-fra" not in sheet  # same_request true: no flag
    assert "## kb-fra" in sheet and 'En 1992, dis "bonjour".' in sheet  # the witness of kb
    assert "Write a poem." in sheet and "Écris un poème." in sheet


def test_run_check_review_keeps_a_sheet_that_was_filled_in(tmp_path, capsys):
    english, translations, _ = write_set(tmp_path)
    assert run_check(args_for(tmp_path, "structure", english, translations)) == 0
    assert run_check(args_for(tmp_path, "review", english, translations)) == 0
    sheet = tmp_path / "checks" / "base-v1" / "fra.review.md"
    assert run_check(args_for(tmp_path, "review", english, translations)) == 0  # same flags: same bytes, accepted
    sheet.write_text(sheet.read_text(encoding="utf-8").replace("avis :", "avis : ok", 1), encoding="utf-8")
    capsys.readouterr()
    assert run_check(args_for(tmp_path, "review", english, translations)) == 1
    assert "avis : ok" in sheet.read_text(encoding="utf-8")
    assert "fra.review.md" in capsys.readouterr().err


def test_run_check_review_without_any_check_file_fails(tmp_path, capsys):
    english, translations, _ = write_set(tmp_path)
    assert run_check(args_for(tmp_path, "review", english, translations)) == 1
    assert "structure" in capsys.readouterr().err


def test_cli_check_takes_the_four_kinds_and_request_needs_a_model(tmp_path, capsys):
    english, translations, _ = write_set(tmp_path)
    common = ["--set", "s", "--lang", "fra", "--english", str(english), "--translations", str(translations)]
    assert main(["check", "request", *common, "--out", str(tmp_path / "checks")]) == 1
    err = capsys.readouterr().err
    assert err.startswith("bench check request:") and "--model" in err and "--author" in err and "--thinking" in err
    assert not (tmp_path / "checks").exists()
    with pytest.raises(SystemExit):
        main(["check", "spelling", *common])
    assert main(["check", "structure", *common, "--out", str(tmp_path / "checks")]) == 0
    assert (tmp_path / "checks" / "s" / "fra.structure.jsonl").exists()


def test_run_check_review_flags_a_translation_without_a_verdict(tmp_path):
    english, translations, _ = write_set(tmp_path)
    checks = tmp_path / "checks" / "base-v1"
    checks.mkdir(parents=True)
    write_jsonl(checks / "fra.request.jsonl", [{"id": "ka-fra", "same_request": True, "reason": "same"}])
    write_jsonl(
        checks / "fra.request.log.jsonl",
        [
            {"id": "kb-fra", "status": "truncated", "retries": 0},
            {"id": "ka-fra", "status": "json_error", "retries": 1},
            {"id": "ka-fra", "status": "ok", "retries": 0},  # resumed: the last line of an id is its status
        ],
    )  # kc-fra was never tried: the run was stopped before it
    assert run_check(args_for(tmp_path, "review", english, translations)) == 0
    sheet = (checks / "fra.review.md").read_text(encoding="utf-8")
    assert "## ka-fra" not in sheet
    assert "- request / request : no verdict (truncated in the log)" in sheet[sheet.index("## kb-fra") :]
    assert "- request / request : no verdict (not run)" in sheet[sheet.index("## kc-fra") :]


def test_run_check_review_without_a_request_log_says_not_run(tmp_path):
    english, translations, _ = write_set(tmp_path)
    checks = tmp_path / "checks" / "base-v1"
    checks.mkdir(parents=True)
    write_jsonl(
        checks / "fra.request.jsonl",
        [
            {"id": "ka-fra", "same_request": True, "reason": "same"},
            {"id": "kb-fra", "same_request": False, "reason": "the year changed"},
        ],
    )
    assert run_check(args_for(tmp_path, "review", english, translations)) == 0
    sheet = (checks / "fra.review.md").read_text(encoding="utf-8")
    assert "request : the year changed" in sheet and "request : no verdict (not run)" in sheet
    assert sheet.count("request / request") == 2


# --- bench check request: Gemma 4 reads the English and the translation


def result(text, finish_reason="stop", prompt_tokens=10, completion_tokens=5, wall_s=0.5):
    return GenerateResult(text, prompt_tokens, completion_tokens, finish_reason, wall_s)


def fake_generate(answers):
    """A GenerateFn that pops `answers` (a str, a GenerateResult or an exception), counts and records its calls."""
    queue = list(answers)

    def generate(messages):
        generate.calls += 1
        generate.messages.append(messages)
        answer = queue.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, GenerateResult) else result(answer)

    generate.calls = 0
    generate.messages = []
    return generate


def verdict(same, reason="ok"):
    return json.dumps({"same_request": same, "reason": reason})


PAIRS = [
    {"id": "kb-fra", "english": "In 1992, say hello.", "translation": "En 1993, dis bonjour."},
    {"id": "ka-fra", "english": "Fix x.", "translation": "Corrige x."},
    {"id": "kc-fra", "english": "Write a poem.", "translation": "Écris un poème."},
]


def test_request_messages_put_both_texts_as_data_between_marked_tags():
    m = build_request_messages("CONSIGNE", "Write a poem.", "Écris un poème.", "fra")
    assert m[0] == {"role": "system", "content": "CONSIGNE"}
    assert m[1] == {
        "role": "user",
        "content": '<prompt lang="eng">\nWrite a poem.\n</prompt>\n\n'
        '<prompt lang="fra">\nÉcris un poème.\n</prompt>\n\n' + REQUEST_INSTRUCTION,
    }
    assert REQUEST_INSTRUCTION == (
        "Is the translation between the second pair of tags the same request as the English message between the "
        "first pair? Return the JSON object now."
    )


@pytest.mark.parametrize(
    "text",
    [
        '{"same_request": true, "reason": "same ask"}',
        '```json\n{"same_request": true, "reason": "same ask"}\n```',
        '```\n{"same_request": true, "reason": "same ask"}\n```\n',
        '<|channel>thought\n{"same_request": false, "reason": "draft"}<channel|>'
        '{"same_request": true, "reason": "same ask"}',
    ],
)
def test_parse_request_reads_a_bare_or_fenced_object(text):
    assert parse_request(text) == {"same_request": True, "reason": "same ask"}


def test_parse_request_keeps_the_key_order_and_a_false_verdict():
    parsed = parse_request('{"reason": "a constraint is lost", "same_request": false}')
    assert parsed == {"same_request": False, "reason": "a constraint is lost"}
    assert list(parsed) == ["same_request", "reason"]


@pytest.mark.parametrize(
    "text",
    [
        '{"same_request": "true", "reason": "x"}',
        '{"same_request": 1, "reason": "x"}',
        '{"same_request": null, "reason": "x"}',
        '{"same_request": true, "reason": 3}',
        '{"same_request": true}',
        '{"same_request": true, "reason": "x", "confidence": 0.9}',
        '{"same_request": true, "reason": "x"} {"same_request": false, "reason": "y"}',
        "Yes, it is the same request.",
        "",
        '["same_request", true]',
    ],
)
def test_parse_request_refuses_anything_else(text):
    assert parse_request(text) is None


def test_run_request_writes_a_verdict_per_pair_in_input_order(tmp_path):
    generate = fake_generate([verdict(False, "the year changed"), verdict(True, "same"), verdict(True, "same")])
    assert run_request(PAIRS, generate, tmp_path, "fra", consigne="C") == {"ok": 3}
    assert read_jsonl(tmp_path / "fra.request.jsonl") == [
        {"id": "kb-fra", "same_request": False, "reason": "the year changed"},
        {"id": "ka-fra", "same_request": True, "reason": "same"},
        {"id": "kc-fra", "same_request": True, "reason": "same"},
    ]
    log = read_jsonl(tmp_path / "fra.request.log.jsonl")
    assert [(r["id"], r["status"]) for r in log] == [("kb-fra", "ok"), ("ka-fra", "ok"), ("kc-fra", "ok")]
    assert log[0] == {
        "id": "kb-fra",
        "status": "ok",
        "retries": 0,
        "wall_s": 0.5,
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    assert generate.messages[0] == build_request_messages("C", PAIRS[0]["english"], PAIRS[0]["translation"], "fra")


def test_run_request_retries_a_json_error_once_with_a_nudge(tmp_path):
    generate = fake_generate(["It is the same.", verdict(True, "same"), "nope", "still nope", verdict(True)])
    assert run_request(PAIRS, generate, tmp_path, "fra", consigne="C") == {"ok": 2, "json_error": 1}
    assert generate.calls == 5
    assert generate.messages[1][0] == generate.messages[0][0]
    assert generate.messages[1][1]["content"].startswith(generate.messages[0][1]["content"])
    assert generate.messages[1][1]["content"] != generate.messages[0][1]["content"]
    assert [r["id"] for r in read_jsonl(tmp_path / "fra.request.jsonl")] == ["kb-fra", "kc-fra"]
    log = read_jsonl(tmp_path / "fra.request.log.jsonl")
    assert (log[0]["status"], log[0]["retries"]) == ("ok", 1) and "raw" not in log[0]
    assert (log[1]["status"], log[1]["retries"], log[1]["raw"]) == ("json_error", 1, "still nope")
    assert log[1]["prompt_tokens"] == 20 and log[1]["wall_s"] == 1.0


def test_run_request_does_not_retry_a_cut_answer_or_an_exception(tmp_path):
    generate = fake_generate(
        [result('{"same_request": tr', finish_reason="length"), RuntimeError("boom"), verdict(True)]
    )
    assert run_request(PAIRS, generate, tmp_path, "fra", consigne="C") == {"truncated": 1, "error": 1, "ok": 1}
    assert generate.calls == 3
    log = read_jsonl(tmp_path / "fra.request.log.jsonl")
    assert [(r["status"], r["retries"]) for r in log] == [("truncated", 0), ("error", 0), ("ok", 0)]
    assert log[0]["raw"] == '{"same_request": tr' and "boom" in log[1]["raw"]
    assert [r["id"] for r in read_jsonl(tmp_path / "fra.request.jsonl")] == ["kc-fra"]


def test_run_request_refuses_to_overwrite(tmp_path):
    run_request(PAIRS, fake_generate([verdict(True)] * 3), tmp_path, "fra", consigne="C")
    before = {n: (tmp_path / n).read_bytes() for n in ("fra.request.jsonl", "fra.request.log.jsonl")}
    generate = fake_generate([verdict(False)] * 3)
    with pytest.raises(FileExistsError, match="immutable"):
        run_request(PAIRS, generate, tmp_path, "fra", consigne="C")
    assert generate.calls == 0 and {n: (tmp_path / n).read_bytes() for n in before} == before


def test_run_request_resume_skips_the_verdicts_already_written(tmp_path):
    run_request(PAIRS, fake_generate([verdict(True), "x", "y", verdict(False, "lost")]), tmp_path, "fra", consigne="C")
    generate = fake_generate([verdict(True, "now ok")])
    assert run_request(PAIRS, generate, tmp_path, "fra", resume=True, consigne="C") == {"ok": 1}
    assert generate.calls == 1 and "Fix x." in generate.messages[0][1]["content"]
    rows = read_jsonl(tmp_path / "fra.request.jsonl")
    assert [(r["id"], r["reason"]) for r in rows] == [("kb-fra", "ok"), ("ka-fra", "now ok"), ("kc-fra", "lost")]
    log = read_jsonl(tmp_path / "fra.request.log.jsonl")
    assert [(r["id"], r["status"]) for r in log] == [
        ("kb-fra", "ok"),
        ("ka-fra", "json_error"),
        ("kc-fra", "ok"),
        ("ka-fra", "ok"),
    ]


def test_run_request_resume_with_foreign_ids_raises_and_touches_nothing(tmp_path):
    run_request(PAIRS, fake_generate([verdict(True)] * 3), tmp_path, "fra", consigne="C")
    before = {n: (tmp_path / n).read_bytes() for n in ("fra.request.jsonl", "fra.request.log.jsonl")}
    generate = fake_generate([verdict(True)])
    with pytest.raises(ValueError, match="ka-fra.*kc-fra"):
        run_request(PAIRS[:1], generate, tmp_path, "fra", resume=True, consigne="C")
    assert generate.calls == 0 and {n: (tmp_path / n).read_bytes() for n in before} == before


def test_run_request_refuses_bad_pairs(tmp_path):
    with pytest.raises(ValueError, match="duplicate"):
        run_request([PAIRS[0], PAIRS[0]], fake_generate([]), tmp_path, "fra", consigne="C")
    with pytest.raises(ValueError, match="-fra"):
        run_request([{**PAIRS[0], "id": "kb-rus"}], fake_generate([]), tmp_path, "fra", consigne="C")
    with pytest.raises(ValueError, match="translation"):
        run_request([{**PAIRS[0], "translation": None}], fake_generate([]), tmp_path, "fra", consigne="C")
    assert not (tmp_path / "fra.request.jsonl").exists()


# --- bench check request (the command), with the mlx backend replaced by a fake

BACKEND_META = {
    "model": "org/gemma-8bit",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "quantization": "8bit-gs64",
    "mlx_lm_version": "0.32.2",
    "temperature": 0,
    "thinking": "off",
    "behaviour": "fake: no reasoning.",
}


@pytest.fixture
def at_repo_root(monkeypatch):
    """The default consigne, consigne/check-same-request-v1.md, is relative to the working directory."""
    monkeypatch.chdir(REPO_ROOT)


def fake_backend(monkeypatch, answers, meta=None, described=None):
    """Replace make_mlx_generate and describe_model (`described`: what describe_model says, default `meta`); return
    the fake GenerateFn and the recorded arguments (empty while the model was never loaded)."""
    from s1d1_bench import mlx_backend

    generate = fake_generate(answers)
    seen = {}

    def make(model_id, thinking, max_tokens=1024):
        seen.update(model_id=model_id, thinking=thinking, max_tokens=max_tokens)
        return generate, dict(meta or BACKEND_META, model=model_id, thinking=thinking)

    def describe(model_id, thinking):
        fields = dict(described or meta or BACKEND_META, model=model_id, thinking=thinking)
        fields.pop("mlx_lm_version")  # known only once mlx_lm is imported
        return Path("unused"), None, fields

    monkeypatch.setattr(mlx_backend, "make_mlx_generate", make)
    monkeypatch.setattr(mlx_backend, "describe_model", describe)
    return generate, seen


def request_cmd(*argv):
    args = build_parser().parse_args(["check", "request", *map(str, argv)])
    return args.handler(args)


def request_base(tmp_path, english, translations, model="org/gemma-8bit", thinking="off"):
    return [
        "--set",
        "base-v1",
        "--lang",
        "fra",
        "--english",
        english,
        "--translations",
        translations,
        "--out",
        tmp_path / "checks",
        "--model",
        model,
        "--author",
        "steve",
        "--thinking",
        thinking,
    ]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_cmd_request_writes_the_documented_meta_and_reads_the_model_text(tmp_path, monkeypatch, at_repo_root):
    english, translations, _ = write_set(tmp_path)
    # a correction is never applied: the check is on what the translator produced
    write_jsonl(
        translations / "fra.corrections.jsonl",
        [{"id": "kc-fra", "prompt": "Écris un sonnet.", "reason": "r", "author": "a"}],
    )
    generate, seen = fake_backend(monkeypatch, [verdict(False, "year"), verdict(True), verdict(True)])
    assert request_cmd(*request_base(tmp_path, english, translations), "--notes", "first run") == 0
    assert seen == {"model_id": "org/gemma-8bit", "thinking": "off", "max_tokens": 512}
    out = tmp_path / "checks" / "base-v1"
    consigne = REPO_ROOT / "consigne" / "check-same-request-v1.md"
    meta = json.loads((out / "fra.request.meta.json").read_text(encoding="utf-8"))
    assert meta == {
        "lang": "fra",
        "model": "org/gemma-8bit",
        "provider": "local",
        "revision": BACKEND_META["revision"],
        "date": datetime.date.today().isoformat(),
        "temperature": 0,
        "thinking": "off",
        "consigne": "check-same-request-v1",
        "consigne_sha256": sha256(consigne),
        "english_prompts_sha256": sha256(english),
        "translations_sha256": sha256(translations / "fra.jsonl"),
        "author": "steve",
        "replayable": True,
        "notes": "fake: no reasoning. first run",
        "quantization": "8bit-gs64",
        "mlx_lm_version": "0.32.2",
        "max_tokens": 512,
    }
    assert list(meta) == list(json.loads((out / "fra.request.meta.json").read_text(encoding="utf-8")))
    assert generate.messages[0][0]["content"] == consigne.read_bytes().decode("utf-8")
    # the pairs follow the translations file; the English is found by base key
    assert generate.messages[0][1]["content"].startswith(
        '<prompt lang="eng">\nIn 1992, say "hello".\nThen stop.\n</prompt>\n\n<prompt lang="fra">\nEn 1993, dis'
    )
    assert "Écris un poème." in generate.messages[2][1]["content"] and "sonnet" not in str(generate.messages)
    rows = read_jsonl(out / "fra.request.jsonl")
    assert [(r["id"], r["same_request"]) for r in rows] == [("kb-fra", False), ("ka-fra", True), ("kc-fra", True)]


def test_cmd_request_writes_meta_before_generating(tmp_path, monkeypatch, at_repo_root):
    from s1d1_bench import mlx_backend

    english, translations, _ = write_set(tmp_path)
    meta_path = tmp_path / "checks" / "base-v1" / "fra.request.meta.json"
    seen_meta = []

    def answer(messages):
        seen_meta.append(meta_path.is_file())
        return result(verdict(True))

    monkeypatch.setattr(mlx_backend, "make_mlx_generate", lambda m, t, max_tokens=1024: (answer, dict(BACKEND_META)))
    assert request_cmd(*request_base(tmp_path, english, translations)) == 0
    assert seen_meta == [True, True, True]


def first_run(tmp_path, monkeypatch, english, translations):
    """A run whose second pair (ka-fra) ends in json_error: a resume has something left to do."""
    fake_backend(monkeypatch, [verdict(True), "x", "y", verdict(True)])
    assert request_cmd(*request_base(tmp_path, english, translations)) == 0
    return tmp_path / "checks" / "base-v1"


def test_cmd_request_resume_with_the_same_settings_goes_on_and_keeps_the_meta(tmp_path, monkeypatch, at_repo_root):
    english, translations, _ = write_set(tmp_path)
    out = first_run(tmp_path, monkeypatch, english, translations)
    meta_before = (out / "fra.request.meta.json").read_bytes()
    generate, _ = fake_backend(monkeypatch, [verdict(False, "now")])
    assert request_cmd(*request_base(tmp_path, english, translations), "--resume") == 0 and generate.calls == 1
    assert (out / "fra.request.meta.json").read_bytes() == meta_before
    assert [r["id"] for r in read_jsonl(out / "fra.request.jsonl")] == ["kb-fra", "ka-fra", "kc-fra"]


def test_cmd_request_refuses_existing_files_without_resume(tmp_path, monkeypatch, capsys, at_repo_root):
    english, translations, _ = write_set(tmp_path)
    out = first_run(tmp_path, monkeypatch, english, translations)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    generate, seen = fake_backend(monkeypatch, [verdict(True)] * 3)
    capsys.readouterr()
    assert request_cmd(*request_base(tmp_path, english, translations)) == 1
    assert generate.calls == 0 and seen == {}
    err = capsys.readouterr().err
    assert err.startswith("bench check request:") and "immutable" in err and "--resume" in err
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


OTHER_REVISION = {**BACKEND_META, "revision": "f" * 40}


@pytest.mark.parametrize(
    ("change", "meta", "described", "key", "loaded"),
    [
        ({"--consigne": "{consigne}"}, None, None, "consigne_sha256", False),
        ({"--english": "{english}"}, None, None, "english_prompts_sha256", False),
        ({"--translations": "{translations}"}, None, None, "translations_sha256", False),
        ({"--thinking": "cap:256"}, None, None, "thinking", False),
        ({"--max-tokens": "1024"}, None, None, "max_tokens", False),
        ({"--model": "org/other"}, None, None, "model", False),
        ({}, OTHER_REVISION, None, "revision", False),
        # describe_model announced the recorded revision, the loaded weights carry another: the backstop refuses
        ({}, OTHER_REVISION, BACKEND_META, "revision", True),
    ],
)
def test_cmd_request_resume_refuses_other_settings(
    tmp_path, monkeypatch, capsys, at_repo_root, change, meta, described, key, loaded
):
    english, translations, _ = write_set(tmp_path)
    out = first_run(tmp_path, monkeypatch, english, translations)
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    # the same content in other bytes: only the sha256 differs
    consigne = tmp_path / "check-same-request-v2.md"
    consigne.write_bytes((REPO_ROOT / "consigne" / "check-same-request-v1.md").read_bytes() + b"\nOne more rule.\n")
    other_english = tmp_path / "other-eng.jsonl"
    other_english.write_text(english.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    other_translations = tmp_path / "other-translations"
    other_translations.mkdir()
    (other_translations / "fra.jsonl").write_text(
        (translations / "fra.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    (other_translations / "fra.meta.json").write_bytes((translations / "fra.meta.json").read_bytes())
    values = {"consigne": consigne, "english": other_english, "translations": other_translations}
    argv = [str(a) for a in request_base(tmp_path, english, translations)]
    for flag_name, value in change.items():
        value = value.format(**values)
        if flag_name in argv:
            argv[argv.index(flag_name) + 1] = value
        else:
            argv += [flag_name, value]
    generate, seen = fake_backend(monkeypatch, [verdict(True)] * 3, meta=meta, described=described)
    capsys.readouterr()
    assert request_cmd(*argv, "--resume") == 1
    err = capsys.readouterr().err
    assert generate.calls == 0 and bool(seen) is loaded  # what describe_model knows: checked unloaded
    assert key in err and "fra.request.meta.json" in err
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


@pytest.mark.parametrize("notes", ["see /Users/steve/mlx-models/x", "from ~/mlx-models/x", "in {home}/x"])
def test_cmd_request_refuses_a_local_path_in_notes(tmp_path, monkeypatch, capsys, at_repo_root, notes):
    english, translations, _ = write_set(tmp_path)
    generate, _ = fake_backend(monkeypatch, [verdict(True)] * 3)
    code = request_cmd(*request_base(tmp_path, english, translations), "--notes", notes.format(home=Path.home()))
    assert code == 1 and generate.calls == 0 and not (tmp_path / "checks").exists()
    assert "local path" in capsys.readouterr().err


def test_cmd_request_refuses_a_translation_without_its_english_before_loading(
    tmp_path, monkeypatch, capsys, at_repo_root
):
    english, translations, _ = write_set(tmp_path)
    write_jsonl(english, ENGLISH[:2])  # kc-eng is gone
    generate, seen = fake_backend(monkeypatch, [verdict(True)] * 3)
    assert request_cmd(*request_base(tmp_path, english, translations)) == 1
    err = capsys.readouterr().err
    assert seen == {} and not (tmp_path / "checks").exists()
    assert err.startswith("bench check request:") and "kc-eng" in err and "Traceback" not in err


def test_cli_check_request_help_lists_its_arguments(capsys):
    with pytest.raises(SystemExit):
        main(["check", "--help"])
    out = capsys.readouterr().out
    for text in ("--model", "--author", "--thinking", "--max-tokens", "512", "--resume", "--notes", "--consigne"):
        assert text in out
    args = build_parser().parse_args(
        ["check", "request", "--set", "s", "--lang", "fra", "--english", "e", "--translations", "t"]
    )
    assert (args.max_tokens, args.consigne, args.out) == (
        512,
        Path("consigne/check-same-request-v1.md"),
        Path("data/checks"),
    )
    assert (args.model, args.author, args.thinking, args.resume, args.notes) == (None, None, None, False, "")


def test_run_check_labels_names_an_abstention_instead_of_comparing_it(tmp_path):
    english, translations, _ = write_set(tmp_path)
    abstained = {"complexity": None, "task": None, "needs_tools": None, "abstained": "json_error"}
    eng = write_labels(
        tmp_path / "judges" / "eng",
        [
            {"id": "ka-eng", **abstained},
            {"id": "kb-eng", "complexity": "low", "task": "editing", "needs_tools": False},
            {"id": "kc-eng", "complexity": "low", "task": "creative_writing", "needs_tools": False},
        ],
    )
    fra = write_labels(
        tmp_path / "judges" / "fra",
        [
            {"id": "ka-fra", "complexity": "low", "task": "coding_debugging", "needs_tools": True},
            {"id": "kb-fra", "complexity": "high", "task": "translation", "needs_tools": False},
            {"id": "kc-fra", **abstained, "abstained": "truncated"},
        ],
    )
    wit = write_labels(tmp_path / "judges" / "wit", [{"id": "kb-fra", **abstained}])  # a witness that abstained
    args = args_for(tmp_path, "labels", english, translations, labels_english=eng, labels=fra, labels_witness=wit)
    assert run_check(args) == 0
    rows = read_jsonl(tmp_path / "checks" / "base-v1" / "fra.labels.jsonl")
    assert [(r["id"], r["rule"], r["lifted"]) for r in rows] == [
        ("ka-fra", "label_missing", False),
        ("kb-fra", "task", False),  # no witness to lift it
        ("kc-fra", "label_missing", False),
    ]
    assert rows[0]["detail"] == f"abstention (json_error) for ka-eng in {eng}"
    assert rows[1]["detail"] == "eng=editing tr=translation"
    assert rows[2]["detail"] == f"abstention (truncated) for kc-fra in {fra}"
