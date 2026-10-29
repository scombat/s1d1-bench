import json
from types import SimpleNamespace

import pytest

from s1d1_bench.check import label_faults, review_sheet, run_check, structural_faults
from s1d1_bench.cli import main
from s1d1_bench.jsonl import read_jsonl, write_jsonl

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


def test_cli_check_takes_the_four_kinds_and_request_is_not_there_yet(tmp_path, capsys):
    english, translations, _ = write_set(tmp_path)
    common = ["--set", "s", "--lang", "fra", "--english", str(english), "--translations", str(translations)]
    assert main(["check", "request", *common, "--out", str(tmp_path / "checks")]) != 0
    assert "not implemented yet" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["check", "spelling", *common])
    assert main(["check", "structure", *common, "--out", str(tmp_path / "checks")]) == 0
    assert (tmp_path / "checks" / "s" / "fra.structure.jsonl").exists()
