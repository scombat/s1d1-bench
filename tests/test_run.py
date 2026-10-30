import datetime
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s1d1_bench.cli import main
from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.run import build_results, load_folder, lower_median, majority, truths

LANGS = ["eng", "fra", "rus", "zho"]
KEYS = ["k1", "k2"]
# prompts file order, deliberately neither sorted nor grouped by key
IDS = ["k2-fra", "k1-eng", "k1-zho", "k2-eng", "k1-fra", "k2-zho", "k1-rus", "k2-rus"]
MAIN = ["j-b", "j-a", "j-c"]  # not in name order: the --main order is the priority order
GEMMA_MODEL = "mlx-community/gemma-4-31b-it-8bit"


def L(level, task="information_seeking", needs_tools=False, **extra):
    return {"complexity": level, "task": task, "needs_tools": needs_tools, **extra}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta_rows() -> list[dict]:
    rows = []
    for i in IDS:
        key, lang = i.split("-")
        rows.append(
            {"id": i, "base_key": key, "lang": lang, "origin": "pool", "source": "oasst2", "corrected": lang == "fra"}
        )
    return rows


def write_set(root: Path) -> Path:
    data = root / "data"
    data.mkdir(parents=True)
    prompts = data / "set.jsonl"
    write_jsonl(prompts, [{"id": i, "prompt": f"prompt {i}"} for i in IDS])
    rows = list(reversed(meta_rows()))  # the meta file need not follow the prompts order
    for row in rows:
        row["translator"] = None if row["lang"] == "eng" else "m@rev"
    pq.write_table(pa.Table.from_pylist(rows), data / "set.meta.parquet")
    return prompts


def write_folder(folder: Path, labels: dict[str, dict], prompts_sha: str, model="org/model", system=False) -> Path:
    folder.mkdir(parents=True)
    meta = {
        "judge": folder.name,
        "model": model,
        "provider": "system" if system else "local",
        "revision": "rev-" + folder.name,
        "date": "2026-10-29",
        "temperature": 0,
        "thinking": "none" if system else "off",
        "instructions": "none" if system else "v7-single",
        "prompts_sha256": prompts_sha,
        "author": "tester",
        "replayable": True,
        "licence": "CC-BY-4.0",
        "notes": "",
    }
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    write_jsonl(folder / "labels.jsonl", [{"id": i, **lab} for i, lab in labels.items()])
    return folder


def every(level, **kw) -> dict[str, dict]:
    return {i: L(level, **kw) for i in IDS}


def by_lang(levels: dict[str, str | None], **kw) -> dict[str, dict]:
    """Same level for both keys, one per language."""
    return {i: L(levels[i.split("-")[1]], **kw) for i in IDS}


# --- lower_median, majority -------------------------------------------------------------------------------------------


def test_lower_median_six_six_takes_the_lower():
    assert lower_median(["medium", "low"] * 6) == "low"
    assert lower_median(["high", "medium", "high"]) == "high"
    assert lower_median(["expert", "trivial"]) == "trivial"  # two votes: the lower one
    assert lower_median(["low", "medium", "medium", "high"]) == "medium"
    # through the truths: j-a low everywhere, j-b medium everywhere, j-c low in eng/fra and medium in rus/zho
    labels = {
        "j-a": every("low"),
        "j-b": every("medium"),
        "j-c": by_lang({"eng": "low", "fra": "low", "rus": "medium", "zho": "medium"}),
    }
    result = truths(labels, MAIN, meta_rows())
    for i in IDS:
        assert result[i]["truth_all"]["complexity"] == "low"  # 6 low / 6 medium over the key's 12 votes
        assert result[i]["spread"] == 1  # each id has a low and a medium
        assert result[i]["undecidable"] is False
    assert result["k1-eng"]["truth_lang"]["complexity"] == "low"
    assert result["k1-rus"]["truth_lang"]["complexity"] == "medium"


def test_undecidable_when_nulls_are_majority():
    assert lower_median([None] * 7 + ["high"] * 5) is None
    assert lower_median([None] * 6 + ["high"] * 6) == "high"  # half is not a majority
    assert lower_median([None, None, "low"]) is None
    assert lower_median([None, "low", "expert"]) == "low"
    assert lower_median([]) is None
    # k1: 7 of its 12 votes are null (j-a: 4, j-b: 3); k2: 3 of 12
    labels = {
        "j-a": by_lang({"eng": None, "fra": None, "rus": None, "zho": None}),
        "j-b": by_lang({"eng": None, "fra": None, "rus": None, "zho": "high"}),
        "j-c": every("high"),
    }
    labels["j-a"].update({i: L("high") for i in IDS if i.startswith("k2")})
    result = truths(labels, MAIN, meta_rows())
    for i in IDS:
        undecidable = i.startswith("k1")
        assert result[i]["undecidable"] is undecidable
        assert result[i]["truth_all"]["complexity"] == (None if undecidable else "high")
    assert result["k1-eng"]["spread"] == 0  # one non-null vote
    assert result["k1-eng"]["truth_lang"]["complexity"] is None  # 2 nulls of 3
    assert result["k1-zho"]["truth_lang"]["complexity"] == "high"  # 1 null of 3
    assert result["k1-zho"]["lang_disagreement"] is False  # truth_all is null: no disagreement
    # the row stays in the results, with a real null
    table = build_results(meta_rows(), {n: labels[n] for n in MAIN}, {}, result)
    assert table.num_rows == len(IDS)
    row = table.to_pylist()[IDS.index("k1-eng")]
    assert row["undecidable"] is True and row["truth_all.complexity"] is None


def test_all_null_votes_give_no_spread():
    labels = {name: every(None) for name in MAIN}
    result = truths(labels, MAIN, meta_rows())
    assert result["k1-eng"]["spread"] is None
    assert result["k1-eng"]["undecidable"] is True


def test_majority_tie_goes_to_the_first_main_judge():
    assert majority(["x", "y", "y"], ["j1", "j2", "j3"]) == "y"
    assert majority(["x", "y"], ["j1", "j2"]) == "x"
    assert majority(["y", "x", "x", "y"], ["j1", "j2", "j3", "j4"]) == "y"  # 2/2: y voted first
    assert majority([None, "x", None], ["j1", "j2", "j3"]) == "x"  # a null is no vote
    assert majority([None, None], ["j1", "j2"]) is None
    with pytest.raises(ValueError):
        majority(["x"], ["j1", "j2"])
    # through the truths: three different tasks, 4 votes each on a key: the first judge of --main wins (j-b)
    labels = {
        "j-a": every("low", task="coding_debugging", needs_tools=True),
        "j-b": every("low", task="creative_writing", needs_tools=False),
        "j-c": by_lang({lang: "low" for lang in LANGS}, task="math"),
    }
    for i in IDS:  # needs_tools 6 true / 6 false on each key, 2 to 1 on each id
        labels["j-c"][i]["needs_tools"] = i.split("-")[1] in ("eng", "fra")
    result = truths(labels, MAIN, meta_rows())
    for i in IDS:
        assert result[i]["truth_all"]["task"] == "creative_writing"
        assert result[i]["truth_lang"]["task"] == "creative_writing"
        assert result[i]["truth_all"]["needs_tools"] is False  # 6/6: j-b votes first
        assert result[i]["truth_lang"]["needs_tools"] is (i.split("-")[1] in ("eng", "fra"))
    reordered = truths(labels, ["j-a", "j-b", "j-c"], meta_rows())
    assert reordered["k1-eng"]["truth_all"]["task"] == "coding_debugging"
    assert reordered["k1-rus"]["truth_all"]["needs_tools"] is True


def test_truth_all_uses_the_four_versions():
    # k1: the three judges say expert in eng only, low elsewhere; k2: high everywhere
    labels = {name: by_lang({"eng": "expert", "fra": "low", "rus": "low", "zho": "low"}) for name in MAIN}
    for name in MAIN:
        labels[name].update({i: L("high") for i in IDS if i.startswith("k2")})
    labels["j-off"] = every("trivial")  # not a main judge: no vote
    result = truths(labels, MAIN, meta_rows())
    for i in IDS:
        key, lang = i.split("-")
        assert result[i]["truth_all"]["complexity"] == ("low" if key == "k1" else "high")
        assert result[i]["truth_all_nogemma"]["complexity"] == result[i]["truth_all"]["complexity"]  # no Gemma here
    assert result["k1-eng"]["truth_lang"]["complexity"] == "expert"
    assert result["k1-fra"]["truth_lang"]["complexity"] == "low"
    # a key's truth is the same on its four rows, whatever the order of the rows in the file
    assert len({str(result[i]["truth_all"]) for i in IDS if i.startswith("k1")}) == 1


def test_lang_disagreement_flag():
    labels = {name: by_lang({"eng": "expert", "fra": "low", "rus": "low", "zho": "low"}) for name in MAIN}
    labels["j-b"].update({i: L(None) for i in IDS if i.startswith("k2")})
    labels["j-a"].update({i: L(None) for i in IDS if i.startswith("k2")})  # k2: 8 of 12 null, undecidable
    result = truths(labels, MAIN, meta_rows())
    assert result["k1-eng"]["lang_disagreement"] is True
    for i in ("k1-fra", "k1-rus", "k1-zho"):
        assert result[i]["lang_disagreement"] is False
    for i in IDS:
        if i.startswith("k2"):
            assert result[i]["truth_all"]["complexity"] is None
            assert result[i]["lang_disagreement"] is False


def test_gemma_exclusion_changes_the_nogemma_truths():
    labels = {
        "j-b": every("expert", task="math"),  # the Gemma judge, first in --main
        "j-a": every("low", task="coding_debugging"),
        "j-c": by_lang({"eng": "medium", "fra": "medium", "rus": "medium", "zho": "medium"}, task="math"),
    }
    with_gemma = truths(labels, MAIN, meta_rows())
    result = truths(labels, MAIN, meta_rows(), gemma={"j-b"})
    for i in IDS:
        assert result[i]["truth_all"] == with_gemma[i]["truth_all"]
        assert result[i]["truth_all"]["complexity"] == "medium"  # low×4 medium×4 expert×4
        assert result[i]["truth_all_nogemma"]["complexity"] == "low"  # low×4 medium×4
        assert result[i]["truth_lang"]["complexity"] == "medium"
        assert result[i]["truth_lang_nogemma"]["complexity"] == "low"  # two votes: the lower
        assert result[i]["truth_all"]["task"] == "math"
        assert result[i]["truth_all_nogemma"]["task"] == "coding_debugging"  # 4/4: j-a is now the first
        assert with_gemma[i]["truth_all_nogemma"] == with_gemma[i]["truth_all"]
        assert result[i]["spread"] == 3  # the spread is over every main judge, Gemma included


# --- load_folder ------------------------------------------------------------------------------------------------------


def test_refuses_wrong_sha_and_missing_id(tmp_path):
    prompts = write_set(tmp_path)
    good = sha(prompts)
    other = hashlib.sha256(b"another set").hexdigest()
    name, meta, labels = load_folder(write_folder(tmp_path / "judges" / "ok", every("low"), good), good, IDS)
    assert name == "ok" and meta["revision"] == "rev-ok" and labels["k1-eng"] == L("low")

    wrong = write_folder(tmp_path / "judges" / "check-qwen3-off-fra", every("low"), other)
    with pytest.raises(ValueError) as exc:
        load_folder(wrong, good, IDS)
    assert "check-qwen3-off-fra" in str(exc.value) and good in str(exc.value) and other in str(exc.value)

    partial = every("low")
    del partial["k1-zho"], partial["k2-rus"]
    incomplete = write_folder(tmp_path / "judges" / "incomplete", partial, good)
    with pytest.raises(ValueError) as exc:
        load_folder(incomplete, good, IDS)
    assert "incomplete" in str(exc.value) and "k1-zho" in str(exc.value)  # the first missing in prompts order

    extra = write_folder(tmp_path / "judges" / "extra", {**every("low"), "k9-eng": L("low")}, good)
    with pytest.raises(ValueError, match="k9-eng"):
        load_folder(extra, good, IDS)

    (tmp_path / "judges" / "empty").mkdir()
    with pytest.raises(ValueError, match="empty"):
        load_folder(tmp_path / "judges" / "empty", good, IDS)


def test_load_folder_reads_a_classifier_with_its_optional_fields(tmp_path):
    prompts = write_set(tmp_path)
    rows = {i: L("low", task=None, needs_tools=None, score=0.3) for i in IDS}
    folder = write_folder(tmp_path / "classifiers" / "laya", rows, sha(prompts), system=True)
    _, _, labels = load_folder(folder, sha(prompts), IDS)
    assert labels["k1-eng"] == {
        "complexity": "low",
        "task": None,
        "needs_tools": None,
        "score": 0.3,
        "needs_tools_probability": None,
    }


# --- the command ------------------------------------------------------------------------------------------------------


def full_tree(tmp_path: Path) -> Path:
    """3 main judges (j-b is Gemma), one extra judge, three classifiers; return the prompts file."""
    prompts = write_set(tmp_path)
    good = sha(prompts)
    judges = tmp_path / "judges"
    write_folder(judges / "j-b", every("medium"), good, model=GEMMA_MODEL)
    write_folder(judges / "j-a", every("low"), good)
    write_folder(judges / "j-c", every("high", needs_tools=True), good)
    write_folder(judges / "j-off", every("trivial"), good)
    (judges / "j-off" / "log.jsonl").write_text('{"ignored": true}\n', encoding="utf-8")
    (judges / "MODELS.md").write_text("not a folder\n", encoding="utf-8")
    classifiers = tmp_path / "classifiers"
    scored = {i: L("low", needs_tools=False, score=0.2, needs_tools_probability=0.1) for i in IDS}
    write_folder(classifiers / "s1d1-small-int8", scored, good, system=True)
    write_folder(classifiers / "random", every("expert", task="math"), good, system=True)
    (classifiers / "random" / "draws.jsonl").write_text("not even json\n", encoding="utf-8")
    write_folder(classifiers / "laya", every("medium", task=None, needs_tools=None, score=0.5), good, system=True)
    return prompts


def run_cli(tmp_path, prompts, *extra, main_list="j-b,j-a,j-c"):
    argv = ["run", "--prompts", str(prompts), "--judges", str(tmp_path / "judges" / "*"), "--main", main_list]
    argv += ["--classifiers", str(tmp_path / "classifiers" / "*"), "--out", str(tmp_path / "results" / "set.parquet")]
    return main(argv + list(extra))


TRUTH_COLUMNS = [
    f"{t}.{f}"
    for t in ("truth_all", "truth_lang", "truth_all_nogemma", "truth_lang_nogemma")
    for f in ("complexity", "task", "needs_tools")
]


def test_columns_and_row_count(tmp_path):
    prompts = full_tree(tmp_path)
    assert run_cli(tmp_path, prompts) == 0
    table = pq.read_table(tmp_path / "results" / "set.parquet")
    folder_columns = []
    for name in ["j-b", "j-a", "j-c", "j-off", "laya", "random", "s1d1-small-int8"]:
        folder_columns += [f"{name}.complexity", f"{name}.task", f"{name}.needs_tools"]
        if name == "laya":
            folder_columns += ["laya.score"]
        if name == "s1d1-small-int8":
            folder_columns += ["s1d1-small-int8.score", "s1d1-small-int8.needs_tools_probability"]
    assert table.column_names == [
        "id", "base_key", "lang", "origin", "source", "corrected",
        *TRUTH_COLUMNS, "undecidable", "spread", "lang_disagreement",
        *folder_columns,
    ]  # fmt: skip
    assert table.num_rows == len(IDS) == 2 * 4
    assert table.column("id").to_pylist() == IDS
    assert table.column("base_key").to_pylist() == [i.split("-")[0] for i in IDS]
    assert table.column("corrected").to_pylist() == [i.endswith("fra") for i in IDS]
    schema = table.schema
    assert schema.field("undecidable").type == pa.bool_() and schema.field("lang_disagreement").type == pa.bool_()
    assert schema.field("truth_all.needs_tools").type == pa.bool_()
    assert pa.types.is_integer(schema.field("spread").type)
    assert schema.field("laya.score").type == pa.float64()
    rows = table.to_pylist()
    assert {r["truth_all.complexity"] for r in rows} == {"medium"}  # low, medium, high ×4
    assert {r["truth_all_nogemma.complexity"] for r in rows} == {"low"}  # j-b is Gemma
    assert {r["spread"] for r in rows} == {2}
    assert {r["truth_all.needs_tools"] for r in rows} == {False}
    assert rows[0]["laya.task"] is None and rows[0]["laya.needs_tools"] is None and rows[0]["laya.score"] == 0.5
    assert rows[0]["s1d1-small-int8.needs_tools_probability"] == 0.1
    assert rows[0]["random.task"] == "math" and rows[0]["j-off.complexity"] == "trivial"


def test_run_json_records_the_inputs(tmp_path):
    prompts = full_tree(tmp_path)
    assert run_cli(tmp_path, prompts) == 0
    record = json.loads((tmp_path / "results" / "set.run.json").read_text(encoding="utf-8"))
    assert record["prompts"] == str(prompts)
    assert record["prompts_sha256"] == sha(prompts)
    assert record["main"] == ["j-b", "j-a", "j-c"]
    assert record["gemma"] == ["j-b"]
    assert record["date"] == datetime.date.today().isoformat()
    assert record["rows"] == len(IDS)
    assert [j["name"] for j in record["judges"]] == ["j-b", "j-a", "j-c", "j-off"]
    assert record["judges"][0] == {"name": "j-b", "model": GEMMA_MODEL, "revision": "rev-j-b", "date": "2026-10-29"}
    assert [c["name"] for c in record["classifiers"]] == ["laya", "random", "s1d1-small-int8"]
    assert "lower" in record["rules"] and "null" in record["rules"]
    text = (tmp_path / "results" / "set.run.json").read_text(encoding="utf-8")
    assert text == json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def test_run_rewrites_both_files(tmp_path):
    prompts = full_tree(tmp_path)
    out = tmp_path / "results" / "set.parquet"
    assert run_cli(tmp_path, prompts) == 0
    first = (out.read_bytes(), out.with_name("set.run.json").read_bytes())
    assert run_cli(tmp_path, prompts, main_list="j-a,j-b,j-c") == 0  # a second call replaces both
    second = json.loads(out.with_name("set.run.json").read_text(encoding="utf-8"))
    assert second["main"] == ["j-a", "j-b", "j-c"]
    assert pq.read_table(out).column_names[21] == "j-a.complexity"
    assert out.read_bytes() != first[0]


def test_run_refuses_a_folder_of_another_set_and_writes_nothing(tmp_path, capsys):
    prompts = full_tree(tmp_path)
    other = hashlib.sha256(b"base-v1-fra.jsonl").hexdigest()
    write_folder(tmp_path / "judges" / "check-qwen3-off-fra", every("low"), other)
    assert run_cli(tmp_path, prompts) == 1
    err = capsys.readouterr().err
    assert "check-qwen3-off-fra" in err and sha(prompts) in err and other in err
    assert not (tmp_path / "results").exists()


def test_run_refuses_a_main_judge_missing_an_id(tmp_path, capsys):
    prompts = full_tree(tmp_path)
    labels = tmp_path / "judges" / "j-a" / "labels.jsonl"
    lines = labels.read_text(encoding="utf-8").splitlines(keepends=True)
    labels.write_text("".join(line for line in lines if '"k2-zho"' not in line), encoding="utf-8")
    out = tmp_path / "results" / "set.parquet"
    out.parent.mkdir()
    out.write_bytes(b"old")
    assert run_cli(tmp_path, prompts) == 1
    err = capsys.readouterr().err
    assert "j-a" in err and "k2-zho" in err
    assert out.read_bytes() == b"old" and not out.with_name("set.run.json").exists()  # nothing written


def test_run_refuses_a_main_judge_not_loaded(tmp_path, capsys):
    prompts = full_tree(tmp_path)
    assert run_cli(tmp_path, prompts, main_list="j-b,j-a,j-z") == 1
    assert "j-z" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_run_refuses_a_glob_without_folders(tmp_path, capsys):
    prompts = write_set(tmp_path)
    argv = ["run", "--prompts", str(prompts), "--judges", str(tmp_path / "judges" / "*"), "--main", "a,b,c"]
    assert main(argv + ["--out", str(tmp_path / "results" / "set.parquet")]) == 1
    assert "judges" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


def test_a_folder_matched_twice_is_loaded_once(tmp_path):
    prompts = full_tree(tmp_path)
    argv = ["run", "--prompts", str(prompts), "--main", "j-b,j-a,j-c", "--out", str(tmp_path / "r" / "x.parquet")]
    argv += ["--judges", str(tmp_path / "judges" / "*"), "--judges", str(tmp_path / "judges" / "j-a")]
    assert main(argv) == 0
    table = pq.read_table(tmp_path / "r" / "x.parquet")
    assert table.column_names.count("j-a.complexity") == 1
    assert not any(c.startswith("laya.") for c in table.column_names)  # no --classifiers: no classifier column
