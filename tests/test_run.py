import datetime
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s1d1_bench.cli import main
from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.run import build_results, even_floor_median, load_folder, majority, plain_lower_median, truths

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


def A(status="json_error") -> dict:
    """An abstention row as read_labels gives it."""
    return {"complexity": None, "task": None, "needs_tools": None, "abstained": status}


def every(level, **kw) -> dict[str, dict]:
    return {i: L(level, **kw) for i in IDS}


def by_lang(levels: dict[str, str | None], **kw) -> dict[str, dict]:
    """Same level for both keys, one per language."""
    return {i: L(levels[i.split("-")[1]], **kw) for i in IDS}


# --- even_floor_median, majority --------------------------------------------------------------------------------------


def test_even_floor_median_six_six_takes_the_lower():
    assert even_floor_median(["medium", "low"] * 6) == "low"
    assert even_floor_median(["high", "medium", "high"]) == "high"
    assert even_floor_median(["expert", "trivial"]) == "medium"  # two votes: their mean rounded down (docs/34)
    assert even_floor_median(["low", "medium", "medium", "high"]) == "medium"
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
    assert even_floor_median([None] * 7 + ["high"] * 5) is None
    assert even_floor_median([None] * 6 + ["high"] * 6) == "high"  # half is not a majority
    assert even_floor_median([None, None, "low"]) is None
    assert even_floor_median([None, "low", "expert"]) == "medium"  # two votes: (1 + 4) // 2
    assert even_floor_median([]) is None
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


def test_even_count_takes_the_mean_of_the_two_middle_votes_rounded_down():
    assert even_floor_median(["low", "high"]) == "medium"  # docs/34: two judges, mean rounded down
    assert even_floor_median(["medium", "low"]) == "low"  # adjacent: the lower one
    assert even_floor_median(["low"] * 6 + ["high"] * 6) == "medium"
    assert even_floor_median(["low"] * 6 + ["medium"] * 6) == "low"
    assert even_floor_median(["trivial", "high"]) == "low"  # (0 + 3) // 2
    assert even_floor_median(["low", "expert"]) == "medium"  # (1 + 4) // 2
    assert even_floor_median(["trivial", "low", "high", "expert"]) == "medium"  # the two middle: low, high
    assert even_floor_median([None, "low", "high"]) == "medium"  # two non-null votes, one null of three
    # odd counts: the middle vote, unchanged
    assert even_floor_median(["low", "high", "high"]) == "high"
    assert even_floor_median(["low", "low", "expert"]) == "low"
    assert even_floor_median(["trivial", "medium", "expert"]) == "medium"
    assert even_floor_median(["low", "high", "expert", None, None]) == "high"
    # the plain lower median, kept for the diagnostic of `bench run`
    assert plain_lower_median(["low", "high"]) == "low"
    assert plain_lower_median(["low"] * 6 + ["high"] * 6) == "low"
    assert plain_lower_median([None] * 7 + ["high"] * 5) is None
    # through the truths: j-a low, j-b high, j-c low in eng/fra and high in rus/zho: 6 low / 6 high on each key
    labels = {
        "j-a": every("low"),
        "j-b": every("high"),
        "j-c": by_lang({"eng": "low", "fra": "low", "rus": "high", "zho": "high"}),
    }
    result = truths(labels, MAIN, meta_rows())
    plain = truths(labels, MAIN, meta_rows(), median=plain_lower_median)
    for i in IDS:
        assert result[i]["truth_all"]["complexity"] == "medium"
        assert plain[i]["truth_all"]["complexity"] == "low"
        assert result[i]["spread"] == 2
    assert result["k1-eng"]["truth_lang"]["complexity"] == "low"  # low, high, low: odd, the middle
    assert result["k1-eng"]["truth_lang_nogemma"]["complexity"] == "low"  # no Gemma given: same as truth_lang
    nogemma = truths(labels, MAIN, meta_rows(), gemma={"j-c"})
    assert nogemma["k1-eng"]["truth_lang_nogemma"]["complexity"] == "medium"  # low, high: two votes


def test_run_prints_how_many_keys_the_plain_lower_median_would_change(tmp_path, capsys):
    prompts = write_set(tmp_path)
    good = sha(prompts)
    write_folder(tmp_path / "judges" / "j-b", every("high"), good, model=GEMMA_MODEL)
    write_folder(tmp_path / "judges" / "j-a", every("low"), good)
    write_folder(tmp_path / "judges" / "j-c", by_lang({"eng": "low", "fra": "low", "rus": "high", "zho": "high"}), good)
    argv = ["run", "--prompts", str(prompts), "--judges", str(tmp_path / "judges" / "*"), "--main", "j-b,j-a,j-c"]
    assert main(argv + ["--out", str(tmp_path / "results" / "set.parquet")]) == 0
    out = capsys.readouterr().out.splitlines()
    # 6 low / 6 high with Gemma (medium, plain low); without j-b: low x6, high x2 (low either way)
    assert out[-1] == (
        "plain lower median instead (diagnostic): truth_all.complexity would differ on 2 of 2 base keys, "
        "truth_all_nogemma.complexity on 0 of 2"
    )
    table = pq.read_table(tmp_path / "results" / "set.parquet")
    assert set(table.column("truth_all.complexity").to_pylist()) == {"medium"}


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
        if name.startswith("j-"):
            folder_columns += [f"{name}.abstained"]  # every judge folder, never a classifier
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
    assert {r["truth_all_nogemma.complexity"] for r in rows} == {"medium"}  # j-b is Gemma: low x4, high x4
    assert {r["truth_lang_nogemma.complexity"] for r in rows} == {"medium"}  # low, high
    assert {r["spread"] for r in rows} == {2}
    assert {r["truth_all.needs_tools"] for r in rows} == {False}
    assert rows[0]["laya.task"] is None and rows[0]["laya.needs_tools"] is None and rows[0]["laya.score"] == 0.5
    assert rows[0]["s1d1-small-int8.needs_tools_probability"] == 0.1
    assert rows[0]["random.task"] == "math" and rows[0]["j-off.complexity"] == "trivial"
    assert schema.field("j-a.abstained").type == pa.bool_()
    assert {r[f"{name}.abstained"] for r in rows for name in ("j-a", "j-b", "j-c", "j-off")} == {False}


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
    assert "rounded down" in record["rules"] and "null" in record["rules"] and "abstention" in record["rules"]
    assert "docs/34 §0.3" in record["rules"] and "decision 3" not in record["rules"]
    assert record["abstentions"] == {"j-b": [], "j-a": [], "j-c": [], "j-off": []}
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


def test_judges_and_classifiers_take_several_values_per_option(tmp_path):
    """`--judges a b c --classifiers x y`, the option repeated, and the two mixed all load the same folders."""
    prompts = full_tree(tmp_path)
    judges, classifiers = tmp_path / "judges", tmp_path / "classifiers"
    base = ["run", "--prompts", str(prompts), "--main", "j-b,j-a,j-c"]
    several = ["--judges", *(str(judges / n) for n in ("j-a", "j-b", "j-c")), "--classifiers"]
    several += [str(classifiers / "laya"), str(classifiers / "random"), str(classifiers / "s1d1-*")]
    mixed = ["--judges", str(judges / "j-a"), str(judges / "j-b"), "--judges", str(judges / "j-c"), "--classifiers"]
    mixed += [str(classifiers / "laya"), "--classifiers", str(classifiers / "random"), str(classifiers / "s1d1-*")]
    for name, tail in (("several", several), ("mixed", mixed)):
        assert main([*base, *tail, "--out", str(tmp_path / name / "set.parquet")]) == 0, name
    record = json.loads((tmp_path / "several" / "set.run.json").read_text(encoding="utf-8"))
    assert [j["name"] for j in record["judges"]] == ["j-b", "j-a", "j-c"]  # the order of --main
    assert [c["name"] for c in record["classifiers"]] == ["laya", "random", "s1d1-small-int8"]
    one = pq.read_table(tmp_path / "several" / "set.parquet")
    two = pq.read_table(tmp_path / "mixed" / "set.parquet")
    assert one.column_names == two.column_names and one.equals(two)


# --- abstentions ------------------------------------------------------------------------------------------------------


def test_an_abstention_is_a_null_vote_toward_undecidable():
    # k1: j-a abstains on its 4 versions, j-b gives a null tier on 3: 7 of 12 null votes -> undecidable
    labels = {
        "j-a": {i: A() if i.startswith("k1") else L("high") for i in IDS},
        "j-b": by_lang({"eng": None, "fra": None, "rus": None, "zho": "high"}),
        "j-c": every("high"),
    }
    labels["j-b"].update({i: L("high") for i in IDS if i.startswith("k2")})
    result = truths(labels, MAIN, meta_rows())
    null_row = L(None, task=None, needs_tools=None)
    nulls = {**labels, "j-a": {i: null_row if i.startswith("k1") else L("high") for i in IDS}}
    assert result == truths(nulls, MAIN, meta_rows())  # exactly as a null vote on every field
    for i in IDS:
        assert result[i]["undecidable"] is i.startswith("k1")
    # 6 of 12: j-a abstains on 4, j-b null on 2: decidable
    labels["j-b"] = by_lang({"eng": None, "fra": None, "rus": "high", "zho": "high"})
    assert truths(labels, MAIN, meta_rows())["k1-eng"]["truth_all"]["complexity"] == "high"


def test_an_abstaining_judge_gives_no_task_tools_or_spread_vote():
    labels = {
        "j-b": every("expert", task="math", needs_tools=True),  # first in --main: wins every tie it is in
        "j-a": every("low", task="coding_debugging"),
        "j-c": every("medium", task="creative_writing"),
    }
    labels["j-b"]["k1-eng"] = A("truncated")
    result = truths(labels, MAIN, meta_rows())
    # k1-eng: j-a and j-c are left, one vote each: the tie goes to j-a, the first of them in --main
    assert result["k1-eng"]["truth_lang"] == {"complexity": "low", "task": "coding_debugging", "needs_tools": False}
    assert result["k1-eng"]["spread"] == 1  # low and medium; j-b's expert elsewhere does not count here
    assert result["k1-fra"]["spread"] == 3
    # truth_all of k1: j-b votes on 3 versions, 3 math / 4 coding_debugging / 4 creative_writing: j-a first
    assert result["k1-eng"]["truth_all"]["task"] == "coding_debugging"
    assert result["k2-eng"]["truth_all"]["task"] == "math"  # 4/4/4: j-b first
    assert result["k1-eng"]["truth_all"]["needs_tools"] is False  # 3 true, 8 false


def test_run_writes_the_abstained_column_and_lists_the_abstentions(tmp_path, capsys):
    prompts = full_tree(tmp_path)
    good = sha(prompts)
    labels = {**every("low"), "k1-zho": A("json_error"), "k2-eng": A("truncated")}
    path = tmp_path / "judges" / "j-a" / "labels.jsonl"
    write_jsonl(path, [{"id": i, **lab} for i, lab in labels.items()])
    labels_off = {**every("trivial"), "k2-rus": A("truncated")}
    write_jsonl(tmp_path / "judges" / "j-off" / "labels.jsonl", [{"id": i, **lab} for i, lab in labels_off.items()])
    assert good == sha(prompts)
    assert run_cli(tmp_path, prompts) == 0
    out = capsys.readouterr().out
    assert "abstentions 3" in out
    rows = pq.read_table(tmp_path / "results" / "set.parquet").to_pylist()
    by_id = {r["id"]: r for r in rows}
    assert by_id["k1-zho"]["j-a.abstained"] is True and by_id["k1-zho"]["j-a.complexity"] is None
    assert by_id["k2-eng"]["j-a.abstained"] is True and by_id["k1-eng"]["j-a.abstained"] is False
    assert by_id["k2-rus"]["j-off.abstained"] is True
    assert sum(r["j-b.abstained"] for r in rows) == 0
    record = json.loads((tmp_path / "results" / "set.run.json").read_text(encoding="utf-8"))
    assert record["abstentions"] == {
        "j-b": [],
        "j-a": [{"id": "k1-zho", "status": "json_error"}, {"id": "k2-eng", "status": "truncated"}],  # prompts order
        "j-c": [],
        "j-off": [{"id": "k2-rus", "status": "truncated"}],
    }


def test_an_id_without_any_row_is_still_refused_next_to_abstentions(tmp_path, capsys):
    prompts = full_tree(tmp_path)
    labels = {**every("low"), "k1-zho": A()}
    del labels["k2-zho"]
    write_jsonl(tmp_path / "judges" / "j-a" / "labels.jsonl", [{"id": i, **lab} for i, lab in labels.items()])
    assert run_cli(tmp_path, prompts) == 1
    err = capsys.readouterr().err
    assert "j-a" in err and "k2-zho" in err and "never fills in" in err
    assert not (tmp_path / "results").exists()


def test_load_folder_keeps_an_abstention_and_refuses_one_in_a_classifier(tmp_path):
    prompts = write_set(tmp_path)
    judge = write_folder(tmp_path / "judges" / "j", {**every("low"), "k1-eng": A()}, sha(prompts))
    _, _, labels = load_folder(judge, sha(prompts), IDS)
    assert labels["k1-eng"] == A() and labels["k1-fra"] == L("low")
    rows = {**every("low", score=0.3), "k1-eng": A()}
    classifier = write_folder(tmp_path / "classifiers" / "c", rows, sha(prompts), system=True)
    with pytest.raises(ValueError, match="abstain"):
        load_folder(classifier, sha(prompts), IDS)


# --- raw decisions (decisions.jsonl of a classifier) ------------------------------------------------------------------


def write_decisions(folder: Path, rows: list[dict]) -> None:
    write_jsonl(folder / "decisions.jsonl", rows)


def raw_row(i: str, tier: str = "MEDIUM", score: float = 0.12) -> dict:
    return {"id": i, "tier": tier, "score": score, "signals": ["code"], "cause": "score"}


def test_run_adds_raw_score_and_raw_tier_from_decisions(tmp_path, capsys):
    prompts = full_tree(tmp_path)
    classifiers = tmp_path / "classifiers"
    # prompts file order is not sorted: the columns must follow the ids, not the order of the decisions file
    raws = [raw_row(i, "SIMPLE" if n % 2 else "REASONING", -0.25 if n == 0 else 0.5 + n) for n, i in enumerate(IDS)]
    write_decisions(classifiers / "laya", list(reversed(raws)))
    assert run_cli(tmp_path, prompts) == 0
    table = pq.read_table(tmp_path / "results" / "set.parquet")
    names = table.column_names
    assert [c for c in names if "raw_" in c] == ["laya.raw_score", "laya.raw_tier"]  # the others have none
    assert names[names.index("laya.score") + 1 : names.index("laya.score") + 3] == ["laya.raw_score", "laya.raw_tier"]
    assert table.schema.field("laya.raw_score").type == pa.float64()
    assert table.schema.field("laya.raw_tier").type == pa.string()
    assert table.column("id").to_pylist() == IDS
    assert table.column("laya.raw_score").to_pylist() == [r["score"] for r in raws]
    assert table.column("laya.raw_score").to_pylist()[0] == -0.25  # negative: kept as is, no clamp, no rounding
    assert table.column("laya.raw_tier").to_pylist() == [r["tier"] for r in raws]
    assert table.column("laya.score").to_pylist() == [0.5] * len(IDS)  # the label score is untouched
    record = json.loads((tmp_path / "results" / "set.run.json").read_text(encoding="utf-8"))
    entries = {c["name"]: c for c in record["classifiers"]}
    assert entries["laya"]["raw_decisions"] is True
    assert "raw_decisions" not in entries["random"] and "raw_decisions" not in entries["s1d1-small-int8"]
    assert all("raw_decisions" not in j for j in record["judges"])
    assert "raw decisions" in capsys.readouterr().out


@pytest.mark.parametrize("case", ["missing", "unknown", "duplicate"])
def test_run_refuses_decisions_with_a_missing_or_unknown_id(tmp_path, capsys, case):
    prompts = full_tree(tmp_path)
    folder = tmp_path / "classifiers" / "laya"
    rows = [raw_row(i) for i in IDS]
    expected = {"missing": "k2-zho", "unknown": "k9-eng", "duplicate": "k1-eng"}[case]
    if case == "missing":
        rows = [r for r in rows if r["id"] != expected]
    elif case == "unknown":
        rows.append(raw_row(expected))
    else:
        rows.append(raw_row(expected, "COMPLEX"))
    write_decisions(folder, rows)
    out = tmp_path / "results" / "set.parquet"
    out.parent.mkdir()
    out.write_bytes(b"old")
    assert run_cli(tmp_path, prompts) == 1
    err = capsys.readouterr().err
    assert "laya" in err and "decisions.jsonl" in err and repr(expected) in err
    assert {"missing": "no decision", "unknown": "not in the prompts file", "duplicate": "duplicate id"}[case] in err
    assert out.read_bytes() == b"old" and not out.with_name("set.run.json").exists()  # nothing written


def test_run_refuses_a_decision_without_a_tier_or_a_score(tmp_path, capsys):
    prompts = full_tree(tmp_path)
    rows = [raw_row(i) for i in IDS]
    del rows[2]["score"]
    write_decisions(tmp_path / "classifiers" / "laya", rows)
    assert run_cli(tmp_path, prompts) == 1
    err = capsys.readouterr().err
    assert "laya" in err and "decisions.jsonl" in err and "score" in err
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_run_refuses_a_non_finite_raw_score(tmp_path, capsys, bad):
    prompts = full_tree(tmp_path)
    rows = [raw_row(i) for i in IDS]
    rows[3]["score"] = bad  # json.dumps writes NaN / Infinity / -Infinity, which json.loads reads back
    write_decisions(tmp_path / "classifiers" / "laya", rows)
    assert run_cli(tmp_path, prompts) == 1
    err = capsys.readouterr().err
    assert "laya" in err and repr(IDS[3]) in err and "finite" in err
    assert not (tmp_path / "results").exists()
