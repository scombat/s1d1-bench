import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s1d1_bench.cli import main
from s1d1_bench.draw import STRATA, read_veto, stratified_draw
from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.rubric import TASKS


def L(level, task="chat", needs_tools=False):
    return {"complexity": level, "task": task, "needs_tools": needs_tools}


def pool(n_per_stratum=30, tools_every=10):
    """n ids per stratum, task cycling over the rubric, one needs_tools true every `tools_every`."""
    labels = {}
    k = 0
    for stratum in STRATA:
        for j in range(n_per_stratum):
            labels[f"{stratum}-{j:02d}"] = L(stratum, TASKS[k % len(TASKS)], k % tools_every == 0)
            k += 1
    return labels


def test_each_stratum_gets_per_stratum_ids_and_nothing_else():
    d = stratified_draw(pool(), per_stratum=5, tools_min=0, task_min=0)
    assert [len(d.picked[s]) for s in STRATA] == [5, 5, 5, 5]
    assert all(i.startswith(s) for s in STRATA for i in d.picked[s])
    assert len(set(d.ids())) == 20
    assert d.shortfalls == []


def test_same_seed_same_draw_other_seed_other_draw():
    a, b, c = (stratified_draw(pool(), seed=s, per_stratum=5) for s in (0, 0, 1))
    assert a.ids() == b.ids()
    assert a.ids() != c.ids()


def test_tools_minimum_is_met_round_robin_over_strata():
    labels = pool(tools_every=3)
    d = stratified_draw(labels, per_stratum=10, tools_min=8, task_min=0)
    # the minimum is a floor (the fill step may add more); the first two picks of every stratum are the round-robin
    assert sum(labels[i]["needs_tools"] for i in d.ids()) >= 8
    assert all(labels[i]["needs_tools"] for s in STRATA for i in d.picked[s][:2])


def test_tools_shortfall_is_reported_not_hidden():
    labels = pool(tools_every=50)  # 3 needs_tools in 120
    d = stratified_draw(labels, per_stratum=10, tools_min=20, task_min=0)
    assert sum(labels[i]["needs_tools"] for i in d.ids()) == 3
    assert d.shortfalls == ["needs_tools: 3 picked, 20 wanted"]


def test_task_minimum_is_met_when_the_candidates_allow_it():
    labels = pool(n_per_stratum=40)
    d = stratified_draw(labels, per_stratum=20, tools_min=0, task_min=4)
    counts = {t: sum(labels[i]["task"] == t for i in d.ids()) for t in TASKS}
    assert min(counts.values()) >= 4
    assert d.shortfalls == []


def test_task_shortfall_names_the_task():
    labels = pool(n_per_stratum=5)  # 20 ids, 19 tasks: at most 2 of any task
    d = stratified_draw(labels, per_stratum=5, tools_min=0, task_min=4)
    assert any(s.startswith(f"task {TASKS[0]}: ") for s in d.shortfalls)


def test_null_complexity_is_never_drawn():
    labels = pool() | {"n1": L(None), "n2": L(None)}
    d = stratified_draw(labels, per_stratum=30, tools_min=0, task_min=0)
    assert "n1" not in d.ids() and "n2" not in d.ids()


def test_veto_replaces_only_the_vetoed_id():
    labels = pool()
    before = stratified_draw(labels, per_stratum=5, tools_min=0, task_min=0)
    gone = before.picked["low"][2]
    after = stratified_draw(labels, per_stratum=5, tools_min=0, task_min=0, vetoed={gone: "same template"})
    assert gone not in after.ids()
    for s in STRATA:
        if s != "low":
            assert after.picked[s] == before.picked[s]
    assert [i for i in after.picked["low"] if i != gone][:4] == [i for i in before.picked["low"] if i != gone]
    assert len(after.picked["low"]) == 5


def test_read_veto_wants_an_id_a_tab_and_a_reason(tmp_path):
    f = tmp_path / "veto.txt"
    f.write_text("# comment\nlow-01\tsame template as low-00\n\nhigh-02\ttranslation changes the task\n")
    assert read_veto(f) == {"low-01": "same template as low-00", "high-02": "translation changes the task"}
    f.write_text("low-01 no tab\n")
    with pytest.raises(ValueError, match="expected"):
        read_veto(f)


def test_cli_writes_base_keys_and_the_draw_record(tmp_path):
    labels = pool()
    presort = tmp_path / "presort"
    presort.mkdir()
    write_jsonl(presort / "labels.jsonl", [{"id": i} | lab for i, lab in labels.items()])
    meta = tmp_path / "set.meta.parquet"
    pq.write_table(pa.table({"id": list(labels), "base_key": [i.upper() for i in labels]}), meta)
    out = tmp_path / "base.select.txt"
    rc = main(
        [
            "draw",
            "--presort",
            str(presort),
            "--meta",
            str(meta),
            "--out",
            str(out),
            "--per-stratum",
            "5",
            "--tools-min",
            "0",
            "--task-min",
            "0",
        ]
    )
    assert rc == 0
    keys = out.read_text().split()
    assert len(keys) == 20 and all(k == k.upper() for k in keys)
    record = json.loads((tmp_path / "base.draw.json").read_text())
    assert record["seed"] == 0 and record["per_stratum"] == 5 and record["shortfalls"] == []
    assert [k.upper() for k in [i for s in STRATA for i in record["picked"][s]]] == keys


def test_cli_refuses_a_veto_on_an_unknown_id(tmp_path, capsys):
    presort = tmp_path / "presort"
    presort.mkdir()
    write_jsonl(presort / "labels.jsonl", [{"id": "a"} | L("low")])
    meta = tmp_path / "set.meta.parquet"
    pq.write_table(pa.table({"id": ["a"], "base_key": ["A"]}), meta)
    veto = tmp_path / "veto.txt"
    veto.write_text("zzz\tnot here\n")
    out = tmp_path / "o.txt"
    rc = main(["draw", "--presort", str(presort), "--meta", str(meta), "--out", str(out), "--veto", str(veto)])
    assert rc == 1
    assert "zzz" in capsys.readouterr().err
