import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s1d1_bench import pool
from s1d1_bench.cli import main
from s1d1_bench.witness import CONFIGS, run_witness, witness_rows

SHA = "a" * 40
META = [
    {"base_key": "kb", "source": "arena_hard", "source_id": "q2"},
    {"base_key": "ka", "source": "arena_hard", "source_id": "q1"},
    {"base_key": "kd", "source": "dolly", "source_id": "3"},
]


def test_witness_rows_join_on_question_id_and_keep_only_arena():
    meta = [
        {"base_key": "k1", "source": "arena_hard", "source_id": "q1"},
        {"base_key": "k2", "source": "dolly", "source_id": "3"},
    ]
    rows = witness_rows([{"question_id": "q1", "prompt": "Bonjour"}], meta, "fra")
    assert rows == [{"id": "k1-fra", "base_key": "k1", "prompt": "Bonjour"}]


def test_witness_rows_name_a_key_without_witness():
    with pytest.raises(ValueError, match="k1"):
        witness_rows([], [{"base_key": "k1", "source": "arena_hard", "source_id": "q1"}], "rus")


def fake_hub(monkeypatch):
    """No network: the dataset of each config is two rows, `load_split` records how it was called."""
    calls = []

    def load_split(source, config, split, cache):
        calls.append((source, config, split, cache))
        return [
            {"question_id": "q1", "prompt": f"{config}-one", "cluster": "x"},
            {"question_id": "q2", "prompt": f"{config}-two", "cluster": "y"},
            {"question_id": "q3", "prompt": f"{config}-unused", "cluster": "z"},
        ]

    monkeypatch.setattr(pool, "load_split", load_split)
    monkeypatch.setattr(pool, "_hub_info", lambda source: SimpleNamespace(sha=SHA))
    return calls


def write_meta(tmp_path):
    path = tmp_path / "base-v1-eng.meta.parquet"
    pq.write_table(pa.Table.from_pylist(META), path)
    return path


def args_for(tmp_path, meta, langs="fra,rus,zho"):
    return SimpleNamespace(meta=meta, langs=langs, out=tmp_path / "witness", cache=tmp_path / "cache")


def test_run_witness_writes_files_select_and_meta(tmp_path, monkeypatch):
    calls = fake_hub(monkeypatch)
    assert run_witness(args_for(tmp_path, write_meta(tmp_path))) == 0
    out = tmp_path / "witness"
    assert sorted(p.name for p in out.iterdir()) == sorted(
        ["arena.select.txt", *(f"{lang}.{ext}" for lang in CONFIGS for ext in ("jsonl", "meta.json"))]
    )
    assert [c[:3] for c in calls] == [("arena_hard", config, "test") for config in ("fr", "ru", "zh")]
    assert {c[3] for c in calls} == {tmp_path / "cache"}
    assert (out / "arena.select.txt").read_text(encoding="utf-8") == "ka\nkb\n"
    rows = [json.loads(line) for line in (out / "zho.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"id": "ka-zho", "base_key": "ka", "prompt": "zh-one"},
        {"id": "kb-zho", "base_key": "kb", "prompt": "zh-two"},
    ]
    meta = json.loads((out / "rus.meta.json").read_text(encoding="utf-8"))
    assert meta["lang"] == "rus" and meta["config"] == "ru" and meta["split"] == "test"
    assert meta["repo"] == "CohereLabs/m-ArenaHard" and meta["revision"] == SHA and meta["licence"] == "apache-2.0"
    assert meta["translator"] == "Google Translate, per the m-ArenaHard dataset card"
    assert meta["source_prompts_meta"] == "base-v1-eng" and meta["date"]


def test_run_witness_refuses_to_overwrite(tmp_path, monkeypatch, capsys):
    fake_hub(monkeypatch)
    meta = write_meta(tmp_path)
    out = tmp_path / "witness"
    out.mkdir()
    (out / "rus.jsonl").write_text("precious\n", encoding="utf-8")
    assert run_witness(args_for(tmp_path, meta)) == 1
    assert "rus.jsonl" in capsys.readouterr().err
    assert (out / "rus.jsonl").read_text(encoding="utf-8") == "precious\n"
    assert sorted(p.name for p in out.iterdir()) == ["rus.jsonl"]  # nothing else was written


def test_run_witness_reports_a_missing_witness(tmp_path, monkeypatch, capsys):
    fake_hub(monkeypatch)
    meta = tmp_path / "m.meta.parquet"
    pq.write_table(pa.Table.from_pylist(META + [{"base_key": "kz", "source": "arena_hard", "source_id": "q9"}]), meta)
    assert run_witness(args_for(tmp_path, meta)) == 1
    assert "kz" in capsys.readouterr().err
    assert not (tmp_path / "witness").exists()


def test_bench_witness_is_wired_and_listed(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    assert "witness" in capsys.readouterr().out
