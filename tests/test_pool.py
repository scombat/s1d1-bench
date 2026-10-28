import pyarrow as pa
from fixtures.pool_rows import (
    KEY_OF_ROW_0,
    OASST_FIXTURE,
    OASST_LANG_FIXTURE,
    POOL_ROWS,
    ROW_9000_CHARS,
    ROW_EMPTY,
    ROWS,
)

from s1d1_bench.cli import main
from s1d1_bench.jsonl import read_jsonl
from s1d1_bench.keys import normalize, prompt_key
from s1d1_bench.pool import build_pool, dolly_prompt, oasst2_rows, sample_candidates

COLUMNS = ["prompt_key", "prompt", "source", "source_id", "lang", "licence", "source_label", "chars"]
TABLE, _ = build_pool(POOL_ROWS, frozenset())


def test_build_pool_excludes_dedupes_and_counts():
    table, stats = build_pool(ROWS, exclusions=frozenset({KEY_OF_ROW_0}))
    assert stats["dolly"]["excluded"] == 1 and stats["oasst2"]["duplicate"] == 1
    assert table.num_rows == len(ROWS) - 2
    assert set(table.column("licence").to_pylist()) <= {"cc-by-sa-3.0", "apache-2.0"}


def test_build_pool_table_shape_and_first_occurrence_wins():
    table, stats = build_pool(ROWS, frozenset())
    assert table.column_names == COLUMNS
    assert all(f.type == pa.int64() for f in table.schema if f.name == "chars")
    rows = table.to_pylist()
    for r in rows:
        assert r["prompt"] == normalize(r["prompt"]) and r["chars"] == len(r["prompt"])
        assert r["prompt_key"] == prompt_key(r["prompt"])
    assert [r["source_id"] for r in rows if r["source"] == "oasst2" and r["prompt"].startswith("Write a haiku")] == [
        "a1"
    ]
    for s in stats.values():
        assert s["read"] == s["empty"] + s["too_long"] + s["excluded"] + s["duplicate"] + s["kept"]


def test_build_pool_drops_empty_and_too_long():
    _, stats = build_pool([ROW_EMPTY, ROW_9000_CHARS], frozenset())
    assert stats["dolly"]["empty"] == 1 and stats["arena_hard"]["too_long"] == 1


def test_build_pool_stats_have_the_three_sources_even_when_unused():
    _, stats = build_pool([], frozenset())
    assert list(stats) == ["dolly", "oasst2", "arena_hard"]
    assert all(v == dict(read=0, empty=0, too_long=0, excluded=0, duplicate=0, kept=0) for v in stats.values())


def test_a_row_is_dropped_for_one_reason_in_order():
    # empty before too_long/excluded; too_long before excluded; excluded before duplicate
    long_prompt = "y" * 9000
    long_key = prompt_key(long_prompt)
    rows = [
        dict(ROW_9000_CHARS, prompt=long_prompt, source_id="1"),
        dict(ROW_9000_CHARS, prompt=long_prompt, source_id="2"),
    ]
    _, stats = build_pool(rows, frozenset({long_key}))
    assert stats["arena_hard"]["too_long"] == 2 and stats["arena_hard"]["excluded"] == 0

    ok = dict(ROW_9000_CHARS, prompt="short one")
    _, stats = build_pool([ok, dict(ok, source_id="2")], frozenset({prompt_key("short one")}))
    assert stats["arena_hard"]["excluded"] == 2 and stats["arena_hard"]["duplicate"] == 0


def test_max_chars_is_measured_after_normalization():
    padded = dict(ROW_9000_CHARS, prompt="  " + "z" * 8192 + " \r\n")
    table, stats = build_pool([padded], frozenset())
    assert table.num_rows == 1 and stats["arena_hard"]["too_long"] == 0
    _, stats = build_pool([padded], frozenset(), max_chars=8191)
    assert stats["arena_hard"]["too_long"] == 1


def test_dolly_prompt_joins_instruction_and_context():
    assert dolly_prompt({"instruction": "Sum up", "context": "Text."}) == "Sum up\n\nText."
    assert dolly_prompt({"instruction": "Hi", "context": ""}) == "Hi"


def test_oasst2_keeps_only_root_prompter_messages_that_passed_review():
    kept = oasst2_rows(OASST_FIXTURE)
    assert len(kept) == 2 and all(r["source_label"] == "prompter" for r in kept)
    assert [r["source_id"] for r in kept] == ["m1", "m2"]
    assert [r["lang"] for r in kept] == ["eng", "fra"]
    assert all(r["source"] == "oasst2" and r["licence"] == "apache-2.0" for r in kept)


def test_oasst2_lang_codes_map_to_iso_639_3():
    assert [r["lang"] for r in oasst2_rows(OASST_LANG_FIXTURE)] == ["por", "ukr", "zho", "und"]


def test_sample_candidates_is_seeded_and_per_source():
    a = sample_candidates(TABLE, per_source=2, lang="eng", seed=0)
    b = sample_candidates(TABLE, 2, "eng", 0)
    assert a == b and len(a) == 2 * 3
    assert all(set(r) == {"id", "prompt"} for r in a)


def test_sample_candidates_ignores_input_order_and_language():
    shuffled, _ = build_pool(list(reversed(POOL_ROWS)), frozenset())
    # the kept set differs only in which duplicate is first; none in POOL_ROWS among the English rows
    eng = {r["id"] for r in sample_candidates(TABLE, 50, "eng", 0)}
    assert eng == {r["prompt_key"] for r in TABLE.to_pylist() if r["lang"] == "eng"}
    assert sample_candidates(TABLE, 2, "eng", 0) == sample_candidates(shuffled, 2, "eng", 0)
    assert sample_candidates(TABLE, 2, "eng", 0) != sample_candidates(TABLE, 2, "eng", 1)


def test_sample_candidates_takes_all_when_source_is_small():
    assert len(sample_candidates(TABLE, 1000, "eng", 0)) == sum(1 for r in TABLE.to_pylist() if r["lang"] == "eng")


def test_pool_subcommand_defaults():
    from s1d1_bench.cli import build_parser

    args = build_parser().parse_args(["pool"])
    assert (str(args.out), str(args.exclusions), str(args.candidates)) == (
        "data/pool/pool.parquet",
        "data/exclusions/s1d1-seen-keys.txt",
        "data/pool/candidates-eng.jsonl",
    )
    assert args.per_source == 300 and args.seed == 0


def test_run_pool_writes_outputs_with_fake_fetch(tmp_path, monkeypatch):
    import json

    import pyarrow.parquet as pq

    from s1d1_bench import pool

    by_source = {s: [r for r in POOL_ROWS if r["source"] == s] for s in ("dolly", "oasst2", "arena_hard")}
    monkeypatch.setattr(pool, "fetch_rows", lambda source, cache: by_source[source])
    monkeypatch.setattr(pool, "dataset_versions", lambda: {"dolly": {"revision": "r"}})
    monkeypatch.setattr(pool, "licence_check", lambda: {"dolly": {"plan": "cc-by-sa-3.0", "card": "cc-by-sa-3.0"}})
    excl = tmp_path / "ex.txt"
    excl.write_text(KEY_OF_ROW_0 + "\n")
    out = tmp_path / "pool" / "pool.parquet"
    cand = tmp_path / "pool" / "cands.jsonl"
    rc = main(["pool", "--out", str(out), "--exclusions", str(excl), "--candidates", str(cand), "--per-source", "2"])
    assert rc == 0
    assert pq.read_table(out).num_rows == len(POOL_ROWS) - 2  # one excluded, one duplicate
    stats = json.loads((tmp_path / "pool" / "pool.stats.json").read_text())
    assert stats["stats"]["dolly"]["excluded"] == 1 and stats["date"] and stats["licence_check"]["dolly"]["plan"]
    assert len(read_jsonl(cand)) == 6
