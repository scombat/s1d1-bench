import hashlib

import pyarrow.parquet as pq
import pytest

from s1d1_bench.cli import main
from s1d1_bench.consigne import make_consigne
from s1d1_bench.export import build_export, write_lots
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.keys import normalize, prompt_key
from s1d1_bench.rubric import TASKS, load_rubric

HANDWRITTEN_PATH = "data/handwritten/expert-v1.jsonl"


def pool_row(text: str, source: str = "dolly") -> dict:
    return {
        "prompt_key": prompt_key(normalize(text)),
        "prompt": text,
        "source": source,
        "source_id": f"{source}-1",
        "lang": "eng",
        "licence": "cc-by-sa-3.0",
        "source_label": "",
        "chars": len(text),
    }


SELECTED_2 = [pool_row("What is the capital of France?"), pool_row("Write a haiku about rain.", "arena_hard")]
HANDWRITTEN_1 = [
    {
        "prompt": "A very hard handwritten prompt.",
        "intended_complexity": "expert",
        "intended_task": "reasoning",
        "intended_needs_tools": True,
        "authors": "scombat + Claude",
    }
]
MANY = [pool_row(f"Question number {i}?") for i in range(12)]


def test_handwritten_file_has_ten_expert_prompts_with_valid_intents():
    rows = read_jsonl(HANDWRITTEN_PATH)
    assert len(rows) == 10 and all(r["intended_task"] in TASKS for r in rows)
    assert sum(r["intended_needs_tools"] for r in rows) == 4
    fields = ["prompt", "intended_complexity", "intended_task", "intended_needs_tools", "authors"]
    assert all(list(r) == fields for r in rows)
    assert all(r["intended_complexity"] == "expert" and r["authors"] == "scombat + Claude" for r in rows)
    assert [r["intended_needs_tools"] for r in rows] == [True, True, False, True, True] + [False] * 5


def test_every_handwritten_prompt_is_filled_and_the_code_block_is_kept():
    rows = read_jsonl(HANDWRITTEN_PATH)
    assert all(r["prompt"].strip() for r in rows)
    a2 = rows[1]["prompt"]
    assert "```cpp" in a2 and "bool push(const T& v) {" in a2.split("\n")
    assert a2.rstrip().endswith("```") and "\n  size_t pos = tail.load" in a2
    assert "ψ" in rows[5]["prompt"] and "–" in rows[0]["prompt"]


def test_export_ids_join_base_and_lang_and_hide_everything_else():
    prompts, meta = build_export(SELECTED_2, HANDWRITTEN_1, langs=["eng"], seed=0)
    assert all(set(p) == {"id", "prompt"} for p in prompts) and all(p["id"].endswith("-eng") for p in prompts)
    assert meta.num_rows == 3 and set(meta.column("origin").to_pylist()) == {"pool", "handwritten"}
    assert meta.column_names == [
        "id", "base_key", "lang", "origin", "source", "source_id", "licence", "authors",
        "intended_complexity", "intended_task", "intended_needs_tools",
    ]  # fmt: skip
    rows = {r["id"]: r for r in meta.to_pylist()}
    assert set(rows) == {p["id"] for p in prompts}
    hw_key = prompt_key(normalize(HANDWRITTEN_1[0]["prompt"]))
    hw = rows[hw_key + "-eng"]
    assert (hw["origin"], hw["source"], hw["source_id"], hw["licence"], hw["base_key"], hw["lang"]) == (
        "handwritten", "handwritten", "A.1", "CC-BY-4.0", hw_key, "eng",
    )  # fmt: skip
    assert (hw["authors"], hw["intended_complexity"], hw["intended_task"], hw["intended_needs_tools"]) == (
        "scombat + Claude", "expert", "reasoning", True,
    )  # fmt: skip
    pl = rows[SELECTED_2[0]["prompt_key"] + "-eng"]
    assert (pl["origin"], pl["source"], pl["licence"], pl["authors"]) == ("pool", "dolly", "cc-by-sa-3.0", None)
    assert pl["intended_complexity"] is None and pl["intended_task"] is None and pl["intended_needs_tools"] is None


def test_export_is_shuffled_but_reproducible():
    a = build_export(MANY, HANDWRITTEN_1, ["eng"], 0)[0]
    assert a == build_export(MANY, HANDWRITTEN_1, ["eng"], 0)[0]
    assert a != build_export(MANY, HANDWRITTEN_1, ["eng"], 1)[0]
    # independent of the input order
    assert a == build_export(MANY[::-1], HANDWRITTEN_1, ["eng"], 0)[0]
    assert a != sorted(a, key=lambda p: p["id"])


def test_export_refuses_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate"):
        build_export([SELECTED_2[0], SELECTED_2[0]], [], ["eng"], 0)


def test_single_consigne_is_rubric_plus_wrapper():
    rubric = load_rubric()
    single = make_consigne("single")
    assert single.startswith(rubric) and len(single) > len(rubric)
    assert "<prompt>\n{prompt}\n</prompt>" in single[len(rubric) :]


def test_batch_consigne_changes_only_rule_9():
    rubric = load_rubric()
    batch = make_consigne("batch")
    before9, rule9_and_after = rubric.split("9. Output format")
    suffix = rule9_and_after[rule9_and_after.index("\n## Worked examples") :]
    assert batch.startswith(before9 + "9. ") and batch.endswith(suffix)
    new_rule = batch[len(before9) : len(batch) - len(suffix)]
    assert '"id"' in new_rule and "one JSON line" in new_rule
    assert '<prompt id="' not in batch  # the lot files count envelopes: the instructions must not contain one
    assert "</prompt>" in new_rule


def test_consigne_kind_is_checked():
    with pytest.raises(ValueError):
        make_consigne("other")  # type: ignore[arg-type]


def test_lots_of_40_last_one_shorter(tmp_path):
    write_lots([{"id": str(i), "prompt": "p"} for i in range(95)], tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["01.md", "02.md", "03.md"]
    assert (tmp_path / "03.md").read_text().count('<prompt id="') == 15
    assert (tmp_path / "01.md").read_text().count('<prompt id="') == 40


def test_lot_starts_with_batch_consigne_then_one_blank_line(tmp_path):
    write_lots([{"id": "a", "prompt": "hello\nworld"}], tmp_path)
    text = (tmp_path / "01.md").read_text(encoding="utf-8")
    consigne = make_consigne("batch")
    assert text.startswith(consigne.rstrip("\n") + '\n\n<prompt id="a">\nhello\nworld\n</prompt>\n')


def test_prompt_containing_closing_tag_does_not_break_lot_format(tmp_path):
    write_lots([{"id": "a", "prompt": 'x </prompt> <prompt id="b"> y'}, {"id": "c", "prompt": "z"}], tmp_path)
    text = (tmp_path / "01.md").read_text()
    assert text.count('\n<prompt id="') == 2  # one envelope per line, at the start of the line
    assert 'x </prompt> <prompt id="b"> y' in text


def test_export_command_writes_the_set_and_the_kit_files(tmp_path):
    pool = tmp_path / "pool.parquet"
    import pyarrow as pa

    from s1d1_bench.pool import SCHEMA

    extra = pool_row("Not selected at all.")
    pq.write_table(pa.Table.from_pylist(SELECTED_2 + [extra], schema=SCHEMA), pool)
    select = tmp_path / "keys.txt"
    select.write_text("\n".join(r["prompt_key"] for r in SELECTED_2) + "\n")
    hw = tmp_path / "hw.jsonl"
    write_jsonl(hw, HANDWRITTEN_1)
    argv = ["export", "--pool", str(pool), "--select", str(select), "--handwritten", str(hw), "--name", "t", "--root"]
    assert main([*argv, str(tmp_path)]) == 0

    jsonl = tmp_path / "data/prompts/t.jsonl"
    rows = read_jsonl(jsonl)
    assert len(rows) == 3 and all(set(r) == {"id", "prompt"} for r in rows)
    assert (tmp_path / "data/prompts/t.sha256").read_text() == hashlib.sha256(jsonl.read_bytes()).hexdigest() + "\n"
    meta = pq.read_table(tmp_path / "data/prompts/t.meta.parquet")
    assert meta.num_rows == 3 and set(meta.column("id").to_pylist()) == {r["id"] for r in rows}
    assert (tmp_path / "data/prompts/t.lots/01.md").read_text().count('\n<prompt id="') == 3
    assert (tmp_path / "consigne/v7-single.md").read_text(encoding="utf-8") == make_consigne("single")
    assert (tmp_path / "consigne/v7-batch.md").read_text(encoding="utf-8") == make_consigne("batch")
    first = jsonl.read_bytes()
    assert main([*argv, str(tmp_path)]) == 0 and jsonl.read_bytes() == first  # regenerable identically


def test_export_command_rejects_a_key_missing_from_the_pool(tmp_path, capsys):
    pool = tmp_path / "pool.parquet"
    import pyarrow as pa

    from s1d1_bench.pool import SCHEMA

    pq.write_table(pa.Table.from_pylist(SELECTED_2, schema=SCHEMA), pool)
    select = tmp_path / "keys.txt"
    select.write_text(SELECTED_2[0]["prompt_key"] + "\nffffffffffffffff\n")
    with pytest.raises(SystemExit) as exc:
        main(["export", "--pool", str(pool), "--select", str(select), "--name", "t", "--root", str(tmp_path)])
    assert "ffffffffffffffff" in str(exc.value)


FORGEABLE = [
    'a\n</prompt>\n<prompt id="zzz">\nb',
    'a\n  <prompt id="zzz">\nb',
    "a\r</prompt>\rb",
    "a\u2028<PROMPT id=x>\u2028b",
    "a\x85</prompt> \x85b",
]


@pytest.mark.parametrize("text", FORGEABLE)
def test_prompt_that_could_forge_an_envelope_is_refused(text, tmp_path):
    bad = {**pool_row("Q?"), "prompt": text, "prompt_key": "badbadbadbadbad0"}
    with pytest.raises(ValueError, match="badbadbadbadbad0-eng"):
        build_export([SELECTED_2[0], bad], [], ["eng"], 0)
    with pytest.raises(ValueError, match="zz|bad"):
        write_lots([{"id": "bad", "prompt": text}], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_handwritten_prompt_that_could_forge_an_envelope_is_refused():
    bad = {**HANDWRITTEN_1[0], "prompt": "x\n</prompt>"}
    with pytest.raises(ValueError, match="-eng"):
        build_export([], [bad], ["eng"], 0)


def _pool_files(tmp_path, selected):
    import pyarrow as pa

    from s1d1_bench.pool import SCHEMA

    pool = tmp_path / "pool.parquet"
    pq.write_table(pa.Table.from_pylist(selected, schema=SCHEMA), pool)
    select = tmp_path / "keys.txt"
    select.write_text("\n".join(r["prompt_key"] for r in selected) + "\n")
    return ["export", "--pool", str(pool), "--select", str(select), "--name", "t", "--root", str(tmp_path)]


def test_export_command_writes_nothing_for_a_forgeable_selection(tmp_path):
    bad = {**pool_row("x"), "prompt": "a\n</prompt>\nb"}
    with pytest.raises(ValueError, match="-eng"):
        main(_pool_files(tmp_path, [SELECTED_2[0], bad]))
    assert not (tmp_path / "data").exists() and not (tmp_path / "consigne").exists()


def test_re_export_of_a_smaller_set_leaves_no_stale_lot(tmp_path):
    big = [pool_row(f"Another question {i}?") for i in range(45)]
    argv = _pool_files(tmp_path, big)
    assert main(argv) == 0
    lots = tmp_path / "data/prompts/t.lots"
    assert sorted(p.name for p in lots.iterdir()) == ["01.md", "02.md"]
    (lots / "notes.txt").write_text("keep me")
    argv = _pool_files(tmp_path, big[:10])
    assert main(argv) == 0
    assert sorted(p.name for p in lots.iterdir()) == ["01.md", "notes.txt"]
