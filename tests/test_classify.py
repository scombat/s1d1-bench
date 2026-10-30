import hashlib
import json
from collections import Counter

import pytest

from s1d1_bench.classify import ANCHORS, constant_labels, length_labels, nearest_anchor, random_labels
from s1d1_bench.cli import build_parser, main
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import validate_judge_dir
from s1d1_bench.rubric import LEVELS, TASKS


def test_anchors_are_the_documented_ones():
    assert ANCHORS == {"trivial": 0.01, "low": 0.25, "medium": 0.50, "high": 0.75, "expert": 1.00}
    assert list(ANCHORS) == LEVELS


@pytest.mark.parametrize(
    ("score", "tier"),
    [
        (0.0, "trivial"),
        (0.01, "trivial"),
        (0.129, "trivial"),
        (0.13, "trivial"),  # RULING R1 again: the midpoint of trivial and low goes to trivial
        (0.131, "low"),
        (0.25, "low"),
        (0.375, "low"),  # RULING R1: an exact midpoint goes to the lower tier
        (0.376, "medium"),
        (0.50, "medium"),
        (0.625, "medium"),
        (0.626, "high"),
        (0.75, "high"),
        (0.875, "high"),
        (0.876, "expert"),
        (1.0, "expert"),
    ],
)
def test_nearest_anchor_midpoints(score, tier):
    assert nearest_anchor(score) == tier


def test_nearest_anchor_is_the_closest_anchor_everywhere():
    midpoints = {0.13, 0.375, 0.625, 0.875}  # exact ties: float distances are noisy there, fixed by the test above
    for k in range(0, 1001):
        score = k / 1000
        if score in midpoints:
            continue
        closest = min(ANCHORS, key=lambda t: (abs(ANCHORS[t] - score), LEVELS.index(t)))  # ties: lower tier
        assert nearest_anchor(score) == closest, score


IDS = [f"k{n:03d}-eng" for n in range(40)]


def test_random_is_seeded_and_uniform_enough():
    draws = random_labels(IDS, seed=0, draws=10)
    assert len(draws) == 10 and all(len(d) == len(IDS) for d in draws)
    assert draws == random_labels(IDS, seed=0, draws=10)  # same seed, same draws
    assert draws != random_labels(IDS, seed=1, draws=10)
    assert draws[0] != draws[1]  # the ten draws differ from each other
    rows = [row for draw in draws for row in draw]  # 400 rows
    for row in rows:
        assert set(row) == {"id", "complexity", "task", "needs_tools"}  # no score: omitted, not null
        assert row["complexity"] in LEVELS and row["task"] in TASKS and type(row["needs_tools"]) is bool
    assert [r["id"] for r in draws[3]] == IDS
    levels = Counter(r["complexity"] for r in rows)
    assert set(levels) == set(LEVELS) and all(40 <= n <= 120 for n in levels.values())  # expected 80 each
    tools = Counter(r["needs_tools"] for r in rows)
    assert all(150 <= n <= 250 for n in tools.values()) and len(tools) == 2
    assert set(r["task"] for r in rows) == set(TASKS)  # 400 draws over 19 tasks reach them all


def _prompts(lengths_by_lang: dict[str, list[int]]) -> list[dict]:
    return [
        {"id": f"b{n:02d}-{lang}", "prompt": "x" * length}
        for lang, lengths in lengths_by_lang.items()
        for n, length in enumerate(lengths)
    ]


def test_length_deciles_are_per_language():
    short = list(range(1, 21))  # 20 prompts of 1..20 characters
    long = [1000 + 37 * n for n in range(20)]  # 20 prompts of 1000..1703 characters
    prompts = _prompts({"eng": short, "fra": long})
    rows = length_labels(prompts)
    assert [r["id"] for r in rows] == [p["id"] for p in prompts]  # prompt order
    by_lang = {lang: [r["complexity"] for r in rows if r["id"].endswith(lang)] for lang in ("eng", "fra")}
    # the same distribution of tiers although every French prompt is longer than every English one
    assert by_lang["eng"] == by_lang["fra"]
    # 2/3/3/1/1 deciles of two prompts each: 4 trivial, 6 low, 6 medium, 2 high, 2 expert
    assert Counter(by_lang["eng"]) == {"trivial": 4, "low": 6, "medium": 6, "high": 2, "expert": 2}
    assert by_lang["eng"] == sorted(by_lang["eng"], key=LEVELS.index)  # longer prompt, higher or equal tier
    assert all(r["task"] is None and r["needs_tools"] is None for r in rows)
    # score = rank / (n - 1) within the language: 0 for the shortest, 1 for the longest
    scores = [r["score"] for r in rows if r["id"].endswith("-fra")]
    assert scores[0] == 0.0 and scores[-1] == 1.0 and scores == sorted(scores)
    assert scores[1] == pytest.approx(1 / 19)


def test_length_decile_to_tier_map_is_explicit():
    # ten prompts per language: one per decile, so the tier of the k-th shortest shows the decile map
    prompts = _prompts({"eng": list(range(10, 110, 10))})
    tiers = [r["complexity"] for r in length_labels(prompts)]
    assert tiers == ["trivial"] * 2 + ["low"] * 3 + ["medium"] * 3 + ["high", "expert"]
    custom = [r["complexity"] for r in length_labels(prompts, decile_tiers=(1, 1, 1, 1, 6))]
    assert custom == ["trivial", "low", "medium", "high"] + ["expert"] * 6
    with pytest.raises(ValueError, match="10"):
        length_labels(prompts, decile_tiers=(2, 3, 3, 1, 2))


def test_length_ties_and_small_languages_are_deterministic():
    prompts = _prompts({"eng": [5, 5, 5, 5], "fra": [7]})
    rows = length_labels(prompts)
    assert rows == length_labels(prompts)
    fra = next(r for r in rows if r["id"].endswith("-fra"))
    assert fra["score"] == 0.0 and fra["complexity"] == "trivial"  # a single prompt: rank 0, score 0
    eng = [r["score"] for r in rows if r["id"].endswith("-eng")]
    assert eng == [0.0, 1 / 3, 2 / 3, 1.0]  # equal lengths: ranked stably by id


def test_length_language_is_the_suffix_after_the_last_dash():
    prompts = [{"id": "a-b-eng", "prompt": "x"}, {"id": "c-d-eng", "prompt": "xx"}, {"id": "e-fra", "prompt": "x"}]
    scores = {r["id"]: r["score"] for r in length_labels(prompts)}
    assert scores == {"a-b-eng": 0.0, "c-d-eng": 1.0, "e-fra": 0.0}
    with pytest.raises(ValueError, match="nodash"):
        length_labels([{"id": "nodash", "prompt": "x"}])


def test_length_counts_characters_not_bytes():
    prompts = [{"id": "a-eng", "prompt": "é" * 3}, {"id": "b-eng", "prompt": "e" * 4}]
    scores = {r["id"]: r["score"] for r in length_labels(prompts)}
    assert scores == {"a-eng": 0.0, "b-eng": 1.0}  # 3 characters (6 bytes) < 4 characters


def test_constant():
    rows = constant_labels(["a-eng", "b-fra"], "low")
    assert rows == [
        {"id": "a-eng", "complexity": "low", "task": None, "needs_tools": None, "score": 0.25},
        {"id": "b-fra", "complexity": "low", "task": None, "needs_tools": None, "score": 0.25},
    ]
    assert constant_labels(["a-eng"], "expert")[0]["score"] == 1.0
    with pytest.raises(ValueError, match="huge"):
        constant_labels(["a-eng"], "huge")


# ---------- the command ----------

PROMPTS = [
    {"id": f"b{n}-{lang}", "prompt": "p" * (10 * n + 1 + len(lang))} for n in range(6) for lang in ("eng", "fra")
]


@pytest.fixture
def prompts_file(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    write_jsonl(d / "mini.jsonl", PROMPTS)
    return d / "mini.jsonl"


def _classify(adapter, prompts_file, out, *extra):
    return main(["classify", adapter, "--prompts", str(prompts_file), "--out", str(out), "--author", "tester", *extra])


def test_parser_declares_every_option_of_the_brief(prompts_file, tmp_path):
    args = build_parser().parse_args(
        ["classify", "s1d1", "--prompts", str(prompts_file), "--out", str(tmp_path / "o"), "--author", "a"]
    )
    assert (args.seed, args.draws, args.level, args.backend, args.notes) == (0, 10, "low", "auto", "")
    assert args.model is None and args.revision is None and args.laya_model is None
    full = build_parser().parse_args(
        ["classify", "laya", "--prompts", "p", "--out", "o", "--author", "a", "--model", "m", "--revision", "r",
         "--backend", "onnx", "--laya-model", "multilingual", "--level", "high", "--seed", "3", "--draws", "2"]
    )
    assert (full.laya_model, full.level, full.seed, full.draws) == ("multilingual", "high", 3, 2)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["classify", "bogus", "--prompts", "p", "--out", "o", "--author", "a"])


def test_an_adapter_that_is_not_registered_is_refused(prompts_file, tmp_path, capsys):
    from types import SimpleNamespace

    from s1d1_bench.classify import run_classify_cmd

    out = tmp_path / "out"
    args = SimpleNamespace(adapter="bogus", out=str(out), prompts=str(prompts_file))
    assert run_classify_cmd(args) == 2
    assert "not implemented yet" in capsys.readouterr().err
    assert not out.exists()


def test_cli_writes_meta_then_labels_and_refuses_overwrite(prompts_file, tmp_path, capsys):
    out = tmp_path / "classifiers" / "random"
    assert _classify("random", prompts_file, out, "--notes", "extra words") == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    sha = hashlib.sha256(prompts_file.read_bytes()).hexdigest()
    assert meta["prompts_sha256"] == sha and meta["author"] == "tester" and meta["judge"] == "random"
    assert (meta["provider"], meta["instructions"], meta["thinking"]) == ("system", "none", "none")
    assert meta["replayable"] is True and meta["temperature"] == 0 and meta["licence"] == "CC-BY-4.0"
    assert meta["model"] == "witness:random" and "quantization" not in meta
    assert "seed 0" in meta["notes"] and "10 draws" in meta["notes"] and "extra words" in meta["notes"]
    assert isinstance(meta["date"], str) and len(meta["date"]) == 10
    labels = read_jsonl(out / "labels.jsonl")
    draws = read_jsonl(out / "draws.jsonl")
    assert [r["id"] for r in labels] == [p["id"] for p in PROMPTS]
    assert len(draws) == 10 * len(PROMPTS)
    assert [{k: v for k, v in d.items() if k != "draw"} for d in draws[: len(PROMPTS)]] == labels  # draw 0
    assert sorted({d["draw"] for d in draws}) == list(range(10))
    assert validate_judge_dir(out, prompts_file.parent) == []  # draws.jsonl is ignored by validate
    # a second run neither overwrites nor writes anything
    before = {p.name: p.read_bytes() for p in out.iterdir()}
    capsys.readouterr()
    assert _classify("random", prompts_file, out) == 1
    assert "already exist" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in out.iterdir()} == before


def test_random_cli_is_deterministic_and_seed_changes_it(prompts_file, tmp_path):
    a, b, c = (tmp_path / n for n in "abc")
    assert _classify("random", prompts_file, a) == 0
    assert _classify("random", prompts_file, b) == 0
    assert _classify("random", prompts_file, c, "--seed", "1", "--draws", "3") == 0
    assert (a / "labels.jsonl").read_bytes() == (b / "labels.jsonl").read_bytes()
    assert (a / "draws.jsonl").read_bytes() == (b / "draws.jsonl").read_bytes()
    assert (a / "labels.jsonl").read_bytes() != (c / "labels.jsonl").read_bytes()
    assert len(read_jsonl(c / "draws.jsonl")) == 3 * len(PROMPTS)


def test_length_and_constant_cli_validate(prompts_file, tmp_path):
    length, const = tmp_path / "length", tmp_path / "constant-low"
    assert _classify("length", prompts_file, length) == 0
    assert _classify("constant", prompts_file, const) == 0
    for out in (length, const):
        assert validate_judge_dir(out, prompts_file.parent) == []
        assert not (out / "draws.jsonl").exists()
    labels = read_jsonl(length / "labels.jsonl")
    assert all(r["task"] is None and r["needs_tools"] is None and 0 <= r["score"] <= 1 for r in labels)
    meta = json.loads((length / "meta.json").read_text(encoding="utf-8"))
    assert "2/3/3/1/1" in meta["notes"] and "per language" in meta["notes"]
    const_labels = read_jsonl(const / "labels.jsonl")
    assert {(r["complexity"], r["score"]) for r in const_labels} == {("low", 0.25)}
    assert "low" in json.loads((const / "meta.json").read_text(encoding="utf-8"))["notes"]
    high = tmp_path / "constant-high"
    assert _classify("constant", prompts_file, high, "--level", "high") == 0
    assert {r["complexity"] for r in read_jsonl(high / "labels.jsonl")} == {"high"}


def test_cli_refuses_a_bad_prompts_file_and_writes_nothing(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    write_jsonl(bad, [{"id": "a-eng", "prompt": "x"}, {"id": "a-eng", "prompt": "y"}])
    out = tmp_path / "out"
    assert _classify("length", bad, out) == 1
    assert "duplicate" in capsys.readouterr().err and not out.exists()
    nodash = tmp_path / "nodash.jsonl"
    write_jsonl(nodash, [{"id": "nodash", "prompt": "x"}])
    assert _classify("length", nodash, out) == 1
    assert "nodash" in capsys.readouterr().err and not out.exists()


# ---------- litellm-heuristic ----------

LITELLM_PROMPTS = [
    {"id": "thanks-eng", "prompt": "thanks"},
    {
        "id": "proof-eng",
        "prompt": "Prove step by step that the algorithm is correct, think through each case carefully and reason "
        "about the proof.",
    },
]


@pytest.fixture
def litellm_prompts_file(tmp_path):
    d = tmp_path / "litellm-prompts"
    d.mkdir()
    write_jsonl(d / "mini.jsonl", LITELLM_PROMPTS)
    return d / "mini.jsonl"


def test_litellm_heuristic_adapter_writes_name_to_name_tiers_and_raw_decisions(litellm_prompts_file, tmp_path):
    out = tmp_path / "litellm-heuristic"
    assert _classify("litellm-heuristic", litellm_prompts_file, out) == 0
    labels = read_jsonl(out / "labels.jsonl")
    assert labels == [
        {"id": "thanks-eng", "complexity": "low", "task": None, "needs_tools": None},
        {"id": "proof-eng", "complexity": "expert", "task": None, "needs_tools": None},
    ]
    assert all("score" not in row for row in labels)
    decisions = read_jsonl(out / "decisions.jsonl")
    assert [d["id"] for d in decisions] == [p["id"] for p in LITELLM_PROMPTS]
    thanks, proof = decisions
    assert (thanks["tier"], thanks["cause"]) == ("SIMPLE", "heuristic_scorer")
    assert thanks["score"] == pytest.approx(-0.15)  # raw, negative, never clamped or rescaled
    assert thanks["signals"] == ["short (1 tokens)", "simple (thanks)"]
    assert (proof["tier"], proof["cause"]) == ("REASONING", "reasoning_override")
    assert isinstance(proof["signals"], list) and proof["score"] > 0
    assert set(thanks) == set(proof) == {"id", "tier", "score", "signals", "cause"}


def test_litellm_heuristic_folder_validates(litellm_prompts_file, tmp_path):
    out = tmp_path / "litellm-heuristic"
    assert _classify("litellm-heuristic", litellm_prompts_file, out) == 0
    assert validate_judge_dir(out, litellm_prompts_file.parent) == []  # decisions.jsonl is ignored by validate


def test_litellm_heuristic_meta_names_the_tag_and_commit(litellm_prompts_file, tmp_path):
    from s1d1_bench.vendor.litellm_heuristic import COMMIT, TAG

    out = tmp_path / "litellm-heuristic"
    assert _classify("litellm-heuristic", litellm_prompts_file, out, "--notes", "extra words") == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "litellm:complexity_router:heuristic"
    assert meta["revision"] == f"{TAG} {COMMIT}"
    assert meta["replayable"] is True and meta["author"] == "tester"
    for fragment in (
        "uv run bench classify litellm-heuristic --prompts",
        "SIMPLE -> low",
        "COMPLEX -> high",
        "REASONING -> expert",
        "decisions.jsonl",
        "not on the v7 scale",
        "rank metrics only",
        "system_prompt None",
        "default configuration",
        "src/s1d1_bench/vendor/litellm_heuristic/NOTICE.md",
        "extra words",
    ):
        assert fragment in meta["notes"], fragment
    assert "/Users/" not in meta["notes"] and str(tmp_path) not in meta["notes"]
