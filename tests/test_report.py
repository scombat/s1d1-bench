import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from s1d1_bench import metrics
from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.report import (
    CEILING_NOTE,
    FAILS,
    IN_TRUTH,
    OUTSIDE,
    PASSES,
    RAW,
    TABLES,
    V7,
    markdown_table,
    tables,
    two_sigma_rule,
    write_tables,
)
from s1d1_bench.rubric import LEVELS
from s1d1_bench.run import RULES, build_results, truths

LANGS = ["eng", "fra", "rus", "zho"]
MAIN = ["j-a", "j-b", "j-c"]  # j-a is the Gemma judge: out of the *_nogemma truths
OTHERS = ["j-a-off", "j-d-off"]  # j-a-off has j-a's model: the thinking pair


def ids_of(keys):
    return [f"{key}-{lang}" for key in keys for lang in LANGS]


def label(level, task="information_seeking", needs_tools=False, **extra):
    return {"complexity": level, "task": task, "needs_tools": needs_tools, **extra}


def build(tmp_path: Path, keys, votes, preds, decisions=None, folders=None, draws=None, gemma=("j-a",)):
    """The results table, meta table, run record and classifiers root of a fake set.

    `votes`: judge -> id -> label dict (main judges first, in MAIN order); `preds`: classifier -> id -> label dict;
    `decisions`: classifier -> id -> {"tier", "score"} (the raw_score columns); `folders`: the classifier folders made
    under the root (default: every classifier); `draws`: the rows of random/draws.jsonl (default: three draws)."""
    ids = ids_of(keys)
    sources = {key: ("handwritten" if n % 2 else "oasst2") for n, key in enumerate(keys)}
    meta_rows = [
        {
            "id": i,
            "base_key": i.rsplit("-", 1)[0],
            "lang": i.rsplit("-", 1)[1],
            "origin": "handwritten" if sources[i.rsplit("-", 1)[0]] == "handwritten" else "pool",
            "source": sources[i.rsplit("-", 1)[0]],
            "corrected": False,
        }
        for i in ids
    ]
    main = [j for j in votes if j in MAIN]
    truth = truths({j: votes[j] for j in main}, main, meta_rows, list(gemma))
    results = build_results(meta_rows, votes, preds, truth, decisions or {})
    meta = pa.Table.from_pylist(meta_rows)
    model = {"j-a": "org/gemma-x", "j-a-off": "org/gemma-x"}
    run = {
        "prompts": "data/prompts/fake.jsonl",
        "prompts_sha256": "0" * 64,
        "meta": "data/prompts/fake.meta.parquet",
        "rows": len(ids),
        "judges": [
            {"name": j, "model": model.get(j, f"org/{j}"), "revision": "r", "date": "2026-10-30"} for j in votes
        ],
        "classifiers": [{"name": c, "model": f"m/{c}", "revision": "r", "date": "2026-10-30"} for c in preds],
        "main": main,
        "gemma": list(gemma),
        "abstentions": {
            j: [{"id": i, "status": "json_error"} for i in ids if votes[j][i].get("abstained")] for j in votes
        },
        "date": "2026-10-30",
        "rules": RULES,
    }
    root = tmp_path / "classifiers"
    for name in preds if folders is None else folders:
        (root / name).mkdir(parents=True)
    (root / "random").mkdir(parents=True, exist_ok=True)
    if draws is None:
        draws = [{"draw": d, "id": i, "complexity": LEVELS[(d + n) % 5]} for d in range(3) for n, i in enumerate(ids)]
    write_jsonl(root / "random" / "draws.jsonl", draws)
    return results, meta, run, root


KEYS = ["k1", "k2"]


def small_votes():
    """k1: j-a and j-c say high, j-b low (12 votes: 4 low, 8 high -> truth_all high; the 8 votes of j-b and j-c give
    medium, the rest of j-a); k2: all trivial; j-c abstains on k2-zho; the non-main judges vary."""
    votes = {j: {} for j in MAIN + OTHERS}
    for lang in LANGS:
        votes["j-a"][f"k1-{lang}"] = label("high")
        votes["j-b"][f"k1-{lang}"] = label("low")
        votes["j-c"][f"k1-{lang}"] = label("high")
        votes["j-a-off"][f"k1-{lang}"] = label("medium")
        votes["j-d-off"][f"k1-{lang}"] = label("high", task="coding_debugging", needs_tools=True)
        for j in MAIN + OTHERS:
            votes[j][f"k2-{lang}"] = label("trivial")
    votes["j-c"]["k2-zho"] = {"complexity": None, "task": None, "needs_tools": None, "abstained": "json_error"}
    return votes


def small_preds():
    s1d1 = {}
    for lang, level in zip(LANGS, ["high", "high", "medium", "medium"], strict=True):
        s1d1[f"k1-{lang}"] = label(
            level, task="coding_debugging", needs_tools=True, score={"high": 0.7, "medium": 0.55}[level],
            needs_tools_probability=0.8,
        )  # fmt: skip
    for lang, level in zip(LANGS, ["trivial", "trivial", "low", "low"], strict=True):
        s1d1[f"k2-{lang}"] = label(
            level, needs_tools=False, score={"trivial": 0.05, "low": 0.3}[level], needs_tools_probability=0.1
        )
    ids = ids_of(KEYS)
    laya = {i: label("medium" if i.startswith("k1") else "low", task=None, needs_tools=None) for i in ids}
    notask = {i: label("low", task=None, needs_tools=None) for i in ids}
    return {"s1d1-full-int8": s1d1, "laya-rank": laya, "notask": notask}


def small_decisions():
    return {"laya-rank": {i: {"tier": "x", "score": (2.0 if i.startswith("k1") else -1.0)} for i in ids_of(KEYS)}}


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    results, meta, run, root = build(
        tmp_path_factory.mktemp("small"), KEYS, small_votes(), small_preds(), small_decisions()
    )
    return tables(results, meta, run, root, sigma=0.005), (results, meta, run, root)


def rows(table, **where):
    return [r for r in table.to_pylist() if all(r[k] == v for k, v in where.items())]


def one(table, **where):
    found = rows(table, **where)
    assert len(found) == 1, (where, found)
    return found[0]


def test_every_table_is_there(small):
    computed, _ = small
    assert set(TABLES) <= set(computed)
    for name in [
        "agreement_all", "agreement_nogemma", "ceiling", "floor", "exact_vs_constant_low", "judge_pairs",
        "by_language", "lang_change", "truth_lang_vs_all", "by_source", "by_origin", "by_tier", "task",
        "needs_tools", "needs_tools_pr", "rank_metrics", "calibration", "calibration_bins", "paired", "predictions",
        "inputs", "inputs_summary", "skipped",
    ]:  # fmt: skip
        assert name in computed, name
        assert isinstance(computed[name], pa.Table)


def test_known_qwk_exact_within_one_by_hand(small):
    computed, _ = small
    # truth_all: k1 high x4, k2 trivial x4; s1d1: high high medium medium / trivial trivial low low
    # O = (high, high) 2, (high, medium) 2, (trivial, trivial) 2, (trivial, low) 2; E = 1 on rows {trivial, high} x
    # the 4 predicted columns; sum W E = 28/16, sum W O = 4/16 -> kappa = 1 - 4/28 = 6/7
    row = one(computed["agreement_all"], name="s1d1-full-int8")
    assert row["role"] == "candidate"
    assert row["n_ids"] == 8
    assert row["qwk"] == pytest.approx(6 / 7)
    assert row["exact"] == pytest.approx(0.5)
    assert row["within_one"] == pytest.approx(1.0)
    assert row["tiers_never_predicted"] == "expert"
    # the two-coder alpha matches metrics on the same pairs
    truth = [3, 3, 3, 3, 0, 0, 0, 0]
    pred = [3, 3, 2, 2, 0, 0, 1, 1]
    assert row["alpha"] == pytest.approx(metrics.krippendorff_alpha_ordinal(list(zip(truth, pred, strict=True))))


def test_roles_and_abstention_leave_one_id_out_for_that_judge(small):
    computed, _ = small
    table = computed["agreement_all"]
    assert one(table, name="j-a")["role"] == IN_TRUTH
    assert one(table, name="j-d-off")["role"] == OUTSIDE
    assert one(table, name="j-c")["n_ids"] == 7  # its abstention on k2-zho
    assert one(table, name="j-b")["n_ids"] == 8
    # without Gemma, the Gemma main judge is outside that truth
    assert one(computed["agreement_nogemma"], name="j-a")["role"] == OUTSIDE
    assert one(computed["agreement_nogemma"], name="j-b")["role"] == IN_TRUTH


def test_ceiling_uses_the_other_judges_8_votes(small):
    computed, _ = small
    table = computed["ceiling"]
    row = one(table, judge="j-a")
    # the rest of j-a: k1 -> j-b low x4 + j-c high x4 -> medium (R5); k2 trivial. j-a: high x4, trivial x4.
    # exact 4/8; QWK: O = (medium, high) 4, (trivial, trivial) 4 -> sum W E = 28/16, sum W O = 4/16 -> 6/7
    assert row["exact"] == pytest.approx(0.5)
    assert row["qwk"] == pytest.approx(6 / 7)
    assert row["note"] == CEILING_NOTE
    # against truth_all (12 votes, j-a's own included) j-a is exact everywhere: the ceiling is another number
    assert one(computed["agreement_all"], name="j-a")["exact"] == pytest.approx(1.0)
    # j-b: the rest (j-a, j-c) says high on k1, trivial on k2 (j-c abstains once: 7 of 8 votes)
    assert one(table, judge="j-b")["exact"] == pytest.approx(0.5)
    mean = one(table, judge="mean")
    values = [one(table, judge=j)["qwk"] for j in MAIN]
    assert mean["qwk"] == pytest.approx(sum(values) / 3)
    assert {r["judge"] for r in table.to_pylist()} == {*MAIN, "mean"}


def test_rank_only_candidate_has_rank_metrics_but_no_calibration(small):
    computed, _ = small
    rank = one(computed["rank_metrics"], name="laya-rank")
    assert rank["scale"] == RAW
    assert rank["spearman"] == pytest.approx(1.0)  # raw 2.0 on k1 (high), -1.0 on k2 (trivial)
    assert rank["auroc_high"] == pytest.approx(1.0)
    assert one(computed["rank_metrics"], name="s1d1-full-int8")["scale"] == V7
    assert "laya-rank" not in {r["name"] for r in computed["calibration"].to_pylist()}
    assert "laya-rank" not in {r["candidate"] for r in computed["calibration_bins"].to_pylist()}
    assert "notask" not in {r["name"] for r in computed["rank_metrics"].to_pylist()}  # no score at all
    cal = one(computed["calibration"], name="s1d1-full-int8")
    anchors = [0.75] * 4 + [0.01] * 4
    scores = [0.7, 0.7, 0.55, 0.55, 0.05, 0.05, 0.3, 0.3]
    assert cal["mae"] == pytest.approx(metrics.mae(scores, anchors))
    assert cal["ece"] == pytest.approx(metrics.ece(scores, anchors))


def test_candidate_without_task_has_no_task_nor_needs_tools_row(small):
    computed, _ = small
    for name in ("task", "needs_tools"):
        names = {r["name"] for r in computed[name].to_pylist()}
        assert "notask" not in names and "laya-rank" not in names
        assert "s1d1-full-int8" in names
        assert set(MAIN + OTHERS) <= names
    task = one(computed["task"], name="s1d1-full-int8")
    # truth task: information_seeking everywhere; s1d1 says coding_debugging on k1: accuracy 4/8
    assert task["accuracy"] == pytest.approx(0.5)
    needs = one(computed["needs_tools"], name="s1d1-full-int8")
    assert needs["positives"] == 0  # no id needs tools in the truth
    assert math.isnan(needs["auroc"])  # one class empty
    assert "peu de positifs" in needs["note"]
    assert math.isnan(one(computed["needs_tools"], name="j-b")["auroc"])  # no probability for a judge
    assert {r["name"] for r in computed["needs_tools_pr"].to_pylist()} == {"s1d1-full-int8"}


def test_languages_and_language_change(small):
    computed, _ = small
    langs = {r["lang"] for r in rows(computed["by_language"], name="s1d1-full-int8")}
    assert langs == set(LANGS)
    change = one(computed["lang_change"], name="s1d1-full-int8")
    # k1: high high medium medium, k2: trivial trivial low low: both keys change, by one tier
    assert change["n_keys"] == 2
    assert change["lang_change_rate"] == pytest.approx(1.0)
    assert change["lang_range_mean"] == pytest.approx(1.0)
    assert one(computed["lang_change"], name="notask")["lang_change_rate"] == pytest.approx(0.0)
    # j-c has a null version on k2: the key is dropped for it and counted
    jc = one(computed["lang_change"], name="j-c")
    assert (jc["n_keys"], jc["n_keys_dropped"]) == (1, 1)
    overall = one(computed["truth_lang_vs_all"], group="all")
    assert (overall["n_ids"], overall["n_keys"]) == (8, 2)


def test_tier_source_and_origin_tables(small):
    computed, _ = small
    high = one(computed["by_tier"], name="s1d1-full-int8", tier="high")
    assert high["n_ids"] == 4
    assert high["exact"] == pytest.approx(0.5)
    assert high["mean_signed_error"] == pytest.approx(-0.5)
    # one source per key: a single truth tier in each source, so no QWK (blank) but an exact
    source = one(computed["by_source"], name="s1d1-full-int8", source="oasst2")
    assert source["n_ids"] == 4 and math.isnan(source["qwk"])
    assert source["exact"] == pytest.approx(0.5)
    assert {r["origin"] for r in rows(computed["by_origin"], name="j-b")} == {"pool", "handwritten"}


def test_judge_pairs_flag_the_thinking_pair(small):
    computed, _ = small
    table = computed["judge_pairs"]
    assert len(table) == 10  # five judges, every pair
    pair = one(table, judge_a="j-a", judge_b="j-a-off")
    assert pair["thinking_pair"] is True
    assert one(table, judge_a="j-b", judge_b="j-c")["thinking_pair"] is False
    assert one(table, judge_a="j-b", judge_b="j-c")["n_ids"] == 7
    assert one(table, judge_a="j-b", judge_b="j-c")["left_out"] == 1


def test_floor_random_band_and_missing_roles_are_skipped(small):
    computed, (results, meta, run, root) = small
    floor = computed["floor"]
    draws = rows(floor, kind="random draw")
    assert len(draws) == 3
    qwks = [r["qwk"] for r in draws]
    assert one(floor, kind="random min")["qwk"] == pytest.approx(min(qwks))
    assert one(floor, kind="random max")["qwk"] == pytest.approx(max(qwks))
    assert one(floor, kind="random mean")["qwk"] == pytest.approx(sum(qwks) / 3)
    assert all(math.isnan(r["qwk_lo"]) for r in draws)
    skipped = {r["name"] for r in computed["skipped"].to_pylist()}
    assert {"s1d1-full-fp16", "length", "constant-low", "litellm-heuristic"} <= skipped
    assert len(computed["predictions"]) == 0  # both predictions need litellm-heuristic


def test_paired_rows_and_the_inputs(small):
    computed, _ = small
    paired = computed["paired"]
    assert {r["candidate"] for r in paired.to_pylist()} == {"laya-rank", "notask"}
    assert all(r["rule_2sigma"] in (PASSES, FAILS) for r in paired.to_pylist())
    inputs = computed["inputs"]
    assert one(inputs, name="j-c")["abstentions"] == 1
    assert one(inputs, name="s1d1-full-int8")["role"] == "candidate"
    summary = {r["item"]: r["value"] for r in computed["inputs_summary"].to_pylist()}
    assert summary["rows"] == "8 ids"
    assert summary["base keys"] == "2 prompts de base"
    assert summary["undecidable"].startswith("0 ids")


def test_two_sigma_rule():
    sigma = 0.005
    assert two_sigma_rule(0.02, 0.05, sigma) == PASSES
    assert two_sigma_rule(-0.05, -0.02, sigma) == PASSES
    assert two_sigma_rule(-0.01, 0.03, sigma) == FAILS  # the interval straddles 0
    assert two_sigma_rule(0.004, 0.02, sigma) == FAILS  # excludes 0 but its low end is within 2 sigma
    assert two_sigma_rule(0.01, 0.02, sigma) == FAILS  # lo == 2 sigma is not beyond it
    assert two_sigma_rule(math.nan, math.nan, sigma) == FAILS


# --- a larger set: the intervals are metrics.bootstrap's, the paired rule on real draws -----------------------------

WIDE_KEYS = [f"w{n}" for n in range(10)]


def wide_set(tmp_path):
    """Ten keys, truth tier n % 5 for key wn (every judge agrees). Candidates: s1d1-full-int8 one tier off on w1,
    `twin` one tier off on w2 (the paired difference straddles 0), `flat` low everywhere (far below)."""
    ids = ids_of(WIDE_KEYS)
    votes = {j: {i: label(LEVELS[int(i[1]) % 5]) for i in ids} for j in MAIN}

    def tiers(off_key):
        return {
            i: label(LEVELS[min(int(i[1]) % 5 + 1, 4)] if i.startswith(off_key + "-") else LEVELS[int(i[1]) % 5])
            for i in ids
        }

    preds = {
        "s1d1-full-int8": tiers("w1"),
        "twin": tiers("w2"),
        "flat": {i: label("low") for i in ids},
    }
    return build(tmp_path, WIDE_KEYS, votes, preds)


def test_intervals_are_those_of_metrics_bootstrap(tmp_path):
    results, meta, run, root = wide_set(tmp_path)
    computed = tables(results, meta, run, root, sigma=0.005)
    truth = np.array([LEVELS.index(v) for v in results["truth_all.complexity"].to_pylist()])
    pred = {
        c: np.array([LEVELS.index(v) for v in results[f"{c}.complexity"].to_pylist()])
        for c in ("s1d1-full-int8", "twin", "flat")
    }
    keys = list(dict.fromkeys(results["base_key"].to_pylist()))
    groups = {k: [i for i, b in enumerate(results["base_key"].to_pylist()) if b == k] for k in keys}

    def qwk_of(name):
        return lambda idx: metrics.qwk(truth[idx], pred[name][idx])

    row = one(computed["agreement_all"], name="twin")
    assert (row["qwk_lo"], row["qwk_hi"]) == pytest.approx(metrics.bootstrap(keys, groups, qwk_of("twin")))
    exact = metrics.bootstrap(keys, groups, lambda idx: metrics.accuracy(truth[idx], pred["twin"][idx]))
    assert (row["exact_lo"], row["exact_hi"]) == pytest.approx(exact)
    for other in ("twin", "flat"):
        found = one(computed["paired"], candidate=other)
        expected = metrics.paired_difference(keys, groups, qwk_of("s1d1-full-int8"), qwk_of(other))
        assert (found["difference"], found["lo"], found["hi"]) == pytest.approx(expected)


def test_paired_rule_on_real_draws(tmp_path):
    results, meta, run, root = wide_set(tmp_path)
    computed = tables(results, meta, run, root, sigma=0.005)
    twin = one(computed["paired"], candidate="twin")
    assert twin["lo"] < 0 < twin["hi"]  # straddles 0
    assert twin["rule_2sigma"] == FAILS
    flat = one(computed["paired"], candidate="flat")
    assert flat["lo"] > 2 * 0.005
    assert flat["rule_2sigma"] == PASSES
    # the same interval, which excludes 0, does not pass when 2 sigma is wider than its low end
    wide_sigma = tables(results, meta, run, root, sigma=flat["lo"])
    flat_wide = one(wide_sigma["paired"], candidate="flat")
    assert flat_wide["lo"] > 0
    assert flat_wide["rule_2sigma"] == FAILS


def test_refused_when_a_classifier_folder_or_the_random_draws_are_missing(tmp_path):
    results, meta, run, root = build(
        tmp_path / "a", KEYS, small_votes(), small_preds(), small_decisions(), folders=["s1d1-full-int8", "laya-rank"]
    )
    with pytest.raises(ValueError, match="notask"):
        tables(results, meta, run, root)
    results, meta, run, root = build(tmp_path / "b", KEYS, small_votes(), small_preds(), small_decisions())
    (root / "random" / "draws.jsonl").unlink()
    with pytest.raises(ValueError, match="draws.jsonl"):
        tables(results, meta, run, root)


def test_markdown_table_french_formatting():
    table = pa.table(
        {
            "name": ["a|b", "c"],
            "n_ids": [360, 90],
            "qwk": [0.81234, math.nan],
            "qwk_lo": [0.77149, math.nan],
            "qwk_hi": [0.8481, math.nan],
            "verified": [True, False],
            "note": [None, "x"],
        }
    )
    text = markdown_table(table)
    lines = text.splitlines()
    assert lines[0] == "| name | n_ids | qwk | verified | note |"
    assert lines[1] == "|---|---|---|---|---|"
    assert lines[2] == "| a\\|b | 360 | 0,812 [0,771 ; 0,848] | oui | — |"
    assert lines[3] == "| c | 90 | — | non | x |"
    assert markdown_table(table, columns=["name", "qwk_lo"], labels={"name": "candidat"}).splitlines()[0] == (
        "| candidat | qwk_lo |"
    )


def test_write_tables_and_determinism(small, tmp_path):
    computed, inputs = small
    first = write_tables(computed, tmp_path / "one")
    second = write_tables(tables(*inputs, sigma=0.005), tmp_path / "two")
    assert [p.name for p in first] == [p.name for p in second]
    assert all(p.parent == tmp_path / "one" / "tables" for p in first)
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name
    text = (tmp_path / "one" / "tables" / "agreement_all.csv").read_text(encoding="utf-8")
    header, s1d1 = text.splitlines()[0], next(line for line in text.splitlines() if line.startswith("s1d1-full-int8"))
    cells = dict(zip(header.split(","), s1d1.split(","), strict=True))
    # full precision ('.' as the decimal point): the cell reads back as the very float of the table
    assert float(cells["qwk"]) == one(computed["agreement_all"], name="s1d1-full-int8")["qwk"] == pytest.approx(6 / 7)
    by_source = (tmp_path / "one" / "tables" / "by_source.csv").read_text(encoding="utf-8").splitlines()
    head = by_source[0].split(",")
    row = next(line.split(",") for line in by_source[1:] if line.startswith("s1d1-full-int8,candidate,oasst2"))
    assert row[head.index("qwk")] == ""  # nan is written empty
