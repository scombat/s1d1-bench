import csv
import math
import re
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from s1d1_bench import metrics
from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.report import (
    CEILING_NOTE,
    CONSTANT_SCORE,
    FAILS,
    IN_TRUTH,
    MACRO_F1_NOTE,
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
    medium, the rest of j-a); k2: all trivial; j-c abstains on k2-zho; the non-main judges vary, j-a-off votes null
    (not an abstention) on k1-eng."""
    votes = {j: {} for j in MAIN + OTHERS}
    for lang in LANGS:
        votes["j-a"][f"k1-{lang}"] = label("high")
        votes["j-b"][f"k1-{lang}"] = label("low")
        votes["j-c"][f"k1-{lang}"] = label("high")
        votes["j-a-off"][f"k1-{lang}"] = label("medium" if lang != "eng" else None)  # a null vote, no abstention
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


def fake_profile(run: dict) -> dict:
    """The profile of a fake set (`bench profile`): every classifier of its run record on cpu, every judge on the
    GPU through MLX, without peak RSS for the judges."""
    rows = []
    for n, entry in enumerate(run["classifiers"]):
        rows.append(
            {
                "name": entry["name"], "kind": "classifier", "model": entry["model"], "revision": "r", "device": "cpu",
                "weights_mb": 10.0 * (n + 1), "latency_p50_ms": 1.0 + n, "latency_p95_ms": 3.0 + n,
                "latency_mean_ms": 1.5 + n, "peak_rss_mb": 100.0 + n, "note": "fake classifier",
            }
        )  # fmt: skip
    for n, entry in enumerate(run["judges"]):
        rows.append(
            {
                "name": entry["name"], "kind": "judge", "model": entry["model"], "revision": "r", "device": "gpu-mlx",
                "weights_mb": 1000.0 * (n + 1), "latency_p50_ms": 1000.0 * (n + 1),
                "latency_p95_ms": 1500.0 * (n + 1), "latency_mean_ms": 1100.0 * (n + 1), "peak_rss_mb": None,
                "note": "fake judge",
            }
        )  # fmt: skip
    meta = {"machine": "fake machine, Fake M0", "batch_size": 1, "warmup": 10, "date": "2026-10-31", "skipped": []}
    return {"meta": meta, "rows": rows}


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    results, meta, run, root = build(
        tmp_path_factory.mktemp("small"), KEYS, small_votes(), small_preds(), small_decisions()
    )
    return tables(results, meta, run, root, sigma=0.005, profile=fake_profile(run)), (results, meta, run, root)


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
        "confusion", "judge_spread_by_tier", "inputs", "inputs_summary", "skipped",
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
    assert rank["note"] is None  # a score that varies
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
    assert task["note"] == MACRO_F1_NOTE
    needs = one(computed["needs_tools"], name="s1d1-full-int8")
    assert needs["n_ids_positive"] == 0  # no id needs tools in the truth
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
    assert one(table, judge_a="j-b", judge_b="j-c")["n_ids_left_out"] == 1


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
    jc = one(inputs, name="j-c")
    assert (jc["n_ids_abstained"], jc["n_ids_null_vote"], jc["null_votes_ids"]) == (1, 1, "k2-zho")
    off = one(inputs, name="j-a-off")  # a null vote that is not an abstention is counted too
    assert (off["n_ids_abstained"], off["n_ids_null_vote"], off["null_votes_ids"]) == (0, 1, "k1-eng")
    assert one(inputs, name="j-b")["null_votes_ids"] == ""
    assert one(inputs, name="s1d1-full-int8")["null_votes_ids"] is None
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


def test_profile_table_takes_every_row_of_the_json(small):
    computed, (_, _, run, _) = small
    profile = fake_profile(run)
    found = computed["profile"].to_pylist()
    assert [r["name"] for r in found] == [r["name"] for r in profile["rows"]]  # the order of the file
    for row, source in zip(found, profile["rows"], strict=True):
        assert {key: row[key] for key in source} == source, row["name"]  # every key of the row, untouched
    assert one(computed["profile"], name="j-a")["role"] == IN_TRUTH
    assert one(computed["profile"], name="j-a-off")["role"] == OUTSIDE
    assert one(computed["profile"], name="s1d1-full-int8")["role"] == "candidate"
    assert {(r["machine"], r["batch_size"]) for r in found} == {("fake machine, Fake M0", 1)}  # from its meta


def test_tier_distribution_one_row_per_candidate_and_tier_plus_the_truth(small):
    computed, (_, _, run, _) = small
    found = computed["tier_distribution"].to_pylist()
    names = ["truth_all"] + [entry["name"] for entry in run["classifiers"]]
    assert [r["name"] for r in found] == [name for name in names for _ in LEVELS]  # the truth first
    assert [r["tier"] for r in found] == LEVELS * len(names)
    for name in names:
        assert math.isclose(sum(r["share"] for r in found if r["name"] == name), 1.0), name
    # k1: truth high (4 ids), k2: truth trivial (4 ids); s1d1: high, high, medium, medium, trivial, trivial, low, low
    truth = {r["tier"]: r["share"] for r in found if r["name"] == "truth_all"}
    assert truth == {"trivial": 0.5, "low": 0.0, "medium": 0.0, "high": 0.5, "expert": 0.0}
    s1d1 = {r["tier"]: (r["n_ids"], r["n_ids_total"]) for r in found if r["name"] == "s1d1-full-int8"}
    assert s1d1 == {"trivial": (2, 8), "low": (2, 8), "medium": (2, 8), "high": (2, 8), "expert": (0, 8)}
    assert {r["role"] for r in found} == {"truth", "candidate"}


def test_tables_without_a_profile_have_no_profile_table(small):
    computed, inputs = small
    bare = tables(*inputs, sigma=0.005)
    assert list(bare) == [name for name in TABLES if name != "profile"]  # the profile table only is left out
    assert bare["tier_distribution"].to_pylist() == computed["tier_distribution"].to_pylist()
    with pytest.raises(ValueError, match="rows"):
        tables(*inputs, sigma=0.005, profile={"meta": {}})
    with pytest.raises(ValueError, match="latency_p50_ms"):
        row = {"name": "x", "kind": "classifier", "device": "cpu", "weights_mb": 1}
        tables(*inputs, sigma=0.005, profile={"rows": [row]})


# --- a larger set: the intervals are metrics.bootstrap's, the paired rule on real draws -----------------------------

WIDE_KEYS = [f"w{n}" for n in range(10)]


def wide_set(tmp_path):
    """Ten keys, truth tier n % 5 for key wn (every judge agrees). Candidates: s1d1-full-int8 one tier off on w1,
    `twin` one tier off on w2 (the paired difference straddles 0), `flat` low everywhere (far below), score 0.25."""
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
        "flat": {i: label("low", score=0.25) for i in ids},
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
    # the number of draws used goes with the interval: hidden unless asked for
    used = table.append_column("qwk_draws_used", pa.array([1000, 0]))
    assert markdown_table(used).splitlines()[0] == "| name | n_ids | qwk | verified | note |"
    assert markdown_table(used, columns=["qwk_draws_used"]).splitlines()[2] == "| 1000 |"


def test_write_tables_and_determinism(small, tmp_path):
    computed, inputs = small
    first = write_tables(computed, tmp_path / "one")
    second = write_tables(tables(*inputs, sigma=0.005, profile=fake_profile(inputs[2])), tmp_path / "two")
    assert [p.name for p in first] == [p.name for p in second]
    assert all(p.parent == tmp_path / "one" / "tables" for p in first)
    for a, b in zip(first, second, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name

    def read(name):
        with open(tmp_path / "one" / "tables" / f"{name}.csv", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    s1d1 = next(row for row in read("agreement_all") if row["name"] == "s1d1-full-int8")
    # full precision ('.' as the decimal point): the cell reads back as the very float of the table
    assert float(s1d1["qwk"]) == one(computed["agreement_all"], name="s1d1-full-int8")["qwk"] == pytest.approx(6 / 7)
    row = next(r for r in read("by_source") if (r["name"], r["source"]) == ("s1d1-full-int8", "oasst2"))
    assert row["qwk"] == ""  # nan is written empty
    assert row["qwk_draws_used"] == "0"


def test_rank_metrics_mark_a_constant_score(tmp_path):
    computed = tables(*wide_set(tmp_path), sigma=0.005)
    flat = one(computed["rank_metrics"], name="flat")
    assert flat["note"] == CONSTANT_SCORE
    assert flat["scale"] == "constant (anchor of low)"  # its one value, 0.25, is the v7 anchor of low: not a raw score
    assert math.isnan(flat["spearman"])
    assert flat["auroc_high"] == pytest.approx(0.5)  # no discrimination: kept, and marked


def test_subgroup_qwk_drops_the_draws_with_a_single_truth_tier(tmp_path):
    """Three keys: k1 (oasst2, truth high), k2 (handwritten, medium), k3 (oasst2, trivial). In oasst2, a draw that holds
    k1 but not k3 (or the reverse) has a single truth tier there: metrics.qwk would give exactly 0 whatever the
    prediction. Those draws are dropped (nan); only the draws holding both k1 and k3 make the interval."""
    keys = ["k1", "k2", "k3"]
    truth_of = {"k1": "high", "k2": "medium", "k3": "trivial"}
    ids = ids_of(keys)
    votes = {j: {i: label(truth_of[i[:2]]) for i in ids} for j in MAIN}
    predicted = {
        "k1": ["high", "high", "medium", "medium"],
        "k2": ["medium"] * 4,
        "k3": ["trivial", "trivial", "low", "low"],
    }
    preds = {"s1d1-full-int8": {f"{k}-{lang}": label(predicted[k][n]) for k in keys for n, lang in enumerate(LANGS)}}
    results, meta, run, root = build(tmp_path, keys, votes, preds)
    row = one(tables(results, meta, run, root)["by_source"], name="s1d1-full-int8", source="oasst2")
    truth = np.array([LEVELS.index(v) for v in results["truth_all.complexity"].to_pylist()])
    pred = np.array([LEVELS.index(v) for v in results["s1d1-full-int8.complexity"].to_pylist()])
    base = results["base_key"].to_pylist()
    groups = {k: [i for i, b in enumerate(base) if b == k] for k in keys}
    oasst2 = set(groups["k1"] + groups["k3"])
    kept, degenerate = [], 0
    for draw in metrics.bootstrap_draws(keys, groups, 1000, 0):
        rows_in = [i for i in draw if i in oasst2]
        if len(set(truth[rows_in].tolist())) >= 2:
            kept.append(metrics.qwk(truth[rows_in], pred[rows_in]))
        else:
            degenerate += 1
    assert 0 < degenerate < 1000  # both kinds of draw occur
    assert row["qwk_draws_used"] == len(kept)
    assert (row["qwk_lo"], row["qwk_hi"]) == pytest.approx(metrics.percentile_interval(kept))
    assert row["qwk_lo"] > 0
    # counting the degenerate draws as 0 would have pulled the low end down to 0
    assert metrics.percentile_interval(kept + [0.0] * degenerate)[0] == 0.0
    assert row["exact_draws_used"] == 1000 - sum(
        1 for draw in metrics.bootstrap_draws(keys, groups, 1000, 0) if not oasst2 & set(draw)
    )


def test_confusion_and_judge_spread_tables(small):
    computed, _ = small
    confusion = computed["confusion"]
    assert len(confusion) == 3 * 25  # every candidate, the full 5 x 5 grid
    cells = {(r["truth_tier"], r["pred_tier"]): r for r in rows(confusion, candidate="s1d1-full-int8")}
    # truth high: high 2, medium 2; truth trivial: trivial 2, low 2 (row-normalised per truth tier)
    assert (cells["high", "high"]["n_ids"], cells["high", "high"]["row_share"]) == (2, pytest.approx(0.5))
    assert (cells["high", "medium"]["n_ids"], cells["high", "medium"]["row_share"]) == (2, pytest.approx(0.5))
    assert (cells["trivial", "low"]["n_ids"], cells["trivial", "low"]["row_share"]) == (2, pytest.approx(0.5))
    assert cells["medium", "medium"]["n_ids"] == 0 and cells["medium", "medium"]["row_share"] == 0.0
    assert sum(r["n_ids"] for r in rows(confusion, candidate="s1d1-full-int8")) == 8
    spread = computed["judge_spread_by_tier"]
    assert len(spread) == 25  # 5 tiers x spread 0..4
    # k1 (truth high): j-a high, j-b low, j-c high -> spread 2 on its 4 ids; k2 (trivial): all trivial -> 0
    high = one(spread, tier="high", spread=2)
    assert (high["n_ids"], high["n_ids_tier"], high["mean_spread_tier"], high["n_ids_null_spread_tier"]) == (
        4, 4, pytest.approx(2.0), 0,
    )  # fmt: skip
    assert one(spread, tier="trivial", spread=0)["n_ids"] == 4
    assert math.isnan(one(spread, tier="expert", spread=0)["mean_spread_tier"])


# --- 10b: the figures, the index and the command ----------------------------------------------------------------------

TEMPLATE = Path(__file__).resolve().parents[1] / "report" / "TEMPLATE.md"
CAVEATS = (
    "Plafond = accord de chaque juge principal avec la médiane des autres",
    "il ne dépasse pas la grille",
    "Plancher = les témoins",
    "(a) Si la grille elle-même correspond à « quel modèle suffit »",
    "(b) Le routage",
    "(c) La distribution réelle des prompts",
    "(d) Les prompts agentiques et multi-tours",
    "(e) Les langues autres que les quatre choisies",
)


def fake_report_inputs(tmp_path: Path):
    """The amendment's fake set on disk: 2 keys x 4 languages x the 3 main judges x 3 classifiers (s1d1-like, rank-only
    through `.raw_score`, one without task), the classifiers root with random/draws.jsonl; returns the CLI arguments."""
    import json

    import pyarrow.parquet as pq

    votes = {j: v for j, v in small_votes().items() if j in MAIN}
    results, meta, run, root = build(tmp_path, KEYS, votes, small_preds(), small_decisions())
    (tmp_path / "results").mkdir()
    pq.write_table(results, tmp_path / "results" / "fake.parquet")
    (tmp_path / "results" / "fake.run.json").write_text(json.dumps(run), encoding="utf-8")
    profile = json.dumps(fake_profile(run))
    (tmp_path / "results" / "fake.profile.json").write_text(profile, encoding="utf-8")  # read by default
    pq.write_table(meta, tmp_path / "fake.meta.parquet")
    results_path, meta_path = tmp_path / "results" / "fake.parquet", tmp_path / "fake.meta.parquet"
    args = ["report", "--results", str(results_path), "--meta", str(meta_path), "--classifiers", str(root)]
    args += ["--template", str(TEMPLATE)]
    return args, (results, meta, run, root)


def markdown_tables(text: str) -> list[list[list[str]]]:
    """Every Markdown table of `text` as a list of rows (header first), each a list of cells."""
    found, current = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            if not line.startswith("|---"):
                current.append([c.strip() for c in line.strip("|").split(" | ")])
        elif current:
            found.append(current)
            current = []
    return found


def test_fill_refuses_a_hole_without_value_and_a_value_without_hole():
    from s1d1_bench.report_index import fill

    assert fill("a {x} b {x} {y_2}", {"x": "1", "y_2": "{z}"}) == "a 1 b 1 {z}"  # a value is never filled again
    with pytest.raises(ValueError, match="unknown_hole"):
        fill("a {unknown_hole}", {})
    with pytest.raises(ValueError, match="extra"):
        fill("a {x}", {"x": "1", "extra": "2"})


def test_write_index_raises_when_a_hole_of_the_template_stays_empty(small, tmp_path):
    from s1d1_bench.report_figures import NAMES, Figure
    from s1d1_bench.report_index import write_index

    computed, (_, _, run, root) = small
    absent = [Figure(name, None, None, ("rien à tracer",)) for name in NAMES]
    with pytest.raises(ValueError, match="figure_qwk_bars"):  # no figure given: its hole has no value
        write_index(tmp_path, computed, [], run, TEMPLATE)
    template = tmp_path / "template.md"
    template.write_text(TEMPLATE.read_text(encoding="utf-8") + "\n{unknown_hole}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown_hole"):
        write_index(tmp_path, computed, absent, run, template)
    assert not (tmp_path / "index.md").exists()
    # the real template, every figure absent: written, nothing left unfilled
    text = write_index(tmp_path, computed, absent, run, TEMPLATE, classifiers_root=root).read_text(encoding="utf-8")
    from s1d1_bench.report_index import HOLE

    assert HOLE.search(text) is None
    assert "_Figure `qwk-bars` non tracée : rien à tracer._" in text


def test_write_index_refuses_a_superiority_sentence(small, tmp_path):
    from s1d1_bench.report_figures import NAMES, Figure
    from s1d1_bench.report_index import write_index

    computed, (_, _, run, _) = small
    absent = [Figure(name, None, None, ()) for name in NAMES]
    with pytest.raises(ValueError, match="meilleur que"):  # the set name lands in the title
        write_index(tmp_path, computed, absent, run | {"prompts": "data/prompts/meilleur que.jsonl"}, TEMPLATE)
    with pytest.raises(ValueError, match="better than"):  # the first section is in English now
        write_index(tmp_path, computed, absent, run | {"prompts": "data/prompts/better than.jsonl"}, TEMPLATE)
    template = TEMPLATE.read_text(encoding="utf-8").lower()
    assert "meilleur que" not in template and "mieux que" not in template and "better than" not in template


def test_report_command_on_the_fake_set(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from s1d1_bench.cli import main
    from s1d1_bench.report import french
    from s1d1_bench.report_figures import NAMES
    from s1d1_bench.report_index import HOLE

    args, inputs = fake_report_inputs(tmp_path)
    out = tmp_path / "out"
    assert main([*args, "--out", str(out)]) == 0
    assert "index.md" in capsys.readouterr().out
    assert {p.stem for p in (out / "tables").glob("*.csv")} == set(TABLES)
    svgs = sorted((out / "figures").glob("*.svg"))
    assert {p.stem for p in svgs} == set(NAMES)  # every figure has something to draw here
    for svg in svgs:
        assert svg.with_suffix(".csv").is_file(), svg.name
        assert b"<dc:date>" not in svg.read_bytes()  # no date in the metadata
    assert {p.stem for p in (out / "figures").glob("*.csv")} == {p.stem for p in svgs}

    text = (out / "index.md").read_text(encoding="utf-8")
    assert HOLE.search(text) is None
    for caveat in CAVEATS:
        assert caveat in text, caveat
    assert "meilleur que" not in text.lower() and "mieux que" not in text.lower()
    # the sentence of docs/32 § 5 with its numbers: s1d1 QWK 6/7 against truth_all, the ceiling mean, 8 ids, 2 keys
    computed = tables(*inputs, sigma=0.005)
    mean = one(computed["ceiling"], judge="mean")
    assert (
        "Sur 8 ids, soit 2 prompts de base en 4 langues, `s1d1-full-int8` s'accorde avec la médiane de\n> 3 juges"
        in text
    )
    assert f"QWK {french(6 / 7)} " in text and french(6 / 7) == "0,857"
    assert f"le plafond inter-juges est {french(mean['qwk'])}" in text
    assert "Laya english absent de ce run" in text  # an absent candidate fills its hole, said as such
    assert "La règle des 2 σ" in text and "borne basse > 2 σ = 0,010" in text
    # the absent roles and panels are said, never a crash
    assert "Prediction 1 was not evaluated" in text and "Prediction 2 was not evaluated" in text
    assert "figure `confusion` : `laya-multilingual-tier` absent : pas de panneau de confusion" in text
    assert "plancher, `length` : pas de ligne de plancher" in text
    # the captions under the figures, filled from the tables and the profile (fake: s1d1 1 ms on cpu, judges 1-3 s)
    assert "`s1d1-full-int8` répond par prompt en 1,0 ms sur le CPU." in text  # judge-level QWK: the headline only
    assert "Les variantes int8 de s1d1, sur le CPU, répondent en 1,0 ms ; leur QWK se lit point par point." in text
    assert "Un juge principal met, sur le GPU du même Mac et en lisant la consigne, 1,0 à 3,0 s." in text
    assert "la cause de\ncette queue n'y est pas mesurée" in text  # no causal claim on the p95 tail
    assert "Les poids de `s1d1-full-int8` font 10 Mo ;" in text
    assert "ceux des variantes de s1d1 tiennent dans 10 Mo" in text
    assert "les juges principaux demandent 1,0 à 3,0 Go." in text
    assert "ne sont pas des modèles : ils n'ont\nni latence ni poids, et ne sont ni dans cette figure" in text
    truth_line = "- `truth_all` : 2 paliers sur 5, jamais `low`, `medium` et `expert`"
    assert f"{truth_line} ; plus grande part : `trivial` 50,0 %" in text

    # the rank-only candidate has neither MAE nor ECE anywhere, the candidate without task no task row
    for path in [*(out / "tables").glob("*.csv"), *(out / "figures").glob("*.csv")]:
        with open(path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = set(reader.fieldnames or [])
            rows_here = list(reader)
        if columns & {"mae", "ece"}:
            assert all("laya-rank" not in row.values() for row in rows_here), path.name
        if path.stem in ("task", "needs_tools"):
            assert all(row["name"] != "notask" for row in rows_here), path.name
    for table in markdown_tables(text):
        header, body = table[0], table[1:]
        names = {row[0] for row in body}
        if "MAE" in header or "ECE" in header:
            assert "laya-rank" not in names and "s1d1-full-int8" in names
        if "F1 macro" in header or "ids positifs" in header:
            assert "notask" not in names
    assert "laya-rank" in {row[0] for t in markdown_tables(text) if "AUROC (high ou expert)" in t[0] for row in t[1:]}


def test_figure_csv_headers_are_english_like_the_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from s1d1_bench.cli import main
    from s1d1_bench.report_figures import HEADERS

    args, _ = fake_report_inputs(tmp_path)
    out = tmp_path / "out"
    assert main([*args, "--out", str(out)]) == 0
    headers = {}
    for path in sorted((out / "figures").glob("*.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            headers[path.stem] = next(csv.reader(f))
    for stem, header in headers.items():
        assert len(set(header)) == len(header), stem
        assert set(header) <= set(HEADERS.values()), stem  # every header is one of the readable English words
    assert headers["qwk-bars"] == [
        "kind", "name", "family", "value", "lower bound", "upper bound", "draws used",
    ]  # fmt: skip
    with open(out / "figures" / "language-disagreements.csv", encoding="utf-8", newline="") as f:
        first = next(csv.DictReader(f))
    assert (first["group"], first["value"]) == ("all", "all")
    with open(out / "figures" / "qwk-bars.csv", encoding="utf-8", newline="") as f:
        names = {row["kind"]: row["name"] for row in csv.DictReader(f)}
    assert names["ceiling (mean)"] == "mean"
    with open(out / "tables" / "ceiling.csv", encoding="utf-8", newline="") as f:
        assert "judge" in next(csv.reader(f))  # the tables keep their names: only the figure CSVs are translated


def test_report_svg_byte_identical_across_two_runs(tmp_path):
    pytest.importorskip("matplotlib")
    from s1d1_bench.cli import main

    args, _ = fake_report_inputs(tmp_path)
    assert main([*args, "--out", str(tmp_path / "one")]) == 0
    assert main([*args, "--out", str(tmp_path / "two")]) == 0
    first = sorted((tmp_path / "one").rglob("*.*"))
    assert any(p.suffix == ".svg" for p in first)
    for path in first:
        twin = tmp_path / "two" / path.relative_to(tmp_path / "one")
        assert path.read_bytes() == twin.read_bytes(), path.name


def test_figures_skip_a_figure_with_nothing_to_draw(small, tmp_path):
    pytest.importorskip("matplotlib")
    from s1d1_bench.report_figures import figures

    computed, _ = small
    empty = dict(computed)
    empty["calibration_bins"] = computed["calibration_bins"].slice(0, 0)
    stale = tmp_path / "figures" / "reliability-s1d1.svg"
    stale.parent.mkdir(parents=True)
    stale.write_text("old", encoding="utf-8")
    drawn = {f.name: f for f in figures(empty, tmp_path)}
    reliability = drawn["reliability-s1d1"]
    assert reliability.svg is None and reliability.csv is None and reliability.notes
    assert not stale.exists()  # a stale file of a figure not drawn is removed
    confusion = drawn["confusion"]
    assert confusion.svg.is_file()
    assert any("litellm-heuristic" in note for note in confusion.notes)  # the panels of absent candidates


def test_report_command_refusals(tmp_path, capsys, monkeypatch):
    import sys

    from s1d1_bench.cli import main

    args, (_, _, _, root) = fake_report_inputs(tmp_path)
    out = ["--out", str(tmp_path / "out")]
    # matplotlib missing: exit 1 with the hint, before any computation
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    assert main([*args, *out]) == 1
    assert "uv run --group report" in capsys.readouterr().err
    monkeypatch.undo()
    pytest.importorskip("matplotlib")
    (root / "notask").rmdir()
    assert main([*args, *out]) == 1
    assert "notask" in capsys.readouterr().err
    (root / "notask").mkdir()
    (tmp_path / "results" / "fake.run.json").unlink()
    assert main([*args, *out]) == 1
    assert "fake.run.json" in capsys.readouterr().err
    assert not (tmp_path / "out" / "index.md").exists()


PROFILE_FIGURES = ("quality-vs-latency", "quality-vs-footprint")


def _figure_names(path: Path) -> set[str]:
    with open(path, encoding="utf-8", newline="") as f:
        return {row["name"] for row in csv.DictReader(f)}


def test_report_command_without_a_profile_skips_the_profile_table_and_figures(tmp_path):
    pytest.importorskip("matplotlib")
    from s1d1_bench.cli import main
    from s1d1_bench.report_figures import NAMES

    args, _ = fake_report_inputs(tmp_path)
    (tmp_path / "results" / "fake.profile.json").unlink()
    out = tmp_path / "out"
    (out / "tables").mkdir(parents=True)
    (out / "tables" / "profile.csv").write_text("stale", encoding="utf-8")
    assert main([*args, "--out", str(out)]) == 0
    assert not (out / "tables" / "profile.csv").exists()  # a stale profile table is removed
    assert {p.stem for p in (out / "tables").glob("*.csv")} == set(TABLES) - {"profile"}
    assert {p.stem for p in (out / "figures").glob("*.svg")} == set(NAMES) - set(PROFILE_FIGURES)
    assert (out / "figures" / "tier-distribution.svg").is_file()  # it needs no profile
    text = (out / "index.md").read_text(encoding="utf-8")
    for name in PROFILE_FIGURES:
        assert f"_Figure `{name}` non tracée : pas de profil" in text, name
    assert "`s1d1-full-int8` répond par prompt en non mesuré (pas de profil)." in text
    assert "répondent en non mesuré (pas de profil) ;" in text
    assert "_(pas de profil : `results/<jeu>.profile.json`" in text  # in place of the profile table


def test_report_command_reads_a_named_profile_and_refuses_a_missing_one(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    from s1d1_bench.cli import main

    args, _ = fake_report_inputs(tmp_path)
    moved = tmp_path / "elsewhere.profile.json"
    (tmp_path / "results" / "fake.profile.json").rename(moved)
    out = tmp_path / "out"
    assert main([*args, "--profile", str(moved), "--out", str(out)]) == 0
    assert (out / "tables" / "profile.csv").is_file()
    assert (out / "figures" / "quality-vs-latency.svg").is_file()
    capsys.readouterr()
    assert main([*args, "--profile", str(tmp_path / "nowhere.profile.json"), "--out", str(out)]) == 1
    assert "nowhere.profile.json: missing (the profile of bench profile)" in capsys.readouterr().err


def test_a_run_candidate_missing_from_the_profile_or_the_inverse_is_said(tmp_path):
    """Review focus 3: the profile lacks a candidate and a main judge of the run, and has a model the run does not
    know; the report says which, under the two profile figures, and draws them without those points."""
    pytest.importorskip("matplotlib")
    import json

    from s1d1_bench.cli import main

    args, (_, _, run, _) = fake_report_inputs(tmp_path)
    profile = fake_profile(run)
    ghost = profile["rows"][0] | {"name": "ghost-model"}
    profile["rows"] = [r for r in profile["rows"] if r["name"] not in ("laya-rank", "j-b")] + [ghost]
    (tmp_path / "results" / "fake.profile.json").write_text(json.dumps(profile), encoding="utf-8")
    out = tmp_path / "out"
    assert main([*args, "--out", str(out)]) == 0
    text = (out / "index.md").read_text(encoding="utf-8")
    for figure in PROFILE_FIGURES:
        names = _figure_names(out / "figures" / f"{figure}.csv")
        assert {"s1d1-full-int8", "notask", "j-a", "j-c"} <= names, figure
        assert not names & {"laya-rank", "j-b", "ghost-model"}, figure
        for name in ("laya-rank", "j-b"):
            assert f"- figure `{figure}` : `{name}` absent du profil : pas de point" in text, (figure, name)
        assert f"- figure `{figure}` : `ghost-model` est dans le profil mais pas dans le run : pas de point" in text
    assert one(tables(*_inputs_of(tmp_path), profile=profile)["profile"], name="ghost-model")["role"] == (
        "not in the run"
    )


def _inputs_of(tmp_path: Path):
    """The results, meta, run record and classifiers root that `fake_report_inputs` wrote under `tmp_path`."""
    import json

    import pyarrow.parquet as pq

    run = json.loads((tmp_path / "results" / "fake.run.json").read_text(encoding="utf-8"))
    results = pq.read_table(tmp_path / "results" / "fake.parquet")
    return results, pq.read_table(tmp_path / "fake.meta.parquet"), run, tmp_path / "classifiers"


def test_report_without_numpy_exits_1_with_the_hint(tmp_path, capsys, monkeypatch):
    import sys

    from s1d1_bench.cli import main

    monkeypatch.delitem(sys.modules, "s1d1_bench.report")
    monkeypatch.setitem(sys.modules, "numpy", None)
    code = main(["report", "--results", "r.parquet", "--meta", "m.parquet", "--out", str(tmp_path)])
    assert code == 1
    err = capsys.readouterr().err
    assert "numpy" in err and "uv run --group report bench report" in err


def _slug(heading: str) -> str:
    """The anchor GitHub gives a Markdown heading: lower case, punctuation dropped, spaces as hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def test_index_links_its_tables_and_says_what_it_does_not_measure(tmp_path):
    pytest.importorskip("matplotlib")
    from s1d1_bench.cli import main

    args, _ = fake_report_inputs(tmp_path)
    out = tmp_path / "out"
    assert main([*args, "--out", str(out)]) == 0
    text = (out / "index.md").read_text(encoding="utf-8")
    anchors = {_slug(line.lstrip("#")) for line in text.splitlines() if line.startswith("#")}
    links = re.findall(r"\]\(([^)\s]+)\)", text)
    for link in links:
        if link.startswith("#"):
            assert link[1:] in anchors, link
        elif link.startswith("../../docs/"):
            # leaves the report folder: resolved from where the committed report lives, not from the temporary one
            assert (Path(__file__).resolve().parents[1] / "report" / "base-v1" / link).is_file(), link
        else:
            assert (out / link).is_file(), link
    # the numbers of the predictions and of the sentence sit next to the links of their tables
    for table in ("predictions", "by_language", "agreement_all", "ceiling", "paired", "by_tier"):
        assert f"(tables/{table}.csv)" in text, table
    assert "#les-deux-prédictions-en-tableau" in links
    # the breakdowns are for the tier only; the three analyses of docs/32 not produced are listed
    assert "la tâche, `needs_tools` et le score le sont globalement seulement" in text
    missing = text.split("## Ce qui manque")[1]
    for item in ("(i) la part des prompts notés 0,70 à 0,80", "(ii) les confusions entre tâches voisines", "(iii)"):
        assert item in missing, item
    assert "Rien" not in missing


def test_predictions_wording_reads_the_point_values_and_the_overlap():
    from s1d1_bench.report_index import _predictions_first

    def row(prediction, name, lang, qwk, lo, hi, verified):
        return {"prediction": prediction, "name": name, "lang": lang, "qwk": qwk, "qwk_lo": lo, "qwk_hi": hi,
                "verified": verified}  # fmt: skip

    rows = [
        row("prediction_2", "length", "all", 0.54, 0.42, 0.65, False),
        row("prediction_2", "litellm-heuristic", "all", 0.09, 0.03, 0.17, False),
        row("prediction_2", "s1d1-full-int8", "all", 0.81, 0.73, 0.85, False),
        row("prediction_1", "litellm-heuristic", "eng", 0.20, 0.09, 0.33, False),
        row("prediction_1", "litellm-heuristic", "fra", 0.25, 0.10, 0.40, False),
        row("prediction_1", "litellm-heuristic", "rus", 0.30, 0.12, 0.45, False),
        row("prediction_1", "litellm-heuristic", "zho", 0.04, -0.02, 0.08, False),
    ]
    computed = {
        "predictions": pa.Table.from_pylist(rows),
        "by_language": pa.Table.from_pylist([{"name": "litellm-heuristic", "n_ids_all": 90}]),
    }
    text = _predictions_first(computed, "360 ids", "90 prompts de base")
    first, second = text.split("\n\n")[:2]
    assert first.startswith("**Prediction 2 does not hold: `litellm-heuristic` (0.090 [0.030, 0.170]) is below")
    assert "should sit between `length` and `s1d1-full-int8`" in first
    assert "1,000 draws of the 90 base prompts" in first  # the French count of the summary, said in English
    assert "QWK(" not in text  # said in words, not as a formula repeated
    assert "the CIs of `litellm-heuristic` and `length` do not overlap" in first
    assert second.startswith("**Prediction 1 does not hold: the QWKs in fra and rus are not below")
    assert "the CIs of fra and rus overlap that of eng; the CI of zho does not overlap that of eng" in second
    assert "−0.020" in second  # the true minus sign in prose numbers


def test_report_command_refuses_a_missing_template_before_computing(tmp_path, capsys, monkeypatch):
    pytest.importorskip("matplotlib")
    from s1d1_bench import report
    from s1d1_bench.cli import main

    args, _ = fake_report_inputs(tmp_path)
    args[args.index("--template") + 1] = str(tmp_path / "nowhere.md")
    monkeypatch.setattr(report, "tables", lambda *a, **k: pytest.fail("the tables were computed"))
    assert main([*args, "--out", str(tmp_path / "out")]) == 1
    assert "nowhere.md: missing (the template of index.md)" in capsys.readouterr().err


@pytest.mark.parametrize("key", ["main", "judges", "classifiers"])
def test_report_command_refuses_a_run_record_without_a_key(tmp_path, capsys, key):
    pytest.importorskip("matplotlib")
    import json

    from s1d1_bench.cli import main

    args, (_, _, run, _) = fake_report_inputs(tmp_path)
    del run[key]
    (tmp_path / "results" / "fake.run.json").write_text(json.dumps(run), encoding="utf-8")
    assert main([*args, "--out", str(tmp_path / "out")]) == 1
    assert f"the run record has no key {key!r}" in capsys.readouterr().err


@pytest.mark.parametrize("column", ["source", "origin"])
def test_report_command_refuses_a_meta_table_without_a_column(tmp_path, capsys, column):
    pytest.importorskip("matplotlib")
    import pyarrow.parquet as pq

    from s1d1_bench.cli import main

    args, (_, meta, _, _) = fake_report_inputs(tmp_path)
    pq.write_table(meta.drop_columns([column]), tmp_path / "fake.meta.parquet")
    assert main([*args, "--out", str(tmp_path / "out")]) == 1
    assert f"the meta table has no column {column!r}" in capsys.readouterr().err


def test_caption_spans_say_the_unit_once_when_both_ends_share_it():
    from s1d1_bench.report_index import _latency, _span, _weight

    assert _span([9.114, 1.646, 5.8], _latency, "-") == "1,6 à 9,1 ms"
    assert _span([0.123, 1821.0], _latency, "-") == "0,1 ms à 1,8 s"
    assert _span([42786.5, 1821.0], _latency, "-") == "1,8 à 42,8 s"
    assert _span([24.7, 615.1], _weight, "-") == "25 à 615 Mo"
    assert _span([12076.2, 33763.1], _weight, "-") == "12,1 à 33,8 Go"
    assert _span([310.3], _weight, "-") == "310 Mo" and _span([], _weight, "none") == "none"


def test_constant_scores_are_named_from_the_data_with_their_anchor():
    from s1d1_bench.report import constant_scale
    from s1d1_bench.report_index import _constant_scores

    assert constant_scale(0.25) == "constant (anchor of low)"
    assert constant_scale(0.4) == "constant"  # one value, but no tier's anchor
    raw = {"name": "length", "scale": RAW}
    assert _constant_scores([raw]) == "aucun candidat n'a un score constant."
    low = {"name": "constant-low", "scale": constant_scale(0.25)}
    assert _constant_scores([raw, low]) == (
        "`constant-low` a un score constant (l'ancre de low) : ni rang ni AUROC ne le départagent."
    )
    odd = {"name": "odd", "scale": constant_scale(0.4)}
    assert _constant_scores([low, odd]) == (
        "`constant-low` (l'ancre de low) et `odd` ont un score constant : ni rang ni AUROC ne les départagent."
    )


def test_markdown_numbers_take_the_true_minus_sign_and_the_csv_keeps_the_hyphen(tmp_path):
    table = pa.table({"qwk": [-0.1234], "qwk_lo": [-0.2], "qwk_hi": [0.05]})
    assert markdown_table(table).splitlines()[2] == "| −0,123 [−0,200 ; 0,050] |"
    (path,) = write_tables({"t": table}, tmp_path)
    assert path.read_text(encoding="utf-8").splitlines()[1] == "-0.1234,-0.2,0.05"
