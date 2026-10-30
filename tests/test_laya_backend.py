import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1d1_bench import laya_backend
from s1d1_bench.classify import TIER_TO_LEVEL
from s1d1_bench.cli import main
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import validate_judge_dir
from s1d1_bench.run import read_decisions

COMMIT = "7b928d8" + "c" * 33
LABELS = ("SIMPLE", "MEDIUM", "COMPLEX", "REASONING")
RESEARCH_FILE = Path("/Users/steve/work/s1d1-model-router/bench/laya_scores.py")


def probabilities(text: str) -> dict[str, float]:
    """Four probabilities that depend on the text only (its length decides how much weight goes to which level)."""
    weights = [1 + (len(text) * (k + 3)) % 5 for k in range(4)]
    total = sum(weights)
    return {label: w / total for label, w in zip(LABELS, weights, strict=True)}


class FakeAgent:
    device = "cpu"


class FakeRouter:
    """laya.Router with the two calls the adapter makes: `load(name)` and `predict_batch(requests)`."""

    instances: list = []
    calls: list = []
    truncate_long = 10**9  # a state longer than this counts as truncated
    answer = None  # when set: a function state -> probabilities, instead of `probabilities`

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        FakeRouter.instances.append(self)

    def load(self, name):
        self.loaded = name
        return FakeAgent()

    def predict_batch(self, requests):
        FakeRouter.calls.append(list(requests))
        answer = FakeRouter.answer or probabilities
        return [
            {
                "answers": {"tier": {"type": "choice", "probabilities": answer(r["state"])}},
                "usage": {"truncated": len(r["state"]) > FakeRouter.truncate_long},
            }
            for r in requests
        ]


@pytest.fixture
def fake_laya(monkeypatch):
    FakeRouter.instances, FakeRouter.calls, FakeRouter.truncate_long, FakeRouter.answer = [], [], 10**9, None
    monkeypatch.setitem(sys.modules, "laya", SimpleNamespace(Router=FakeRouter, __version__="0.4.15"))
    infos = []

    def model_info(repo, revision=None):
        infos.append((repo, revision))
        return SimpleNamespace(sha=revision if revision and len(revision) == 40 else COMMIT)

    monkeypatch.setattr(laya_backend, "model_info", model_info)
    monkeypatch.setenv("HF_TOKEN", "hf_not_for_laya")
    return infos


def expected(text: str) -> float:
    """The expected level on 0..3, not divided."""
    p = probabilities(text)
    return 0 * p["SIMPLE"] + 1 * p["MEDIUM"] + 2 * p["COMPLEX"] + 3 * p["REASONING"]


def most_probable(text: str) -> str:
    p = probabilities(text)
    return min(LABELS, key=lambda label: (-p[label], LABELS.index(label)))


# ---------- the question ----------


def test_questions_are_the_research_repo_ones_byte_for_byte():
    if not RESEARCH_FILE.is_file():
        pytest.skip("the research repo is not readable here")
    spec = importlib.util.spec_from_file_location("laya_scores_research", RESEARCH_FILE)
    research = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(research)
    assert laya_backend.INSTRUCTIONS == research.INSTRUCTIONS
    assert laya_backend.CRITERIA == research.CRITERIA and list(laya_backend.CRITERIA) == list(research.CRITERIA)
    assert laya_backend.QUESTIONS == research.QUESTIONS
    assert json.dumps(laya_backend.QUESTIONS) == json.dumps(research.QUESTIONS)


def test_the_question_is_one_choice_tier_with_four_criteria():
    q = laya_backend.QUESTIONS["tier"]
    assert list(laya_backend.QUESTIONS) == ["tier"] and q["type"] == "choice"
    assert list(q["criteria"]) == list(LABELS) and q["instructions"] == laya_backend.INSTRUCTIONS


# ---------- the readings of the probabilities ----------


def test_expected_level_is_sum_of_i_times_p_on_0_to_3():
    one = {label: 0.0 for label in LABELS}
    assert laya_backend.expected_level({**one, "SIMPLE": 1.0}) == 0.0
    assert laya_backend.expected_level({**one, "REASONING": 1.0}) == 3.0
    assert laya_backend.expected_level({**one, "MEDIUM": 1.0}) == 1.0
    assert laya_backend.expected_level({**one, "COMPLEX": 1.0}) == 2.0
    real = {"SIMPLE": 0.4529, "MEDIUM": 0.2896, "COMPLEX": 0.1605, "REASONING": 0.097}  # a real answer of Laya
    assert laya_backend.expected_level(real) == pytest.approx(0.2896 + 2 * 0.1605 + 3 * 0.097)
    assert laya_backend.expected_level({label: 0.25 for label in LABELS}) == pytest.approx(1.5)
    assert not hasattr(laya_backend, "expected_score")  # no normalised score any more


def test_most_probable_takes_the_highest_and_the_lowest_name_on_an_exact_tie():
    assert laya_backend.most_probable({"SIMPLE": 0.1, "MEDIUM": 0.2, "COMPLEX": 0.3, "REASONING": 0.4}) == "REASONING"
    assert laya_backend.most_probable({"SIMPLE": 0.4, "MEDIUM": 0.3, "COMPLEX": 0.2, "REASONING": 0.1}) == "SIMPLE"
    assert laya_backend.most_probable({"SIMPLE": 0.1, "MEDIUM": 0.4, "COMPLEX": 0.4, "REASONING": 0.1}) == "MEDIUM"
    assert laya_backend.most_probable({"SIMPLE": 0.1, "MEDIUM": 0.1, "COMPLEX": 0.4, "REASONING": 0.4}) == "COMPLEX"
    assert laya_backend.most_probable({label: 0.25 for label in LABELS}) == "SIMPLE"


# ---------- the classifier function ----------


def test_classifier_gives_the_four_probabilities_in_criteria_order(fake_laya):
    classify, meta = laya_backend.make_laya_classifier("english", None)
    prompts = ["a", "bb" * 7, "hello there, what is up", "x" * 300, "y" * 41, "z" * 5]
    out = classify(prompts)
    assert len(out) == len(prompts)
    for text, row in zip(prompts, out, strict=True):
        assert set(row) == {"probabilities"}
        assert list(row["probabilities"]) == list(laya_backend.CRITERIA) == list(LABELS)
        assert row["probabilities"] == pytest.approx(probabilities(text))
    assert len({laya_backend.most_probable(r["probabilities"]) for r in out}) > 1  # the fake spans several tiers


def test_classifier_keeps_the_order_of_the_keys_whatever_laya_returns(fake_laya):
    FakeRouter.answer = lambda state: {"REASONING": 0.1, "COMPLEX": 0.2, "SIMPLE": 0.3, "MEDIUM": 0.4}
    classify, _ = laya_backend.make_laya_classifier("english", None)
    assert list(classify(["anything"])[0]["probabilities"]) == list(LABELS)


def test_classifier_asks_the_question_of_the_chosen_checkpoint(fake_laya):
    classify, _ = laya_backend.make_laya_classifier("multilingual", None)
    classify(["one", "two"])
    assert FakeRouter.instances[0].loaded == "multilingual"
    for request in FakeRouter.calls[0]:
        assert request["questions"] is laya_backend.QUESTIONS and request["model"] == "multilingual"
        assert set(request) == {"state", "questions", "model"}


def test_classifier_sorts_by_length_in_batches_of_32_and_restores_the_order(fake_laya):
    classify, _ = laya_backend.make_laya_classifier("english", None)
    prompts = [("p" * ((n * 37) % 91 + 1)) for n in range(70)]
    out = classify(prompts)
    assert [len(batch) for batch in FakeRouter.calls] == [32, 32, 6]
    seen = [r["state"] for batch in FakeRouter.calls for r in batch]
    assert seen == sorted(prompts, key=len) and sorted(seen) == sorted(prompts)
    # back in the order of the prompts
    levels = [laya_backend.expected_level(r["probabilities"]) for r in out]
    assert levels == pytest.approx([expected(p) for p in prompts])


def test_classifier_empty_input(fake_laya):
    classify, _ = laya_backend.make_laya_classifier("english", None)
    assert classify([]) == [] and FakeRouter.calls == []


def test_truncated_prompts_are_counted_in_meta(fake_laya):
    FakeRouter.truncate_long = 50
    classify, meta = laya_backend.make_laya_classifier("english", None)
    assert meta["truncated"] == 0
    classify(["short", "x" * 60, "y" * 70])
    assert meta["truncated"] == 2


# ---------- revision, model id, token ----------


def test_revision_is_resolved_pinned_and_written(fake_laya):
    _, meta = laya_backend.make_laya_classifier("english", None)
    assert fake_laya == [("convaiinnovations/laya", None)]
    assert FakeRouter.instances[0].kwargs == {"revision": COMMIT}  # the weights are those of the recorded commit
    assert meta["commit"] == COMMIT and meta["laya_version"] == "0.4.15"
    assert meta["revision"] == f"laya 0.4.15 convaiinnovations/laya@{COMMIT}"
    assert meta["model"] == "convaiinnovations/laya" and meta["device"] == "cpu"
    given = "a" * 40
    _, meta = laya_backend.make_laya_classifier("multilingual", given)
    assert FakeRouter.instances[1].kwargs == {"revision": given} and meta["commit"] == given
    assert meta["model"] == "convaiinnovations/laya/multilingual"


def test_unknown_model_is_refused_before_anything_loads(fake_laya):
    with pytest.raises(ValueError, match="english, multilingual"):
        laya_backend.make_laya_classifier("typed-decisions", None)
    assert FakeRouter.instances == [] and fake_laya == []


def test_the_hub_is_read_without_a_token(fake_laya, monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    laya_backend.make_laya_classifier("english", None)
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1" and "HF_TOKEN" not in os.environ


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


def _classify(prompts_file, out, *extra):
    return main(["classify", "laya", "--prompts", str(prompts_file), "--out", str(out), "--author", "tester", *extra])


def test_cli_writes_a_valid_classifier_folder(fake_laya, prompts_file, tmp_path):
    out = tmp_path / "classifiers" / "laya-english"
    assert _classify(prompts_file, out, "--laya-model", "english") == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "convaiinnovations/laya"
    assert meta["revision"] == f"laya 0.4.15 convaiinnovations/laya@{COMMIT}"
    assert meta["replayable"] is True and meta["temperature"] == 0 and meta["judge"] == "laya-english"
    assert (meta["provider"], meta["instructions"], meta["thinking"]) == ("system", "none", "none")
    assert "quantization" not in meta and meta["author"] == "tester"
    for needle in (
        "bench classify laya", "--laya-model english", f"--revision {COMMIT}", "uv run --group laya",
        "bench/laya_scores.py", "2026-10-30", "LiteLLM",
        "SIMPLE -> low, MEDIUM -> medium, COMPLEX -> high, REASONING -> expert", "never trivial",
        "labels.jsonl has no score", "decisions.jsonl holds", "0..3", "four probabilities",
        "laya-english and laya-multilingual", "do not enter the results", "most probable",
        "cpu", "0 prompts truncated",
    ):  # fmt: skip
        assert needle in meta["notes"], needle
    assert "/Users/" not in json.dumps(meta)
    labels = read_jsonl(out / "labels.jsonl")
    assert [r["id"] for r in labels] == [p["id"] for p in PROMPTS]
    for row, item in zip(labels, PROMPTS, strict=True):
        assert row == {
            "id": item["id"], "complexity": TIER_TO_LEVEL[most_probable(item["prompt"])], "task": None,
            "needs_tools": None,
        }  # fmt: skip
        assert "score" not in row and row["complexity"] != "trivial"
    assert validate_judge_dir(out, prompts_file.parent) == []


def test_cli_three_prompts_name_to_name_with_decisions_aligned_by_id(fake_laya, tmp_path):
    answers = {
        "easy": {"SIMPLE": 0.7, "MEDIUM": 0.2, "COMPLEX": 0.05, "REASONING": 0.05},
        "hard": {"SIMPLE": 0.1, "MEDIUM": 0.2, "COMPLEX": 0.6, "REASONING": 0.1},
        "deep": {"SIMPLE": 0.0, "MEDIUM": 0.1, "COMPLEX": 0.2, "REASONING": 0.7},
    }
    FakeRouter.answer = lambda state: answers[state]
    d = tmp_path / "prompts"
    d.mkdir()
    items = [{"id": "p1-eng", "prompt": "deep"}, {"id": "p2-eng", "prompt": "easy"}, {"id": "p3-eng", "prompt": "hard"}]
    write_jsonl(d / "three.jsonl", items)
    out = tmp_path / "laya-english-tier"
    assert _classify(d / "three.jsonl", out, "--laya-model", "english") == 0
    labels = read_jsonl(out / "labels.jsonl")
    assert labels == [
        {"id": "p1-eng", "complexity": "expert", "task": None, "needs_tools": None},
        {"id": "p2-eng", "complexity": "low", "task": None, "needs_tools": None},
        {"id": "p3-eng", "complexity": "high", "task": None, "needs_tools": None},
    ]
    assert all("score" not in row for row in labels)
    decisions = read_jsonl(out / "decisions.jsonl")
    assert [r["id"] for r in decisions] == ["p1-eng", "p2-eng", "p3-eng"]
    assert [r["tier"] for r in decisions] == ["REASONING", "SIMPLE", "COMPLEX"]
    assert [r["score"] for r in decisions] == pytest.approx([0.1 + 0.4 + 2.1, 0.2 + 0.1 + 0.15, 0.2 + 1.2 + 0.3])
    assert [list(r["probabilities"]) for r in decisions] == [list(LABELS)] * 3
    assert [r["probabilities"] for r in decisions] == [answers["deep"], answers["easy"], answers["hard"]]
    assert set(decisions[0]) == {"id", "tier", "score", "probabilities"}
    assert validate_judge_dir(out, d) == []
    read = read_decisions(out / "decisions.jsonl", [i["id"] for i in items])
    assert read == {
        "p1-eng": {"tier": "REASONING", "score": pytest.approx(2.6)},
        "p2-eng": {"tier": "SIMPLE", "score": pytest.approx(0.45)},
        "p3-eng": {"tier": "COMPLEX", "score": pytest.approx(1.7)},
    }


def test_cli_exact_ties_go_to_the_lowest_name(fake_laya, tmp_path):
    answers = {
        "tie-mc": {"SIMPLE": 0.1, "MEDIUM": 0.4, "COMPLEX": 0.4, "REASONING": 0.1},
        "tie-all": {label: 0.25 for label in LABELS},
        "tie-cr": {"SIMPLE": 0.0, "MEDIUM": 0.2, "COMPLEX": 0.4, "REASONING": 0.4},
    }
    FakeRouter.answer = lambda state: answers[state]
    d = tmp_path / "prompts"
    d.mkdir()
    items = [{"id": f"p{n}-eng", "prompt": state} for n, state in enumerate(answers)]
    write_jsonl(d / "ties.jsonl", items)
    out = tmp_path / "laya-ties"
    assert _classify(d / "ties.jsonl", out, "--laya-model", "english") == 0
    assert [r["complexity"] for r in read_jsonl(out / "labels.jsonl")] == ["medium", "low", "high"]
    assert [r["tier"] for r in read_jsonl(out / "decisions.jsonl")] == ["MEDIUM", "SIMPLE", "COMPLEX"]
    assert [r["score"] for r in read_jsonl(out / "decisions.jsonl")] == pytest.approx([1.5, 1.5, 0.2 + 0.8 + 1.2])


def test_cli_multilingual_names_its_checkpoint(fake_laya, prompts_file, tmp_path):
    out = tmp_path / "laya-multilingual"
    assert _classify(prompts_file, out, "--laya-model", "multilingual", "--notes", "extra words") == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "convaiinnovations/laya/multilingual" and "--laya-model multilingual" in meta["notes"]
    assert meta["notes"].endswith("extra words")
    assert FakeRouter.instances[0].loaded == "multilingual"
    assert len(read_jsonl(out / "decisions.jsonl")) == len(PROMPTS)
    assert validate_judge_dir(out, prompts_file.parent) == []


def test_cli_needs_a_laya_model(fake_laya, prompts_file, tmp_path, capsys):
    out = tmp_path / "out"
    assert _classify(prompts_file, out) == 1
    assert "--laya-model" in capsys.readouterr().err and not out.exists()
    assert FakeRouter.instances == []


def test_cli_names_the_group_to_install_when_laya_is_missing(prompts_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "laya", None)  # `import laya` raises ImportError
    out = tmp_path / "out"
    assert _classify(prompts_file, out, "--laya-model", "english") == 1
    err = capsys.readouterr().err
    assert "laya" in err and "--group laya" in err and not out.exists()


# ---------- the real model, once ----------


@pytest.mark.slow
def test_english_checkpoint_scores_two_prompts_from_the_hub():
    pytest.importorskip("laya")
    classify, meta = laya_backend.make_laya_classifier("english", None)
    assert meta["laya_version"] and len(meta["commit"]) == 40
    out = classify([
        "What is the capital of Australia?",
        "Prove that there are infinitely many primes and discuss the tradeoffs of three proof strategies.",
    ])
    assert len(out) == 2
    for row in out:
        assert list(row["probabilities"]) == list(LABELS)
        assert 0.0 <= laya_backend.expected_level(row["probabilities"]) <= 3.0
        assert laya_backend.most_probable(row["probabilities"]) in TIER_TO_LEVEL
    print(meta, out)
