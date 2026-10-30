import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from s1d1_bench import laya_backend
from s1d1_bench.classify import ANCHORS, nearest_anchor
from s1d1_bench.cli import main
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import validate_judge_dir

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
    p = probabilities(text)
    return (0 * p["SIMPLE"] + 1 * p["MEDIUM"] + 2 * p["COMPLEX"] + 3 * p["REASONING"]) / 3


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


# ---------- the score ----------


def test_score_is_the_expected_level_normalised_into_0_1():
    one = {label: 0.0 for label in LABELS}
    assert laya_backend.expected_score({**one, "SIMPLE": 1.0}) == 0.0
    assert laya_backend.expected_score({**one, "REASONING": 1.0}) == 1.0
    assert laya_backend.expected_score({**one, "MEDIUM": 1.0}) == pytest.approx(1 / 3)
    assert laya_backend.expected_score({**one, "COMPLEX": 1.0}) == pytest.approx(2 / 3)
    real = {"SIMPLE": 0.4529, "MEDIUM": 0.2896, "COMPLEX": 0.1605, "REASONING": 0.097}  # a real answer of Laya
    assert laya_backend.expected_score(real) == pytest.approx((0.2896 + 2 * 0.1605 + 3 * 0.097) / 3)
    assert laya_backend.expected_score({label: 0.25 for label in LABELS}) == pytest.approx(0.5)


# ---------- the classifier function ----------


def test_classifier_gives_score_nearest_anchor_and_null_task(fake_laya):
    classify, meta = laya_backend.make_laya_classifier("english", None)
    prompts = ["a", "bb" * 7, "hello there, what is up", "x" * 300, "y" * 41, "z" * 5]
    out = classify(prompts)
    assert len(out) == len(prompts)
    for text, row in zip(prompts, out, strict=True):
        score = expected(text)
        assert row == {
            "complexity": nearest_anchor(score), "task": None, "needs_tools": None, "score": pytest.approx(score),
        }  # fmt: skip
        assert 0.0 <= row["score"] <= 1.0 and math.isfinite(row["score"])
        assert min(ANCHORS, key=lambda t: abs(ANCHORS[t] - score)) == row["complexity"]
    assert "needs_tools_probability" not in out[0]
    assert len({r["complexity"] for r in out}) > 1  # the fake spans several tiers, the mapping is not a constant


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
    assert [r["score"] for r in out] == pytest.approx([expected(p) for p in prompts])


def test_classifier_empty_input(fake_laya):
    classify, _ = laya_backend.make_laya_classifier("english", None)
    assert classify([]) == [] and FakeRouter.calls == []


def test_midpoint_scores_go_to_the_lower_tier(fake_laya):
    # 0.875 MEDIUM + 0.125 COMPLEX: expected level 1.125, score exactly 0.375, the midpoint of low and medium
    FakeRouter.answer = lambda state: {"SIMPLE": 0.0, "MEDIUM": 0.875, "COMPLEX": 0.125, "REASONING": 0.0}
    classify, _ = laya_backend.make_laya_classifier("english", None)
    row = classify(["anything"])[0]
    assert row["score"] == 0.375 and row["complexity"] == "low"


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
        "bench/laya_scores.py", "2026-10-30", "LiteLLM", "expected level", "nearest anchor",
        "cpu", "0 prompts truncated",
    ):  # fmt: skip
        assert needle in meta["notes"], needle
    assert "/Users/" not in json.dumps(meta)
    labels = read_jsonl(out / "labels.jsonl")
    assert [r["id"] for r in labels] == [p["id"] for p in PROMPTS]
    for row, item in zip(labels, PROMPTS, strict=True):
        score = expected(item["prompt"])
        assert row == {
            "id": item["id"], "complexity": nearest_anchor(score), "task": None, "needs_tools": None,
            "score": pytest.approx(score),
        }  # fmt: skip
    assert validate_judge_dir(out, prompts_file.parent) == []


def test_cli_multilingual_names_its_checkpoint(fake_laya, prompts_file, tmp_path):
    out = tmp_path / "laya-multilingual"
    assert _classify(prompts_file, out, "--laya-model", "multilingual", "--notes", "extra words") == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "convaiinnovations/laya/multilingual" and "--laya-model multilingual" in meta["notes"]
    assert meta["notes"].endswith("extra words")
    assert FakeRouter.instances[0].loaded == "multilingual"
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
        assert 0.0 <= row["score"] <= 1.0 and row["complexity"] in ANCHORS
        assert row["task"] is None and row["needs_tools"] is None
    print(meta, out)
