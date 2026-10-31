import json
import math
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from s1d1_bench import routellm_backend
from s1d1_bench.cli import main
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import validate_judge_dir
from s1d1_bench.rubric import LEVELS
from s1d1_bench.run import read_decisions

COMMIT = "86237e3" + "d" * 33
FOLDER = "/cache/models--routellm--bert/snapshots/" + COMMIT


def logits_of(text: str) -> np.ndarray:
    """Three float32 logits that depend on the text only."""
    return np.array([(len(text) % 7) - 3, (len(text) % 5) - 2, (len(text) % 3) - 1], dtype=np.float32)


class FakeParameter:
    dtype = "torch.float32"


class FakeModel:
    instances: list = []
    seen: list = []

    def __init__(self, folder, num_labels):
        self.folder, self.num_labels = folder, num_labels
        self.evaluated = False
        FakeModel.instances.append(self)

    def eval(self):
        self.evaluated = True

    def parameters(self):
        return iter([FakeParameter()])

    def __call__(self, **inputs):
        FakeModel.seen.append(inputs)
        logits = logits_of(inputs["text"])
        return SimpleNamespace(logits=SimpleNamespace(numpy=lambda: logits[None, :]))


class FakeTokenizer:
    model_max_length = 20
    calls: list = []

    @classmethod
    def from_pretrained(cls, folder):
        cls.folder = folder
        return cls()

    def __call__(self, prompt, **kwargs):
        FakeTokenizer.calls.append((prompt, kwargs))
        return {"input_ids": list(range(len(prompt) // 3 + 1)), "text": prompt}


@pytest.fixture
def fake_routellm(monkeypatch):
    FakeModel.instances, FakeModel.seen, FakeTokenizer.calls = [], [], []
    downloads, infos = [], []
    fake_model_class = SimpleNamespace(from_pretrained=lambda folder, num_labels: FakeModel(folder, num_labels))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoModelForSequenceClassification=fake_model_class, AutoTokenizer=FakeTokenizer),
    )

    def snapshot_download(repo, revision, allow_patterns, token):
        downloads.append((repo, revision, tuple(allow_patterns), token))
        return FOLDER

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))

    def model_info(repo, revision=None):
        infos.append((repo, revision))
        return SimpleNamespace(sha=revision if revision and len(revision) == 40 else COMMIT)

    monkeypatch.setattr(routellm_backend, "model_info", model_info)
    monkeypatch.setenv("HF_TOKEN", "hf_not_for_routellm")
    return SimpleNamespace(downloads=downloads, infos=infos)


# ---------- the bounds ----------


@pytest.mark.parametrize(
    ("rate", "tier"),
    [
        (0.0, "trivial"),
        (0.05, "trivial"),
        (0.199999, "trivial"),
        (0.2, "low"),  # the lower bound belongs to the tier above it
        (0.399999, "low"),
        (0.4, "medium"),
        (0.599999, "medium"),
        (0.6, "high"),
        (0.799999, "high"),
        (0.8, "expert"),
        (0.95, "expert"),
        (1.0, "expert"),
    ],
)
def test_tier_bounds_are_five_equal_intervals_with_the_lower_bound_in(rate, tier):
    assert routellm_backend.tier_of_win_rate(rate) == tier


def test_every_tier_is_reachable_and_the_bounds_are_equal():
    assert [routellm_backend.tier_of_win_rate(k / 10 + 0.05) for k in range(10)] == [
        t for t in LEVELS for _ in range(2)
    ]
    assert routellm_backend.TIER_BOUNDS == (0.2, 0.4, 0.6, 0.8)


@pytest.mark.parametrize("bad", [-0.01, 1.000001, math.nan, math.inf, True, None, "0.5"])
def test_a_rate_outside_the_unit_interval_is_refused_not_clipped(bad):
    with pytest.raises(ValueError, match="probability in"):
        routellm_backend.tier_of_win_rate(bad)


# ---------- the router's reading of the logits ----------


def test_strong_win_rate_is_one_minus_the_last_two_softmax_probabilities():
    rate, probabilities = routellm_backend.strong_win_rate(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    p0 = math.exp(2) / (math.exp(2) + 2)
    assert rate == pytest.approx(p0, abs=1e-6) and probabilities[0] == pytest.approx(p0, abs=1e-6)
    assert probabilities[1] == probabilities[2] == pytest.approx(1 / (math.exp(2) + 2), abs=1e-6)
    equal, _ = routellm_backend.strong_win_rate(np.zeros(3, dtype=np.float32))
    assert equal == pytest.approx(1 / 3, abs=1e-6)
    low, _ = routellm_backend.strong_win_rate(np.array([-30.0, 5.0, 5.0], dtype=np.float32))
    assert low == pytest.approx(0.0, abs=1e-6)
    high, _ = routellm_backend.strong_win_rate(np.array([30.0, -5.0, -5.0], dtype=np.float32))
    assert high == pytest.approx(1.0, abs=1e-6)


def test_strong_win_rate_does_not_overflow_on_large_logits():
    rate, probabilities = routellm_backend.strong_win_rate(np.array([900.0, 899.0, -900.0], dtype=np.float32))
    assert 0 <= rate <= 1 and math.isfinite(rate) and sum(probabilities) == pytest.approx(1.0, abs=1e-5)


# ---------- the routers ----------


def test_mf_is_excluded_with_its_reason_before_anything_loads(fake_routellm):
    with pytest.raises(ValueError, match=r"excluded.*OpenAI's text-embedding-3-small.*no paid call"):
        routellm_backend.make_routellm_classifier("mf", None)
    assert FakeModel.instances == [] and fake_routellm.infos == [] and fake_routellm.downloads == []


def test_an_unknown_router_is_refused(fake_routellm):
    with pytest.raises(ValueError, match="expected one of bert"):
        routellm_backend.make_routellm_classifier("sw_ranking", None)
    assert FakeModel.instances == [] and fake_routellm.infos == []


def test_the_router_is_loaded_from_the_pinned_commit_without_the_training_states(fake_routellm):
    _, meta = routellm_backend.make_routellm_classifier("bert", None)
    assert fake_routellm.infos == [("routellm/bert_gpt4_augmented", None)]
    ((repo, revision, patterns, token),) = fake_routellm.downloads
    assert (repo, revision, token) == ("routellm/bert_gpt4_augmented", COMMIT, False)
    assert "model.safetensors" in patterns and not any("optimizer" in p for p in patterns)
    (model,) = FakeModel.instances
    assert (model.folder, model.num_labels, model.evaluated) == (FOLDER, 3, True)
    assert FakeTokenizer.folder == FOLDER
    assert meta == {
        "model": "routellm/bert_gpt4_augmented", "revision": COMMIT, "device": "cpu", "precision": "float32",
        "max_length": 20, "truncated": 0,
    }  # fmt: skip
    given = "a" * 40
    _, meta = routellm_backend.make_routellm_classifier("bert", given)
    assert meta["revision"] == given and fake_routellm.downloads[1][1] == given


def test_the_hub_is_read_without_a_token(fake_routellm, monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    routellm_backend.make_routellm_classifier("bert", None)
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1" and "HF_TOKEN" not in os.environ


def test_classify_runs_one_prompt_at_a_time_as_routellm_does_and_keeps_the_order(fake_routellm):
    classify, meta = routellm_backend.make_routellm_classifier("bert", None)
    prompts = ["a", "bb" * 7, "hello there, what is up", "x" * 300, "y" * 41]
    out = classify(prompts)
    assert len(out) == len(FakeModel.seen) == len(prompts)  # one call of the model per prompt
    for text, row in zip(prompts, out, strict=True):
        rate, probabilities = routellm_backend.strong_win_rate(logits_of(text))
        assert row == {"strong_win_rate": rate, "probabilities": probabilities}
    calls = [(p, kw) for p, kw in FakeTokenizer.calls if kw]
    assert [p for p, _ in calls] == prompts  # given as they are
    assert all(kw == {"return_tensors": "pt", "padding": True, "truncation": True} for _, kw in calls)
    assert classify([]) == []


def test_prompts_over_the_token_budget_are_counted(fake_routellm):
    classify, meta = routellm_backend.make_routellm_classifier("bert", None)
    assert meta["truncated"] == 0
    classify(["short", "x" * 61, "y" * 200])  # the fake counts len // 3 + 1 tokens, the budget is 20
    assert meta["truncated"] == 2


# ---------- the command ----------

PROMPTS = [
    {"id": f"b{n}-{lang}", "prompt": "p" * (10 * n + 1 + len(lang))} for n in range(6) for lang in ("eng", "fra")
]
RATES = [0.0, 0.05, 0.199999, 0.2, 0.399999, 0.4, 0.599999, 0.6, 0.799999, 0.8, 0.95, 1.0]


@pytest.fixture
def prompts_file(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    write_jsonl(d / "mini.jsonl", PROMPTS)
    return d / "mini.jsonl"


def _classify(prompts_file, out, *extra):
    return main(
        ["classify", "routellm", "--prompts", str(prompts_file), "--out", str(out), "--author", "tester", *extra]
    )


def test_cli_writes_a_valid_folder_with_the_mapping_at_the_bounds(monkeypatch, prompts_file, tmp_path):
    info = {
        "model": "routellm/bert_gpt4_augmented", "revision": COMMIT, "device": "cpu", "precision": "float32",
        "max_length": 512, "truncated": 3,
    }  # fmt: skip

    def classify(prompts):
        assert prompts == [p["prompt"] for p in PROMPTS]
        return [{"strong_win_rate": r, "probabilities": [round(1 - r, 6), 0.0, r]} for r in RATES]

    monkeypatch.setattr(routellm_backend, "make_routellm_classifier", lambda router, revision: (classify, info))
    out = tmp_path / "classifiers" / "routellm-bert"
    assert _classify(prompts_file, out, "--routellm-router", "bert") == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "routellm/bert_gpt4_augmented" and meta["revision"] == COMMIT
    assert (meta["provider"], meta["instructions"], meta["thinking"]) == ("system", "none", "none")
    assert meta["quantization"] == "float32" and meta["judge"] == "routellm-bert" and meta["author"] == "tester"
    for needle in (
        "uv run --group routellm bench classify routellm --routellm-router bert", f"--revision {COMMIT}",
        f"lm-sys/RouteLLM@{routellm_backend.ROUTELLM_COMMIT}", "probability that the strong model", "no calibration",
        "trivial [0 ; 0.2), low [0.2 ; 0.4), medium [0.4 ; 0.6), high [0.6 ; 0.8), expert [0.8 ; 1]",
        "does not predict a complexity tier", "task and needs_tools are null", "3 prompts truncated",
    ):  # fmt: skip
        assert needle in meta["notes"], needle
    assert "/Users/" not in json.dumps(meta)
    labels = read_jsonl(out / "labels.jsonl")
    assert [r["id"] for r in labels] == [p["id"] for p in PROMPTS]
    expected = ["trivial", "trivial", "trivial", "low", "low", "medium", "medium", "high", "high", "expert",
                "expert", "expert"]  # fmt: skip
    assert [r["complexity"] for r in labels] == expected
    assert [r["score"] for r in labels] == RATES
    assert all(r["task"] is None and r["needs_tools"] is None and "needs_tools_probability" not in r for r in labels)
    decisions = read_decisions(out / "decisions.jsonl", [p["id"] for p in PROMPTS])
    assert [d["tier"] for d in decisions.values()] == expected and [d["score"] for d in decisions.values()] == RATES
    assert read_jsonl(out / "decisions.jsonl")[3]["probabilities"] == [0.8, 0.0, 0.2]  # the raw output is kept
    assert validate_judge_dir(out, prompts_file.parent) == []


def test_cli_end_to_end_with_the_fake_router(fake_routellm, prompts_file, tmp_path):
    out = tmp_path / "classifiers" / "routellm-bert"
    assert _classify(prompts_file, out, "--routellm-router", "bert", "--revision", "e" * 40) == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["revision"] == "e" * 40 and "float32" in meta["notes"]
    for row, item in zip(read_jsonl(out / "labels.jsonl"), PROMPTS, strict=True):
        rate, _ = routellm_backend.strong_win_rate(logits_of(item["prompt"]))
        assert row["score"] == rate and row["complexity"] == routellm_backend.tier_of_win_rate(rate)
    assert validate_judge_dir(out, prompts_file.parent) == []


def test_cli_needs_a_router(fake_routellm, prompts_file, tmp_path, capsys):
    out = tmp_path / "out"
    assert _classify(prompts_file, out) == 1
    assert "--routellm-router" in capsys.readouterr().err and not out.exists()


def test_cli_refuses_mf_and_says_why(fake_routellm, prompts_file, tmp_path, capsys):
    out = tmp_path / "out"
    assert _classify(prompts_file, out, "--routellm-router", "mf") == 1
    err = capsys.readouterr().err
    assert "excluded" in err and "OpenAI" in err and not out.exists()


def test_cli_names_the_group_to_install_when_torch_is_missing(prompts_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # an import of torch raises ImportError
    out = tmp_path / "out"
    assert _classify(prompts_file, out, "--routellm-router", "bert") == 1
    assert "--group routellm" in capsys.readouterr().err and not out.exists()


def test_cli_does_not_overwrite_a_folder(fake_routellm, prompts_file, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    assert _classify(prompts_file, out, "--routellm-router", "bert") == 1
    assert "already exists" in capsys.readouterr().err
