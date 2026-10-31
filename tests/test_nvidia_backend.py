import json
import math
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from s1d1_bench import nvidia_backend
from s1d1_bench.classifier_file import read_classifier_labels
from s1d1_bench.cli import main
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import validate_judge_dir
from s1d1_bench.rubric import LEVELS
from s1d1_bench.run import read_decisions

COMMIT = "fea1121" + "c" * 33
CONFIG = {
    "target_sizes": {
        "task_type": 12,
        "creativity_scope": 3,
        "reasoning": 2,
        "contextual_knowledge": 2,
        "number_of_few_shots": 6,
        "domain_knowledge": 4,
        "no_label_reason": 1,
        "constraint_ct": 2,
    },
    "task_type_map": {
        "0": "Brainstorming",
        "1": "Chatbot",
        "2": "Classification",
        "3": "Closed QA",
        "4": "Code Generation",
        "5": "Extraction",
        "6": "Open QA",
        "7": "Other",
        "8": "Rewrite",
        "9": "Summarization",
        "10": "Text Generation",
        "11": "Unknown",
    },
    "weights_map": {
        "creativity_scope": [2, 1, 0],
        "reasoning": [0, 1],
        "contextual_knowledge": [0, 1],
        "number_of_few_shots": [0, 1, 2, 3, 4, 5],
        "domain_knowledge": [3, 1, 2, 0],
        "no_label_reason": [0],
        "constraint_ct": [1, 0],
    },
    "divisor_map": {
        "creativity_scope": 2,
        "reasoning": 1,
        "contextual_knowledge": 1,
        "number_of_few_shots": 1,
        "domain_knowledge": 3,
        "no_label_reason": 1,
        "constraint_ct": 1,
    },
}  # fmt: skip  (the config.json of the card's repo, revision fea1121)
SIZES = list(CONFIG["target_sizes"].values())


def one_hot(size: int, index: int) -> np.ndarray:
    out = np.zeros(size, dtype=np.float32)
    out[index] = 1.0
    return out


def softmaxes(task: int, creativity: int = 2, reasoning: int = 0, contextual: int = 0, few: int = 0, domain: int = 3,
              constraint: int = 1) -> list[np.ndarray]:  # fmt: skip
    """The eight softmax vectors, each a one-hot on the given index (indices of the card's weights_map)."""
    picks = [task, creativity, reasoning, contextual, few, domain, 0, constraint]
    return [one_hot(size, pick) for size, pick in zip(SIZES, picks, strict=True)]


# ---------- the bounds ----------


@pytest.mark.parametrize(
    ("score", "tier"),
    [
        (0.0, "trivial"), (0.05348, "trivial"), (0.199999, "trivial"),
        (0.2, "low"),  # the lower bound belongs to the tier above it
        (0.399999, "low"), (0.4, "medium"), (0.599999, "medium"), (0.6, "high"), (0.799999, "high"),
        (0.8, "expert"), (0.95, "expert"), (1.0, "expert"),
    ],
)  # fmt: skip
def test_tier_bounds_are_five_equal_intervals_with_the_lower_bound_in(score, tier):
    assert nvidia_backend.tier_of_score(score) == tier


def test_every_tier_is_reachable_and_the_bounds_are_those_of_routellm():
    from s1d1_bench import routellm_backend

    assert [nvidia_backend.tier_of_score(k / 10 + 0.05) for k in range(10)] == [t for t in LEVELS for _ in range(2)]
    assert nvidia_backend.TIER_BOUNDS == routellm_backend.TIER_BOUNDS == (0.2, 0.4, 0.6, 0.8)


@pytest.mark.parametrize("bad", [-0.01, 1.000001, math.nan, math.inf, True, None, "0.5"])
def test_a_score_outside_the_unit_interval_is_refused_not_clipped(bad):
    with pytest.raises(ValueError, match="number in"):
        nvidia_backend.tier_of_score(bad)


def test_only_english_is_covered():
    assert nvidia_backend.COVERED_LANGUAGES == ("eng",)
    assert [nvidia_backend.covered(lang) for lang in ("eng", "fra", "rus", "zho", "")] == [
        True,
        False,
        False,
        False,
        False,
    ]


# ---------- the card's reading of the heads ----------


def test_heads_follow_the_cards_arithmetic():
    # creativity "Low" (index 1: weight 1 / 2), reasoning "Yes", contextual "No", domain "Medium" (index 2: weight
    # 2 / 3), constraints "1.0" (index 0: weight 1), three few shots (index 3: weight 3)
    heads = nvidia_backend.heads_of_probabilities(
        softmaxes(task=4, creativity=1, reasoning=1, contextual=0, few=3, domain=2, constraint=0), CONFIG
    )
    assert heads["task_type_1"] == "Code Generation" and heads["task_type_prob"] == 1.0
    assert heads["creativity_scope"] == 0.5 and heads["reasoning"] == 1.0 and heads["contextual_knowledge"] == 0.0
    assert heads["number_of_few_shots"] == 3.0 and heads["domain_knowledge"] == 0.6667
    assert heads["no_label_reason"] == 0.0 and heads["constraint_ct"] == 1.0
    expected = round(0.35 * 0.5 + 0.25 * 1.0 + 0.15 * 1.0 + 0.15 * 0.6667 + 0.05 * 0.0 + 0.05 * 3.0, 5)
    assert heads["prompt_complexity_score"] == expected == 0.8250


def test_the_score_is_the_card_weighted_sum_of_the_six_dimensions():
    assert nvidia_backend.SCORE_WEIGHTS == {
        "creativity_scope": 0.35, "reasoning": 0.25, "constraint_ct": 0.15, "domain_knowledge": 0.15,
        "contextual_knowledge": 0.05, "number_of_few_shots": 0.05,
    }  # fmt: skip
    assert sum(nvidia_backend.SCORE_WEIGHTS.values()) == pytest.approx(1.0)
    # the card's example: 0.0826, 0.0632, 0.056, 0, 0.9803, 0.5578 -> 0.27822 (0.27823 with the unrounded weights)
    heads = {"creativity_scope": 0.0826, "reasoning": 0.0632, "contextual_knowledge": 0.056, "number_of_few_shots": 0,
             "domain_knowledge": 0.9803, "constraint_ct": 0.5578}  # fmt: skip
    total = sum(w * heads[k] for k, w in nvidia_backend.SCORE_WEIGHTS.items())
    assert round(total, 5) == pytest.approx(0.27823, abs=2e-5)


def test_the_second_task_is_na_under_a_tenth_and_few_shots_under_005_are_zero():
    task = np.zeros(12, dtype=np.float32)
    task[4], task[10], task[6] = 0.95, 0.04, 0.01
    probabilities = softmaxes(task=0)
    probabilities[0] = task
    probabilities[4] = np.array([0.97, 0.03, 0, 0, 0, 0], dtype=np.float32)  # expected few shots 0.03 < 0.05
    heads = nvidia_backend.heads_of_probabilities(probabilities, CONFIG)
    assert (heads["task_type_1"], heads["task_type_2"], heads["task_type_prob"]) == ("Code Generation", "NA", 0.95)
    assert heads["number_of_few_shots"] == 0
    task[10] = 0.15
    task[4] = 0.8
    assert nvidia_backend.heads_of_probabilities(probabilities, CONFIG)["task_type_2"] == "Text Generation"
    probabilities[4] = np.array([0.5, 0.5, 0, 0, 0, 0], dtype=np.float32)  # 0.5 >= 0.05 stays
    assert nvidia_backend.heads_of_probabilities(probabilities, CONFIG)["number_of_few_shots"] == 0.5


def test_the_heads_come_out_in_the_cards_key_order():
    assert list(nvidia_backend.heads_of_probabilities(softmaxes(task=0), CONFIG)) == [
        "task_type_1", "task_type_2", "task_type_prob", "creativity_scope", "reasoning", "contextual_knowledge",
        "number_of_few_shots", "domain_knowledge", "no_label_reason", "constraint_ct", "prompt_complexity_score",
    ]  # fmt: skip


# ---------- the loading, with a minimal fake of torch, transformers and the Hub ----------


class Tensor:
    """The few tensor operations of the card's `MeanPooling` and of the loop, on a numpy array."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def unsqueeze(self, dim):
        return Tensor(np.expand_dims(self.a, dim))

    def expand(self, size):
        return Tensor(np.broadcast_to(self.a, tuple(size)))

    def float(self):
        return self

    def size(self):
        return self.a.shape

    def sum(self, dim):
        return Tensor(self.a.sum(dim))

    def __mul__(self, other):
        return Tensor(self.a * other.a)

    def __truediv__(self, other):
        return Tensor(self.a / other.a)

    def __getitem__(self, index):
        return Tensor(self.a[index])

    def numpy(self):
        return self.a


class FakeParameter:
    dtype = "torch.float32"


class FakeModule:
    def __init__(self):
        self.modules = {}
        self.evaluated = False

    def add_module(self, name, module):
        self.modules[name] = module

    def parameters(self):
        return iter([FakeParameter()])

    def eval(self):
        self.evaluated = True

    def load_state_dict(self, state, strict):
        FakeTorch.loaded.append((state, strict))

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class FakeLinear(FakeModule):
    def __init__(self, i, o):
        super().__init__()
        self.o = o

    def forward(self, x):  # logits that depend on the pooled text only: a ramp shifted by the pooled value
        shift = float(x.a.sum()) % 3
        return Tensor((np.arange(self.o, dtype=np.float32)[None, :] * 0.5) - shift)


class FakeBackbone(FakeModule):
    config = SimpleNamespace(hidden_size=768)

    def forward(self, input_ids, attention_mask):
        n = attention_mask.a.shape[1]
        ramp = np.arange(n, dtype=np.float32)[None, :, None] * np.ones((1, 1, 4), dtype=np.float32)
        return SimpleNamespace(last_hidden_state=Tensor(ramp))


class FakeTokenizer:
    model_max_length = 512
    calls: list = []

    @classmethod
    def from_pretrained(cls, folder):
        cls.folder = folder
        return cls()

    def __call__(self, prompt, **kwargs):
        FakeTokenizer.calls.append((prompt, kwargs))
        if not kwargs:  # the count of tokens, to know whether the prompt is cut
            return {"input_ids": list(range(len(prompt) // 3 + 1))}
        n = 6
        return {"input_ids": Tensor(np.ones((1, n))), "attention_mask": Tensor(np.ones((1, n)))}


class FakeTorch(SimpleNamespace):
    loaded: list = []


FOLDER = "/cache/models--nvidia--ptcc/snapshots/" + COMMIT
BACKBONE_FOLDER = "/cache/models--microsoft--deberta-v3-base/snapshots/" + nvidia_backend.BACKBONE_REVISION


@pytest.fixture
def fake_nvidia(monkeypatch, tmp_path):
    FakeTokenizer.calls, FakeTorch.loaded = [], []
    folder = tmp_path / "nvidia-snapshot"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    downloads, infos, builds = [], [], []
    fake_torch = FakeTorch(
        nn=SimpleNamespace(Module=FakeModule, Linear=FakeLinear),
        no_grad=nullcontext,
        sum=lambda x, dim: x.sum(dim),
        clamp=lambda x, min: Tensor(np.maximum(x.a, min)),
        softmax=lambda x, dim: Tensor(np.exp(x.a - x.a.max(axis=dim, keepdims=True))
                                      / np.exp(x.a - x.a.max(axis=dim, keepdims=True)).sum(axis=dim, keepdims=True)),
    )  # fmt: skip
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "safetensors", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=lambda path: {"file": path}))
    fake_auto_model = SimpleNamespace(from_config=lambda config: builds.append(config) or FakeBackbone())
    fake_auto_config = SimpleNamespace(from_pretrained=lambda path: ("config", path))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoConfig=fake_auto_config, AutoModel=fake_auto_model, AutoTokenizer=FakeTokenizer),
    )

    def snapshot_download(repo, revision, allow_patterns, token):
        downloads.append((repo, revision, tuple(allow_patterns), token))
        return str(folder) if repo == nvidia_backend.REPO else BACKBONE_FOLDER

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download))

    def model_info(repo, revision=None):
        infos.append((repo, revision))
        return SimpleNamespace(sha=revision if revision and len(revision) == 40 else COMMIT)

    monkeypatch.setattr(nvidia_backend, "model_info", model_info)
    monkeypatch.setenv("HF_TOKEN", "hf_not_for_nvidia")
    return SimpleNamespace(downloads=downloads, infos=infos, builds=builds, folder=folder)


def test_the_model_is_loaded_from_the_pinned_commit_and_the_backbone_from_its_config_only(fake_nvidia):
    _, meta = nvidia_backend.make_nvidia_classifier(None)
    assert fake_nvidia.infos == [(nvidia_backend.REPO, None)]
    nvidia_download, backbone_download = fake_nvidia.downloads
    assert nvidia_download[:2] == (nvidia_backend.REPO, COMMIT) and nvidia_download[3] is False
    assert "model.safetensors" in nvidia_download[2] and "spm.model" in nvidia_download[2]
    assert backbone_download == (
        "microsoft/deberta-v3-base",
        nvidia_backend.BACKBONE_REVISION,
        ("config.json",),
        False,
    )  # the backbone weights are not downloaded: they are in the card repo's weights
    assert fake_nvidia.builds == [("config", BACKBONE_FOLDER)]
    ((state, strict),) = FakeTorch.loaded
    assert strict is True and state == {"file": os.path.join(fake_nvidia.folder, "model.safetensors")}
    assert FakeTokenizer.folder == str(fake_nvidia.folder)
    assert meta == {
        "model": nvidia_backend.REPO, "revision": COMMIT, "backbone": "microsoft/deberta-v3-base",
        "backbone_revision": nvidia_backend.BACKBONE_REVISION, "device": "cpu", "precision": "float32",
        "max_length": 512, "truncated": 0,
    }  # fmt: skip
    given = "a" * 40
    _, meta = nvidia_backend.make_nvidia_classifier(given)
    assert meta["revision"] == given and fake_nvidia.downloads[2][1] == given


def test_the_hub_is_read_without_a_token(fake_nvidia, monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    nvidia_backend.make_nvidia_classifier(None)
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1" and "HF_TOKEN" not in os.environ


def test_classify_runs_one_prompt_at_a_time_as_the_card_does_and_keeps_the_order(fake_nvidia):
    classify, meta = nvidia_backend.make_nvidia_classifier(None)
    prompts = ["a", "bb" * 7, "hello there, what is up", "x" * 300, "y" * 41]
    out = classify(prompts)
    assert len(out) == len(prompts) and all("prompt_complexity_score" in row for row in out)
    assert all(len(row) == 11 and 0 <= row["prompt_complexity_score"] <= 1 for row in out)
    calls = [(p, kw) for p, kw in FakeTokenizer.calls if kw]
    assert [p for p, _ in calls] == [[p] for p in prompts]  # one prompt per call, as they are (no "Prompt: " prefix)
    assert all(
        kw == {"return_tensors": "pt", "add_special_tokens": True, "max_length": 512, "padding": "max_length",
               "truncation": True}
        for _, kw in calls
    )  # fmt: skip
    assert classify([]) == []


def test_prompts_over_the_token_budget_are_counted(fake_nvidia):
    classify, meta = nvidia_backend.make_nvidia_classifier(None)
    assert meta["truncated"] == 0
    classify(["short", "x" * 1600])  # the fake counts len // 3 + 1 tokens, the budget is 512
    assert meta["truncated"] == 1


# ---------- the command ----------

LANGS = ("eng", "fra", "rus", "zho")
KEYS = [f"k{n:02d}" for n in range(9)]
PROMPTS = [
    {"id": f"{key}-{lang}", "prompt": f"{lang} prompt {key} " + "p" * n} for n, key in enumerate(KEYS) for lang in LANGS
]
SCORES = [0.0, 0.05, 0.199999, 0.2, 0.399999, 0.4, 0.599999, 0.6, 0.95]  # one per English prompt, at the bounds


@pytest.fixture
def prompts_file(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    write_jsonl(d / "mini.jsonl", PROMPTS)
    meta = pa.table({"id": [p["id"] for p in PROMPTS], "lang": [p["id"].rsplit("-", 1)[1] for p in PROMPTS]})
    pq.write_table(meta, d / "mini.meta.parquet")
    return d / "mini.jsonl"


def heads_for(score: float) -> dict:
    return {
        "task_type_1": "Open QA", "task_type_2": "NA", "task_type_prob": 0.9, "creativity_scope": 0.1,
        "reasoning": 0.2, "contextual_knowledge": 0.0, "number_of_few_shots": 0, "domain_knowledge": 0.3,
        "no_label_reason": 0.0, "constraint_ct": 0.4, "prompt_complexity_score": score,
    }  # fmt: skip


INFO = {
    "model": nvidia_backend.REPO, "revision": COMMIT, "backbone": "microsoft/deberta-v3-base",
    "backbone_revision": nvidia_backend.BACKBONE_REVISION, "device": "cpu", "precision": "float32", "max_length": 512,
    "truncated": 2,
}  # fmt: skip


def _classify(prompts_file, out, *extra):
    return main(["classify", "nvidia", "--prompts", str(prompts_file), "--out", str(out), "--author", "tester", *extra])


def test_cli_runs_the_english_ids_only_and_leaves_the_others_not_predicted(monkeypatch, prompts_file, tmp_path):
    seen = []

    def classify(prompts):
        seen.append(prompts)
        return [heads_for(s) for s in SCORES]

    monkeypatch.setattr(nvidia_backend, "make_nvidia_classifier", lambda revision: (classify, INFO))
    out = tmp_path / "classifiers" / "nvidia-ptcc-eng"
    assert _classify(prompts_file, out) == 0
    assert seen == [[p["prompt"] for p in PROMPTS if p["id"].endswith("-eng")]]  # 9 calls' worth, no other language
    labels = read_jsonl(out / "labels.jsonl")
    assert [r["id"] for r in labels] == [p["id"] for p in PROMPTS]  # every id has a line, in the file's order
    english = [r for r in labels if r["id"].endswith("-eng")]
    others = [r for r in labels if not r["id"].endswith("-eng")]
    assert (len(english), len(others)) == (9, 27)
    assert [r["complexity"] for r in english] == [
        "trivial", "trivial", "trivial", "low", "low", "medium", "medium", "high", "expert"
    ]  # fmt: skip
    assert [r["score"] for r in english] == SCORES
    assert all(r["task"] is None and r["needs_tools"] is None for r in labels)
    assert all(r == {"id": r["id"], "complexity": None, "task": None, "needs_tools": None} for r in others)
    assert validate_judge_dir(out, prompts_file.parent) == []
    # `bench run` reads it as a classifier: null predictions, 9 scored ids
    loaded = read_classifier_labels(out / "labels.jsonl")
    assert sum(v["complexity"] is not None for v in loaded.values()) == 9
    decisions = read_decisions(out / "decisions.jsonl", [p["id"] for p in PROMPTS])
    assert [d["tier"] for i, d in decisions.items() if i.endswith("-eng")] == [r["complexity"] for r in english]
    assert all(d == {"tier": None, "score": None} for i, d in decisions.items() if not i.endswith("-eng"))
    raw = read_jsonl(out / "decisions.jsonl")
    assert raw[0]["task_type_1"] == "Open QA" and set(raw[0]) == {"id", "tier", "score", *heads_for(0)}  # every head
    assert raw[1] == {"id": "k00-fra", "tier": None, "score": None, "not_predicted": "language not covered"}


def test_cli_meta_says_what_the_model_is_and_the_choices(monkeypatch, prompts_file, tmp_path):
    monkeypatch.setattr(
        nvidia_backend,
        "make_nvidia_classifier",
        lambda revision: (lambda prompts: [heads_for(0.3)] * len(prompts), INFO),
    )
    out = tmp_path / "classifiers" / "nvidia-ptcc-eng"
    assert _classify(prompts_file, out, "--notes", "Run on a test machine.") == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == nvidia_backend.REPO and meta["revision"] == COMMIT and meta["quantization"] == "float32"
    assert (meta["provider"], meta["instructions"], meta["thinking"]) == ("system", "none", "none")
    assert meta["judge"] == "nvidia-ptcc-eng" and meta["author"] == "tester" and meta["licence"] == "CC-BY-4.0"
    for needle in (
        f"uv run --group routellm bench classify nvidia --revision {COMMIT}", nvidia_backend.CARD_REVISION,
        "microsoft/deberta-v3-base@" + nvidia_backend.BACKBONE_REVISION, "strict=True", "English only",
        "(9 of 36)", "never run",
        "trivial [0 ; 0.2), low [0.2 ; 0.4), medium [0.4 ; 0.6), high [0.6 ; 0.8), expert [0.8 ; 1]",
        "task and needs_tools are null", "without the 'Prompt: ' prefix", "2 prompts truncated",
        "Run on a test machine.",
    ):  # fmt: skip
        assert needle in meta["notes"], needle
    assert "/Users/" not in json.dumps(meta)


def test_cli_reads_the_language_from_the_id_when_there_is_no_meta_table(monkeypatch, prompts_file, tmp_path):
    (prompts_file.parent / "mini.meta.parquet").unlink()
    seen = []
    monkeypatch.setattr(
        nvidia_backend, "make_nvidia_classifier",
        lambda revision: (lambda prompts: seen.append(prompts) or [heads_for(0.3)] * len(prompts), INFO),
    )  # fmt: skip
    out = tmp_path / "out"
    assert _classify(prompts_file, out) == 0
    assert len(seen[0]) == 9 and len(read_jsonl(out / "labels.jsonl")) == 36


def test_cli_refuses_an_id_the_meta_table_does_not_know(monkeypatch, prompts_file, tmp_path, capsys):
    meta = pa.table({"id": [p["id"] for p in PROMPTS[:-1]], "lang": ["eng"] * (len(PROMPTS) - 1)})
    pq.write_table(meta, prompts_file.parent / "mini.meta.parquet")
    monkeypatch.setattr(nvidia_backend, "make_nvidia_classifier", lambda revision: pytest.fail("must not load"))
    out = tmp_path / "out"
    assert _classify(prompts_file, out) == 1
    assert "no language for id" in capsys.readouterr().err and not out.exists()


def test_cli_refuses_a_set_with_no_english_prompt(monkeypatch, tmp_path, capsys):
    path = tmp_path / "fr.jsonl"
    write_jsonl(path, [{"id": "a-fra", "prompt": "bonjour"}])
    monkeypatch.setattr(nvidia_backend, "make_nvidia_classifier", lambda revision: pytest.fail("must not load"))
    out = tmp_path / "out"
    assert _classify(path, out) == 1
    assert "nothing to classify" in capsys.readouterr().err and not out.exists()


def test_cli_end_to_end_with_the_fake_model(fake_nvidia, prompts_file, tmp_path):
    out = tmp_path / "classifiers" / "nvidia-ptcc-eng"
    assert _classify(prompts_file, out, "--revision", "e" * 40) == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["revision"] == "e" * 40 and "float32" in meta["notes"]
    labels = read_jsonl(out / "labels.jsonl")
    assert sum(r["complexity"] is not None for r in labels) == 9 and len(labels) == 36
    for row in labels:
        if row["id"].endswith("-eng"):
            assert row["complexity"] == nvidia_backend.tier_of_score(row["score"])
    assert validate_judge_dir(out, prompts_file.parent) == []


def test_cli_names_the_group_to_install_when_torch_is_missing(prompts_file, tmp_path, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)  # an import of torch raises ImportError
    out = tmp_path / "out"
    assert _classify(prompts_file, out) == 1
    assert "--group routellm" in capsys.readouterr().err and not out.exists()


def test_cli_does_not_overwrite_a_folder(fake_nvidia, prompts_file, tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    assert _classify(prompts_file, out) == 1
    assert "already exists" in capsys.readouterr().err


# ---------- decisions not made, read by `bench run` ----------


def test_a_decision_with_a_null_tier_and_a_null_score_is_kept_as_not_made(tmp_path):
    path = tmp_path / "decisions.jsonl"
    write_jsonl(path, [{"id": "a", "tier": "low", "score": 0.3}, {"id": "b", "tier": None, "score": None}])
    assert read_decisions(path, ["a", "b"]) == {"a": {"tier": "low", "score": 0.3}, "b": {"tier": None, "score": None}}


@pytest.mark.parametrize(
    "row",
    [{"id": "b", "tier": None, "score": 0.3}, {"id": "b", "tier": "low", "score": None}, {"id": "b"},
     {"id": "b", "tier": "low"}, {"id": "b", "score": None}],
)  # fmt: skip
def test_a_half_null_decision_is_still_refused(tmp_path, row):
    path = tmp_path / "decisions.jsonl"
    write_jsonl(path, [row])
    with pytest.raises(ValueError, match="needs a string 'tier' and a numeric 'score'"):
        read_decisions(path, ["b"])


# ---------- the card's own example, with the real weights ----------


@pytest.mark.slow
def test_the_real_model_reproduces_the_card_code_sample():
    """The code sample of the card prints these numbers for 'Prompt: Write a Python script that uses a for loop.'
    Needs the `routellm` group and the weights in the Hub cache (read only, no token)."""
    pytest.importorskip("torch")
    classify, meta = nvidia_backend.make_nvidia_classifier(nvidia_backend.CARD_REVISION)
    (heads,) = classify(["Prompt: Write a Python script that uses a for loop."])
    assert meta["revision"] == nvidia_backend.CARD_REVISION
    assert heads == {
        "task_type_1": "Code Generation", "task_type_2": "Text Generation", "task_type_prob": 0.767,
        "creativity_scope": 0.0826, "reasoning": 0.0632, "contextual_knowledge": 0.056, "number_of_few_shots": 0,
        "domain_knowledge": 0.9803, "no_label_reason": 0.0, "constraint_ct": 0.5578,
        "prompt_complexity_score": pytest.approx(0.27822, abs=2e-5),
    }  # fmt: skip
    # the first table of the card (the mystery prompt, with no prefix)
    (mystery,) = classify(
        ["Write a mystery set in a small town where an everyday object goes missing, causing a ripple of curiosity and "
         "suspicion. Follow the investigation and reveal the surprising truth behind the disappearance."]
    )  # fmt: skip
    assert mystery["task_type_1"] == "Text Generation"
    keys = ("prompt_complexity_score", "creativity_scope", "reasoning", "contextual_knowledge", "domain_knowledge",
            "constraint_ct")  # fmt: skip
    # the card's table gives three decimals, truncated rather than rounded (0.7857 is written 0.785)
    assert [mystery[k] for k in keys] == pytest.approx([0.472, 0.867, 0.056, 0.048, 0.226, 0.785], abs=1e-3)
