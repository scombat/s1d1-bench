import json
import math
from types import SimpleNamespace

import pytest

from s1d1_bench import s1d1_backend
from s1d1_bench.classify import write_run
from s1d1_bench.cli import main
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import validate_judge_dir

np = pytest.importorskip("numpy")

TASK_NAMES = (
    "information_seeking", "reasoning", "planning", "editing", "summarization", "extraction",
    "classification", "translation", "coding_debugging", "math", "data_analysis", "writing",
    "role_playing", "advice_seeking", "brainstorming", "completion", "action_execution", "chat", "other",
)  # fmt: skip
TIERS = (("trivial", 0.01), ("low", 0.25), ("medium", 0.50), ("high", 0.75))
SMALL = "scombat/s1d1-router-6src-small-onnx-int8"
FULL_SHA = "33c7f34" + "a" * 33


def fake_decode(logits, tokens):
    """The decode of s1d1_router.py (same formulas), so that the mapping is checked from the logits."""
    out = []
    for row, n in zip(logits.astype(np.float32), tokens, strict=True):
        score = float(np.clip(0.5 + 0.25 * row[0], 0.01, 1.0))
        p_tools = float(1 / (1 + np.exp(-row[-1])))
        out.append({
            "complexity_score": round(score, 3),
            "complexity_tier": min(TIERS, key=lambda t: abs(t[1] - score))[0],
            "task": TASK_NAMES[int(row[1:1 + len(TASK_NAMES)].argmax())],
            "needs_tools_probability": round(p_tools, 3),
            "needs_tools": p_tools >= 0.5,
            "prompt_tokens": int(n),
        })  # fmt: skip
    return out


def logits_row(text: str) -> np.ndarray:
    """21 logits that depend on the text only: complexity units from its length, the task from the first letter."""
    row = np.zeros(21, dtype=np.float32)
    row[0] = (len(text) % 7 - 2) / 1.0  # -2 .. 4 units
    row[1 + ord(text[0]) % 19] = 5.0
    row[-1] = 2.0 if "tool" in text else -2.0
    return row


class FakeRouter:
    instances: list = []
    batches: list = []

    def __init__(self, repo, device="cpu", backend="auto"):
        self.repo, self.device, self.backend = repo, device, backend
        FakeRouter.instances.append(self)

    def logits(self, prompts):
        FakeRouter.batches.append(list(prompts))
        return np.stack([logits_row(p) for p in prompts]), np.array([len(p) for p in prompts])


@pytest.fixture
def fake_hub(monkeypatch, tmp_path):
    """A router module, a snapshot folder and a model_info without any network."""
    FakeRouter.instances, FakeRouter.batches = [], []
    folder = tmp_path / "snapshot"
    folder.mkdir()
    (folder / "variant.json").write_text(json.dumps({"backend": "onnx", "file": "model_int8.onnx"}), encoding="utf-8")
    module = SimpleNamespace(
        Router=FakeRouter, decode=fake_decode, MAX_LENGTH=1024, HUB_FILES=("*.json",), backend_of=lambda f: "onnx"
    )
    calls = SimpleNamespace(fetch=[], snapshot=[], info=[])
    heads = {SMALL: FULL_SHA, "scombat/s1d1-router": "7fb9a4e" + "b" * 33}

    def model_info(repo, revision=None):
        calls.info.append((repo, revision))
        return SimpleNamespace(sha=revision if revision and len(revision) == 40 else heads.get(repo, "0" * 40))

    monkeypatch.setattr(s1d1_backend, "fetch_router_module", lambda revision: calls.fetch.append(revision) or module)

    def snapshot(repo, revision, patterns):
        calls.snapshot.append((repo, revision, patterns))
        return str(folder)

    monkeypatch.setattr(s1d1_backend, "snapshot", snapshot)
    monkeypatch.setattr(s1d1_backend, "model_info", model_info)
    monkeypatch.setattr(s1d1_backend, "mps_available", lambda: False)
    calls.module, calls.heads, calls.folder = module, heads, folder
    return calls


# ---------- batches ----------


def test_plan_batches_sorts_by_length_and_covers_every_prompt_once():
    prompts = ["x" * n for n in (50, 5, 500, 5, 20)]
    batches = s1d1_backend.plan_batches(prompts, max_prompts=64, max_tokens=16384, max_length=1024)
    assert batches == [[1, 3, 4, 0, 2]]  # one batch; ties keep the file order; shortest first
    assert sorted(i for b in batches for i in b) == list(range(5))


def test_plan_batches_cap_on_prompts_and_on_tokens():
    short = ["ab"] * 150
    sizes = [len(b) for b in s1d1_backend.plan_batches(short, 64, 16384, 1024)]
    assert sizes == [64, 64, 22]
    # 3000 characters = 1000 estimated tokens (1 token per 3 characters): 16 of them fill 16 000 of 16 384 tokens
    longer = ["y" * 3000] * 40
    sizes = [len(b) for b in s1d1_backend.plan_batches(longer, 64, 16384, 1024)]
    assert sizes == [16, 16, 8]
    # a prompt beyond MAX_LENGTH counts for MAX_LENGTH tokens (truncated by the router), not for its length
    huge = ["z" * 100_000] * 17
    assert [len(b) for b in s1d1_backend.plan_batches(huge, 64, 16384, 1024)] == [16, 1]
    # a single prompt is always a batch of its own, whatever its size
    assert s1d1_backend.plan_batches(["w" * 10**6], 64, 16, 1024) == [[0]]
    assert s1d1_backend.plan_batches([], 64, 16384, 1024) == []


def test_plan_batches_padding_cost_follows_the_longest_prompt_of_the_batch():
    prompts = ["a" * 30] * 100 + ["b" * 3000] * 2
    batches = s1d1_backend.plan_batches(prompts, 64, 16384, 1024)
    # the 36 short ones left over do not join the long ones: 38 prompts padded to 1000 tokens cost 38 000 > 16 384
    assert [len(b) for b in batches] == [64, 36, 2]
    assert all(len(prompts[i]) == 30 for b in batches[:2] for i in b)


# ---------- the classifier function ----------


def test_classifier_maps_the_logits_and_restores_the_order(fake_hub):
    classify, meta = s1d1_backend.make_s1d1_classifier(SMALL, "33c7f34", "auto")
    prompts = ["a long prompt that needs tools", "b", "medium sized one", "tool", "zz", "x" * 2500]
    out = classify(prompts)
    assert len(out) == len(prompts)
    for text, row in zip(prompts, out, strict=True):
        expected = fake_decode(logits_row(text)[None, :], np.array([len(text)]))[0]
        assert row == {
            "complexity": expected["complexity_tier"],
            "task": expected["task"],
            "needs_tools": expected["needs_tools"],
            "score": expected["complexity_score"],
            "needs_tools_probability": expected["needs_tools_probability"],
        }
    # the batch of the fake router saw the prompts shortest first
    seen = [p for batch in FakeRouter.batches for p in batch]
    assert seen == sorted(prompts, key=len)
    assert {r["complexity"] for r in out} <= {"trivial", "low", "medium", "high"}  # never expert (capped by decode)
    assert type(out[3]["needs_tools"]) is bool and out[3]["needs_tools"] is True and out[1]["needs_tools"] is False


def test_classifier_batches_at_most_64_prompts(fake_hub):
    classify, _ = s1d1_backend.make_s1d1_classifier(SMALL, "33c7f34", "onnx")
    prompts = [f"prompt number {n}" for n in range(150)]
    out = classify(prompts)
    assert [len(b) for b in FakeRouter.batches] == [64, 64, 22]
    assert [r["task"] for r in out] == [fake_decode(logits_row(p)[None], np.array([1]))[0]["task"] for p in prompts]


def test_classifier_empty_input_loads_nothing_and_returns_nothing(fake_hub):
    classify, _ = s1d1_backend.make_s1d1_classifier(SMALL, "33c7f34", "onnx")
    assert classify([]) == []
    assert FakeRouter.batches == []


# ---------- revisions ----------


def test_revision_is_resolved_at_run_time_and_the_weights_are_pinned_to_it(fake_hub):
    fake_hub.heads["scombat/s1d1-router"] = "a" * 40
    _, meta = s1d1_backend.make_s1d1_classifier("scombat/s1d1-router", None, "torch")
    assert meta["revision"] == "a" * 40 and meta["model"] == "scombat/s1d1-router"
    assert fake_hub.info == [("scombat/s1d1-router", None)]
    assert fake_hub.snapshot == [("scombat/s1d1-router", "a" * 40, ("*.json",))]  # the Router gets the pinned folder
    assert FakeRouter.instances[0].repo == str(fake_hub.folder)
    assert fake_hub.fetch == ["a" * 40]  # the fp16 repo carries its own s1d1_router.py, at the same revision


def test_the_router_file_comes_from_the_main_repo_when_the_weights_are_elsewhere(fake_hub):
    _, meta = s1d1_backend.make_s1d1_classifier(SMALL, "33c7f34", "auto")
    assert meta["revision"] == FULL_SHA
    assert fake_hub.fetch == [fake_hub.heads["scombat/s1d1-router"]]
    assert meta["router_revision"] == fake_hub.heads["scombat/s1d1-router"]


def test_the_router_revision_pins_the_router_file(fake_hub):
    pinned = "c" * 40
    _, meta = s1d1_backend.make_s1d1_classifier(SMALL, "33c7f34", "auto", router_revision=pinned)
    assert meta["router_revision"] == pinned and meta["revision"] == FULL_SHA  # the weights keep their own revision
    assert fake_hub.fetch == [pinned]
    assert ("scombat/s1d1-router", pinned) in fake_hub.info
    _, meta = s1d1_backend.make_s1d1_classifier("scombat/s1d1-router", None, "torch", router_revision=pinned)
    assert meta["router_revision"] == pinned and fake_hub.fetch[-1] == pinned  # even for the repo that carries it


@pytest.mark.parametrize(
    ("repo", "prefix"),
    [
        ("scombat/s1d1-router-6src-small-onnx-int8", "33c7f34"),
        ("scombat/s1d1-router-6src-full-onnx-int8", "068d0d2"),
        ("scombat/s1d1-router-6src-compact-onnx-int8", "7e87870"),
    ],
)
def test_onnx_int8_repos_refuse_a_revision_that_is_not_the_card_one(fake_hub, repo, prefix):
    fake_hub.heads[repo] = "bd27c6b" + "0" * 33  # the head moved on after the card revision
    with pytest.raises(ValueError, match=rf"{prefix}.*bd27c6b|bd27c6b.*{prefix}"):
        s1d1_backend.make_s1d1_classifier(repo, None, "onnx")
    assert fake_hub.snapshot == [] and fake_hub.fetch == [] and FakeRouter.instances == []  # nothing downloaded
    fake_hub.heads[repo] = prefix + "f" * 33  # the card revision itself
    _, meta = s1d1_backend.make_s1d1_classifier(repo, None, "onnx")
    assert meta["revision"].startswith(prefix)


def test_a_given_revision_is_checked_too(fake_hub):
    with pytest.raises(ValueError, match="33c7f34"):
        s1d1_backend.make_s1d1_classifier(SMALL, "deadbeef" + "0" * 32, "onnx")
    assert fake_hub.snapshot == []
    _, meta = s1d1_backend.make_s1d1_classifier(SMALL, FULL_SHA, "onnx")
    assert meta["revision"] == FULL_SHA


def test_tiny_and_the_fp16_reference_have_no_card_revision_to_check(fake_hub):
    fake_hub.heads["scombat/s1d1-router-6src-tiny-onnx-int8"] = "dac13b5" + "0" * 33
    for repo in ("scombat/s1d1-router-6src-tiny-onnx-int8", "scombat/s1d1-router"):
        _, meta = s1d1_backend.make_s1d1_classifier(repo, None, "auto")
        assert meta["revision"] == fake_hub.heads[repo]


# ---------- backend, device, quantization ----------


def test_backend_is_passed_through_and_quantization_names_backend_and_precision(fake_hub):
    _, meta = s1d1_backend.make_s1d1_classifier(SMALL, "33c7f34", "auto")
    router = FakeRouter.instances[-1]
    assert (router.backend, router.device) == ("auto", "cpu")
    assert meta["quantization"] == "onnx-int8" and meta["backend"] == "onnx" and meta["device"] == "cpu"


def test_torch_runs_on_mps_when_available_and_says_its_precision(fake_hub, monkeypatch):
    fake_hub.module.backend_of = lambda f: "torch"
    monkeypatch.setattr(s1d1_backend, "mps_available", lambda: True)

    def torch_router(repo, device="cpu", backend="auto"):
        router = FakeRouter(repo, device, backend)
        router.model = SimpleNamespace(dtype="torch.float16")
        return router

    fake_hub.module.Router = torch_router
    _, meta = s1d1_backend.make_s1d1_classifier("scombat/s1d1-router", None, "torch")
    assert FakeRouter.instances[-1].device == "mps" and FakeRouter.instances[-1].backend == "torch"
    assert meta["quantization"] == "torch-fp16" and meta["device"] == "mps"
    monkeypatch.setattr(s1d1_backend, "mps_available", lambda: False)
    _, meta = s1d1_backend.make_s1d1_classifier("scombat/s1d1-router", None, "torch")
    assert FakeRouter.instances[-1].device == "cpu"
    assert meta["device"] == "cpu"


def test_the_hub_is_read_without_a_token(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_IMPLICIT_TOKEN", raising=False)
    s1d1_backend.hub_env()
    import os

    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


# ---------- the command ----------

PROMPTS = [
    {"id": f"b{n}-{lang}", "prompt": ("use a tool " if n % 2 else "p") * (n + 1) + lang}
    for n in range(6)
    for lang in ("eng", "fra")
]


@pytest.fixture
def prompts_file(tmp_path):
    d = tmp_path / "prompts"
    d.mkdir()
    write_jsonl(d / "mini.jsonl", PROMPTS)
    return d / "mini.jsonl"


def _classify(prompts_file, out, *extra):
    return main(["classify", "s1d1", "--prompts", str(prompts_file), "--out", str(out), "--author", "tester", *extra])


def test_cli_writes_a_valid_folder_with_the_resolved_revision(fake_hub, prompts_file, tmp_path, capsys):
    out = tmp_path / "classifiers" / "s1d1-small-int8"
    assert _classify(prompts_file, out, "--model", SMALL, "--revision", "33c7f34", "--backend", "onnx") == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == SMALL and meta["revision"] == FULL_SHA
    assert meta["quantization"] == "onnx-int8" and meta["replayable"] is True and meta["temperature"] == 0
    assert (meta["provider"], meta["instructions"], meta["thinking"]) == ("system", "none", "none")
    assert meta["judge"] == "s1d1-small-int8" and meta["author"] == "tester"
    for needle in (
        "bench classify s1d1", SMALL, "--backend onnx", "onnx", "cpu", "int8", "scombat/s1d1-router@", "64", "16384",
        "length",
    ):  # fmt: skip
        assert needle in meta["notes"], needle
    labels = read_jsonl(out / "labels.jsonl")
    assert [r["id"] for r in labels] == [p["id"] for p in PROMPTS]
    for row, item in zip(labels, PROMPTS, strict=True):
        expected = fake_decode(logits_row(item["prompt"])[None], np.array([1]))[0]
        assert row["complexity"] == expected["complexity_tier"] and row["task"] == expected["task"]
        assert row["needs_tools"] is expected["needs_tools"]
        assert row["score"] == expected["complexity_score"]
        assert row["needs_tools_probability"] == expected["needs_tools_probability"]
        assert math.isfinite(row["score"])
    assert validate_judge_dir(out, prompts_file.parent) == []


def test_cli_writes_the_router_revision_into_the_notes(fake_hub, prompts_file, tmp_path):
    pinned = "d" * 40
    out = tmp_path / "pinned"
    assert _classify(prompts_file, out, "--model", SMALL, "--revision", "33c7f34", "--router-revision", pinned) == 0
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert f"--router-revision {pinned}" in meta["notes"] and f"scombat/s1d1-router@{pinned}" in meta["notes"]
    assert fake_hub.fetch == [pinned]
    default = tmp_path / "default"  # without the option: the head of the router repo, written all the same
    assert _classify(prompts_file, default, "--model", SMALL, "--revision", "33c7f34") == 0
    head = fake_hub.heads["scombat/s1d1-router"]
    notes = json.loads((default / "meta.json").read_text(encoding="utf-8"))["notes"]
    assert f"--router-revision {head}" in notes and fake_hub.fetch[-1] == head


def test_cli_refuses_on_the_card_revision_and_writes_nothing(fake_hub, prompts_file, tmp_path, capsys):
    fake_hub.heads[SMALL] = "14fd140" + "0" * 33
    out = tmp_path / "out"
    assert _classify(prompts_file, out, "--model", SMALL) == 1
    err = capsys.readouterr().err
    assert "33c7f34" in err and "14fd140" in err and "--revision" in err
    assert not out.exists()


def test_cli_needs_a_model(fake_hub, prompts_file, tmp_path, capsys):
    out = tmp_path / "out"
    assert _classify(prompts_file, out) == 1
    assert "--model" in capsys.readouterr().err and not out.exists()


def test_cli_names_the_group_to_install_when_a_dependency_is_missing(
    fake_hub, prompts_file, tmp_path, capsys, monkeypatch
):
    def missing(revision):
        raise ModuleNotFoundError("No module named 'tokenizers'", name="tokenizers")

    monkeypatch.setattr(s1d1_backend, "fetch_router_module", missing)
    out = tmp_path / "out"
    assert _classify(prompts_file, out, "--model", "scombat/s1d1-router", "--backend", "torch") == 1
    err = capsys.readouterr().err
    assert "tokenizers" in err and "--group s1d1" in err and "s1d1-torch" in err and not out.exists()


# ---------- write_run: the optional quantization ----------


def _args(out, notes=""):
    return SimpleNamespace(out=out, author="tester", notes=notes)


ROWS = [{"id": "a-eng", "complexity": "low", "task": None, "needs_tools": None}]


def test_write_run_carries_the_quantization_only_when_given(tmp_path):
    kwargs = {"model": "m", "revision": "r", "notes": "n"}
    assert write_run(_args(tmp_path / "plain"), "0" * 64, ROWS, **kwargs) == 0
    assert "quantization" not in json.loads((tmp_path / "plain" / "meta.json").read_text(encoding="utf-8"))
    assert write_run(_args(tmp_path / "quant"), "0" * 64, ROWS, **kwargs, quantization="onnx-int8") == 0
    meta = json.loads((tmp_path / "quant" / "meta.json").read_text(encoding="utf-8"))
    assert meta["quantization"] == "onnx-int8"


# ---------- the real router, once ----------


@pytest.mark.slow
def test_small_onnx_int8_classifies_two_prompts_from_the_hub():
    classify, meta = s1d1_backend.make_s1d1_classifier(SMALL, "33c7f34", "auto")
    assert meta["revision"].startswith("33c7f34") and meta["quantization"] == "onnx-int8" and meta["device"] == "cpu"
    prompts = [
        "Quelle est la capitale de l'Australie ?",
        "Write a Python script that scrapes three websites, merges their tables, plots a regression and "
        "summarises the outcome for a non-technical audience, calling the weather API for the dates involved.",
    ]
    out = classify(prompts)
    assert len(out) == 2
    for row in out:
        assert row["complexity"] in ("trivial", "low", "medium", "high")
        assert row["task"] in TASK_NAMES and isinstance(row["needs_tools"], bool)
        assert 0.01 <= row["score"] <= 1.0 and 0.0 <= row["needs_tools_probability"] <= 1.0
    assert out[0]["score"] < out[1]["score"]
    print(meta, out)
