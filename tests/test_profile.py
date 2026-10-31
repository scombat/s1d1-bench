import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from s1d1_bench import profile
from s1d1_bench.cli import build_parser, main
from s1d1_bench.jsonl import write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[1]
ROW_KEYS = {
    "name", "kind", "model", "revision", "device", "weights_mb",
    "latency_p50_ms", "latency_p95_ms", "latency_mean_ms", "peak_rss_mb", "note",
}  # fmt: skip

# ---------- the measurement of a classifier (review focus: after warm-up, batch of 1) ----------


def _counting_classifier(warmup: int, warm_ns: int, measured_ns: int):
    """A fake classifier that records every call and moves a fake clock: `warm_ns` per call during the first `warmup`
    calls, `measured_ns` per call after them."""
    state = {"now": 0, "calls": []}

    def classify(prompts):
        state["calls"].append(list(prompts))
        state["now"] += warm_ns if len(state["calls"]) <= warmup else measured_ns
        return [{"complexity": "low"} for _ in prompts]

    return classify, state


def test_profile_classifier_warms_up_then_measures_every_prompt_alone():
    prompts = [f"prompt {i}" for i in range(360)]
    classify, state = _counting_classifier(10, warm_ns=1_000_000_000, measured_ns=2_000_000)
    result = profile.profile_classifier(classify, prompts, warmup=10, clock=lambda: state["now"])
    calls = state["calls"]
    assert len(calls) == 370  # 10 warm-up calls, not counted, then 360 measured
    assert all(len(call) == 1 for call in calls)  # batch of 1: one prompt per call
    assert calls[:10] == [[p] for p in prompts[:10]]
    assert calls[10:] == [[p] for p in prompts]  # every prompt measured once, in file order
    assert result["n"] == 360
    # the slow warm-up calls (1 s each) are nowhere in the figures: every measured call took 2 ms
    assert (result["p50_ms"], result["p95_ms"], result["mean_ms"]) == (2.0, 2.0, 2.0)
    assert result["peak_rss_mb"] > 0


def test_profile_classifier_cycles_the_prompts_for_a_warm_up_longer_than_the_file():
    classify, state = _counting_classifier(5, warm_ns=7, measured_ns=1_000_000)
    result = profile.profile_classifier(classify, ["a", "b"], warmup=5, clock=lambda: state["now"])
    assert state["calls"] == [["a"], ["b"], ["a"], ["b"], ["a"], ["a"], ["b"]]
    assert result["n"] == 2 and result["mean_ms"] == 1.0


def test_profile_classifier_refuses_a_classifier_that_does_not_answer_one_row_per_prompt():
    with pytest.raises(ValueError, match="1 prompt"):
        profile.profile_classifier(lambda prompts: [], ["a"], warmup=0)


def test_percentile_interpolates_between_ranks():
    values = list(range(1, 101))
    assert profile.percentile(values, 50) == 50.5
    assert profile.percentile(values, 95) == pytest.approx(95.05)
    assert profile.percentile([3.0], 95) == 3.0
    assert profile.percentile([4.0, 1.0, 2.0], 50) == 2.0  # unsorted input


def test_rss_units_bytes_on_macos_kilobytes_on_linux():
    assert profile.rss_mb(250_000_000, "darwin") == 250.0
    assert profile.rss_mb(250_000, "linux") == 256.0


# ---------- judges ----------


def _judge_folder(root: Path, name: str = "j1", thinking: str = "cap:512") -> Path:
    folder = root / name
    folder.mkdir(parents=True)
    rows = [
        {"id": "a-eng", "status": "ok", "retries": 0, "wall_s": 2.0, "prompt_tokens": 1900, "completion_tokens": 100},
        {"id": "b-eng", "status": "ok", "retries": 0, "wall_s": 4.0, "prompt_tokens": 1910, "completion_tokens": 300},
        {"id": "c-eng", "status": "json_error", "retries": 1, "wall_s": 99.0},  # no answer: left out
        {"id": "d-eng", "status": "ok", "retries": 0, "wall_s": 6.0, "prompt_tokens": 1920, "completion_tokens": 200},
        {"id": "e-eng", "status": "ok", "retries": 1, "wall_s": 8.0, "prompt_tokens": 1930, "completion_tokens": 400},
    ]
    write_jsonl(folder / "log.jsonl", rows)
    meta = {"judge": name, "thinking": thinking, "max_tokens": 1024, "mlx_lm_version": "0.32.2"}
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return folder


def test_profile_judge_reads_the_ok_calls_of_the_log(tmp_path):
    result = profile.profile_judge(_judge_folder(tmp_path))
    assert result["n"] == 5 and result["n_ok"] == 4  # the json_error line is left out
    assert result["p50_s"] == 5.0 and result["mean_s"] == 5.0
    assert result["p95_s"] == pytest.approx(7.7)
    assert result["mean_prompt_tokens"] == 1915.0 and result["mean_completion_tokens"] == 250.0


# ---------- weights ----------


def test_weights_bytes_sums_the_weight_files_of_the_folder_only(tmp_path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc").write_bytes(b"x" * 1000)
    folder = tmp_path / "snapshot"
    folder.mkdir()
    (folder / "model.safetensors").symlink_to(blobs / "abc")  # Hub cache layout: a link to its blob
    for name, size in {"model_int8.onnx": 10, "w.bin": 20, "w.npz": 30, "w.gguf": 40, "w.pt": 50}.items():
        (folder / name).write_bytes(b"x" * size)
    (folder / "tokenizer.json").write_bytes(b"x" * 5000)  # not a weight file
    (folder / "multilingual").mkdir()  # a subfolder holds another checkpoint: not counted
    (folder / "multilingual" / "model.safetensors").write_bytes(b"x" * 7000)
    assert profile.weights_bytes(folder) == 1150
    assert profile.weights_bytes(folder / "multilingual") == 7000


# ---------- the file ----------


def test_write_profile_is_deterministic(tmp_path):
    rows = [{"name": "b", "kind": "judge", "note": "z"}, {"name": "a", "kind": "classifier", "note": "y"}]
    meta = {"warmup": 10, "batch_size": 1, "skipped": []}
    out = tmp_path / "p.json"
    profile.write_profile(out, rows, meta)
    first = out.read_bytes()
    profile.write_profile(out, rows, meta)
    assert out.read_bytes() == first
    assert first.decode() == json.dumps({"meta": meta, "rows": rows}, sort_keys=True, indent=2) + "\n"


# ---------- which classifier, how to reload it ----------


def _meta(name: str) -> dict:
    return json.loads((REPO_ROOT / "classifiers" / name / "meta.json").read_text(encoding="utf-8"))


def test_classifier_spec_reads_the_replay_line_of_the_real_folders():
    assert profile.classifier_spec(_meta("s1d1-compact-int8")) == {
        "adapter": "s1d1",
        "model": "scombat/s1d1-router-6src-compact-onnx-int8",
        "revision": "7e87870132a19d12b32dbad1f8233801b8145b9f",
        "backend": "onnx",
        "router_revision": "7fb9a4e3d03e3d942a98648268ff76c6152c18d6",
    }
    assert profile.classifier_spec(_meta("s1d1-full-fp16"))["backend"] == "torch"
    assert profile.classifier_spec(_meta("laya-multilingual-tier")) == {
        "adapter": "laya",
        "laya_model": "multilingual",
        "commit": "7b928d828b7b0e022f929d9bd2e44165aa270148",
    }
    assert profile.classifier_spec(_meta("laya-english-tier"))["laya_model"] == "english"
    assert profile.classifier_spec(_meta("litellm-heuristic")) == {"adapter": "litellm-heuristic"}
    assert profile.classifier_spec(_meta("routellm-bert")) == {
        "adapter": "routellm",
        "router": "bert",
        "revision": "86237e3df400762178ea98379477b8296e66d5e4",
    }
    assert profile.classifier_spec(_meta("nvidia-ptcc-eng")) == {
        "adapter": "nvidia",
        "revision": "fea1121511eafabaf7dd6fc66863dcb04f74defb",
    }
    for witness in ("random", "length", "constant-low"):
        assert profile.classifier_spec(_meta(witness)) is None


# ---------- the command ----------


def _setup(tmp_path: Path, classifiers: dict[str, dict], judges: tuple[str, ...] = ("j1",)) -> dict:
    """A prompts file, classifier folders (name -> meta.json), judge folders, an mlx weights folder per judge model
    and the run.json naming them all."""
    prompts = tmp_path / "prompts.jsonl"
    write_jsonl(prompts, [{"id": f"k{i}-eng", "prompt": f"what is {i} + {i}?"} for i in range(3)])
    digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
    for name, meta in classifiers.items():
        (tmp_path / "classifiers" / name).mkdir(parents=True)
        (tmp_path / "classifiers" / name / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    weights = tmp_path / "mlx" / "org--model"
    weights.mkdir(parents=True)
    (weights / "model-00001-of-00002.safetensors").write_bytes(b"x" * 1_500_000)
    (weights / "model-00002-of-00002.safetensors").write_bytes(b"x" * 500_000)
    (weights / "REVISION").write_text("f" * 40, encoding="utf-8")
    for name in judges:
        _judge_folder(tmp_path / "judges", name)
    run = {
        "prompts_sha256": digest,
        "classifiers": [{"name": n, "model": m["model"], "revision": m["revision"]} for n, m in classifiers.items()],
        "judges": [{"name": n, "model": "org/model", "revision": "f" * 40} for n in judges],
    }
    (tmp_path / "run.json").write_text(json.dumps(run), encoding="utf-8")
    argv = ["profile", "--prompts", str(prompts), "--run", str(tmp_path / "run.json")]
    argv += ["--classifiers-dir", str(tmp_path / "classifiers"), "--judges-dir", str(tmp_path / "judges")]
    argv += ["--models-dir", str(tmp_path / "mlx"), "--out", str(tmp_path / "out.json")]
    return {"argv": argv, "out": tmp_path / "out.json"}


WITNESS = {"model": "witness:constant-low", "revision": "n/a", "notes": "Witness: the tier low for every prompt."}
S1D1 = {
    "model": "scombat/s1d1-router-6src-tiny-onnx-int8",
    "revision": "d" * 40,
    "notes": f"Replay: bench classify s1d1 --model scombat/s1d1-router-6src-tiny-onnx-int8 --revision {'d' * 40} "
    f"--backend onnx. s1d1_router.py of scombat/s1d1-router@{'7' * 40}, backend onnx, device cpu.",
}
LITELLM = {"model": "litellm:complexity_router:heuristic", "revision": "v1.106.6 61c6ab7", "notes": "Replay: ..."}


def _fake_worker(calls: list):
    def run_worker(spec):
        calls.append(spec)
        return {
            "device": "cpu", "weights_bytes": 24_742_151, "weight_files": ["model_int8.onnx"],
            "quantization": "onnx-int8", "work": "tokenisation, forward pass, decoding",
            "n": 3, "p50_ms": 1.5, "p95_ms": 2.5, "mean_ms": 1.75, "peak_rss_mb": 321.04,
        }  # fmt: skip

    return run_worker


def test_profile_skips_the_witnesses_and_profiles_the_models_and_judges(tmp_path, monkeypatch):
    setup = _setup(tmp_path, {"constant-low": WITNESS, "s1d1-tiny-int8": S1D1}, judges=("j1", "j2"))
    calls: list = []
    monkeypatch.setattr(profile, "run_worker", _fake_worker(calls))
    assert main(setup["argv"]) == 0
    data = json.loads(setup["out"].read_text(encoding="utf-8"))
    assert [row["name"] for row in data["rows"]] == ["s1d1-tiny-int8", "j1", "j2"]  # the names of run.json
    assert all(set(row) == ROW_KEYS for row in data["rows"])
    assert data["meta"]["skipped"] == [{"name": "constant-low", "reason": "witness: not a model"}]
    assert [spec["adapter"] for spec in calls] == ["s1d1"]  # the witness never reached a worker
    assert calls[0]["revision"] == "d" * 40 and calls[0]["router_revision"] == "7" * 40
    assert calls[0]["warmup"] == 10 and calls[0]["prompts"] == setup["argv"][2]
    s1d1, judge = data["rows"][0], data["rows"][1]
    assert (s1d1["kind"], s1d1["device"], s1d1["weights_mb"], s1d1["peak_rss_mb"]) == ("classifier", "cpu", 24.7, 321.0)
    assert (s1d1["latency_p50_ms"], s1d1["latency_p95_ms"], s1d1["latency_mean_ms"]) == (1.5, 2.5, 1.75)
    assert (s1d1["model"], s1d1["revision"]) == (S1D1["model"], S1D1["revision"])
    assert (judge["kind"], judge["device"], judge["peak_rss_mb"]) == ("judge", "gpu-mlx", None)
    assert judge["weights_mb"] == 2.0
    assert (judge["latency_p50_ms"], judge["latency_mean_ms"]) == (5000.0, 5000.0)
    assert "1 900" in judge["note"] and "cap:512" in judge["note"] and "4 ok" in judge["note"]
    meta = data["meta"]
    assert (meta["warmup"], meta["batch_size"], meta["prompts"]) == (10, 1, setup["argv"][2])
    assert meta["prompts_sha256"] == hashlib.sha256(Path(setup["argv"][2]).read_bytes()).hexdigest()
    assert meta["machine"] and meta["date"]


ROUTELLM = {
    "model": "routellm/bert_gpt4_augmented",
    "revision": "a" * 40,
    "notes": f"Replay: uv run --group routellm bench classify routellm --routellm-router bert --revision {'a' * 40}",
}
LLM4B = {"model": "org/model", "revision": "f" * 40, "provider": "local", "notes": "a small LLM, judge format"}


def test_profile_reloads_a_routellm_router_by_its_adapter(tmp_path, monkeypatch):
    setup = _setup(tmp_path, {"routellm-bert": ROUTELLM}, judges=())
    calls: list = []
    monkeypatch.setattr(profile, "run_worker", _fake_worker(calls))
    assert main(setup["argv"]) == 0
    assert calls[0]["adapter"] == "routellm" and calls[0]["router"] == "bert" and calls[0]["revision"] == "a" * 40
    (row,) = json.loads(setup["out"].read_text(encoding="utf-8"))["rows"]
    assert (row["name"], row["kind"], row["model"]) == ("routellm-bert", "classifier", ROUTELLM["model"])


NVIDIA = {"model": "nvidia/prompt-task-and-complexity-classifier", "revision": "b" * 40, "notes": "Replay: ..."}


def test_profile_reloads_the_nvidia_classifier_by_its_adapter(tmp_path, monkeypatch):
    setup = _setup(tmp_path, {"nvidia-ptcc-eng": NVIDIA}, judges=())
    calls: list = []
    monkeypatch.setattr(profile, "run_worker", _fake_worker(calls))
    assert main(setup["argv"]) == 0
    assert calls[0]["adapter"] == "nvidia" and calls[0]["revision"] == "b" * 40
    (row,) = json.loads(setup["out"].read_text(encoding="utf-8"))["rows"]
    assert (row["name"], row["kind"], row["model"]) == ("nvidia-ptcc-eng", "classifier", NVIDIA["model"])
    with pytest.raises(ValueError, match="full revision"):
        profile.classifier_spec({**NVIDIA, "revision": "main"})


def test_the_nvidia_classifier_is_timed_on_the_english_prompts_only(tmp_path, monkeypatch):
    prompts = tmp_path / "p.jsonl"
    write_jsonl(
        prompts,
        [{"id": f"k{i}-{lang}", "prompt": f"{lang} {i}"} for i in range(2) for lang in ("eng", "fra", "rus", "zho")],
    )
    timed: list = []

    def fake_load(spec):
        return (lambda batch: timed.append(batch[0]) or [{}]), {"device": "cpu"}

    monkeypatch.setattr(profile, "_load", fake_load)
    monkeypatch.setattr(profile, "peak_rss_mb", lambda: 1.0)
    result = profile.run_in_process({"adapter": "nvidia", "prompts": str(prompts), "warmup": 0})
    assert timed == ["eng 0", "eng 1"] and result["n"] == 2  # the other languages are never run, so never timed
    timed.clear()
    result = profile.run_in_process({"adapter": "routellm", "prompts": str(prompts), "warmup": 0})
    assert result["n"] == 8  # the other adapters are timed on every prompt, as before


def test_profile_times_a_judge_format_candidate_from_its_log_and_keeps_it_a_candidate(tmp_path, monkeypatch):
    setup = _setup(tmp_path, {"llm4b": LLM4B, "s1d1-tiny-int8": S1D1}, judges=())
    shutil.copy(
        _judge_folder(tmp_path / "scratch", "j") / "log.jsonl", tmp_path / "classifiers" / "llm4b" / "log.jsonl"
    )
    calls: list = []
    monkeypatch.setattr(profile, "run_worker", _fake_worker(calls))
    assert main(setup["argv"]) == 0
    data = json.loads(setup["out"].read_text(encoding="utf-8"))
    assert [row["name"] for row in data["rows"]] == ["llm4b", "s1d1-tiny-int8"]
    assert [spec["adapter"] for spec in calls] == ["s1d1"]  # the LLM never reached a worker: it is timed from its log
    llm = data["rows"][0]
    assert (llm["kind"], llm["device"], llm["peak_rss_mb"], llm["weights_mb"]) == ("classifier", "gpu-mlx", None, 2.0)
    assert (llm["latency_p50_ms"], llm["latency_mean_ms"]) == (5000.0, 5000.0)
    assert data["meta"]["skipped"] == []
    assert main([*setup["argv"], "--only", "s1d1-tiny-int8"]) == 0
    skipped = json.loads(setup["out"].read_text(encoding="utf-8"))["meta"]["skipped"]
    assert skipped == [{"name": "llm4b", "reason": "not selected by --only"}]


def test_profile_refuses_a_name_that_is_not_in_the_run(tmp_path, monkeypatch, capsys):
    setup = _setup(tmp_path, {"s1d1-tiny-int8": S1D1})
    monkeypatch.setattr(profile, "run_worker", _fake_worker([]))
    assert main([*setup["argv"], "--only", "s1d1-tiny-int8,s1d1-huge-int8"]) == 1
    assert "s1d1-huge-int8" in capsys.readouterr().err
    assert not setup["out"].exists()


def test_profile_only_lists_the_others_as_skipped(tmp_path, monkeypatch):
    setup = _setup(tmp_path, {"constant-low": WITNESS, "s1d1-tiny-int8": S1D1})
    calls: list = []
    monkeypatch.setattr(profile, "run_worker", _fake_worker(calls))
    assert main([*setup["argv"], "--only", "j1"]) == 0
    data = json.loads(setup["out"].read_text(encoding="utf-8"))
    assert [row["name"] for row in data["rows"]] == ["j1"] and calls == []
    assert data["meta"]["skipped"] == [
        {"name": "constant-low", "reason": "witness: not a model"},
        {"name": "s1d1-tiny-int8", "reason": "not selected by --only"},
    ]


def test_profile_lists_a_model_that_cannot_load_as_skipped_and_goes_on(tmp_path, monkeypatch, capsys):
    setup = _setup(tmp_path, {"s1d1-tiny-int8": S1D1})

    def broken(spec):
        raise RuntimeError("model_int8.onnx is not in the local Hub cache")

    monkeypatch.setattr(profile, "run_worker", broken)
    assert main(setup["argv"]) == 1  # the file is written, the exit code says that a model is missing
    data = json.loads(setup["out"].read_text(encoding="utf-8"))
    assert [row["name"] for row in data["rows"]] == ["j1"]
    assert data["meta"]["skipped"] == [
        {"name": "s1d1-tiny-int8", "reason": "could not load: model_int8.onnx is not in the local Hub cache"}
    ]
    assert "s1d1-tiny-int8" in capsys.readouterr().err


def test_profile_refuses_prompts_that_are_not_those_of_the_run(tmp_path, capsys):
    setup = _setup(tmp_path, {"constant-low": WITNESS})
    Path(setup["argv"][2]).write_text('{"id": "x-eng", "prompt": "other"}\n', encoding="utf-8")
    assert main(setup["argv"]) == 1
    assert "prompts_sha256" in capsys.readouterr().err and not setup["out"].exists()


def test_profile_runs_the_litellm_heuristic_in_a_fresh_process(tmp_path):
    # the real worker, in a subprocess, with the one classifier that needs no model and no dependency group
    setup = _setup(tmp_path, {"litellm-heuristic": LITELLM}, judges=())
    assert main([*setup["argv"], "--warmup", "2"]) == 0
    data = json.loads(setup["out"].read_text(encoding="utf-8"))
    (row,) = data["rows"]
    assert (row["name"], row["device"], row["weights_mb"]) == ("litellm-heuristic", "cpu", 0.0)
    assert 0 < row["latency_p50_ms"] < 1000 and row["peak_rss_mb"] > 0
    assert data["meta"]["warmup"] == 2 and data["meta"]["skipped"] == []


def test_profile_help_lists_its_options(capsys):
    with pytest.raises(SystemExit):
        main(["profile", "--help"])
    out = capsys.readouterr().out
    for option in ("--prompts", "--run", "--judges-dir", "--models-dir", "--out", "--only", "--classifiers-dir"):
        assert option in out
    args = build_parser().parse_args(["profile", "--prompts", "p", "--run", "r", "--out", "o"])
    assert (args.warmup, args.classifiers_dir, args.judges_dir) == (10, Path("classifiers"), Path("judges"))
    assert args.models_dir == Path(os.path.expanduser("~/mlx-models"))


# ---------- offline: bench profile never asks the Hub ----------


def test_model_info_offline_takes_a_full_commit_without_the_network(monkeypatch):
    from s1d1_bench import s1d1_backend

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert s1d1_backend.model_info("scombat/s1d1-router", "7" * 40).sha == "7" * 40
    for revision in (None, "7fb9a4e", "main"):
        with pytest.raises(ValueError, match="HF_HUB_OFFLINE"):
            s1d1_backend.model_info("scombat/s1d1-router", revision)
