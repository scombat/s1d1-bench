"""`bench profile`: latency per prompt, peak memory and weight size of every model of a run.

The bench measures agreement; this command measures what a deployment pays for it, for each candidate of a
`results/<set>.run.json` (docs/36 § 0 decision 2). Output: `results/<set>.profile.json`, derived and regenerable (the
latencies move a little from one run to the next: it is a measurement, not a computation).

- Classifiers: each classifier of `run.json` whose `meta.json` names a model (s1d1, Laya, LiteLLM's heuristic) is
  reloaded by its adapter with the `model`, `revision`, `backend`, router revision and Laya checkpoint of its
  `meta.json` (the `notes` give the "Replay" line), warmed up on `warmup` calls (10; the first prompts of the file),
  then timed on one call per prompt of the file, one prompt per call (batch of 1), with `time.perf_counter_ns` around
  `classify([prompt])` (tokenisation, forward pass and decoding). p50 and p95 interpolate linearly between ranks (as
  numpy's default). The witnesses (`witness:*` models) are no models: they are left out and listed in `meta.skipped`.
- Isolation: each classifier runs in a fresh Python process (`python -m s1d1_bench.profile <spec>`), so that its peak
  RSS (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, read after the timed calls) is its own: the interpreter, the
  libraries, the model load and the inference of that one model, nothing of the others. It is not the size of the
  model alone (docs/36 § 5); memory held by the GPU driver for MPS may not be fully counted in it.
- Offline: the process runs with `HF_HUB_OFFLINE=1`, so the weights come from the local Hub cache at the pinned
  commit or the model fails (nothing is downloaded); a model that fails to load is listed in `meta.skipped` with the
  reason, the others go on, and the exit code is 1.
- Weights: the size on disk of the weight files (`WEIGHT_SUFFIXES`) of the pinned snapshot in the local Hub cache,
  folder only (Laya's english head is the root of its repo, its multilingual head the `multilingual` subfolder);
  LiteLLM's heuristic has none (0).
- Routers of RouteLLM (`routellm/...` models) are reloaded by the routellm adapter, on the CPU, like the others.
- NVIDIA's classifier (`nvidia/...`, English only) is reloaded by the nvidia adapter, on the CPU, and timed on the
  English prompts of the file only (the ids ending `-eng`), the languages it does not cover being never run.
- A candidate in the judge format (`provider: local` in its `meta.json`: a small LLM that reads the v7 instruction
  through `bench judge`) is profiled like a judge, from its `log.jsonl` and its weights in `<models-dir>`, but its row
  says `kind: classifier`: it is a candidate of the report, not an instrument.
- Judges: their latency comes from their `log.jsonl` (`wall_s` of the calls that ended `ok`, one prompt per call by
  construction, each reading the v7 instruction of about 1 900 tokens before the prompt, on the GPU through MLX) and
  their weights from `<models-dir>/<org>--<name>`; their peak RSS cannot be measured after the fact (null).
- Units: milliseconds, and MB = 10^6 bytes.
"""

import datetime
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

from s1d1_bench.jsonl import read_jsonl

WEIGHT_SUFFIXES = (".safetensors", ".onnx", ".bin", ".npz", ".gguf", ".pt")
WARMUP = 10
BATCH_SIZE = 1
RESULT_MARK = "S1D1_PROFILE_RESULT "  # prefix of the worker's answer on stdout (libraries may print there too)
WORKER_TIMEOUT_S = 3600
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
FULL_SHA = r"[0-9a-f]{40}"


# ---------- measures ----------


def percentile(values: list[float], q: float) -> float:
    """The `q`-th percentile of `values`, linear interpolation between the closest ranks (numpy's default)."""
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of no value")
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def rss_mb(maxrss: int, system: str = sys.platform) -> float:
    """`ru_maxrss` in MB (10^6 bytes): bytes on macOS, kilobytes on Linux."""
    return maxrss / 1e6 if system == "darwin" else maxrss * 1024 / 1e6


def peak_rss_mb() -> float:
    """Peak resident memory of this process so far (interpreter and libraries included)."""
    import resource

    return rss_mb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def profile_classifier(
    classify: Callable[[list[str]], list],
    prompts: list[str],
    warmup: int = WARMUP,
    *,
    clock: Callable[[], int] = time.perf_counter_ns,
) -> dict:
    """Warm `classify` up on `warmup` calls (the first prompts, cycling when the file is shorter), then time one call
    per prompt, one prompt per call. Returns `n` (prompts timed), `p50_ms`, `p95_ms`, `mean_ms` and `peak_rss_mb`
    (the process's, read after the timed calls). Raises ValueError when a call does not answer one row."""
    if not prompts:
        raise ValueError("no prompt to time")

    def call(prompt: str) -> None:
        answer = classify([prompt])
        if len(answer) != 1:
            raise ValueError(f"the classifier answered {len(answer)} rows for 1 prompt")

    for i in range(warmup):
        call(prompts[i % len(prompts)])
    times_ms = []
    for prompt in prompts:
        start = clock()
        call(prompt)
        times_ms.append((clock() - start) / 1e6)
    return {
        "n": len(times_ms),
        "p50_ms": percentile(times_ms, 50),
        "p95_ms": percentile(times_ms, 95),
        "mean_ms": sum(times_ms) / len(times_ms),
        "peak_rss_mb": peak_rss_mb(),
    }


def profile_judge(folder: Path) -> dict:
    """The latency of a judge from its `log.jsonl`: `n` lines, `n_ok` calls that ended `ok` (the others are left
    out), `p50_s`, `p95_s`, `mean_s` of their `wall_s`, and their mean `prompt_tokens` and `completion_tokens`."""
    rows = read_jsonl(Path(folder) / "log.jsonl")
    ok = [row for row in rows if row.get("status") == "ok"]
    if not ok:
        raise ValueError(f"{folder}/log.jsonl has no ok call")
    walls = [float(row["wall_s"]) for row in ok]
    return {
        "n": len(rows),
        "n_ok": len(ok),
        "p50_s": percentile(walls, 50),
        "p95_s": percentile(walls, 95),
        "mean_s": sum(walls) / len(walls),
        "mean_prompt_tokens": sum(row["prompt_tokens"] for row in ok) / len(ok),
        "mean_completion_tokens": sum(row["completion_tokens"] for row in ok) / len(ok),
    }


def weight_files(path: Path) -> list[Path]:
    """The weight files (`WEIGHT_SUFFIXES`) directly in the folder `path`, not in its subfolders, sorted by name."""
    return sorted(p for p in Path(path).iterdir() if p.suffix in WEIGHT_SUFFIXES and p.is_file())


def weights_bytes(path: Path) -> int:
    """Size on disk of the weight files of the folder `path` (links followed: the Hub cache links to its blobs)."""
    return sum(p.stat().st_size for p in weight_files(path))


def write_profile(out: Path, rows: list[dict], meta: dict) -> None:
    """`{"meta": meta, "rows": rows}`, keys sorted, 2-space indent, a final newline: same input, same bytes."""
    text = json.dumps({"meta": meta, "rows": rows}, sort_keys=True, indent=2) + "\n"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(text, encoding="utf-8", newline="\n")


def machine() -> str:
    """`platform.platform()`, and the processor name when `sysctl` knows it (macOS)."""
    name = platform.platform()
    try:
        brand = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        brand = ""
    return f"{name}, {brand}" if brand else name


# ---------- how to reload a classifier ----------


def _found(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def classifier_spec(meta: dict) -> dict | None:
    """How to reload the classifier of a `meta.json`: the adapter and its arguments, read from `model`, `revision`
    and the "Replay" line of `notes`. None for a witness (no model). Raises ValueError for an unknown model or a
    Replay line that does not give what the adapter needs."""
    model, revision, notes = meta.get("model", ""), meta.get("revision", ""), meta.get("notes", "")
    if model.startswith("witness:"):
        return None
    if model.startswith("litellm:"):
        return {"adapter": "litellm-heuristic"}
    if model.startswith("scombat/s1d1-router"):
        backend = _found(r"--backend (\w+)", notes)
        router = _found(rf"--router-revision ({FULL_SHA})", notes)
        router = router or _found(rf"s1d1_router\.py of scombat/s1d1-router@({FULL_SHA})", notes)
        if not (backend and router and re.fullmatch(FULL_SHA, revision)):
            raise ValueError(f"{model}: meta.json does not give the backend, the router commit and a full revision")
        return {"adapter": "s1d1", "model": model, "revision": revision, "backend": backend, "router_revision": router}
    if model.startswith("convaiinnovations/laya"):
        laya_model = _found(r"--laya-model (\w+)", notes)
        commit = _found(rf"--revision ({FULL_SHA})", notes) or _found(rf"@({FULL_SHA})", revision)
        if not (laya_model and commit):
            raise ValueError(f"{model}: meta.json does not give the Laya checkpoint and its commit")
        return {"adapter": "laya", "laya_model": laya_model, "commit": commit}
    if model.startswith("routellm/"):
        router = _found(r"--routellm-router (\w+)", notes)
        if not (router and re.fullmatch(FULL_SHA, revision)):
            raise ValueError(f"{model}: meta.json does not give the RouteLLM router and a full revision")
        return {"adapter": "routellm", "router": router, "revision": revision}
    if model.startswith("nvidia/"):
        if not re.fullmatch(FULL_SHA, revision):
            raise ValueError(f"{model}: meta.json does not give a full revision")
        return {"adapter": "nvidia", "revision": revision}
    raise ValueError(f"no adapter for the model {model!r}")


# ---------- the worker: one classifier in a fresh process ----------


def _local_snapshot(repo: str, commit: str) -> Path:
    """The folder of `repo` at `commit` in the local Hub cache (`<cache>/models--<org>--<name>/snapshots/<commit>`,
    the layout of `huggingface_hub`); raises FileNotFoundError when it is not there. Read from the layout rather than
    with `snapshot_download(local_files_only=True)`, which refuses a snapshot holding only the files a backend
    fetched (no README, no licence)."""
    from huggingface_hub import constants

    folder = Path(constants.HF_HUB_CACHE) / f"models--{repo.replace('/', '--')}" / "snapshots" / commit
    if not folder.is_dir():
        raise FileNotFoundError(f"{repo}@{commit} is not in the local Hub cache")
    return folder


def _load(spec: dict) -> tuple[Callable[[list[str]], list], dict]:
    """(classify, facts): the classifier of `spec`, and its `device`, `weights_bytes`, `weight_files`, `quantization`
    (or Laya's version) and `work` (what one call does, for the note)."""
    if spec["adapter"] == "litellm-heuristic":
        from s1d1_bench.vendor.litellm_heuristic import classify as heuristic

        facts = {
            "device": "cpu", "weights_bytes": 0, "weight_files": [], "quantization": "pure Python rules",
            "work": "LiteLLM's heuristic scorer",
        }  # fmt: skip
        return (lambda prompts: [heuristic(p) for p in prompts]), facts
    if spec["adapter"] == "s1d1":
        from s1d1_bench import s1d1_backend

        classify, info = s1d1_backend.make_s1d1_classifier(
            spec["model"], spec["revision"], spec["backend"], spec["router_revision"]
        )
        folder = _local_snapshot(spec["model"], info["revision"])
        facts = {
            "device": info["device"], "quantization": info["quantization"],
            "work": "tokenisation, forward pass, decoding",
        }  # fmt: skip
    elif spec["adapter"] == "laya":
        from s1d1_bench import laya_backend

        classify, info = laya_backend.make_laya_classifier(spec["laya_model"], spec["commit"])
        folder = _local_snapshot(laya_backend.REPO, info["commit"])
        if spec["laya_model"] == "multilingual":
            folder = folder / "multilingual"
        facts = {
            "device": info["device"], "quantization": f"laya {info['laya_version']}",
            "work": "one predict_batch request: tokenisation, forward pass, reading of the tier answer",
        }  # fmt: skip
    elif spec["adapter"] == "routellm":
        from s1d1_bench import routellm_backend

        classify, info = routellm_backend.make_routellm_classifier(spec["router"], spec["revision"])
        folder = _local_snapshot(routellm_backend.ROUTERS[spec["router"]], info["revision"])
        facts = {
            "device": info["device"], "quantization": info["precision"],
            "work": "tokenisation, forward pass, softmax of the three logits",
        }  # fmt: skip
    elif spec["adapter"] == "nvidia":
        from s1d1_bench import nvidia_backend

        classify, info = nvidia_backend.make_nvidia_classifier(spec["revision"])
        folder = _local_snapshot(nvidia_backend.REPO, info["revision"])
        facts = {
            "device": info["device"], "quantization": info["precision"],
            "work": "tokenisation padded to 512 tokens, forward pass, softmax of the eight heads",
        }  # fmt: skip
    else:
        raise ValueError(f"unknown adapter {spec['adapter']!r}")
    facts.update(weights_bytes=weights_bytes(folder), weight_files=[p.name for p in weight_files(folder)])
    return classify, facts


def run_in_process(spec: dict) -> dict:
    """Load the classifier of `spec`, time it on the prompts of `spec["prompts"]`: the facts of `_load` and the
    figures of `profile_classifier`."""
    rows = read_jsonl(Path(spec["prompts"]))
    if spec["adapter"] == "nvidia":  # English only: the other languages are never run, so never timed
        from s1d1_bench import nvidia_backend

        rows = [row for row in rows if nvidia_backend.covered(row["id"].rsplit("-", 1)[-1])]
    prompts = [row["prompt"] for row in rows]
    classify, facts = _load(spec)
    return {**facts, **profile_classifier(classify, prompts, spec["warmup"])}


def run_worker(spec: dict) -> dict:
    """`run_in_process(spec)` in a fresh Python process, offline: its peak RSS is that of this classifier alone.
    Raises RuntimeError with the end of its error output when it fails."""
    env = {**os.environ, **OFFLINE_ENV}
    done = subprocess.run(
        [sys.executable, "-m", "s1d1_bench.profile", json.dumps(spec)],
        capture_output=True,
        text=True,
        env=env,
        timeout=WORKER_TIMEOUT_S,
    )
    answers = [line for line in done.stdout.splitlines() if line.startswith(RESULT_MARK)]
    if done.returncode != 0 or not answers:
        tail = [line for line in done.stderr.strip().splitlines() if line.strip()][-1:] or [f"exit {done.returncode}"]
        raise RuntimeError(tail[0].strip())
    return json.loads(answers[-1][len(RESULT_MARK) :])


def _worker_main(argv: list[str]) -> int:
    result = run_in_process(json.loads(argv[0]))
    print(RESULT_MARK + json.dumps(result), flush=True)
    return 0


# ---------- rows ----------


def _mb(n_bytes: int | None) -> float | None:
    return None if n_bytes is None else round(n_bytes / 1e6, 1)


def classifier_row(name: str, meta: dict, result: dict, warmup: int) -> dict:
    files = ", ".join(result["weight_files"])
    weights = f"weights: {files} of the pinned revision in the local Hub cache" if files else "no weights (rules only)"
    note = (
        f"time of classify([prompt]) per prompt ({result['work']}), batch of 1, {result['n']} "
        f"prompts after {warmup} warm-up calls, time.perf_counter_ns; {result['quantization']} on {result['device']}; "
        f"{weights}; peak RSS: ru_maxrss of a fresh process that loaded this model only (interpreter, libraries and "
        "model load included)"
    )
    if result["device"] == "mps":
        note += "; memory held by the GPU driver (MPS) may not be fully counted in the RSS"
    return {
        "name": name,
        "kind": "classifier",
        "model": meta["model"],
        "revision": meta["revision"],
        "device": result["device"],
        "weights_mb": _mb(result["weights_bytes"]),
        "latency_p50_ms": round(result["p50_ms"], 3),
        "latency_p95_ms": round(result["p95_ms"], 3),
        "latency_mean_ms": round(result["mean_ms"], 3),
        "peak_rss_mb": round(result["peak_rss_mb"], 1),
        "note": note,
    }


def judge_row(entry: dict, folder: Path, models_dir: Path, kind: str = "judge") -> dict:
    stats = profile_judge(folder)
    meta_path = folder / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    weights_dir = models_dir / entry["model"].replace("/", "--")
    if weights_dir.is_dir():
        weights = weights_bytes(weights_dir)
        weights_note = f"weights: the {len(weight_files(weights_dir))} weight files of {weights_dir.name}"
        local = weights_dir / "REVISION"
        if local.is_file() and local.read_text(encoding="utf-8").strip() != entry["revision"]:
            weights_note += f" (local REVISION {local.read_text(encoding='utf-8').strip()}, not that of the run)"
    else:
        weights, weights_note = None, f"weights: {weights_dir.name} not found under the models folder"
    mlx = f" (mlx-lm {meta['mlx_lm_version']})" if meta.get("mlx_lm_version") else ""
    note = (
        f"wall_s of log.jsonl over its {stats['n_ok']} ok calls (of {stats['n']} lines; errors left out), one prompt "
        f"per call by construction, GPU through MLX{mlx}; each call reads the v7 instruction (about 1 900 tokens) with "
        f"the prompt, mean {stats['mean_prompt_tokens']:.0f} prompt tokens, and generates a mean "
        f"{stats['mean_completion_tokens']:.0f} tokens (thinking {meta.get('thinking', 'unknown')}, max_tokens "
        f"{meta.get('max_tokens', 'unknown')}); {weights_note}; peak RSS not measurable after the fact"
    )
    return {
        "name": entry["name"],
        "kind": kind,
        "model": entry["model"],
        "revision": entry["revision"],
        "device": "gpu-mlx",
        "weights_mb": _mb(weights),
        "latency_p50_ms": round(stats["p50_s"] * 1000, 3),
        "latency_p95_ms": round(stats["p95_s"] * 1000, 3),
        "latency_mean_ms": round(stats["mean_s"] * 1000, 3),
        "peak_rss_mb": None,
        "note": note,
    }


# ---------- the command ----------


def _fail(message: str) -> int:
    print(f"bench profile: {message}", file=sys.stderr)
    return 1


def run_profile_cmd(args) -> int:
    """`bench profile`: write `--out` for the classifiers and judges of `--run` (or those of `--only`)."""
    run = json.loads(Path(args.run).read_text(encoding="utf-8"))
    prompts_path = Path(args.prompts)
    digest = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    if run.get("prompts_sha256") and run["prompts_sha256"] != digest:
        return _fail(f"{prompts_path} is not the prompts file of {args.run} (prompts_sha256 differs)")
    classifiers, judges = run.get("classifiers", []), run.get("judges", [])
    names = [entry["name"] for entry in classifiers + judges]
    only = [name.strip() for name in args.only.split(",") if name.strip()] if args.only else None
    unknown = sorted(set(only or []) - set(names))
    if unknown:
        return _fail(f"{', '.join(unknown)}: not a classifier nor a judge of {args.run} ({', '.join(names)})")

    rows: list[dict] = []
    skipped: list[dict] = []
    failed = False
    for entry in classifiers:
        name = entry["name"]
        meta_path = Path(args.classifiers_dir) / name / "meta.json"
        if not meta_path.is_file():
            skipped.append({"name": name, "reason": f"no meta.json in {args.classifiers_dir}/{name}"})
            failed = True
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("provider") == "local":  # a candidate in the judge format: profiled like a judge, kept a candidate
            if only is not None and name not in only:
                skipped.append({"name": name, "reason": "not selected by --only"})
                continue
            rows.append(judge_row(entry, meta_path.parent, Path(args.models_dir).expanduser(), kind="classifier"))
            continue
        try:
            spec = classifier_spec(meta)
        except ValueError as exc:
            skipped.append({"name": name, "reason": str(exc)})
            failed = True
            continue
        if spec is None:
            skipped.append({"name": name, "reason": "witness: not a model"})
            continue
        if only is not None and name not in only:
            skipped.append({"name": name, "reason": "not selected by --only"})
            continue
        print(f"{name}: loading and timing ({spec['adapter']})", file=sys.stderr, flush=True)
        try:
            result = run_worker({**spec, "prompts": str(args.prompts), "warmup": args.warmup})
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            skipped.append({"name": name, "reason": f"could not load: {exc}"})
            print(f"bench profile: {name} skipped, could not load: {exc}", file=sys.stderr)
            failed = True
            continue
        rows.append(classifier_row(name, meta, result, args.warmup))
    for entry in judges:
        if only is not None and entry["name"] not in only:
            skipped.append({"name": entry["name"], "reason": "not selected by --only"})
            continue
        rows.append(judge_row(entry, Path(args.judges_dir) / entry["name"], Path(args.models_dir).expanduser()))

    meta = {
        "date": datetime.date.today().isoformat(),
        "machine": machine(),
        "warmup": args.warmup,
        "batch_size": BATCH_SIZE,
        "prompts": str(args.prompts),
        "prompts_sha256": digest,
        "run": str(args.run),
        "skipped": skipped,
    }
    write_profile(Path(args.out), rows, meta)
    print(f"{args.out}: {len(rows)} rows, {len(skipped)} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_worker_main(sys.argv[1:]))
