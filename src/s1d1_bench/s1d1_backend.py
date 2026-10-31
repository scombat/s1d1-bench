"""The s1d1 adapter of `bench classify`: the Hub router, run through its own `s1d1_router.py`.

The router file is downloaded from the Hub repo `scombat/s1d1-router` and imported by path, never copied here; the
weights are downloaded (read only, no token) at a pinned revision into the Hub cache and handed to its `Router` as a
local folder, so the revision written in meta.json is the one that was run.

- Router file: `--router-revision` pins the revision of `s1d1_router.py` (default: the revision of the weights for
  `scombat/s1d1-router` itself, else the head of that repo); the commit run is written in meta.json notes.
- Revision: `HfApi.model_info(repo, revision=...).sha`, the head of the repo at run time unless `--revision` is given.
  The `-onnx-int8` repos are compared with the revision of their card (`CARD_REVISIONS`, the commit of the int8
  weights with `reduce_range`); a run on another revision is refused before anything is downloaded.
- Batches: prompts sorted by length in characters (stable: equal lengths keep the file order), then cut into batches
  of at most 64 prompts or about 16 384 padded tokens (a token is estimated at 3 characters, at most `MAX_LENGTH`;
  a batch is padded to its longest prompt, so it costs `len(batch) * tokens(longest)`). The truncation at
  `MAX_LENGTH` is done by the router file itself. The results are put back in the order of the prompts.
- Output per prompt: `complexity` (the `complexity_tier` of `decode`, capped at high), `task`, `needs_tools`,
  `score` (`complexity_score`) and `needs_tools_probability`.
"""

import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

ROUTER_REPO = "scombat/s1d1-router"  # the main repo: the fp16 reference, and the home of s1d1_router.py
ROUTER_FILE = "s1d1_router.py"
# the revisions of the cards of the int8 repos: the commits of the corrected (reduce_range) weights. `tiny` has none.
CARD_REVISIONS = {
    "scombat/s1d1-router-6src-full-onnx-int8": "068d0d2",
    "scombat/s1d1-router-6src-compact-onnx-int8": "7e87870",
    "scombat/s1d1-router-6src-small-onnx-int8": "33c7f34",
}
BATCH_PROMPTS = 64
BATCH_TOKENS = 16384
CHARS_PER_TOKEN = 3  # estimate used to size the batches only
MIN_TOKENS = 8


# ---------- the Hub, read only ----------


def hub_env() -> None:
    """Public repos, read only: never send a token, even if one is stored on the machine (as `pool.py` does)."""
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def hub_offline() -> bool:
    """True when `HF_HUB_OFFLINE` is set (as `bench profile` does): the Hub is never asked, only its local cache."""
    return os.environ.get("HF_HUB_OFFLINE", "").strip().upper() in {"1", "ON", "YES", "TRUE"}


def model_info(repo: str, revision: str | None = None):
    """The Hub description of `repo` at `revision` (its head by default): `.sha` is the full commit.

    Offline (`hub_offline()`), a full 40-character commit is taken as it is, without the network, and anything else
    (a head, a branch, a short commit) raises ValueError, since only the Hub could resolve it. Online, nothing
    changes."""
    if hub_offline():
        if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
            return SimpleNamespace(sha=revision)
        raise ValueError(
            f"HF_HUB_OFFLINE is set: {repo} needs a full 40-character commit to be read from the local cache, "
            f"not {revision!r}"
        )
    hub_env()
    from huggingface_hub import HfApi

    return HfApi(token=False).model_info(repo, revision=revision)


def snapshot(repo: str, revision: str, patterns) -> str:
    """The local folder of `repo` at `revision` (downloaded into the Hub cache once), `patterns` files only."""
    hub_env()
    from huggingface_hub import snapshot_download

    return snapshot_download(repo, revision=revision, allow_patterns=list(patterns), token=False)


def fetch_router_module(revision: str | None):
    """`s1d1_router.py` of `scombat/s1d1-router` at `revision` (None: the head), imported by path."""
    hub_env()
    import importlib.util

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(ROUTER_REPO, ROUTER_FILE, revision=revision, token=False)
    spec = importlib.util.spec_from_file_location("s1d1_router", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


# ---------- revisions ----------


def resolve_revision(repo: str, revision: str | None) -> str:
    """The full commit run: `revision` resolved on the Hub (or the head). An `-onnx-int8` repo whose commit does not
    start with its card revision is refused: the weights of the card are the ones the bench reports."""
    sha = model_info(repo, revision).sha
    expected = CARD_REVISIONS.get(repo)
    if expected and not sha.startswith(expected):
        raise ValueError(
            f"{repo} resolves to {sha[:7]}, not to the revision of its card ({expected}): nothing was run. "
            f"Check what changed since the card revision, then pin it with --revision {expected} if the weights are "
            "the same."
        )
    return sha


# ---------- batches ----------


def plan_batches(prompts: list[str], max_prompts: int, max_tokens: int, max_length: int) -> list[list[int]]:
    """Indices of `prompts` in batches: sorted by length in characters (equal lengths in the order of the list), at
    most `max_prompts` per batch and `len(batch) * tokens(longest)` at most `max_tokens`, a token being 3 characters
    and a prompt never more than `max_length` tokens. A prompt too big for the budget is a batch of its own."""

    def tokens(i: int) -> int:
        return min(max_length, max(MIN_TOKENS, -(-len(prompts[i]) // CHARS_PER_TOKEN)))

    batches: list[list[int]] = []
    current: list[int] = []
    for i in sorted(range(len(prompts)), key=lambda k: (len(prompts[k]), k)):
        # ascending lengths: the prompt being added is the longest of the batch
        if current and (len(current) >= max_prompts or (len(current) + 1) * tokens(i) > max_tokens):
            batches.append(current)
            current = []
        current.append(i)
    if current:
        batches.append(current)
    return batches


# ---------- the classifier ----------

_DTYPES = {"float16": "fp16", "float32": "fp32", "bfloat16": "bf16", "float64": "fp64"}


def _dtype_label(backend: str, router, folder: str) -> str:
    """fp16, fp32, int8...: from the torch model, else from the `variant.json` of the folder (file or recipe)."""
    dtype = getattr(getattr(router, "model", None), "dtype", None)
    if backend == "torch" and dtype is not None:
        return _DTYPES.get(str(dtype).removeprefix("torch."), str(dtype))
    try:
        info = json.loads((Path(folder) / "variant.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    text = f"{info.get('file', '')} {info.get('recipe', '')}".lower()
    for label in ("int8", "fp16", "fp32"):
        if label in text:
            return label
    return "fp32"


def make_s1d1_classifier(
    repo: str, revision: str | None, backend: str, router_revision: str | None = None
) -> tuple[Callable[[list[str]], list[dict]], dict]:
    """(classify, meta): `classify(prompts)` rates a list of prompts by sorted batches and returns, in the order of
    the prompts, dicts of `complexity`, `task`, `needs_tools`, `score` and `needs_tools_probability`. `meta` holds
    `model` (the repo), `revision` (resolved), `router_revision` (of `s1d1_router.py`), `backend`, `device` and
    `quantization` (backend and precision, e.g. `onnx-int8`). Raises ValueError, before any download, for an
    `-onnx-int8` repo whose revision is not the one of its card.

    `router_revision` is the revision of `s1d1_router.py` on the Hub (`ROUTER_REPO`); by default the file is the one
    of the weights when they come from `ROUTER_REPO`, else the head of that repo."""
    resolved = resolve_revision(repo, revision)
    if router_revision:
        router_revision = model_info(ROUTER_REPO, router_revision).sha
    else:
        router_revision = resolved if repo == ROUTER_REPO else model_info(ROUTER_REPO).sha
    module = fetch_router_module(router_revision)
    folder = snapshot(repo, resolved, module.HUB_FILES)
    chosen = module.backend_of(folder) if backend == "auto" else backend
    device = "mps" if chosen == "torch" and mps_available() else "cpu"  # torch runs on mps, onnx on cpu
    router = module.Router(folder, device=device, backend=backend)
    meta = {
        "model": repo,
        "revision": resolved,
        "router_revision": router_revision,
        "backend": chosen,
        "device": device,
        "quantization": f"{chosen}-{_dtype_label(chosen, router, folder)}",
    }

    def classify(prompts: list[str]) -> list[dict]:
        out: list[dict | None] = [None] * len(prompts)
        for batch in plan_batches(prompts, BATCH_PROMPTS, BATCH_TOKENS, module.MAX_LENGTH):
            logits, counts = router.logits([prompts[i] for i in batch])
            for i, row in zip(batch, module.decode(logits, counts), strict=True):
                out[i] = {
                    "complexity": row["complexity_tier"],
                    "task": row["task"],
                    "needs_tools": row["needs_tools"],
                    "score": row["complexity_score"],
                    "needs_tools_probability": row["needs_tools_probability"],
                }
        return out  # type: ignore[return-value]

    return classify, meta
