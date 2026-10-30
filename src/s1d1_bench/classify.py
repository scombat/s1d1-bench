"""`bench classify`: write a classifier folder (`classifiers/<name>/`) for a prompts file.

Three witnesses live here, systems that read no model and give the floor a real classifier must clear:

- `random`: five tiers, the 19 tasks and `needs_tools` drawn uniformly at random with a seeded generator
  (`random.Random(seed)`). The generator is consumed draw by draw, then prompt by prompt in the order of the prompts
  file, then tier, task, `needs_tools`: the same file and seed always give the same bytes. `labels.jsonl` holds draw 0;
  `draws.jsonl` holds every draw, one line per prompt and per draw (`draw`, `id`, then the three fields), draw-major.
  `bench validate` only reads `meta.json` and `labels.jsonl`, so the extra file is ignored by it. No `score`.
- `length`: the length of the prompt in characters, ranked within its language (the id suffix after the last `-`).
  Ranks run 0..n-1 by (length, id), equal lengths keeping the order of their ids. `score` = rank / (n - 1) (0 for a
  language of a single prompt). Decile = rank * 10 // n; the deciles map to tiers with `decile_tiers`, the number of
  deciles that give trivial, low, medium, high, expert: (2, 3, 3, 1, 1) gives deciles 0-1 trivial, 2-4 low, 5-7 medium,
  8 high, 9 expert. `task` and `needs_tools` are null (the witness does not predict them).
- `constant`: one tier for every prompt (`--level`, low by default), `score` = its anchor, `task` and `needs_tools`
  null.

The system adapters are dispatched from `run_classify_cmd` too: `s1d1` (the Hub router, see `s1d1_backend`), `laya`
(Laya asked LiteLLM's tier question, see `laya_backend`: its most probable tier maps name to name like LiteLLM's
heuristic, and its expected level and four probabilities stay in `decisions.jsonl`, never in `labels.jsonl`) and
`litellm-heuristic` (LiteLLM's own heuristic complexity scorer, vendored verbatim in
`s1d1_bench.vendor.litellm_heuristic`: its four tiers map name to name onto low, medium, high, expert; its raw score,
which is not on the v7 scale, stays in `decisions.jsonl` and never reaches `labels.jsonl`).
"""

import datetime
import hashlib
import random
import shutil
import sys
from collections.abc import Callable
from pathlib import Path

from s1d1_bench.classifier_file import write_classifier_dir
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.judge_file import LICENCE
from s1d1_bench.judge_runner import _prompt_fault, local_path_keys
from s1d1_bench.rubric import LEVELS, TASKS

ANCHORS = {"trivial": 0.01, "low": 0.25, "medium": 0.50, "high": 0.75, "expert": 1.00}
DECILE_TIERS = (2, 3, 3, 1, 1)  # deciles giving trivial, low, medium, high, expert (docs/34 § 3)
DECILES = 10
# LiteLLM's ComplexityTier names, translated name to name, with no arithmetic
TIER_TO_LEVEL = {"SIMPLE": "low", "MEDIUM": "medium", "COMPLEX": "high", "REASONING": "expert"}


def nearest_anchor(score: float) -> str:
    """The tier whose anchor is nearest to `score`; an exact midpoint between two anchors goes to the lower tier
    (0.375 is low, 0.875 is high). The midpoints are (a + b) / 2 of neighbouring anchors, compared with `<=`."""
    for lower, upper in zip(LEVELS, LEVELS[1:], strict=False):
        if score <= (ANCHORS[lower] + ANCHORS[upper]) / 2:
            return lower
    return LEVELS[-1]


def random_labels(ids: list[str], seed: int, draws: int) -> list[list[dict]]:
    """`draws` lists of one row per id (id, complexity, task, needs_tools), uniform on the five tiers, the tasks of
    the rubric and {True, False}; `random.Random(seed)`, no score."""
    rng = random.Random(seed)
    return [
        [
            {
                "id": i,
                "complexity": rng.choice(LEVELS),
                "task": rng.choice(TASKS),
                "needs_tools": rng.choice([True, False]),
            }
            for i in ids
        ]
        for _ in range(draws)
    ]


def _language(prompt_id: str) -> str:
    if "-" not in prompt_id:
        raise ValueError(f"id {prompt_id!r} has no '-<lang>' suffix: the length witness ranks prompts per language")
    return prompt_id.rsplit("-", 1)[1]


def _decile_tier_table(decile_tiers: tuple[int, ...]) -> list[str]:
    if len(decile_tiers) != len(LEVELS) or sum(decile_tiers) != DECILES or min(decile_tiers) < 0:
        raise ValueError(f"decile_tiers {decile_tiers} must give {len(LEVELS)} counts that sum to {DECILES} deciles")
    return [level for level, count in zip(LEVELS, decile_tiers, strict=True) for _ in range(count)]


def length_labels(prompts: list[dict], decile_tiers: tuple[int, ...] = DECILE_TIERS) -> list[dict]:
    """One row per prompt (in prompt order): tier from the decile of its length in characters within its language,
    `score` = rank / (n - 1) in that language, `task` and `needs_tools` None."""
    tier_of_decile = _decile_tier_table(decile_tiers)
    by_lang: dict[str, list[dict]] = {}
    for item in prompts:
        by_lang.setdefault(_language(item["id"]), []).append(item)
    placed: dict[str, dict] = {}
    for items in by_lang.values():
        n = len(items)
        for rank, item in enumerate(sorted(items, key=lambda p: (len(p["prompt"]), p["id"]))):
            placed[item["id"]] = {
                "id": item["id"],
                "complexity": tier_of_decile[rank * DECILES // n],
                "task": None,
                "needs_tools": None,
                "score": rank / (n - 1) if n > 1 else 0.0,
            }
    return [placed[item["id"]] for item in prompts]


def constant_labels(ids: list[str], level: str) -> list[dict]:
    """The same tier for every id, `score` = its anchor, `task` and `needs_tools` None."""
    if level not in ANCHORS:
        raise ValueError(f"level {level!r} not in {LEVELS}")
    return [{"id": i, "complexity": level, "task": None, "needs_tools": None, "score": ANCHORS[level]} for i in ids]


# ---------- the command ----------


def _fail(message: str) -> int:
    print(f"bench classify: {message}", file=sys.stderr)
    return 1


def write_run(
    args,
    digest: str,
    rows: list[dict],
    *,
    model: str,
    revision: str,
    notes: str,
    extra: dict[str, list[dict]] | None = None,
    quantization: str | None = None,
) -> int:
    """Write the classifier folder `args.out` (meta.json, then labels.jsonl, then the `extra` jsonl files) and say so.
    Shared by every adapter. Returns the exit code; nothing is left behind on failure."""
    out = Path(args.out)
    meta = {
        "judge": out.name,
        "model": model,
        "revision": revision,
        "date": datetime.date.today().isoformat(),
        "temperature": 0,
        "prompts_sha256": digest,
        "author": args.author,
        "replayable": True,
        "licence": LICENCE,
        "notes": " ".join(part for part in (notes, args.notes) if part),
    }
    if quantization is not None:
        meta["quantization"] = quantization  # optional key of meta.json, written after the required ones
    leaks = local_path_keys(meta)
    if leaks:
        return _fail(f"meta.json would hold a local path in {', '.join(leaks)}: meta.json is published, remove it")
    meta.update(provider="system", thinking="none", instructions="none")  # forced anyway by write_classifier_dir
    try:
        write_classifier_dir(out, rows, meta)
    except (FileExistsError, ValueError) as exc:
        return _fail(str(exc))
    try:
        for name, extra_rows in (extra or {}).items():
            write_jsonl(out / name, extra_rows)
    except Exception:
        shutil.rmtree(out)
        raise
    print(f"{out}: {len(rows)} labels")
    return 0


def _random(args, prompts: list[dict], digest: str) -> int:
    draws = random_labels([p["id"] for p in prompts], args.seed, args.draws)
    notes = (
        f"Witness: one of the five tiers, one of the {len(TASKS)} tasks and needs_tools drawn uniformly at random, "
        f"seed {args.seed}, {args.draws} draws (random.Random, prompts in file order). labels.jsonl is draw 0, "
        "draws.jsonl holds all the draws. No score."
    )
    every_draw = [{"draw": d, **row} for d, draw in enumerate(draws) for row in draw]
    return write_run(
        args, digest, draws[0] if draws else [], model="witness:random", revision=f"seed {args.seed}", notes=notes,
        extra={"draws.jsonl": every_draw},
    )


def _length(args, prompts: list[dict], digest: str) -> int:
    notes = (
        "Witness: length of the prompt in characters, ranked per language (id suffix after the last '-'); "
        f"deciles to tiers {'/'.join(map(str, DECILE_TIERS))} (deciles giving trivial, low, medium, high, expert); "
        "score = rank / (n - 1) in the language, equal lengths ranked by id. task and needs_tools are null."
    )
    return write_run(args, digest, length_labels(prompts), model="witness:length", revision="n/a", notes=notes)


def _constant(args, prompts: list[dict], digest: str) -> int:
    notes = (
        f"Witness: the tier {args.level} for every prompt, score {ANCHORS[args.level]} (its anchor). "
        "task and needs_tools are null."
    )
    return write_run(
        args, digest, constant_labels([p["id"] for p in prompts], args.level),
        model=f"witness:constant-{args.level}", revision="n/a", notes=notes,
    )


def _s1d1(args, prompts: list[dict], digest: str) -> int:
    if not args.model:
        return _fail("the s1d1 adapter needs --model, the Hub repo id of the weights (scombat/s1d1-router...)")
    try:
        from s1d1_bench import s1d1_backend

        classify, info = s1d1_backend.make_s1d1_classifier(args.model, args.revision, args.backend)
        results = classify([p["prompt"] for p in prompts])
    except ImportError as exc:  # a dependency group is not installed: nothing was written
        return _fail(
            f"{exc}. The s1d1 adapter needs the s1d1 dependency group (onnx, mlx backends) or s1d1-torch (torch): "
            "uv run --group s1d1 bench classify s1d1 ..., or --group s1d1-torch --group s1d1."
        )
    rows = [{"id": p["id"], **result} for p, result in zip(prompts, results, strict=True)]
    notes = (
        f"Replay: bench classify s1d1 --model {info['model']} --revision {info['revision']} --backend {args.backend}. "
        f"s1d1_router.py of scombat/s1d1-router@{info['router_revision']}, backend {info['backend']}, device "
        f"{info['device']}, precision {info['quantization']}. Prompts sorted by length in characters, batches of at "
        f"most {s1d1_backend.BATCH_PROMPTS} prompts or about {s1d1_backend.BATCH_TOKENS} tokens, truncated at the "
        "router's MAX_LENGTH; tier = complexity_tier of decode() (capped at high), score = complexity_score."
    )
    return write_run(
        args, digest, rows, model=info["model"], revision=info["revision"], notes=notes,
        quantization=info["quantization"],
    )


def _laya(args, prompts: list[dict], digest: str) -> int:
    if not args.laya_model:
        return _fail("the laya adapter needs --laya-model english or multilingual, the checkpoint to question")
    try:
        from s1d1_bench import laya_backend

        classify, info = laya_backend.make_laya_classifier(args.laya_model, args.revision)
        results = classify([p["prompt"] for p in prompts])
    except ImportError as exc:  # the laya dependency group is not installed: nothing was written
        return _fail(
            f"{exc}. The laya adapter needs the laya dependency group: uv run --group laya bench classify laya ..."
        )
    tiers = [laya_backend.most_probable(r["probabilities"]) for r in results]
    rows = [
        {"id": p["id"], "complexity": TIER_TO_LEVEL[tier], "task": None, "needs_tools": None}
        for p, tier in zip(prompts, tiers, strict=True)
    ]
    raw = [
        {
            "id": p["id"], "tier": tier, "score": laya_backend.expected_level(r["probabilities"]),
            "probabilities": r["probabilities"],
        }
        for p, tier, r in zip(prompts, tiers, results, strict=True)
    ]  # fmt: skip
    notes = (
        f"Replay: uv run --group laya bench classify laya --laya-model {args.laya_model} --revision {info['commit']} "
        f"(laya {info['laya_version']}, device {info['device']}). The question is LiteLLM's complexity router `tier` "
        "question, one choice question with its four criteria and its instructions, as docs/27 reads it: copied "
        "identically from bench/laya_scores.py of the research repo, read on 2026-10-30 (the LiteLLM file was read on "
        f"2026-10-25). Laya runs locally, batches of {laya_backend.BATCH} prompts sorted by length. tier = the most "
        "probable of the four (an exact tie goes to the lowest name: SIMPLE < MEDIUM < COMPLEX < REASONING), "
        "translated name to name: SIMPLE -> low, MEDIUM -> medium, COMPLEX -> high, REASONING -> expert (never "
        "trivial). labels.jsonl has no score; decisions.jsonl holds, by id, the tier (LiteLLM's name), the expected "
        "level on 0..3 (sum of i * p_i, not divided) as score, and the four "
        "probabilities. The two folders laya-english and laya-multilingual (nearest anchor of the expected score) stay "
        "in the repo but do not enter the results. "
        "task and needs_tools are null: Laya does not predict them. "
        f"{info['truncated']} prompts truncated at the token budget of the checkpoint."
    )
    return write_run(
        args, digest, rows, model=info["model"], revision=info["revision"], notes=notes,
        extra={"decisions.jsonl": raw},
    )


def _litellm_heuristic(args, prompts: list[dict], digest: str) -> int:
    from s1d1_bench.vendor.litellm_heuristic import COMMIT, TAG, classify

    decisions = [(p["id"], classify(p["prompt"])) for p in prompts]
    rows = [
        {"id": i, "complexity": TIER_TO_LEVEL[d.tier], "task": None, "needs_tools": None} for i, d in decisions
    ]
    raw = [
        {"id": i, "tier": d.tier, "score": d.score, "signals": list(d.signals), "cause": d.cause}
        for i, d in decisions
    ]
    notes = (
        "Replay: uv run bench classify litellm-heuristic --prompts ... --out ... --author ... (no model, no network). "
        "LiteLLM's heuristic complexity scorer, tiers translated name to name: SIMPLE -> low, MEDIUM -> medium, "
        "COMPLEX -> high, REASONING -> expert (no trivial). The raw score of LiteLLM is in decisions.jsonl with the "
        "tier, the signals and the cause; it is not on the v7 scale, so labels.jsonl has no score and the score is "
        "for rank metrics only. Each prompt alone, system_prompt None, LiteLLM's default configuration. task and "
        f"needs_tools are null. Code copied verbatim ({TAG}), see src/s1d1_bench/vendor/litellm_heuristic/NOTICE.md."
    )
    return write_run(
        args, digest, rows, model="litellm:complexity_router:heuristic", revision=f"{TAG} {COMMIT}", notes=notes,
        extra={"decisions.jsonl": raw},
    )


# adapter -> function (args, prompts, sha256 of the prompts file) -> exit code
ADAPTERS: dict[str, Callable[[object, list[dict], str], int]] = {
    "random": _random,
    "length": _length,
    "constant": _constant,
    "s1d1": _s1d1,
    "laya": _laya,
    "litellm-heuristic": _litellm_heuristic,
}


def run_classify_cmd(args) -> int:
    """`bench classify <adapter>`: write the classifier folder `--out` for the prompts of `--prompts`."""
    adapter = ADAPTERS.get(args.adapter)
    if adapter is None:
        print(f"bench classify: the {args.adapter} adapter is not implemented yet", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists():
        return _fail(f"{out} already exists; classifier files are immutable. Use a new --out.")
    prompts_path = Path(args.prompts)
    digest = hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    prompts = read_jsonl(prompts_path)
    problem = _prompt_fault(prompts)
    if problem:
        return _fail(f"{prompts_path}: {problem}")
    try:
        return adapter(args, prompts, digest)
    except ValueError as exc:  # e.g. an id with no language suffix: before anything is written
        return _fail(str(exc))
