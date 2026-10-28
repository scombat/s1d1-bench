"""`bench pool`: the open prompt pool (Dolly-15k, OASST2 first messages, m-ArenaHard English).

`build_pool` is pure (no network); `fetch_rows` is the only function that touches the Hub, and it imports
`datasets` lazily so that importing this module (and the tests) never needs the `pool` dependency group.
"""

import os
import random
import sys
from datetime import date
from functools import cache
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from s1d1_bench.jsonl import write_jsonl
from s1d1_bench.keys import load_exclusions, normalize, prompt_key

SOURCES = ("dolly", "oasst2", "arena_hard")
MAX_CHARS = 8192
STAT_FIELDS = ("read", "empty", "too_long", "excluded", "duplicate", "kept")

# the licences of the plan (docs/32); `licence_check` compares them with the dataset cards on the Hub
LICENCES = {"dolly": "cc-by-sa-3.0", "oasst2": "apache-2.0", "arena_hard": "apache-2.0"}
REPOS = {
    "dolly": "databricks/databricks-dolly-15k",
    "oasst2": "OpenAssistant/oasst2",
    "arena_hard": "CohereLabs/m-ArenaHard",
}
ARENA_CONFIG, ARENA_SPLIT = "en", "test"
OASST_SPLITS = ("train", "validation")
OASST_COLUMNS = ["message_id", "parent_id", "text", "role", "lang", "review_result", "deleted"]

SCHEMA = pa.schema(
    [
        ("prompt_key", pa.string()),
        ("prompt", pa.string()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("lang", pa.string()),
        ("licence", pa.string()),
        ("source_label", pa.string()),
        ("chars", pa.int64()),
    ]
)

# OASST2 `lang` (ISO 639-1, with the region codes pt-BR and uk-UA) -> ISO 639-3. Exactly the 28 codes present in the
# train and validation splits at revision 179dd21f; the region is dropped by taking the primary subtag.
ISO_639_3 = {
    "ar": "ara", "bn": "ben", "ca": "cat", "cs": "ces", "da": "dan", "de": "deu", "en": "eng", "eo": "epo",
    "es": "spa", "eu": "eus", "fi": "fin", "fr": "fra", "he": "heb", "hu": "hun", "id": "ind", "it": "ita",
    "ja": "jpn", "ko": "kor", "nl": "nld", "pl": "pol", "pt": "por", "ru": "rus", "sv": "swe", "th": "tha",
    "tr": "tur", "uk": "ukr", "vi": "vie", "zh": "zho",
}  # fmt: skip


def iso_639_3(code: str | None) -> str:
    return ISO_639_3.get((code or "").split("-")[0].lower(), "und")


def dolly_prompt(row: dict) -> str:
    """Instruction, then the context (when there is one) after a blank line."""
    instruction, context = row["instruction"], row.get("context") or ""
    return f"{instruction}\n\n{context}" if context.strip() else instruction


def oasst2_rows(messages: list[dict]) -> list[dict]:
    """First prompts only: root `prompter` messages, not deleted, that passed review."""
    return [
        {
            "source": "oasst2",
            "source_id": m["message_id"],
            "prompt": m["text"],
            "lang": iso_639_3(m["lang"]),
            "licence": LICENCES["oasst2"],
            "source_label": "prompter",
        }
        for m in messages
        if m["role"] == "prompter" and m["parent_id"] is None and not m["deleted"] and m["review_result"]
    ]


def build_pool(rows: list[dict], exclusions: frozenset[str], max_chars: int = MAX_CHARS) -> tuple[pa.Table, dict]:
    """Normalize, filter and dedupe. A row is dropped for exactly one reason, tested in this order:
    empty, too_long (after normalization), excluded (key in `exclusions`), duplicate (first occurrence wins)."""
    stats = {s: dict.fromkeys(STAT_FIELDS, 0) for s in SOURCES}
    seen: set[str] = set()
    kept: list[dict] = []
    for row in rows:
        counts = stats[row["source"]]
        counts["read"] += 1
        prompt = normalize(row["prompt"])
        if not prompt:
            counts["empty"] += 1
            continue
        if len(prompt) > max_chars:
            counts["too_long"] += 1
            continue
        key = prompt_key(prompt)
        if key in exclusions:
            counts["excluded"] += 1
            continue
        if key in seen:
            counts["duplicate"] += 1
            continue
        seen.add(key)
        counts["kept"] += 1
        kept.append(
            {
                "prompt_key": key,
                "prompt": prompt,
                "source": row["source"],
                "source_id": str(row["source_id"]),
                "lang": row["lang"],
                "licence": row["licence"],
                "source_label": row["source_label"],
                "chars": len(prompt),
            }
        )
    return pa.Table.from_pylist(kept, schema=SCHEMA), stats


def sample_candidates(table: pa.Table, per_source: int, lang: str, seed: int) -> list[dict]:
    """Seeded sample of `per_source` prompts of language `lang` from each source, independent of row order."""
    rows = table.to_pylist()
    out = []
    for source in SOURCES:
        pool = sorted((r for r in rows if r["source"] == source and r["lang"] == lang), key=lambda r: r["prompt_key"])
        for r in random.Random(seed).sample(pool, min(per_source, len(pool))):
            out.append({"id": r["prompt_key"], "prompt": r["prompt"]})
    return out


# ---- Hub access (the only part that needs network and the `pool` dependency group) ----


def _hub_env() -> None:
    # public data, read-only: never send a token, even if one is stored on the machine
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


@cache
def _hub_info(source: str):
    _hub_env()
    from huggingface_hub import HfApi

    return HfApi(token=False).dataset_info(REPOS[source])


def _load(source: str, config: str | None, split: str, cache: Path):
    _hub_env()
    from datasets import load_dataset

    return load_dataset(
        REPOS[source], config, split=split, revision=_hub_info(source).sha, cache_dir=str(cache), token=False
    )


def fetch_rows(source: str, cache: Path) -> list[dict]:
    """Download (public, no token, no remote code) and adapt one source to the pool row format."""
    if source == "dolly":
        ds = _load(source, None, "train", cache)
        return [
            {
                "source": source,
                "source_id": str(i),
                "prompt": dolly_prompt(r),
                "lang": "eng",
                "licence": LICENCES[source],
                "source_label": r["category"],
            }
            for i, r in enumerate(ds)
        ]
    if source == "oasst2":
        rows: list[dict] = []
        for split in OASST_SPLITS:
            ds = _load(source, None, split, cache).select_columns(OASST_COLUMNS)
            rows += oasst2_rows(list(ds))
        return rows
    if source == "arena_hard":
        ds = _load(source, ARENA_CONFIG, ARENA_SPLIT, cache)
        return [
            {
                "source": source,
                "source_id": r["question_id"],
                "prompt": r["prompt"],
                "lang": "eng",
                "licence": LICENCES[source],
                "source_label": r["cluster"],
            }
            for r in ds
        ]
    raise ValueError(f"unknown source {source!r}")


def dataset_versions() -> dict:
    """Hub commit sha of each dataset as loaded, and the version of `datasets` that loaded it."""
    import datasets

    versions = {s: {"repo": REPOS[s], "revision": _hub_info(s).sha} for s in SOURCES}
    versions["arena_hard"].update(config=ARENA_CONFIG, split=ARENA_SPLIT)
    versions["oasst2"]["splits"] = list(OASST_SPLITS)
    versions["datasets_version"] = datasets.__version__
    return versions


def licence_check() -> dict:
    """{source: {plan, card}}: the licence the plan assumes vs the one on the dataset card."""
    return {s: {"plan": LICENCES[s], "card": (_hub_info(s).card_data or {}).get("license")} for s in SOURCES}


def run_pool(args) -> int:
    out, cand = Path(args.out), Path(args.candidates)
    out.parent.mkdir(parents=True, exist_ok=True)
    cand.parent.mkdir(parents=True, exist_ok=True)
    cache = out.parent / "cache"
    exclusions = load_exclusions(args.exclusions)

    rows: list[dict] = []
    for source in SOURCES:
        fetched = fetch_rows(source, cache)
        print(f"{source}: {len(fetched)} rows", file=sys.stderr)
        rows += fetched
    table, stats = build_pool(rows, exclusions)
    pq.write_table(table, out, compression="zstd")
    n_cand = write_jsonl(cand, sample_candidates(table, args.per_source, "eng", args.seed))

    check = licence_check()
    for source, c in check.items():
        if c["plan"] != c["card"]:
            print(f"WARNING licence mismatch for {source}: plan {c['plan']}, card {c['card']}", file=sys.stderr)
    import json

    report = {
        "date": date.today().isoformat(),
        "max_chars": MAX_CHARS,
        "exclusions": {"file": Path(args.exclusions).name, "keys": len(exclusions)},
        "sampling": {"lang": "eng", "per_source": args.per_source, "seed": args.seed, "candidates": n_cand},
        "stats": stats,
        "datasets": dataset_versions(),
        "licence_check": check,
    }
    (out.parent / "pool.stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for source, s in stats.items():
        print(f"{source}: " + " ".join(f"{k}={s[k]}" for k in STAT_FIELDS), file=sys.stderr)
    return 0
