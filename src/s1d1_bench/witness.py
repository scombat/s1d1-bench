"""`bench witness`: the m-ArenaHard translations (Google Translate, per the dataset card) of the Arena-Hard prompts
of a prompts set, as witnesses for the model translations. Public data, read without a token."""

import datetime
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

from s1d1_bench import pool
from s1d1_bench.jsonl import write_jsonl

CONFIGS = {"fra": "fr", "rus": "ru", "zho": "zh"}  # ISO 639-3 -> m-ArenaHard config
SPLIT = "test"
TRANSLATOR = "Google Translate, per the m-ArenaHard dataset card"


def witness_rows(arena: list[dict], meta_rows: list[dict], lang: str) -> list[dict]:
    """The m-ArenaHard row (one language) of each Arena-Hard prompt of `meta_rows`, by `question_id`, sorted by id.
    A base key without a witness raises ValueError naming it."""
    by_question = {r["question_id"]: r["prompt"] for r in arena}
    rows, missing = [], []
    for m in meta_rows:
        if m["source"] != "arena_hard":
            continue
        if m["source_id"] in by_question:
            key = m["base_key"]
            rows.append({"id": f"{key}-{lang}", "base_key": key, "prompt": by_question[m["source_id"]]})
        else:
            missing.append(m["base_key"])
    if missing:
        raise ValueError(f"no m-ArenaHard witness in {lang} for {', '.join(sorted(missing))}")
    return sorted(rows, key=lambda r: r["id"])


def run_witness(args) -> int:
    """`bench witness`: write `<lang>.jsonl` and `<lang>.meta.json` per language, and `arena.select.txt` (the
    Arena-Hard base keys), into `--out`. Refuses to overwrite a `<lang>.jsonl`."""

    def fail(message: str) -> int:
        print(f"bench witness: {message}", file=sys.stderr)
        return 1

    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    unknown = [lang for lang in langs if lang not in CONFIGS]
    if unknown or not langs:
        return fail(f"--langs takes {', '.join(CONFIGS)}; got {args.langs!r}")
    out, meta_path = Path(args.out), Path(args.meta)
    existing = [f"{lang}.jsonl" for lang in langs if (out / f"{lang}.jsonl").exists()]
    if existing:
        return fail(f"{out}: {', '.join(existing)} already exist; a witness file is never overwritten.")

    meta_rows = pq.read_table(meta_path, columns=["base_key", "source", "source_id"]).to_pylist()
    cache = Path(args.cache)
    revision = pool._hub_info("arena_hard").sha
    try:
        rows = {
            lang: witness_rows(list(pool.load_split("arena_hard", CONFIGS[lang], SPLIT, cache)), meta_rows, lang)
            for lang in langs
        }
    except ValueError as exc:
        return fail(str(exc))

    out.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        write_jsonl(out / f"{lang}.jsonl", rows[lang])
        meta = {
            "lang": lang,
            "repo": pool.REPOS["arena_hard"],
            "config": CONFIGS[lang],
            "split": SPLIT,
            "revision": revision,
            "licence": pool.LICENCES["arena_hard"],
            "translator": TRANSLATOR,
            "source_prompts_meta": meta_path.name.removesuffix(".meta.parquet"),
            "date": datetime.date.today().isoformat(),
        }
        (out / f"{lang}.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
        )
    keys = sorted({m["base_key"] for m in meta_rows if m["source"] == "arena_hard"})
    (out / "arena.select.txt").write_text("".join(f"{k}\n" for k in keys), encoding="utf-8", newline="\n")
    for lang in langs:
        print(f"{lang}: {len(rows[lang])} witnesses", file=sys.stderr)
    return 0
