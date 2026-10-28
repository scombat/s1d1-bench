"""`bench export`: the paired prompt file the judges see, its hidden metadata, the chat lots and the instructions."""

import hashlib
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from s1d1_bench.consigne import make_consigne
from s1d1_bench.jsonl import read_jsonl, write_jsonl
from s1d1_bench.keys import normalize, prompt_key

LOT_SIZE = 40
HANDWRITTEN_LICENCE = "CC-BY-4.0"

META_SCHEMA = pa.schema(
    [
        ("id", pa.string()),
        ("base_key", pa.string()),
        ("lang", pa.string()),
        ("origin", pa.string()),
        ("source", pa.string()),
        ("source_id", pa.string()),
        ("licence", pa.string()),
        ("authors", pa.string()),
        ("intended_complexity", pa.string()),
        ("intended_task", pa.string()),
        ("intended_needs_tools", pa.bool_()),
    ]
)


def build_export(
    selected: list[dict], handwritten: list[dict], langs: list[str], seed: int
) -> tuple[list[dict], pa.Table]:
    """The shuffled prompts (`id`, `prompt` only) and their metadata, from pool rows and handwritten rows.

    `id` is the base key plus `-` plus the language code. The order depends on `seed` and on the ids only, never on
    the order of the inputs.
    """
    entries: list[tuple[dict, dict]] = []
    for row in selected:
        lang = row.get("lang", "eng")
        if lang not in langs:
            continue
        meta = {
            "base_key": row["prompt_key"],
            "lang": lang,
            "origin": "pool",
            "source": row["source"],
            "source_id": row["source_id"],
            "licence": row["licence"],
            "authors": None,
            "intended_complexity": None,
            "intended_task": None,
            "intended_needs_tools": None,
        }
        entries.append((meta, row))
    for position, row in enumerate(handwritten, start=1):
        if "eng" not in langs:
            continue
        meta = {
            "base_key": prompt_key(normalize(row["prompt"])),
            "lang": "eng",
            "origin": "handwritten",
            "source": "handwritten",
            "source_id": f"A.{position}",
            "licence": HANDWRITTEN_LICENCE,
            "authors": row["authors"],
            "intended_complexity": row["intended_complexity"],
            "intended_task": row["intended_task"],
            "intended_needs_tools": row["intended_needs_tools"],
        }
        entries.append((meta, row))

    rows = []
    for meta, row in entries:
        meta["id"] = f"{meta['base_key']}-{meta['lang']}"
        rows.append((meta, row["prompt"]))
    rows.sort(key=lambda r: r[0]["id"])
    ids = [meta["id"] for meta, _ in rows]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate ids in the export: {', '.join(duplicates)}")
    random.Random(seed).shuffle(rows)

    prompts = [{"id": meta["id"], "prompt": text} for meta, text in rows]
    table = pa.Table.from_pylist([meta for meta, _ in rows], schema=META_SCHEMA)
    return prompts, table


def write_lots(prompts: list[dict], out_dir: Path | str, size: int = LOT_SIZE) -> list[Path]:
    """Lots of `size` prompts, `01.md`, `02.md`, ...: the batch instructions, a blank line, then each prompt between
    an opening line `<prompt id="ID">` and a closing line `</prompt>`. The prompt text is written as is."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    head = make_consigne("batch").rstrip("\n") + "\n\n"
    paths = []
    for n, start in enumerate(range(0, len(prompts), size), start=1):
        blocks = [f'<prompt id="{p["id"]}">\n{p["prompt"]}\n</prompt>\n' for p in prompts[start : start + size]]
        path = out_dir / f"{n:02d}.md"
        path.write_text(head + "\n".join(blocks), encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


def write_consigne(out_dir: Path | str) -> None:
    """`v7-single.md` and `v7-batch.md`, rewritten each time (identical while the rubric does not change)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("single", "batch"):
        (out_dir / f"v7-{kind}.md").write_text(make_consigne(kind), encoding="utf-8", newline="\n")


def run_export(args) -> int:
    root = Path(args.root)

    def at(path: Path | str) -> Path:  # relative paths are relative to the root
        return root / path

    keys = [line.strip() for line in at(args.select).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(set(keys)) != len(keys):
        raise SystemExit(f"bench export: {args.select} lists a prompt key twice")
    pool = {r["prompt_key"]: r for r in pq.read_table(at(args.pool)).to_pylist()}
    missing = [k for k in keys if k not in pool]
    if missing:
        listed = ", ".join(missing)
        raise SystemExit(f"bench export: {len(missing)} selected key(s) not in the pool {args.pool}: {listed}")
    handwritten = read_jsonl(at(args.handwritten)) if args.handwritten else []
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]

    prompts, meta = build_export([pool[k] for k in keys], handwritten, langs, args.seed)

    out = at("data/prompts")
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{args.name}.jsonl"
    write_jsonl(jsonl, prompts)
    pq.write_table(meta, out / f"{args.name}.meta.parquet", compression="zstd")
    (out / f"{args.name}.sha256").write_text(hashlib.sha256(jsonl.read_bytes()).hexdigest() + "\n", encoding="ascii")
    lots = write_lots(prompts, out / f"{args.name}.lots")
    write_consigne(at("consigne"))
    print(f"{args.name}: {len(prompts)} prompts, {len(lots)} lot(s)", file=sys.stderr)
    return 0
