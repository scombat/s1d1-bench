"""`bench export`: the paired prompt file the judges see, its hidden metadata, the chat lots and the instructions."""

import hashlib
import json
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
        ("translator", pa.string()),  # null for eng
        ("corrected", pa.bool_()),  # True when a human correction replaced the model's text; False for eng
    ]
)
CORRECTION_KEYS = ("id", "prompt", "reason", "author")


def check_envelope_safe(prompts: list[dict]) -> None:
    """Refuse prompts with a line that could pass for an envelope line of a lot file (`</prompt>`, or one starting
    with `<prompt`): the prompt text is never escaped or altered, so such a set cannot be exported."""
    bad = [
        p["id"]
        for p in prompts
        if any(
            (line := raw.strip().lower()) == "</prompt>" or line.startswith("<prompt")
            for raw in p["prompt"].splitlines()
        )
    ]
    if bad:
        raise ValueError(f"prompts with a line that could forge a lot envelope (ids: {', '.join(bad)})")


def _read_corrections(path: Path) -> dict[str, str]:
    """The `id` -> corrected text of a `<lang>.corrections.jsonl` (empty when the file does not exist). Every line
    carries the four keys `id`, `prompt`, `reason`, `author` as non-empty strings; an id given twice is refused."""
    if not path.exists():
        return {}
    fixes: dict[str, str] = {}
    for n, row in enumerate(read_jsonl(path), start=1):
        bad = [k for k in CORRECTION_KEYS if not isinstance(row.get(k), str) or not row[k].strip()]
        if bad:
            raise ValueError(f"{path}, line {n}: {', '.join(bad)} missing or empty")
        if row["id"] in fixes:
            raise ValueError(f"{path}: {row['id']} corrected twice")
        fixes[row["id"]] = row["prompt"]
    return fixes


def read_translations(dirs: list[Path], langs: list[str], corrections: bool = True) -> dict[str, list[dict]]:
    """For each language of `langs` but `eng`, the rows of `<lang>.jsonl` of every folder of `dirs` (`id`,
    `base_key`, `prompt`, plus `translator` from `<lang>.meta.json` and `corrected`).

    With `corrections`, the `<lang>.corrections.jsonl` of a folder replaces the text of the rows of that folder it
    names and sets `corrected`; the model's file is only read, never rewritten. Several folders are the waves of a
    translation: a model's file is never completed, a new wave is one more folder, and an id in two folders is refused.
    """
    out: dict[str, list[dict]] = {}
    for lang in langs:
        if lang == "eng":
            continue
        rows: list[dict] = []
        seen: dict[str, Path] = {}  # id -> the folder that gave it
        for folder in map(Path, dirs):
            path = folder / f"{lang}.jsonl"
            if not path.exists():
                raise FileNotFoundError(f"no {lang}.jsonl in the translation folder {folder}")
            meta_path = folder / f"{lang}.meta.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"no {lang}.meta.json in the translation folder {folder}")
            translator = json.loads(meta_path.read_text(encoding="utf-8")).get("translator")
            if not translator:
                raise ValueError(f"{meta_path} names no translator")
            fixes = _read_corrections(folder / f"{lang}.corrections.jsonl") if corrections else {}
            ids = set()
            for n, row in enumerate(read_jsonl(path), start=1):
                if not all(isinstance(row.get(k), str) for k in ("id", "base_key", "prompt")):
                    raise ValueError(f"{path}, line {n}: id, base_key and prompt are mandatory strings")
                if row["id"] != f"{row['base_key']}-{lang}":
                    raise ValueError(f"{path}: id {row['id']} is not <base_key>-{lang}")
                if row["id"] in seen:
                    raise ValueError(f"{row['id']} is given twice, in {seen[row['id']]} and {folder}")
                seen[row["id"]] = folder
                ids.add(row["id"])
                rows.append(
                    {
                        "id": row["id"],
                        "base_key": row["base_key"],
                        "prompt": fixes.get(row["id"], row["prompt"]),
                        "translator": translator,
                        "corrected": row["id"] in fixes,
                    }
                )
            unknown = sorted(set(fixes) - ids)
            if unknown:
                raise ValueError(f"{folder}/{lang}.corrections.jsonl corrects ids not in {path}: {', '.join(unknown)}")
        out[lang] = rows
    return out


def build_export(
    selected: list[dict],
    handwritten: list[dict],
    langs: list[str],
    seed: int,
    translations: dict[str, list[dict]] | None = None,
) -> tuple[list[dict], pa.Table]:
    """The shuffled prompts (`id`, `prompt` only) and their metadata, from pool rows and handwritten rows, and for a
    language other than `eng`, from `translations[lang]` (rows of `read_translations`).

    A translated row takes the metadata of its `base_key` (a selected or handwritten prompt); each of them must be
    translated into each requested language, and a translation of anything else is refused.
    `id` is the base key plus `-` plus the language code. The order depends on `seed` and on the ids only, never on
    the order of the inputs.
    """
    bases: list[tuple[dict, dict]] = []  # (metadata but id and lang, source row)
    for row in selected:
        meta = {
            "base_key": row["prompt_key"],
            "origin": "pool",
            "source": row["source"],
            "source_id": row["source_id"],
            "licence": row["licence"],
            "authors": None,
            "intended_complexity": None,
            "intended_task": None,
            "intended_needs_tools": None,
        }
        bases.append((meta, row))
    for position, row in enumerate(handwritten, start=1):
        meta = {
            "base_key": prompt_key(normalize(row["prompt"])),
            "origin": "handwritten",
            "source": "handwritten",
            "source_id": f"A.{position}",
            "licence": HANDWRITTEN_LICENCE,
            "authors": row["authors"],
            "intended_complexity": row["intended_complexity"],
            "intended_task": row["intended_task"],
            "intended_needs_tools": row["intended_needs_tools"],
        }
        bases.append((meta, row))

    entries: list[tuple[dict, str]] = []
    for meta, row in bases:  # the prompts as written, in their own language (handwritten ones are English)
        lang = row.get("lang", "eng")
        if lang in langs:
            entries.append(({**meta, "lang": lang, "translator": None, "corrected": False}, row["prompt"]))
    by_key = {meta["base_key"]: (meta, row.get("lang", "eng")) for meta, row in bases}
    for lang in langs:
        if lang == "eng":
            continue
        rows = (translations or {}).get(lang, [])
        unknown = sorted({r["base_key"] for r in rows} - by_key.keys())
        if unknown:
            raise ValueError(f"{lang} translations of keys neither selected nor handwritten: {', '.join(unknown)}")
        translated = {r["base_key"] for r in rows}
        missing = sorted(k for k, (_, own) in by_key.items() if own != lang and k not in translated)
        if missing:
            raise ValueError(f"no {lang} translation for {len(missing)} key(s): {', '.join(missing)}")
        for r in rows:
            meta = {**by_key[r["base_key"]][0], "lang": lang, "translator": r["translator"]}
            entries.append(({**meta, "corrected": r["corrected"]}, r["prompt"]))

    rows = []
    for meta, text in entries:
        meta["id"] = f"{meta['base_key']}-{meta['lang']}"
        rows.append((meta, text))
    rows.sort(key=lambda r: r[0]["id"])
    ids = [meta["id"] for meta, _ in rows]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate ids in the export: {', '.join(duplicates)}")
    random.Random(seed).shuffle(rows)

    prompts = [{"id": meta["id"], "prompt": text} for meta, text in rows]
    check_envelope_safe(prompts)
    table = pa.Table.from_pylist([meta for meta, _ in rows], schema=META_SCHEMA)
    return prompts, table


def write_lots(prompts: list[dict], out_dir: Path | str, size: int = LOT_SIZE) -> list[Path]:
    """Lots of `size` prompts, `01.md`, `02.md`, ...: the batch instructions, a blank line, then each prompt between
    an opening line `<prompt id="ID">` and a closing line `</prompt>`. The prompt text is written as is."""
    check_envelope_safe(prompts)
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


def _drop_unselected(translations: dict[str, list[dict]], unselected: set[str]) -> dict[str, list[dict]]:
    """The translations without those of pool keys left out of the selection (a vetoed prompt keeps its translation
    in the first wave's file, which is never rewritten), said on stderr. Any other unknown key is left to
    `build_export`, which refuses it."""
    kept = {lang: [r for r in rows if r["base_key"] not in unselected] for lang, rows in translations.items()}
    dropped = sorted(r["id"] for rows in translations.values() for r in rows if r["base_key"] in unselected)
    if dropped:
        listed = ", ".join(dropped[:10]) + (f", ... ({len(dropped) - 10} more)" if len(dropped) > 10 else "")
        what = f"{len(dropped)} translation(s) of pool keys that are not selected (vetoed?)"
        print(f"bench export: ignored {what}: {listed}", file=sys.stderr)
    return kept


def _refuse_to_rewrite_a_judged_file(jsonl: Path, prompts: list[dict], judges_dir: Path) -> None:
    """A prompts file that a judge folder pins by SHA-256 is never rewritten with other content (the CI would then
    refuse that folder). Re-exporting the very same content stays allowed."""
    if not jsonl.exists():
        return
    old = hashlib.sha256(jsonl.read_bytes()).hexdigest()
    metas = {d.parent.name: json.loads(d.read_text(encoding="utf-8")) for d in judges_dir.glob("*/meta.json")}
    pinned = [name for name, meta in metas.items() if meta.get("prompts_sha256") == old]
    if not pinned:
        return
    probe = jsonl.with_name(jsonl.name + ".probe")
    try:
        write_jsonl(probe, prompts)
        new = hashlib.sha256(probe.read_bytes()).hexdigest()
    finally:
        probe.unlink(missing_ok=True)
    if new != old:
        raise SystemExit(
            f"bench export: {jsonl.name} is pinned by judge folder(s) {', '.join(sorted(pinned))} and the new content "
            "differs; export under another --name (the corrections are applied by the final export)"
        )


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
    foreign = [lang for lang in langs if lang != "eng"]
    if foreign and not args.translations:
        raise SystemExit(
            f"bench export: --langs {','.join(foreign)} needs --translations <dir>: only eng prompts are in the pool"
        )
    translations = read_translations([at(d) for d in args.translations], langs) if args.translations else None
    if translations:
        handwritten_keys = {prompt_key(normalize(row["prompt"])) for row in handwritten}
        translations = _drop_unselected(translations, pool.keys() - set(keys) - handwritten_keys)

    prompts, meta = build_export([pool[k] for k in keys], handwritten, langs, args.seed, translations)

    out = at("data/prompts")
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{args.name}.jsonl"
    _refuse_to_rewrite_a_judged_file(jsonl, prompts, at("judges"))
    write_jsonl(jsonl, prompts)
    pq.write_table(meta, out / f"{args.name}.meta.parquet", compression="zstd")
    (out / f"{args.name}.sha256").write_text(hashlib.sha256(jsonl.read_bytes()).hexdigest() + "\n", encoding="ascii")
    lots_dir = out / f"{args.name}.lots"
    for stale in lots_dir.glob("*.md"):  # a smaller re-export must not keep the lots of a bigger one
        stale.unlink()
    lots = write_lots(prompts, lots_dir)
    write_consigne(at("consigne"))
    print(f"{args.name}: {len(prompts)} prompts, {len(lots)} lot(s)", file=sys.stderr)
    return 0
