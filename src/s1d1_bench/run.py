"""`bench run`: the judge and classifier folders of one prompts set merged into one table, one row per id.

The truths are made from the main judges only (`--main`): the lower ordinal median of their tier votes, and the
majority of their `task` and `needs_tools` votes. `truth_all` takes the votes of the four language versions of a base
prompt (rule 5 of the rubric: same labels), `truth_lang` the votes on the id alone; each also without the Gemma judges.
The table and its `<out stem>.run.json` record are derived data, rewritten at every call (docs/34, decision 6).
"""

import datetime
import glob
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Collection
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from s1d1_bench.classifier_file import read_classifier_labels
from s1d1_bench.jsonl import read_jsonl
from s1d1_bench.judge_file import OPTIONAL_LABEL_KEYS, SYSTEM, read_labels
from s1d1_bench.rubric import LEVELS

META_COLUMNS = ["id", "base_key", "lang", "origin", "source", "corrected"]
TRUTHS = ["truth_all", "truth_lang", "truth_all_nogemma", "truth_lang_nogemma"]
FIELDS = ["complexity", "task", "needs_tools"]
GEMMA = "gemma"  # a main judge whose meta.json `model` holds this (any case) is left out of the *_nogemma truths
RULES = (
    "complexity: lower ordinal median (trivial<low<medium<high<expert) of the non-null votes, the vote at index "
    "(n-1)//2 once sorted; null when the null votes are strictly more than half (7 of 12 is undecidable, 6 of 12 is "
    "not). truth_all: the main judges' votes on the 4 language versions of the base key (3 judges = 12 votes), "
    "repeated on each version; truth_lang: their votes on the id. task, needs_tools: majority of the non-null votes, "
    "a tie going to the value voted first in priority order (--main order, then the ids of the base key in prompts "
    "file order). *_nogemma: the same without the main judges whose model contains 'gemma'. undecidable: "
    "truth_all.complexity is null. spread: max - min tier index over the main judges' non-null votes on the id. "
    "lang_disagreement: truth_lang.complexity and truth_all.complexity both non-null and different."
)
_TYPES = {"complexity": pa.string(), "task": pa.string(), "needs_tools": pa.bool_()}


def lower_median(levels: list[str | None]) -> str | None:
    """The lower ordinal median of the non-null levels; None when the nulls are strictly more than half of the votes
    (or there is no vote at all)."""
    votes = sorted((v for v in levels if v is not None), key=LEVELS.index)
    nulls = len(levels) - len(votes)
    if not votes or 2 * nulls > len(levels):
        return None
    return votes[(len(votes) - 1) // 2]


def majority(values: list, order: list[str]):
    """The value with the most votes among the non-null `values`; None when every vote is null.

    `order` names the voter of each value (`values[i]` is the vote of `order[i]`) and lists the voters from the
    highest priority to the lowest: a tie goes to the tied value voted first in that list.
    """
    if len(values) != len(order):
        raise ValueError(f"{len(values)} values for {len(order)} voters")
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None
    best = max(counts.values())
    return next(v for v in values if v is not None and counts[v] == best)


def _truth(votes: list[tuple[str, dict]]) -> dict:
    """`votes`: (voter, labels) in priority order."""
    voters = [voter for voter, _ in votes]
    return {
        "complexity": lower_median([labels["complexity"] for _, labels in votes]),
        "task": majority([labels["task"] for _, labels in votes], voters),
        "needs_tools": majority([labels["needs_tools"] for _, labels in votes], voters),
    }


def truths(
    labels_by_judge: dict[str, dict[str, dict]], main: list[str], meta_rows: list[dict], gemma: Collection[str] = ()
) -> dict[str, dict]:
    """id -> truth_all, truth_lang, truth_all_nogemma, truth_lang_nogemma (each complexity, task, needs_tools),
    undecidable, spread, lang_disagreement.

    Only the judges of `main` vote, in that priority order; those in `gemma` are left out of the *_nogemma truths.
    `meta_rows` (id, base_key, lang) in prompts file order: it gives the ids of each base key and their order.
    """
    ids_of_key: dict[str, list[str]] = {}
    for row in meta_rows:
        ids_of_key.setdefault(row["base_key"], []).append(row["id"])
    judges = {"": list(main), "_nogemma": [j for j in main if j not in gemma]}
    by_key: dict[str, dict] = {}
    for key, ids in ids_of_key.items():
        by_key[key] = {
            suffix: _truth([(f"{j}/{i}", labels_by_judge[j][i]) for j in names for i in ids])
            for suffix, names in judges.items()
        }
    result: dict[str, dict] = {}
    for row in meta_rows:
        i = row["id"]
        entry = {}
        for suffix, names in judges.items():
            entry[f"truth_all{suffix}"] = by_key[row["base_key"]][suffix]
            entry[f"truth_lang{suffix}"] = _truth([(j, labels_by_judge[j][i]) for j in names])
        levels = [labels_by_judge[j][i]["complexity"] for j in main]
        ranks = [LEVELS.index(level) for level in levels if level is not None]
        lang_level, all_level = entry["truth_lang"]["complexity"], entry["truth_all"]["complexity"]
        entry["undecidable"] = all_level is None
        entry["spread"] = max(ranks) - min(ranks) if ranks else None
        entry["lang_disagreement"] = lang_level is not None and all_level is not None and lang_level != all_level
        result[i] = {k: entry[k] for k in [*TRUTHS, "undecidable", "spread", "lang_disagreement"]}
    return result


def load_folder(path: Path | str, prompts_sha: str, ids: list[str]) -> tuple[str, dict, dict[str, dict]]:
    """(name, meta, labels) of a judge or classifier folder made on the prompts file of SHA-256 `prompts_sha`, whose
    ids are `ids` (in file order). Refused (ValueError naming the folder) when its `prompts_sha256` is another one,
    when an id has no label (never filled in) or a label has an unknown id. A classifier folder (provider "system")
    also gives `score` and `needs_tools_probability` (None when absent)."""
    folder = Path(path)
    name = Path(os.path.abspath(folder)).name
    for needed in ("meta.json", "labels.jsonl"):
        if not (folder / needed).is_file():
            raise ValueError(f"{folder}: no {needed}")
    try:
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{folder}: meta.json is not valid JSON ({exc})") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"{folder}: meta.json is not a JSON object")
    found = meta.get("prompts_sha256")
    if found != prompts_sha:
        raise ValueError(
            f"{folder}: made on another prompts file: prompts_sha256 expected {prompts_sha} (the --prompts file), "
            f"found {found}"
        )
    reader = read_classifier_labels if meta.get("provider") == SYSTEM else read_labels
    try:
        labels = reader(folder / "labels.jsonl")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"{folder}: cannot read labels.jsonl ({exc!r}); run `bench validate` on it") from exc
    missing = [i for i in ids if i not in labels]
    if missing:
        raise ValueError(
            f"{folder}: no label for id {missing[0]!r} ({len(missing)} id(s) missing); run never fills in a label"
        )
    wanted = set(ids)
    unknown = [i for i in labels if i not in wanted]
    if unknown:
        raise ValueError(f"{folder}: label for id {unknown[0]!r}, which is not in the prompts file")
    for i, row in labels.items():
        if row["complexity"] is not None and row["complexity"] not in LEVELS:
            raise ValueError(f"{folder}: id {i!r}: complexity {row['complexity']!r} not in {LEVELS} or null")
    return name, meta, labels


def _column(name: str, values: list, kind: pa.DataType, columns: dict[str, pa.Array]) -> None:
    if name in columns:
        raise ValueError(f"two columns named {name!r}: two folders share a name")
    columns[name] = pa.array(values, type=kind)


def build_results(
    meta_rows: list[dict], judges: dict[str, dict[str, dict]], classifiers: dict[str, dict[str, dict]], truths: dict
) -> pa.Table:
    """One row per entry of `meta_rows` (prompts file order): the meta columns, the truths, then one column per folder
    and output, the judges then the classifiers in the order of the dicts. A classifier's `score` and
    `needs_tools_probability` columns are there only when at least one of its labels has the value."""
    columns: dict[str, pa.Array] = {}
    ids = [row["id"] for row in meta_rows]
    for name in META_COLUMNS:
        _column(name, [row[name] for row in meta_rows], pa.bool_() if name == "corrected" else pa.string(), columns)
    for truth in TRUTHS:
        for field in FIELDS:
            _column(f"{truth}.{field}", [truths[i][truth][field] for i in ids], _TYPES[field], columns)
    _column("undecidable", [truths[i]["undecidable"] for i in ids], pa.bool_(), columns)
    _column("spread", [truths[i]["spread"] for i in ids], pa.int64(), columns)
    _column("lang_disagreement", [truths[i]["lang_disagreement"] for i in ids], pa.bool_(), columns)
    for folders, optional in ((judges, []), (classifiers, OPTIONAL_LABEL_KEYS)):
        for name, labels in folders.items():
            for field in FIELDS:
                _column(f"{name}.{field}", [labels[i][field] for i in ids], _TYPES[field], columns)
            for key in optional:
                values = [labels[i].get(key) for i in ids]
                if any(v is not None for v in values):
                    _column(f"{name}.{key}", values, pa.float64(), columns)
    return pa.table(columns)


def _meta_rows(prompts: Path, ids: list[str]) -> tuple[Path, list[dict]]:
    """The meta columns of each id, in prompts file order, from `<prompts stem>.meta.parquet` next to the file."""
    path = prompts.with_name(prompts.stem + ".meta.parquet")
    if not path.is_file():
        raise ValueError(f"no {path.name} next to {prompts}")
    table = pq.read_table(path)
    absent = [c for c in META_COLUMNS if c not in table.column_names]
    if absent:
        raise ValueError(f"{path}: no column {', '.join(absent)}")
    by_id: dict[str, dict] = {}
    for row in table.select(META_COLUMNS).to_pylist():
        if row["id"] in by_id:
            raise ValueError(f"{path}: duplicate id {row['id']!r}")
        by_id[row["id"]] = row
    missing = [i for i in ids if i not in by_id]
    extra = sorted(set(by_id) - set(ids))
    if missing or extra:
        raise ValueError(f"{path}: not the ids of {prompts.name} (missing {missing[:1]}, extra {extra[:1]})")
    return path, [by_id[i] for i in ids]


def _folders(patterns: list[str] | None, option: str) -> list[Path]:
    """The folders the globs match (relative to the cwd), each once, in pattern then name order."""
    found: list[Path] = []
    seen: set[str] = set()
    for pattern in patterns or []:
        matches = sorted(Path(p) for p in glob.glob(pattern) if Path(p).is_dir())
        if not matches:
            raise ValueError(f"{option} {pattern!r} matches no folder")
        for folder in matches:
            real = os.path.realpath(folder)
            if real not in seen:
                seen.add(real)
                found.append(folder)
    return found


def _load_all(dirs: list[Path], digest: str, ids: list[str], metas: dict[str, dict]) -> dict[str, dict[str, dict]]:
    labels_by_name: dict[str, dict[str, dict]] = {}
    for folder in dirs:
        name, meta, labels = load_folder(folder, digest, ids)
        if name in metas:
            raise ValueError(f"two folders are named {name!r} (the column names would clash)")
        metas[name] = meta
        labels_by_name[name] = labels
    return labels_by_name


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _run(args) -> int:
    prompts = Path(args.prompts)
    digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
    ids = [row["id"] for row in read_jsonl(prompts)]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{prompts}: duplicate ids")
    meta_path, meta_rows = _meta_rows(prompts, ids)
    main = [name.strip() for name in args.main.split(",") if name.strip()]
    if not main or len(set(main)) != len(main):
        raise ValueError(f"--main {args.main!r}: give distinct judge folder names, comma-separated")
    metas: dict[str, dict] = {}
    judges = _load_all(_folders(args.judges, "--judges"), digest, ids, metas)
    classifiers = _load_all(_folders(args.classifiers, "--classifiers"), digest, ids, metas)
    for name in main:
        if name not in judges:
            raise ValueError(f"--main {name!r} is not among the judge folders loaded ({', '.join(judges) or 'none'})")
    gemma = [name for name in main if GEMMA in str(metas[name].get("model", "")).lower()]
    truth = truths({name: judges[name] for name in main}, main, meta_rows, gemma)
    judge_order = main + sorted(name for name in judges if name not in main)
    classifier_order = sorted(classifiers)
    table = build_results(
        meta_rows, {n: judges[n] for n in judge_order}, {n: classifiers[n] for n in classifier_order}, truth
    )

    def entry(name: str) -> dict:
        return {"name": name, **{k: metas[name].get(k) for k in ("model", "revision", "date")}}

    record = {
        "prompts": str(args.prompts),
        "prompts_sha256": digest,
        "meta": str(meta_path),
        "rows": table.num_rows,
        "judges": [entry(name) for name in judge_order],
        "classifiers": [entry(name) for name in classifier_order],
        "main": main,
        "gemma": gemma,
        "date": datetime.date.today().isoformat(),
        "rules": RULES,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(out, lambda tmp: pq.write_table(table, tmp, compression="zstd"))
    text = json.dumps(record, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    run_json = out.with_name(out.stem + ".run.json")
    _write_atomic(run_json, lambda tmp: tmp.write_text(text, encoding="utf-8", newline="\n"))
    spreads = [truth[i]["spread"] for i in ids if truth[i]["spread"] is not None]
    mean_spread = f"{sum(spreads) / len(spreads):.3f}" if spreads else "-"
    print(
        f"{out}: {table.num_rows} rows, {len(judges)} judges ({len(gemma)} Gemma in --main), {len(classifiers)} "
        f"classifiers; undecidable {sum(truth[i]['undecidable'] for i in ids)}, language disagreements "
        f"{sum(truth[i]['lang_disagreement'] for i in ids)}, mean spread {mean_spread}; record in {run_json}"
    )
    return 0


def run_run_cmd(args) -> int:
    """Write `--out` (parquet) and `<out stem>.run.json`, both rewritten; on a refusal, write neither."""
    try:
        return _run(args)
    except (ValueError, OSError) as exc:
        print(f"bench run: {exc}", file=sys.stderr)
        return 1
