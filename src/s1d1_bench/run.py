"""`bench run`: the judge and classifier folders of one prompts set merged into one table, one row per id.

The truths are made from the main judges only (`--main`): the ordinal median of their non-null tier votes (rule R5:
an odd count takes the middle vote; an even count takes the mean of the two middle tier indices rounded down, docs/34
§0.3, which is the lower median when the two middle votes are equal or adjacent, so 6/6 on adjacent tiers gives the
lower one, and low/high gives medium; null when the null votes are strictly more than half), and the majority of their
`task` and `needs_tools` votes. `truth_all` takes the votes of the four language versions of a base prompt (rule 5 of
the rubric: same labels), `truth_lang` the votes on the id alone; each also without the Gemma judges. An abstention
(a judge's `abstained` row) is a null vote on every field. The table and its `<out stem>.run.json` record are derived
data, rewritten at every call (docs/34, decision 6).

A classifier folder that has a `decisions.jsonl` (its raw output, e.g. a score outside [0, 1]) also gives the columns
`<name>.raw_score` and `<name>.raw_tier`, read as they are; `run.json` then marks the classifier `raw_decisions: true`.
"""

import datetime
import glob
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Collection
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from s1d1_bench.classifier_file import read_classifier_labels
from s1d1_bench.jsonl import read_jsonl
from s1d1_bench.judge_file import ABSTAINED, OPTIONAL_LABEL_KEYS, SYSTEM, abstentions, read_labels
from s1d1_bench.rubric import LEVELS

META_COLUMNS = ["id", "base_key", "lang", "origin", "source", "corrected"]
TRUTHS = ["truth_all", "truth_lang", "truth_all_nogemma", "truth_lang_nogemma"]
FIELDS = ["complexity", "task", "needs_tools"]
DECISIONS = "decisions.jsonl"  # a classifier's raw output: id, tier (a string), score (any number, even negative)
GEMMA = "gemma"  # a main judge whose meta.json `model` holds this (any case) is left out of the *_nogemma truths
RULES = (
    "complexity: ordinal median (trivial<low<medium<high<expert) of the non-null votes once sorted: an odd count "
    "takes the middle vote, an even count the mean of the two middle tier indices (n/2-1 and n/2) rounded down "
    "(low+high -> medium, low+medium -> low; docs/34 §0.3); null when the null votes are strictly more than "
    "half (7 of 12 is undecidable, 6 of 12 is not). truth_all: the main judges' votes on the 4 language versions of"
    " the base key (3 judges = 12 votes), repeated on each version; truth_lang: their votes on the id. task, "
    "needs_tools: majority of the non-null votes, a tie going to the value voted first in priority order (--main "
    "order, then the ids of the base key in prompts file order). *_nogemma: the same without the main judges whose "
    "model contains 'gemma'. undecidable: truth_all.complexity is null. spread: max - min tier index over the main "
    "judges' non-null votes on the id. lang_disagreement: truth_lang.complexity and truth_all.complexity both non-"
    "null and different. An abstention (a judge's row with `abstained`: no answer) is a null vote on every field: out "
    "of every median, majority and spread, counted as a null toward undecidable."
)
_TYPES = {"complexity": pa.string(), "task": pa.string(), "needs_tools": pa.bool_()}


def _sorted_votes(levels: list[str | None]) -> list[int] | None:
    """The tier indices of the non-null votes, sorted; None when the nulls are strictly more than half of the votes
    (or there is no vote at all)."""
    ranks = sorted(LEVELS.index(v) for v in levels if v is not None)
    nulls = len(levels) - len(ranks)
    if not ranks or 2 * nulls > len(levels):
        return None
    return ranks


def even_floor_median(levels: list[str | None]) -> str | None:
    """Rule R5, the median of every truth: the ordinal median of the non-null levels, with the floor applied to an even
    count (hence the name): the middle vote of an odd count, the mean of the two middle tier indices rounded down for an
    even count (docs/34 §0.3: low and high give medium; two adjacent tiers give the lower one). None when the nulls are
    strictly more than half of the votes."""
    ranks = _sorted_votes(levels)
    if ranks is None:
        return None
    n = len(ranks)
    if n % 2:
        return LEVELS[ranks[n // 2]]
    return LEVELS[(ranks[n // 2 - 1] + ranks[n // 2]) // 2]


def plain_lower_median(levels: list[str | None]) -> str | None:
    """The plain lower median: the lower of the two middle votes for an even count (the middle vote for an odd one),
    same null rule. Not a truth: used only for the diagnostic line of `bench run`, which counts the base keys where it
    would change `truth_all`."""
    ranks = _sorted_votes(levels)
    return None if ranks is None else LEVELS[ranks[(len(ranks) - 1) // 2]]


def majority(values: list, order: list[str]):
    """The value with the most votes among the non-null `values`; None when every vote is null.

    `order` names the voter of each value (`values[i]` is the vote of `order[i]`) and lists the voters from the
    highest priority to the lowest: a tie goes to the tied value voted first in that list. The names are not read:
    `values` must already be in that priority order.
    """
    if len(values) != len(order):
        raise ValueError(f"{len(values)} values for {len(order)} voters")
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None
    best = max(counts.values())
    return next(v for v in values if v is not None and counts[v] == best)


def _vote(labels: dict, field: str):
    """A judge's vote on `field`: None (no vote) for an abstention."""
    return None if labels.get(ABSTAINED) else labels[field]


def _truth(votes: list[tuple[str, dict]], median: Callable[[list[str | None]], str | None]) -> dict:
    """`votes`: (voter, labels) in priority order."""
    voters = [voter for voter, _ in votes]
    return {
        "complexity": median([_vote(labels, "complexity") for _, labels in votes]),
        "task": majority([_vote(labels, "task") for _, labels in votes], voters),
        "needs_tools": majority([_vote(labels, "needs_tools") for _, labels in votes], voters),
    }


def truths(
    labels_by_judge: dict[str, dict[str, dict]],
    main: list[str],
    meta_rows: list[dict],
    gemma: Collection[str] = (),
    median: Callable[[list[str | None]], str | None] = even_floor_median,
) -> dict[str, dict]:
    """id -> truth_all, truth_lang, truth_all_nogemma, truth_lang_nogemma (each complexity, task, needs_tools),
    undecidable, spread, lang_disagreement.

    Only the judges of `main` vote, in that priority order; those in `gemma` are left out of the *_nogemma truths.
    `meta_rows` (id, base_key, lang) in prompts file order: it gives the ids of each base key and their order.
    `median` makes every complexity truth (`plain_lower_median` only for the diagnostic of `bench run`).
    """
    ids_of_key: dict[str, list[str]] = {}
    for row in meta_rows:
        ids_of_key.setdefault(row["base_key"], []).append(row["id"])
    judges = {"": list(main), "_nogemma": [j for j in main if j not in gemma]}
    by_key: dict[str, dict] = {}
    for key, ids in ids_of_key.items():
        by_key[key] = {
            suffix: _truth([(f"{j}/{i}", labels_by_judge[j][i]) for j in names for i in ids], median)
            for suffix, names in judges.items()
        }
    result: dict[str, dict] = {}
    for row in meta_rows:
        i = row["id"]
        entry = {}
        for suffix, names in judges.items():
            entry[f"truth_all{suffix}"] = by_key[row["base_key"]][suffix]
            entry[f"truth_lang{suffix}"] = _truth([(j, labels_by_judge[j][i]) for j in names], median)
        levels = [_vote(labels_by_judge[j][i], "complexity") for j in main]
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
    when an id has no row (never filled in; an abstention row is a row, kept with its `abstained` status) or a row has
    an unknown id. A classifier folder (provider "system") also gives `score` and `needs_tools_probability` (None when
    absent)."""
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


def read_decisions(path: Path | str, ids: list[str]) -> dict[str, dict]:
    """`id` -> {"tier", "score"} from a classifier's `decisions.jsonl`, the values untouched (a score may be negative).
    Refused (ValueError naming the file) when a row has no id, a string `tier` or a finite numeric `score`, when an id
    appears twice, when an id of `ids` has no row or a row has an id not in `ids`."""
    decisions: dict[str, dict] = {}
    try:
        rows = read_jsonl(path)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: cannot read it ({exc!r})") from exc
    for number, row in enumerate(rows, start=1):
        i = row.get("id") if isinstance(row, dict) else None
        if not isinstance(i, str):
            raise ValueError(f"{path}: row {number} has no string 'id'")
        score = row.get("score")
        if not isinstance(row.get("tier"), str) or isinstance(score, bool) or not isinstance(score, int | float):
            raise ValueError(f"{path}: id {i!r}: needs a string 'tier' and a numeric 'score'")
        if not math.isfinite(score):
            raise ValueError(f"{path}: id {i!r}: 'score' must be a finite number, not {score!r}")
        if i in decisions:
            raise ValueError(f"{path}: duplicate id {i!r}")
        decisions[i] = {"tier": row["tier"], "score": float(score)}
    missing = [i for i in ids if i not in decisions]
    if missing:
        raise ValueError(f"{path}: no decision for id {missing[0]!r} ({len(missing)} id(s) missing)")
    wanted = set(ids)
    unknown = [i for i in decisions if i not in wanted]
    if unknown:
        raise ValueError(f"{path}: decision for id {unknown[0]!r}, which is not in the prompts file")
    return decisions


def _column(name: str, values: list, kind: pa.DataType, columns: dict[str, pa.Array]) -> None:
    if name in columns:
        raise ValueError(f"two columns named {name!r}: two folders share a name")
    columns[name] = pa.array(values, type=kind)


def build_results(
    meta_rows: list[dict],
    judges: dict[str, dict[str, dict]],
    classifiers: dict[str, dict[str, dict]],
    truths: dict,
    decisions: dict[str, dict[str, dict]] | None = None,
) -> pa.Table:
    """One row per entry of `meta_rows` (prompts file order): the meta columns, the truths, then one column per folder
    and output, the judges then the classifiers in the order of the dicts. Every judge has a `<name>.abstained` bool
    column (true on its abstention rows); a classifier's `score` and `needs_tools_probability` columns are there only
    when at least one of its labels has the value. `decisions` (classifier name -> id -> {"tier", "score"}, from
    `read_decisions`) adds, after the other columns of the classifiers it names, `<name>.raw_score` (float64) and
    `<name>.raw_tier` (string)."""
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
    for folders, optional, judge in ((judges, [], True), (classifiers, OPTIONAL_LABEL_KEYS, False)):
        for name, labels in folders.items():
            for field in FIELDS:
                _column(f"{name}.{field}", [labels[i][field] for i in ids], _TYPES[field], columns)
            if judge:
                _column(f"{name}.{ABSTAINED}", [bool(labels[i].get(ABSTAINED)) for i in ids], pa.bool_(), columns)
            for key in optional:
                values = [labels[i].get(key) for i in ids]
                if any(v is not None for v in values):
                    _column(f"{name}.{key}", values, pa.float64(), columns)
            if not judge and name in (decisions or {}):
                raw = decisions[name]
                _column(f"{name}.raw_score", [raw[i]["score"] for i in ids], pa.float64(), columns)
                _column(f"{name}.raw_tier", [raw[i]["tier"] for i in ids], pa.string(), columns)
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


def _load_all(
    dirs: list[Path],
    digest: str,
    ids: list[str],
    metas: dict[str, dict],
    decisions: dict[str, dict[str, dict]] | None = None,
) -> dict[str, dict[str, dict]]:
    """The labels of each folder by name; when `decisions` is given, the `decisions.jsonl` of the system folders that
    have one is read into it, by name."""
    labels_by_name: dict[str, dict[str, dict]] = {}
    for folder in dirs:
        name, meta, labels = load_folder(folder, digest, ids)
        if name in metas:
            raise ValueError(f"two folders are named {name!r} (the column names would clash)")
        metas[name] = meta
        labels_by_name[name] = labels
        if decisions is not None and meta.get("provider") == SYSTEM and (Path(folder) / DECISIONS).is_file():
            decisions[name] = read_decisions(Path(folder) / DECISIONS, ids)
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
    raw_decisions: dict[str, dict[str, dict]] = {}
    classifiers = _load_all(_folders(args.classifiers, "--classifiers"), digest, ids, metas, raw_decisions)
    for name in main:
        if name not in judges:
            raise ValueError(f"--main {name!r} is not among the judge folders loaded ({', '.join(judges) or 'none'})")
    gemma = [name for name in main if GEMMA in str(metas[name].get("model", "")).lower()]
    truth = truths({name: judges[name] for name in main}, main, meta_rows, gemma)
    judge_order = main + sorted(name for name in judges if name not in main)
    classifier_order = sorted(classifiers)
    table = build_results(
        meta_rows, {n: judges[n] for n in judge_order}, {n: classifiers[n] for n in classifier_order}, truth,
        raw_decisions,
    )  # fmt: skip

    def entry(name: str) -> dict:
        found = {"name": name, **{k: metas[name].get(k) for k in ("model", "revision", "date")}}
        return {**found, "raw_decisions": True} if name in raw_decisions else found

    def abstained(name: str) -> list[dict]:
        """The abstentions of a judge folder, in prompts file order."""
        found = abstentions(judges[name])
        return [{"id": i, "status": found[i]} for i in ids if i in found]

    record = {
        "prompts": str(args.prompts),
        "prompts_sha256": digest,
        "meta": str(meta_path),
        "rows": table.num_rows,
        "judges": [entry(name) for name in judge_order],
        "classifiers": [entry(name) for name in classifier_order],
        "main": main,
        "gemma": gemma,
        "abstentions": {name: abstained(name) for name in judge_order},
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
    n_abstained = sum(len(rows) for rows in record["abstentions"].values())
    print(
        f"{out}: {table.num_rows} rows, {len(judges)} judges ({len(gemma)} Gemma in --main), {len(classifiers)} "
        f"classifiers ({len(raw_decisions)} with raw decisions); abstentions {n_abstained}, undecidable "
        f"{sum(truth[i]['undecidable'] for i in ids)}, language disagreements "
        f"{sum(truth[i]['lang_disagreement'] for i in ids)}, mean spread {mean_spread}; record in {run_json}"
    )
    plain = truths({name: judges[name] for name in main}, main, meta_rows, gemma, median=plain_lower_median)
    first_id = {}  # base key -> its first id: truth_all is the same on every id of a key
    for row in meta_rows:
        first_id.setdefault(row["base_key"], row["id"])

    def changed(column: str) -> int:
        return sum(truth[i][column]["complexity"] != plain[i][column]["complexity"] for i in first_id.values())

    print(
        f"plain lower median instead (diagnostic): truth_all.complexity would differ on {changed('truth_all')} of "
        f"{len(first_id)} base keys, truth_all_nogemma.complexity on {changed('truth_all_nogemma')} of {len(first_id)}"
    )
    return 0


def run_run_cmd(args) -> int:
    """Write `--out` (parquet) and `<out stem>.run.json`, both rewritten; on a refusal, write neither."""
    try:
        return _run(args)
    except (ValueError, OSError) as exc:
        print(f"bench run: {exc}", file=sys.stderr)
        return 1
