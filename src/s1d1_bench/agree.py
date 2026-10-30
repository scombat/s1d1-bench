"""`bench agree`: how often judges give the same label to the same prompts.

An id where either judge of a pair abstained (an `abstained` row: no answer) is left out of that pair: `n` counts the
ids compared, the `abstained` column the common ids left out."""

import json
import sys
from itertools import combinations
from pathlib import Path

from s1d1_bench.judge_file import ABSTAINED, read_labels
from s1d1_bench.rubric import LEVELS

TABLE_HEADER = ["judge A", "judge B", "n", "abstained", "tier exact", "tier ±1", "task", "needs_tools"]


def _tier_distance(a: str | None, b: str | None) -> int | None:
    """Distance in levels; 0 for two nulls; None (a disagreement however large the tolerance) for null vs a level."""
    if a is None or b is None:
        return 0 if a is b else None
    return abs(LEVELS.index(a) - LEVELS.index(b))


def _rate(hits: int, n: int) -> float | None:
    return hits / n if n else None


def pairwise_agreement(labels_by_judge: dict[str, dict[str, dict]]) -> dict[tuple[str, str], dict]:
    """For each unordered pair of judges (ordered as in `labels_by_judge`), the agreement on the ids both answered;
    `left_out` counts the ids both have where either abstained."""
    result: dict[tuple[str, str], dict] = {}
    for name_a, name_b in combinations(labels_by_judge, 2):
        labels_a, labels_b = labels_by_judge[name_a], labels_by_judge[name_b]
        both = [i for i in labels_a if i in labels_b]
        common = [i for i in both if not (labels_a[i].get(ABSTAINED) or labels_b[i].get(ABSTAINED))]
        distances = [_tier_distance(labels_a[i]["complexity"], labels_b[i]["complexity"]) for i in common]
        result[(name_a, name_b)] = {
            "n": len(common),
            "left_out": len(both) - len(common),
            "tier_exact": _rate(sum(d == 0 for d in distances), len(common)),
            "tier_within_1": _rate(sum(d is not None and d <= 1 for d in distances), len(common)),
            "task": _rate(sum(labels_a[i]["task"] == labels_b[i]["task"] for i in common), len(common)),
            "needs_tools": _rate(
                sum(labels_a[i]["needs_tools"] == labels_b[i]["needs_tools"] for i in common), len(common)
            ),
        }
    return result


def _percent(rate: float | None) -> str:
    return "-" if rate is None else f"{100 * rate:.1f}%"


def markdown_table(agreement: dict[tuple[str, str], dict]) -> str:
    lines = ["| " + " | ".join(TABLE_HEADER) + " |", "|" + "|".join("---" for _ in TABLE_HEADER) + "|"]
    for (name_a, name_b), row in agreement.items():
        cells = [name_a, name_b, str(row["n"]), str(row["left_out"])]
        cells += [_percent(row[k]) for k in ("tier_exact", "tier_within_1", "task", "needs_tools")]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def run_agree(args) -> int:
    folders = [Path(p) for p in args.judge_dirs]
    if len(folders) < 2:
        print("bench agree: give at least two judge directories", file=sys.stderr)
        return 1
    labels_by_judge: dict[str, dict[str, dict]] = {}
    for folder in folders:
        path = folder / "labels.jsonl"
        if not path.is_file():
            print(f"bench agree: no labels.jsonl in {folder}", file=sys.stderr)
            return 1
        if folder.name in labels_by_judge:
            print(f"bench agree: two directories are named {folder.name!r}", file=sys.stderr)
            return 1
        try:
            labels_by_judge[folder.name] = read_labels(path)
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            print(f"bench agree: cannot read {path}: {exc}", file=sys.stderr)
            return 1
    print(markdown_table(pairwise_agreement(labels_by_judge)))
    return 0
