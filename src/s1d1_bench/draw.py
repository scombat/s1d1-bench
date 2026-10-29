"""`bench draw`: the stratified random draw of the base prompts from a pre-sort judge file (docs/32 § 3.1, amended
2026-10-29). Nobody picks a prompt: the pre-sort judge sorts, a seeded shuffle orders each stratum, and the draw takes
from the front of each order. A veto (two rules, documented per id) only removes an id from its order, so the next
prompt of that stratum steps in and every other pick stays the same.

Order of the draw, all deterministic given the seed:
1. each stratum's ids are sorted, shuffled once with `random.Random(seed)` (strata in `STRATA` order), then the vetoed
   and `null`-complexity ids are dropped;
2. tools: round-robin over the strata, the next `needs_tools: true` id of each stratum is taken until `tools_min` are
   picked or none is left;
3. tasks: for each task of the rubric (in rubric order) with fewer than `task_min` picks, the next id of that task is
   taken from the stratum with the most room (ties: `STRATA` order) until the task has `task_min` or none is left;
4. fill: each stratum is filled to `per_stratum` from the front of its order.
A stratum that cannot be filled, a tools or task minimum that the candidates cannot meet, is a shortfall: it is
reported, not silently fixed.
"""

import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pyarrow.parquet as pq

from s1d1_bench.judge_file import read_labels
from s1d1_bench.rubric import TASKS

STRATA = ("trivial", "low", "medium", "high")


@dataclass
class Draw:
    seed: int
    per_stratum: int
    tools_min: int
    task_min: int
    vetoed: dict[str, str]
    picked: dict[str, list[str]] = field(default_factory=dict)  # stratum -> ids, in draw order
    shortfalls: list[str] = field(default_factory=list)

    def ids(self) -> list[str]:
        return [i for stratum in STRATA for i in self.picked.get(stratum, [])]


def read_veto(path: Path | str) -> dict[str, str]:
    """`<id>\\t<reason>` per line (blank lines and `#` comments skipped); every id needs a reason."""
    vetoed: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        prompt_id, sep, reason = line.partition("\t")
        if not sep or not reason.strip():
            raise ValueError(f"{path}:{n}: expected '<id><TAB><reason>'")
        vetoed[prompt_id.strip()] = reason.strip()
    return vetoed


def stratified_draw(
    labels: dict[str, dict],
    seed: int = 0,
    per_stratum: int = 20,
    tools_min: int = 20,
    task_min: int = 4,
    vetoed: dict[str, str] | None = None,
) -> Draw:
    """Draw `per_stratum` ids per stratum of `STRATA` from `labels` (id -> complexity, task, needs_tools)."""
    vetoed = dict(vetoed or {})
    draw = Draw(seed, per_stratum, tools_min, task_min, vetoed)
    rng = random.Random(seed)
    orders: dict[str, list[str]] = {}
    for stratum in STRATA:
        order = sorted(i for i, lab in labels.items() if lab["complexity"] == stratum)
        rng.shuffle(order)
        orders[stratum] = [i for i in order if i not in vetoed]
    picked: dict[str, list[str]] = {stratum: [] for stratum in STRATA}
    taken: set[str] = set()

    def take(stratum: str, want) -> str | None:
        """The first id of the stratum's order not yet taken that satisfies `want`, if the stratum has room."""
        if len(picked[stratum]) >= per_stratum:
            return None
        for i in orders[stratum]:
            if i not in taken and want(labels[i]):
                picked[stratum].append(i)
                taken.add(i)
                return i
        return None

    # 2. tools, round-robin
    tools = 0
    progress = True
    while tools < tools_min and progress:
        progress = False
        for stratum in STRATA:
            if tools >= tools_min:
                break
            if take(stratum, lambda lab: lab["needs_tools"] is True) is not None:
                tools += 1
                progress = True
    if tools < tools_min:
        draw.shortfalls.append(f"needs_tools: {tools} picked, {tools_min} wanted")

    # 3. tasks, from the stratum with the most room
    for task in TASKS:
        count = sum(labels[i]["task"] == task for i in taken)
        while count < task_min:
            by_room = sorted(STRATA, key=lambda s: (-(per_stratum - len(picked[s])), STRATA.index(s)))
            got = next((i for s in by_room if (i := take(s, lambda lab, t=task: lab["task"] == t)) is not None), None)
            if got is None:
                break
            count += 1
        if count < task_min:
            draw.shortfalls.append(f"task {task}: {count} picked, {task_min} wanted")

    # 4. fill
    for stratum in STRATA:
        while len(picked[stratum]) < per_stratum and take(stratum, lambda _lab: True) is not None:
            pass
        if len(picked[stratum]) < per_stratum:
            draw.shortfalls.append(f"stratum {stratum}: {len(picked[stratum])} picked, {per_stratum} wanted")
    draw.picked = picked
    return draw


def summary(draw: Draw, labels: dict[str, dict]) -> str:
    ids = draw.ids()
    lines = [f"{len(ids)} ids drawn (seed {draw.seed}, {draw.per_stratum} per stratum, {len(draw.vetoed)} vetoed)"]
    for stratum in STRATA:
        picked = draw.picked.get(stratum, [])
        tools = sum(labels[i]["needs_tools"] is True for i in picked)
        lines.append(f"  {stratum}: {len(picked)} (needs_tools {tools})")
    counts = {task: sum(labels[i]["task"] == task for i in ids) for task in TASKS}
    lines.append("  tasks: " + ", ".join(f"{t} {n}" for t, n in counts.items() if n))
    for s in draw.shortfalls:
        lines.append(f"  shortfall: {s}")
    return "\n".join(lines)


def run_draw(args) -> int:
    labels_path = Path(args.presort) / "labels.jsonl"
    if not labels_path.is_file():
        print(f"bench draw: no labels.jsonl in {args.presort}", file=sys.stderr)
        return 1
    labels = read_labels(labels_path)
    base_key = dict(zip(*pq.read_table(args.meta, columns=["id", "base_key"]).to_pydict().values(), strict=True))
    unknown = sorted(i for i in labels if i not in base_key)
    if unknown:
        print(f"bench draw: {len(unknown)} labelled id(s) not in {args.meta}: {unknown[:3]}", file=sys.stderr)
        return 1
    try:
        vetoed = read_veto(args.veto) if args.veto else {}
    except ValueError as exc:
        print(f"bench draw: {exc}", file=sys.stderr)
        return 1
    foreign = sorted(i for i in vetoed if i not in labels)
    if foreign:
        print(f"bench draw: vetoed id(s) not in the pre-sort: {foreign[:3]}", file=sys.stderr)
        return 1
    draw = stratified_draw(labels, args.seed, args.per_stratum, args.tools_min, args.task_min, vetoed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(f"{base_key[i]}\n" for i in draw.ids()), encoding="utf-8")
    record = asdict(draw) | {"presort": str(args.presort), "meta": str(args.meta), "veto": str(args.veto or "")}
    stem = out.name[: -len(".select.txt")] if out.name.endswith(".select.txt") else out.stem
    text = json.dumps(record, indent=2, ensure_ascii=False) + "\n"
    out.with_name(stem + ".draw.json").write_text(text, encoding="utf-8")
    print(summary(draw, labels))
    return 0
