"""`bench report`, the numbers: every table of the report, computed from the results table of `bench run`.

`tables` returns one pyarrow table per name of `TABLES`: the agreement of each candidate and each judge with
`truth_all` and `truth_all_nogemma` (QWK, exact, within one tier, ordinal alpha), the ceiling (each main judge against
the truth of the two others) and the floor (`length`, `constant-low`, the ten draws of `random`), the judge pairs, the
tables by language, source, origin and truth tier, `task` and `needs_tools`, the rank metrics of the scores, the
calibration of the s1d1 scores against the anchor of the truth tier, the paired differences with the 2 sigma rule, the
two predictions written before the figures (docs/34 § 7.4) and the inputs. `write_tables` writes them as CSV,
`markdown_table` renders one in French for the index. `run_report_cmd` is the command: these tables, then the figures
(`report_figures`, matplotlib) and `index.md` (`report_index`, from `report/TEMPLATE.md`).

Conventions:

- Units, in every count column's name: `n_ids` counts prompt versions (360 ids, one per language), `n_keys` counts base
  prompts (90 prompts de base, one per `base_key`).
- Intervals: 95 % bootstrap over the base keys, the four versions of a key travelling together: the draws of
  `metrics.bootstrap` (n = 1000, seed 0), made once and shared by every statistic, so the interval of a paired
  difference is that of `metrics.paired_difference` (`metrics.bootstrap_draws`, `metrics.percentile_interval`). A
  statistic undefined on a draw (nan) drops that draw. Columns `<m>`, `<m>_lo`, `<m>_hi` and `<m>_draws_used` (how many
  of the 1000 draws the interval is made of). A subgroup (a language, a source, a tier) is measured on the rows of each
  draw that fall in it: the draw is over all the keys, never within the subgroup, so a draw may hold no row of it (nan).
- QWK is nan on rows whose truth holds a single tier, on all the rows as on a draw: there `metrics.qwk` is 0 (or 0/0)
  by construction, whatever the prediction, which is no agreement measure; such draws are dropped and
  `qwk_draws_used` says how many are left.
- macro-F1 (`task`): the unweighted mean of the per-task F1 over the labels present in the truth or the prediction of
  the rows measured, the union taken anew on each draw (the default of `metrics.macro_f1`).
- Truth tier: `truth_all.complexity` (an index of `rubric.LEVELS`); a null truth (undecidable) is left out of every tier
  metric and counted in `inputs_summary`. A judge's abstention (`<judge>.abstained`) or null vote leaves that id out
  for that judge only. The truth of `task` and `needs_tools` is `truth_all.task` and `truth_all.needs_tools`.
- Roles: candidates are the classifiers of the run record, judges its judges; the main judges (`main`) are inside the
  truth (`judge-in-truth`), the others outside it (`judge-outside-truth`, the thinking-effect measure); for
  `truth_all_nogemma` the Gemma main judges are outside. Judges are the measuring instrument: no table compares a
  candidate with a judge, except the ceiling. The named roles (`HEADLINE`, `REFERENCE`, `FLOOR`, `CONSTANT`, `RANDOM`,
  `LITELLM`) are skipped when their candidate is absent, with one row in the table `skipped`. A rank-only candidate is
  told by `metrics.rank_only`, a candidate without `task` or `needs_tools` by `metrics.not_predicted`; MAE, ECE and
  the reliability bins are for the `s1d1-` candidates that have a `<name>.score` only (the other scores are not on the
  v7 scale).

What is read: the results table (`bench run`), the meta table (`source`, `origin` by id), the run record
(`<results stem>.run.json`: judges, classifiers, main, gemma, abstentions) and the classifiers root, which must hold a
folder for every classifier of the run record and `random/draws.jsonl` (the ten draws of the random witness).
"""

import csv
import json
import math
import sys
from collections.abc import Callable, Sequence
from functools import partial
from itertools import combinations
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from s1d1_bench import metrics
from s1d1_bench.classify import ANCHORS
from s1d1_bench.jsonl import read_jsonl
from s1d1_bench.rubric import LEVELS
from s1d1_bench.run import FIELDS, even_floor_median

HEADLINE = ("s1d1-full-int8", "s1d1-full-fp16")  # the headline numbers of the report
REFERENCE = "s1d1-full-int8"  # minuend of every paired difference
FLOOR = "length"  # the informed witness
CONSTANT = "constant-low"
RANDOM = "random"  # its ten draws: <classifiers root>/random/draws.jsonl
LITELLM = "litellm-heuristic"  # its exact agreement next to constant-low's, and the two predictions
CALIBRATED = "s1d1-"  # the prefix of the candidates whose `.score` is on the v7 scale
N_BOOT, SEED = 1000, 0
BINS = 10
HIGH = LEVELS.index("high")  # "high or expert": the positive class of the rank AUROC
LOW = LEVELS.index("low")
CANDIDATE, IN_TRUTH, OUTSIDE = "candidate", "judge-in-truth", "judge-outside-truth"
CEILING_NOTE = "plafond sur 8 votes, pas 12"
FEW_POSITIVES = "peu de positifs: l'IC le dit"
MACRO_F1_NOTE = (
    "macro-F1 : moyenne des F1 par tâche sur l'union des tâches de la vérité et de la prédiction, refaite à chaque "
    "tirage"
)
CONSTANT_SCORE = "score constant"  # rank_metrics: Spearman nan, AUROC 0,5 (no discrimination)
V7, RAW = "v7 score", "raw (not on the v7 scale; ranks only)"
PASSES, FAILS = "passe", "ne passe pas"
RULE_2SIGMA = (
    "un écart apparié passe seulement si son IC 95 % exclut zéro et se tient au-delà de 2 σ : borne basse > 2 σ, ou "
    "borne haute < -2 σ (lecture stricte, docs/18)"
)
PREDICTION_1 = (
    "les listes de mots-clés de litellm-heuristic sont anglaises : son QWK en fra, rus et zho est chacun au-dessous de "
    "son QWK en eng"
)
PREDICTION_2 = "QWK(length) < QWK(litellm-heuristic) < QWK(s1d1-full-int8), contre truth_all"
TABLES = (
    "agreement_all", "agreement_nogemma", "ceiling", "floor", "exact_vs_constant_low", "judge_pairs", "by_language",
    "lang_change", "truth_lang_vs_all", "by_source", "by_origin", "by_tier", "task", "needs_tools", "needs_tools_pr",
    "rank_metrics", "calibration", "calibration_bins", "paired", "predictions", "confusion", "judge_spread_by_tier",
    "inputs", "inputs_summary", "skipped",
)  # fmt: skip


class Estimate(NamedTuple):
    """A statistic on all the rows, its 95 % interval, and its value on each bootstrap draw (nan where undefined)."""

    value: float
    lo: float
    hi: float
    draws: np.ndarray


_NAN = Estimate(math.nan, math.nan, math.nan, np.full(N_BOOT, math.nan))


def _interval(draws: np.ndarray) -> tuple[float, float]:
    return metrics.percentile_interval(draws.tolist())  # the interval rule of metrics.bootstrap, nan draws dropped


_SUFFIXES = ("", "_lo", "_hi", "_draws_used")


def _columns(name: str, estimate: Estimate) -> dict:
    """`<name>`, `<name>_lo`, `<name>_hi` and `<name>_draws_used` (the draws where the statistic is defined)."""
    used = int(np.count_nonzero(~np.isnan(estimate.draws)))
    return {name: estimate.value, f"{name}_lo": estimate.lo, f"{name}_hi": estimate.hi, f"{name}_draws_used": used}


class _Resampler:
    """The bootstrap draws of `metrics.bootstrap` over the base keys of the rows, made once.

    `draws[d]` holds the row indices of draw d (a key drawn twice gives its rows twice), `counts[d, i]` how many times
    row i is in it: a mean over the rows of a draw is then one matrix product for all the draws at once."""

    def __init__(self, base_keys: Sequence[str]):
        groups: dict[str, list[int]] = {}
        for i, key in enumerate(base_keys):
            groups.setdefault(key, []).append(i)
        self.keys, self.groups = list(groups), groups
        self.rows = np.array([i for key in self.keys for i in groups[key]], dtype=int)
        self.draws = [np.asarray(d, dtype=int) for d in metrics.bootstrap_draws(self.keys, groups, N_BOOT, SEED)]
        self.counts = np.stack([np.bincount(d, minlength=len(base_keys)) for d in self.draws]).astype(float)

    def mean(self, values: np.ndarray, mask: np.ndarray) -> Estimate:
        """The mean of `values` over the rows where `mask` is true (nan for none), on all rows and on each draw."""
        weights = mask.astype(float)
        totals = np.where(mask, values, 0.0).astype(float)
        n = weights.sum()
        value = float(totals.sum() / n) if n else math.nan
        den = self.counts @ weights
        with np.errstate(invalid="ignore", divide="ignore"):
            draws = np.where(den > 0, (self.counts @ totals) / den, math.nan)
        return Estimate(value, *_interval(draws), draws)

    def stat(self, fn: Callable[[np.ndarray], float], mask: np.ndarray) -> Estimate:
        """`fn` (row indices -> float) on the rows where `mask` is true, on all rows and on each draw; nan for none."""

        def safe(indices: np.ndarray) -> float:
            return float(fn(indices)) if len(indices) else math.nan

        value = safe(self.rows[mask[self.rows]])
        draws = np.array([safe(d[mask[d]]) for d in self.draws], dtype=float)
        return Estimate(value, *_interval(draws), draws)


def _on(metric: Callable, *arrays: np.ndarray, as_lists: bool = False) -> Callable[[np.ndarray], float]:
    """The statistic `metric(*arrays restricted to the row indices)`; `as_lists` hands it Python lists (the metrics that
    loop over the items in Python run faster on lists than on numpy scalars)."""
    if as_lists:
        return lambda idx: metric(*(a[idx].tolist() for a in arrays))
    return lambda idx: metric(*(a[idx] for a in arrays))


def _codes(values: Sequence, vocabulary: Sequence) -> np.ndarray:
    """The index of each value in `vocabulary`, -1 for a null."""
    position = {v: i for i, v in enumerate(vocabulary)}
    return np.array([-1 if v is None else position[v] for v in values], dtype=int)


def _flags(values: Sequence) -> np.ndarray:
    """1 for true, 0 for false, -1 for a null."""
    return np.array([-1 if v is None else int(bool(v)) for v in values], dtype=int)


def _floats(values: Sequence) -> np.ndarray:
    return np.array([math.nan if v is None else float(v) for v in values], dtype=float)


def two_sigma_rule(lo: float, hi: float, sigma: float) -> str:
    """`PASSES` only when the interval [lo, hi] of a paired difference excludes zero AND lies beyond 2 sigma: lo > 2
    sigma, or hi < -2 sigma (strict reading); `FAILS` otherwise, a nan bound included."""
    if math.isnan(lo) or math.isnan(hi):
        return FAILS
    return PASSES if lo > 2 * sigma or hi < -2 * sigma else FAILS


class _Report:
    def __init__(self, results: pa.Table, meta: pa.Table, run: dict, classifiers_root: Path, sigma: float):
        self.sigma = sigma
        self.run = run
        self.candidates = [entry["name"] for entry in run["classifiers"]]
        self.random_draws_path = self._check_folders(classifiers_root)  # refused before any computation
        self.columns = set(results.column_names)
        data = results.to_pydict()
        self.ids = data["id"]
        self.n = len(self.ids)
        self.base_key = data["base_key"]
        self.lang = np.array(data["lang"], dtype=object)
        self.langs = sorted(set(data["lang"]))
        by_id = {row["id"]: row for row in meta.select(["id", "source", "origin"]).to_pylist()}
        missing = [i for i in self.ids if i not in by_id]
        if missing:
            raise ValueError(f"the meta table has no row for id {missing[0]!r} ({len(missing)} id(s) missing)")
        self.group = {g: np.array([by_id[i][g] for i in self.ids], dtype=object) for g in ("source", "origin")}
        self.judges = [entry["name"] for entry in run["judges"]]
        self.main = list(run["main"])
        self.gemma = list(run.get("gemma", []))
        needed = [f"{name}.{field}" for name in self.judges + self.candidates for field in FIELDS]
        absent = [c for c in needed if c not in self.columns]
        if absent:
            raise ValueError(f"the results table has no column {absent[0]!r}: it does not match the run record")
        self.truth = {
            t: _codes(data[f"{t}.complexity"], LEVELS) for t in ("truth_all", "truth_lang", "truth_all_nogemma")
        }
        tasks = sorted({v for c in self.columns if c.endswith(".task") for v in data[c] if v is not None})
        self.truth_task = _codes(data["truth_all.task"], tasks)
        self.truth_needs = _flags(data["truth_all.needs_tools"])
        self.lang_disagreement = np.array(data["lang_disagreement"], dtype=bool)
        self.tier, self.task, self.needs, self.task_predicted, self.needs_predicted = {}, {}, {}, {}, {}
        for name in self.judges + self.candidates:
            abstained = np.array(data.get(f"{name}.abstained") or [False] * self.n, dtype=bool)
            self.tier[name] = np.where(abstained, -1, _codes(data[f"{name}.complexity"], LEVELS))
            self.task[name] = np.where(abstained, -1, _codes(data[f"{name}.task"], tasks))
            self.needs[name] = np.where(abstained, -1, _flags(data[f"{name}.needs_tools"]))
            self.task_predicted[name] = name in self.judges or not metrics.not_predicted(results[f"{name}.task"])
            self.needs_predicted[name] = name in self.judges or not metrics.not_predicted(
                results[f"{name}.needs_tools"]
            )
        self.data = data
        self.resampler = _Resampler(self.base_key)
        self.random_draws = self._random_draws(self.random_draws_path)
        self.skipped: list[dict] = []
        self.qwk_all: dict[str, Estimate] = {}  # candidate -> its QWK against truth_all, for the paired differences

    # --- inputs ---------------------------------------------------------------------------------------------------

    def _check_folders(self, root: Path) -> Path:
        """The path of the random draws, once the root has a folder for every classifier of the run record."""
        for name in self.candidates:
            if not (root / name).is_dir():
                raise ValueError(f"{root / name}: no folder for the classifier {name!r} of the run record")
        path = root / RANDOM / "draws.jsonl"
        if not path.is_file():
            raise ValueError(f"{path}: missing (the draws of the random witness give the floor band)")
        return path

    def _random_draws(self, path: Path) -> dict[int, np.ndarray]:
        """Draw number -> the tier index of each row (results order), from `random/draws.jsonl`."""
        by_draw: dict[int, dict[str, str | None]] = {}
        for row in read_jsonl(path):
            by_draw.setdefault(int(row["draw"]), {})[row["id"]] = row["complexity"]
        draws = {}
        for draw, levels in sorted(by_draw.items()):
            missing = [i for i in self.ids if i not in levels]
            if missing:
                raise ValueError(f"{path}: draw {draw} has no row for id {missing[0]!r}")
            draws[draw] = _codes([levels[i] for i in self.ids], LEVELS)
        return draws

    def skip(self, role: str, name: str, consequence: str) -> None:
        self.skipped.append({"role": role, "name": name, "consequence": consequence})

    def role(self, name: str, truth: str = "truth_all") -> str:
        if name in self.candidates:
            return CANDIDATE
        inside = self.main if truth != "truth_all_nogemma" else [j for j in self.main if j not in self.gemma]
        return IN_TRUTH if name in inside else OUTSIDE

    def entities(self) -> list[str]:
        """Every candidate, then every judge, in run record order."""
        return self.candidates + self.judges

    # --- the statistics -------------------------------------------------------------------------------------------

    def qwk(self, truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> Estimate:
        """QWK of `pred` against `truth` on the rows of `mask`; nan on rows (all of them, or a draw's) whose truth holds
        a single tier, where `metrics.qwk` is 0 or 0/0 by construction."""

        def statistic(indices: np.ndarray) -> float:
            t = truth[indices]
            return math.nan if t.min() == t.max() else metrics.qwk(t, pred[indices])

        return self.resampler.stat(statistic, mask)

    def agreement(self, truth: np.ndarray, pred: np.ndarray, mask=None, alpha=False, keep=None) -> dict:
        """n_ids, QWK, exact, within one (and the ordinal alpha of the two coders) with their intervals, on the rows
        where both are non-null (and `mask`). The QWK estimate is kept in `qwk_all[keep]` when `keep` names a candidate
        (for the paired differences)."""
        valid = (truth >= 0) & (pred >= 0)
        if mask is not None:
            valid &= mask
        out: dict = {"n_ids": int(valid.sum())}
        qwk = self.qwk(truth, pred, valid)
        if keep is not None:
            self.qwk_all[keep] = qwk
        out |= _columns("qwk", qwk)
        out |= _columns("exact", self.resampler.mean((truth == pred).astype(float), valid))
        out |= _columns("within_one", self.resampler.mean((np.abs(truth - pred) <= 1).astype(float), valid))
        if alpha:
            pairs = np.stack([truth, pred], axis=1)  # one unit per row, the two coders truth and prediction
            alpha = self.resampler.stat(_on(metrics.krippendorff_alpha_ordinal, pairs, as_lists=True), valid)
            out |= _columns("alpha", alpha)
        return out

    # --- the tables -----------------------------------------------------------------------------------------------

    def agreement_table(self, truth_name: str) -> pa.Table:
        truth = self.truth[truth_name]
        result = []
        for name in self.entities():
            pred = self.tier[name]
            valid = (truth >= 0) & (pred >= 0)
            never = [level for k, level in enumerate(LEVELS) if not (pred[valid] == k).any()]
            row = {"name": name, "role": self.role(name, truth_name)}
            keep = name if truth_name == "truth_all" and name in self.candidates else None
            row |= self.agreement(truth, pred, alpha=True, keep=keep)
            row["tiers_never_predicted"] = ", ".join(never)
            result.append(row)
        return pa.Table.from_pylist(result)

    def ceiling(self) -> pa.Table:
        keys, groups = self.resampler.keys, self.resampler.groups
        votes = {
            j: [[LEVELS[c] if c >= 0 else None for c in self.tier[j][groups[key]]] for key in keys] for j in self.main
        }
        result, qwks, exacts = [], [], []
        for judge in self.main:
            rest = metrics.one_vs_rest_truth(votes, judge, even_floor_median)
            truth = np.full(self.n, -1, dtype=int)
            for key, level in zip(keys, rest, strict=True):
                truth[groups[key]] = -1 if level is None else LEVELS.index(level)
            pred = self.tier[judge]
            valid = (truth >= 0) & (pred >= 0)
            qwk = self.qwk(truth, pred, valid)
            exact = self.resampler.mean((truth == pred).astype(float), valid)
            qwks.append(qwk)
            exacts.append(exact)
            row = {"judge": judge, "n_ids": int(valid.sum()), **_columns("qwk", qwk), **_columns("exact", exact)}
            result.append(row | {"note": CEILING_NOTE})
        if result:

            def mean(estimates: list[Estimate]) -> Estimate:
                draws = np.mean([e.draws for e in estimates], axis=0)
                return Estimate(float(np.mean([e.value for e in estimates])), *_interval(draws), draws)

            row = {"judge": "mean", "n_ids": None, **_columns("qwk", mean(qwks)), **_columns("exact", mean(exacts))}
            result.append(row | {"note": CEILING_NOTE})
        return pa.Table.from_pylist(result)

    def floor(self, agreement_all: pa.Table) -> pa.Table:
        by_name = {row["name"]: row for row in agreement_all.to_pylist()}
        metric_columns = [f"{m}{s}" for m in ("qwk", "exact", "within_one") for s in _SUFFIXES]
        result = []
        for name, kind in ((FLOOR, "informed witness"), (CONSTANT, "constant witness")):
            if name not in by_name:
                self.skip("floor", name, "no floor row for it")
                continue
            row = by_name[name]
            result.append(
                {"name": name, "kind": kind, "draw": None, "n_ids": row["n_ids"]} | {c: row[c] for c in metric_columns}
            )
        truth = self.truth["truth_all"]
        draws = []
        for draw, pred in self.random_draws.items():
            valid = (truth >= 0) & (pred >= 0)
            t, p = truth[valid], pred[valid]
            values = {
                "qwk": metrics.qwk(t, p),
                "exact": metrics.accuracy(t.tolist(), p.tolist()),
                "within_one": metrics.within_one(t, p),
            }
            draws.append(values)
            row = {"name": RANDOM, "kind": "random draw", "draw": draw, "n_ids": int(valid.sum())}
            result.append(row | {c: None if c.endswith("_draws_used") else math.nan for c in metric_columns} | values)
        for kind, summary in (("random min", min), ("random max", max), ("random mean", lambda v: sum(v) / len(v))):
            if draws:
                row = {"name": RANDOM, "kind": kind, "draw": None, "n_ids": None}
                values = {m: float(summary([d[m] for d in draws])) for m in ("qwk", "exact", "within_one")}
                result.append(
                    row | {c: None if c.endswith("_draws_used") else math.nan for c in metric_columns} | values
                )
        return pa.Table.from_pylist(result)

    def exact_vs_constant_low(self) -> pa.Table:
        truth = self.truth["truth_all"]
        result = []
        for name in (LITELLM, CONSTANT):
            if name not in self.candidates:
                self.skip("exact next to constant-low", name, "no row in exact_vs_constant_low")
                continue
            pred = self.tier[name]
            valid = (truth >= 0) & (pred >= 0)
            exact = self.resampler.mean((truth == pred).astype(float), valid)
            share = float((pred[valid] == LOW).mean()) if valid.any() else math.nan
            row = {"name": name, "n_ids": int(valid.sum()), **_columns("exact", exact), "share_low_of_ids": share}
            result.append(row)
        return pa.Table.from_pylist(result)

    def judge_pairs(self) -> pa.Table:
        models = {entry["name"]: entry.get("model") for entry in self.run["judges"]}
        result = []
        for a, b in combinations(self.judges, 2):
            ta, tb = self.tier[a], self.tier[b]
            both = (ta >= 0) & (tb >= 0)
            row = {"judge_a": a, "judge_b": b, "role_a": self.role(a), "role_b": self.role(b)}
            row |= {"n_ids": int(both.sum()), "n_ids_left_out": int(self.n - both.sum())}
            row |= _columns("qwk", self.qwk(ta, tb, both))
            row |= _columns("exact", self.resampler.mean((ta == tb).astype(float), both))
            row["thinking_pair"] = models[a] is not None and models[a] == models[b]
            result.append(row)
        return pa.Table.from_pylist(result)

    def by_language(self) -> pa.Table:
        result = []
        for name in self.entities():
            for lang in self.langs:
                here = self.lang == lang
                row = {"name": name, "role": self.role(name), "lang": lang}
                for suffix, truth in (("all", "truth_all"), ("lang", "truth_lang")):
                    found = self.agreement(self.truth[truth], self.tier[name], here)
                    row[f"n_ids_{suffix}"] = found["n_ids"]
                    row |= {f"{m}_{suffix}{s}": found[f"{m}{s}"] for m in ("qwk", "exact") for s in _SUFFIXES}
                result.append(row)
        return pa.Table.from_pylist(result)

    def lang_change(self) -> pa.Table:
        """Per entity: the share of base keys whose four versions do not all get the same tier, and the mean range of
        the tier index over the versions; a key with a null version is dropped for that entity (n_keys_dropped)."""
        groups = self.resampler.groups
        first = np.array([groups[key][0] for key in self.resampler.keys], dtype=int)
        result = []
        for name in self.entities():
            changed = np.zeros(self.n)
            spread = np.zeros(self.n)
            kept = np.zeros(self.n, dtype=bool)
            for key, row in zip(self.resampler.keys, first, strict=True):
                tiers = self.tier[name][groups[key]]
                if (tiers >= 0).all():
                    kept[row] = True
                    changed[row] = float(len(set(tiers.tolist())) > 1)
                    spread[row] = float(tiers.max() - tiers.min())
            entry = {"name": name, "role": self.role(name), "n_keys": int(kept.sum())}
            entry["n_keys_dropped"] = len(first) - int(kept.sum())
            entry |= _columns("lang_change_rate", self.resampler.mean(changed, kept))
            entry |= _columns("lang_range_mean", self.resampler.mean(spread, kept))
            result.append(entry)
        return pa.Table.from_pylist(result)

    def truth_lang_vs_all(self) -> pa.Table:
        def row(group: str, value: str, mask: np.ndarray) -> dict:
            keys = {self.base_key[i] for i in np.flatnonzero(mask)}
            disagreeing = {self.base_key[i] for i in np.flatnonzero(mask & self.lang_disagreement)}
            return {
                "group": group,
                "value": value,
                "n_ids": int(mask.sum()),
                "n_ids_disagreement": int((mask & self.lang_disagreement).sum()),
                "n_keys": len(keys),
                "n_keys_disagreement": len(disagreeing),
            }

        result = [row("all", "all", np.ones(self.n, dtype=bool))]
        result += [row("lang", lang, self.lang == lang) for lang in self.langs]
        result += [row("source", s, self.group["source"] == s) for s in sorted(set(self.group["source"]))]
        return pa.Table.from_pylist(result)

    def by_group(self, group: str) -> pa.Table:
        truth = self.truth["truth_all"]
        result = []
        for name in self.entities():
            for value in sorted(set(self.group[group])):
                found = self.agreement(truth, self.tier[name], self.group[group] == value)
                row = {"name": name, "role": self.role(name), group: value, "n_ids": found["n_ids"]}
                result.append(row | {f"{m}{s}": found[f"{m}{s}"] for m in ("exact", "qwk") for s in _SUFFIXES})
        return pa.Table.from_pylist(result)

    def by_tier(self) -> pa.Table:
        truth = self.truth["truth_all"]
        result = []
        for name in self.entities():
            pred = self.tier[name]
            for k, level in enumerate(LEVELS):
                valid = (truth == k) & (pred >= 0)
                row = {"name": name, "role": self.role(name), "tier": level, "n_ids": int(valid.sum())}
                row |= _columns("exact", self.resampler.mean((pred == k).astype(float), valid))
                row |= _columns("mean_signed_error", self.resampler.mean((pred - truth).astype(float), valid))
                result.append(row)
        return pa.Table.from_pylist(result)

    def task_table(self) -> pa.Table:
        truth = self.truth_task
        result = []
        for name in self.entities():
            if not self.task_predicted[name]:
                continue
            pred = self.task[name]
            valid = (truth >= 0) & (pred >= 0)
            row = {"name": name, "role": self.role(name), "n_ids": int(valid.sum())}
            row |= _columns("accuracy", self.resampler.mean((truth == pred).astype(float), valid))
            row |= _columns("macro_f1", self.resampler.stat(_on(metrics.macro_f1, truth, pred, as_lists=True), valid))
            result.append(row | {"note": MACRO_F1_NOTE})
        return pa.Table.from_pylist(result)

    def needs_tools(self) -> tuple[pa.Table, pa.Table]:
        truth = self.truth_needs
        result, curve = [], []
        for name in self.entities():
            if not self.needs_predicted[name]:
                continue
            pred = self.needs[name]
            valid = (truth >= 0) & (pred >= 0)
            y, p = truth.astype(bool), pred.astype(bool)
            row = {"name": name, "role": self.role(name), "n_ids": int(valid.sum())}
            row["n_ids_positive"] = int((truth[valid] == 1).sum())
            row |= _columns("f1", self.resampler.stat(_on(metrics.f1, y, p, as_lists=True), valid))
            row |= _columns("mcc", self.resampler.stat(_on(metrics.mcc, y, p, as_lists=True), valid))
            row |= _columns("accuracy", self.resampler.mean((truth == pred).astype(float), valid))
            column = f"{name}.needs_tools_probability"
            if name in self.candidates and column in self.columns:
                probability = _floats(self.data[column])
                scored = (truth >= 0) & ~np.isnan(probability)
                row |= _columns("auroc", self.resampler.stat(_on(metrics.auroc, y, probability), scored))
                precision, recall, thresholds = metrics.pr_curve(y[scored], probability[scored])
                curve += [
                    {"name": name, "threshold": float(t), "precision": float(pr), "recall": float(r)}
                    for pr, r, t in zip(precision, recall, thresholds, strict=True)
                ]
            else:
                row |= _columns("auroc", _NAN)
            result.append(row | {"note": FEW_POSITIVES})
        empty_curve = pa.table(
            {
                k: pa.array([], type=t)
                for k, t in (
                    ("name", pa.string()),
                    ("threshold", pa.float64()),
                    ("precision", pa.float64()),
                    ("recall", pa.float64()),
                )
            }
        )
        return pa.Table.from_pylist(result), (pa.Table.from_pylist(curve) if curve else empty_curve)

    def calibrated(self, name: str) -> bool:
        return name.startswith(CALIBRATED) and f"{name}.score" in self.columns

    def rank_metrics(self) -> pa.Table:
        truth = self.truth["truth_all"]
        high = truth >= HIGH
        result = []
        for name in self.candidates:
            column = f"{name}.raw_score" if metrics.rank_only(self.columns, name) else f"{name}.score"
            if column not in self.columns:
                continue
            score = _floats(self.data[column])
            valid = (truth >= 0) & ~np.isnan(score)
            row = {"name": name, "score_column": column, "scale": V7 if self.calibrated(name) else RAW}
            row["n_ids"] = int(valid.sum())
            row |= _columns("spearman", self.resampler.stat(_on(metrics.spearman, score, truth), valid))
            row |= _columns("auroc_high", self.resampler.stat(_on(metrics.auroc, high, score), valid))
            row["note"] = CONSTANT_SCORE if len(np.unique(score[valid])) == 1 else None
            result.append(row)
        return pa.Table.from_pylist(result)

    def calibration(self) -> tuple[pa.Table, pa.Table]:
        truth = self.truth["truth_all"]
        anchors = np.array([ANCHORS[level] for level in LEVELS])
        target = np.where(truth >= 0, anchors[np.maximum(truth, 0)], math.nan)
        result, bins = [], []
        for name in self.candidates:
            if not self.calibrated(name):
                continue
            score = _floats(self.data[f"{name}.score"])
            valid = (truth >= 0) & ~np.isnan(score)
            row = {"name": name, "n_ids": int(valid.sum())}
            row |= _columns("mae", self.resampler.stat(_on(metrics.mae, score, target), valid))
            row |= _columns("spearman", self.resampler.stat(_on(metrics.spearman, score, target), valid))
            row |= _columns("ece", self.resampler.stat(_on(partial(metrics.ece, bins=BINS), score, target), valid))
            result.append(row)
            for low, high, n, mean_score, mean_anchor in metrics.reliability(score[valid], target[valid], BINS):
                bins.append(
                    {
                        "candidate": name,
                        "bin_low": low,
                        "bin_high": high,
                        "n_ids": n,
                        "mean_score": mean_score,
                        "mean_anchor": mean_anchor,
                    }
                )
        empty_bins = pa.table(
            {"candidate": pa.array([], type=pa.string())}
            | {
                k: pa.array([], type=pa.float64())
                for k in ("bin_low", "bin_high", "n_ids", "mean_score", "mean_anchor")
            }
        )
        return pa.Table.from_pylist(result), (pa.Table.from_pylist(bins) if bins else empty_bins)

    def paired(self) -> pa.Table:
        if REFERENCE not in self.qwk_all:
            self.skip("reference of the paired differences", REFERENCE, "no paired difference")
            return pa.table({"reference": pa.array([], type=pa.string())})
        reference = self.qwk_all[REFERENCE]
        result = []
        for name in self.candidates:
            if name == REFERENCE:
                continue
            other = self.qwk_all[name]
            draws = reference.draws - other.draws
            lo, hi = _interval(draws)
            result.append(
                {
                    "reference": REFERENCE,
                    "candidate": name,
                    "difference": reference.value - other.value,
                    "lo": lo,
                    "hi": hi,
                    "draws_used": int(np.count_nonzero(~np.isnan(draws))),
                    "sigma": self.sigma,
                    "threshold": 2 * self.sigma,
                    "rule_2sigma": two_sigma_rule(lo, hi, self.sigma),
                }
            )
        return pa.Table.from_pylist(result)

    def predictions(self, agreement_all: pa.Table, by_language: pa.Table) -> pa.Table:
        result = []
        if LITELLM in self.candidates:
            found = [r for r in by_language.to_pylist() if r["name"] == LITELLM]
            qwk = {r["lang"]: r["qwk_all"] for r in found}
            others = [lang for lang in self.langs if lang != "eng"]
            # point estimates; a nan QWK makes the comparison False, so the prediction is not verified
            verified = "eng" in qwk and bool(others) and all(qwk[lang] < qwk["eng"] for lang in others)
            for r in found:
                result.append(
                    {
                        "prediction": "prediction_1",
                        "statement": PREDICTION_1,
                        "name": LITELLM,
                        "lang": r["lang"],
                        "qwk": r["qwk_all"],
                        "qwk_lo": r["qwk_all_lo"],
                        "qwk_hi": r["qwk_all_hi"],
                        "qwk_draws_used": r["qwk_all_draws_used"],
                        "verified": verified,
                    }
                )
        else:
            self.skip("prediction 1", LITELLM, "prediction 1 not evaluated")
        chain = [FLOOR, LITELLM, REFERENCE]
        absent = [name for name in chain if name not in self.candidates]
        if absent:
            for name in absent:
                self.skip("prediction 2", name, "prediction 2 not evaluated")
        else:
            rows = {r["name"]: r for r in agreement_all.to_pylist()}
            values = [rows[name]["qwk"] for name in chain]
            verified = values[0] < values[1] < values[2]  # False when any is nan
            for name in chain:
                result.append(
                    {
                        "prediction": "prediction_2",
                        "statement": PREDICTION_2,
                        "name": name,
                        "lang": "all",
                        "qwk": rows[name]["qwk"],
                        "qwk_lo": rows[name]["qwk_lo"],
                        "qwk_hi": rows[name]["qwk_hi"],
                        "qwk_draws_used": rows[name]["qwk_draws_used"],
                        "verified": verified,
                    }
                )
        if not result:
            return pa.table({"prediction": pa.array([], type=pa.string()), "verified": pa.array([], type=pa.bool_())})
        return pa.Table.from_pylist(result)

    def confusion(self) -> pa.Table:
        """Long format, every candidate: one row per (truth tier, predicted tier) of the 5 x 5 grid, `n_ids` and
        `row_share` (the share of the ids of that truth tier, `metrics.confusion`: a truth tier without any id gives
        0 everywhere); undecidable and null predictions are left out."""
        truth = self.truth["truth_all"]
        result = []
        for name in self.candidates:
            pred = self.tier[name]
            valid = (truth >= 0) & (pred >= 0)
            counts = np.zeros((len(LEVELS), len(LEVELS)), dtype=int)
            np.add.at(counts, (truth[valid], pred[valid]), 1)
            shares = metrics.confusion(truth[valid].tolist(), pred[valid].tolist(), list(range(len(LEVELS))))
            for i, j in np.ndindex(counts.shape):
                result.append(
                    {
                        "candidate": name,
                        "truth_tier": LEVELS[i],
                        "pred_tier": LEVELS[j],
                        "n_ids": int(counts[i, j]),
                        "row_share": float(shares[i, j]),
                    }
                )
        if not result:
            return pa.table({"candidate": pa.array([], type=pa.string())})
        return pa.Table.from_pylist(result)

    def judge_spread_by_tier(self) -> pa.Table:
        """Per truth tier and per spread value 0..4 (`spread` of the results: max - min tier index over the main
        judges' non-null votes on the id): `n_ids`; and per tier `n_ids_tier` (ids with a spread), `mean_spread_tier`
        and `n_ids_null_spread_tier` (ids of the tier with a null spread, left out). Undecidable ids are left out."""
        truth = self.truth["truth_all"]
        spread = np.array([-1 if v is None else int(v) for v in self.data["spread"]], dtype=int)
        result = []
        for k, level in enumerate(LEVELS):
            here = (truth == k) & (spread >= 0)
            mean = float(spread[here].mean()) if here.any() else math.nan
            nulls = int(((truth == k) & (spread < 0)).sum())
            for value in range(len(LEVELS)):
                result.append(
                    {
                        "tier": level,
                        "spread": value,
                        "n_ids": int((here & (spread == value)).sum()),
                        "n_ids_tier": int(here.sum()),
                        "mean_spread_tier": mean,
                        "n_ids_null_spread_tier": nulls,
                    }
                )
        return pa.Table.from_pylist(result)

    def inputs(self) -> tuple[pa.Table, pa.Table]:
        abstentions = self.run.get("abstentions", {})
        # the ids whose complexity vote is null, abstention or not (a judge may answer null on one field)
        nulls = {j: [i for i, c in zip(self.ids, self.tier[j], strict=True) if c < 0] for j in self.judges}
        result = []
        for kind, entries in (("judge", self.run["judges"]), ("classifier", self.run["classifiers"])):
            for entry in entries:
                name = entry["name"]
                result.append(
                    {
                        "name": name,
                        "kind": kind,
                        "role": self.role(name),
                        "model": entry.get("model"),
                        "revision": entry.get("revision"),
                        "date": entry.get("date"),
                        "n_ids_abstained": len(abstentions.get(name, [])) if kind == "judge" else None,
                        "n_ids_null_vote": len(nulls[name]) if kind == "judge" else None,
                        "null_votes_ids": ", ".join(nulls[name]) if kind == "judge" else None,
                    }
                )
        undecidable = [i for i, t in zip(self.ids, self.truth["truth_all"], strict=True) if t < 0]
        summary = [
            ("prompts", str(self.run.get("prompts"))),
            ("prompts_sha256", str(self.run.get("prompts_sha256"))),
            ("meta", str(self.run.get("meta"))),
            ("date of the run", str(self.run.get("date"))),
            ("rows", f"{self.n} ids"),
            ("base keys", f"{len(self.resampler.keys)} prompts de base"),
            ("languages", ", ".join(self.langs)),
            ("main judges", ", ".join(self.main)),
            ("gemma", ", ".join(self.gemma)),
            ("undecidable", f"{len(undecidable)} ids" + (f" ({', '.join(undecidable)})" if undecidable else "")),
            ("abstentions", f"{sum(len(v) for v in abstentions.values())} ids"),
            ("bootstrap", f"{N_BOOT} tirages des {len(self.resampler.keys)} prompts de base, graine {SEED}, IC 95 %"),
            ("random draws", f"{len(self.random_draws)} tirages"),
            ("sigma", repr(self.sigma)),
        ]
        return pa.Table.from_pylist(result), pa.Table.from_pylist([{"item": k, "value": v} for k, v in summary])

    def skipped_table(self) -> pa.Table:
        if not self.skipped:
            return pa.table({k: pa.array([], type=pa.string()) for k in ("role", "name", "consequence")})
        return pa.Table.from_pylist(self.skipped)


def tables(
    results: pa.Table, meta: pa.Table, run: dict, classifiers_root: Path | str, sigma: float = 0.005
) -> dict[str, pa.Table]:
    """Every table of the report (`TABLES`), from the results table of `bench run`, the meta table, its run record and
    the classifiers root. Refused (ValueError naming the folder or file) when the root has no folder for a classifier
    of the run record or no `random/draws.jsonl`, or when the tables do not match the run record."""
    report = _Report(results, meta, run, Path(classifiers_root), sigma)
    for name in HEADLINE:
        if name not in report.candidates:
            report.skip("headline", name, "no headline number for it")
    out: dict[str, pa.Table] = {}
    out["agreement_all"] = report.agreement_table("truth_all")
    out["agreement_nogemma"] = report.agreement_table("truth_all_nogemma")
    out["ceiling"] = report.ceiling()
    out["floor"] = report.floor(out["agreement_all"])
    out["exact_vs_constant_low"] = report.exact_vs_constant_low()
    out["judge_pairs"] = report.judge_pairs()
    out["by_language"] = report.by_language()
    out["lang_change"] = report.lang_change()
    out["truth_lang_vs_all"] = report.truth_lang_vs_all()
    out["by_source"] = report.by_group("source")
    out["by_origin"] = report.by_group("origin")
    out["by_tier"] = report.by_tier()
    out["task"] = report.task_table()
    out["needs_tools"], out["needs_tools_pr"] = report.needs_tools()
    out["rank_metrics"] = report.rank_metrics()
    out["calibration"], out["calibration_bins"] = report.calibration()
    out["paired"] = report.paired()
    out["predictions"] = report.predictions(out["agreement_all"], out["by_language"])
    out["confusion"] = report.confusion()
    out["judge_spread_by_tier"] = report.judge_spread_by_tier()
    out["inputs"], out["inputs_summary"] = report.inputs()
    out["skipped"] = report.skipped_table()
    return {name: out[name] for name in TABLES}


# --- output -------------------------------------------------------------------------------------------------------


def _csv_cell(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return repr(value)  # the shortest string that reads back as the same float
    return str(value)


def write_csv(path: Path, columns: Sequence[str], rows: Sequence[dict]) -> Path:
    """Write `rows` (dicts) under the header `columns` as CSV: UTF-8, `.` as the decimal point, floats at full
    precision, nan and null as an empty cell, booleans true/false."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(columns)
        for row in rows:
            writer.writerow([_csv_cell(row.get(c)) for c in columns])
    return path


def write_tables(computed: dict[str, pa.Table], out_dir: Path | str) -> list[Path]:
    """Write each table as `out_dir/tables/<name>.csv` (`write_csv`); return the paths in the order of `computed`."""
    folder = Path(out_dir) / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    return [write_csv(folder / f"{name}.csv", t.column_names, t.to_pylist()) for name, t in computed.items()]


def french(value, digits: int = 3) -> str:
    """A value as the report writes it: floats at `digits` decimals with a comma, booleans oui/non, nan and null as
    a dash, `|` escaped for Markdown."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    if isinstance(value, bool):
        return "oui" if value else "non"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}".replace(".", ",")
    return str(value).replace("|", "\\|")


def markdown_table(
    table: pa.Table, columns: Sequence[str] | None = None, digits: int = 3, labels: dict[str, str] | None = None
) -> str:
    """`table` as a Markdown table with French numbers (`0,812 [0,771 ; 0,848]`): a column `<m>` that has `<m>_lo` and
    `<m>_hi` beside it shows its interval in the same cell, and those two columns and `<m>_draws_used` are not shown
    on their own unless `columns` names them. Floats at `digits` decimals with a comma, booleans oui/non, nan and
    null as a dash; `labels` renames headers."""
    names = table.column_names
    paired = {c for c in names if f"{c}_lo" in names and f"{c}_hi" in names}
    if columns is None:
        hidden = {f"{c}{suffix}" for c in paired for suffix in _SUFFIXES[1:]}
        columns = [c for c in names if c not in hidden]
    labels = labels or {}

    def cell(row: dict, column: str) -> str:
        value = row[column]
        text = french(value, digits)
        if column in paired and isinstance(value, float) and not math.isnan(value):
            lo, hi = row[f"{column}_lo"], row[f"{column}_hi"]
            if lo is not None and hi is not None and not (math.isnan(lo) or math.isnan(hi)):
                text += f" [{french(lo, digits)} ; {french(hi, digits)}]"
        return text

    lines = ["| " + " | ".join(labels.get(c, c) for c in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(cell(row, c) for c in columns) + " |" for row in table.to_pylist()]
    return "\n".join(lines)


# --- the command --------------------------------------------------------------------------------------------------

MATPLOTLIB_HINT = (
    "matplotlib is missing: it comes with the report dependency group (numpy with it), "
    "uv run --group report bench report ..."
)


def run_report_cmd(args) -> int:
    """`bench report`: `--out/tables/*.csv`, `--out/figures/*.svg` (each with its CSV) and `--out/index.md`, from
    `--results`, its `<stem>.run.json`, `--meta` and the `--classifiers` root. Exit 1 with a message on a refusal (a
    missing input, a classifier folder or `random/draws.jsonl` missing from the root) or when matplotlib is missing;
    the check of matplotlib comes first, before any computation."""
    try:
        import matplotlib  # noqa: F401  (the figures need it; checked before the tables take their time)
    except ImportError:
        print(f"bench report: {MATPLOTLIB_HINT}", file=sys.stderr)
        return 1
    from s1d1_bench.report_figures import figures
    from s1d1_bench.report_index import write_index

    results_path = Path(args.results)
    run_path = results_path.with_name(results_path.stem + ".run.json")
    try:
        if not run_path.is_file():
            raise ValueError(f"{run_path}: missing (the run record of {results_path}, written by bench run)")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        computed = tables(pq.read_table(results_path), pq.read_table(args.meta), run, args.classifiers, args.sigma)
        written = write_tables(computed, args.out)
        drawn = figures(computed, args.out, args.sigma)
        index = write_index(args.out, computed, drawn, run, args.template, classifiers_root=args.classifiers)
    except (ValueError, OSError) as exc:
        print(f"bench report: {exc}", file=sys.stderr)
        return 1
    n_figures = sum(f.svg is not None for f in drawn)
    print(f"bench report: {len(written)} tables, {n_figures} figures (SVG + CSV), {index}")
    return 0
