"""The metrics of the report, in numpy only: agreement, classification, ranking, calibration, bootstrap.

Every function states its conventions (which side is the truth, ties, empty or degenerate inputs) in its docstring. The
rule everywhere: a metric that is undefined for the data (an empty class, a constant vector, a zero denominator) is
`nan`, never a made-up 0 or 1, unless the docstring says otherwise (`f1`, `mcc`, `macro_f1` follow the usual
zero-division convention and say so). Tier indices are ints 0..4 in `rubric.LEVELS` order; the first argument of a pair
is the reference (the truth) and the second the prediction.
"""

import math
from collections.abc import Callable, Collection, Hashable, Sequence

import numpy as np

_LOW, _HIGH = 2.5, 97.5  # the 95 % interval of the bootstrap, as percentiles


def _paired(a: Sequence, b: Sequence) -> None:
    if len(a) != len(b):
        raise ValueError(f"{len(a)} values against {len(b)}")


def _average_ranks(x: np.ndarray) -> np.ndarray:
    """The ranks 1..n of `x`, a group of equal values getting the mean of the ranks it spans (1, 2, 2, 4 -> 1, 2.5,
    2.5, 4)."""
    if np.isnan(x).any():
        raise ValueError("nan in the values to rank")
    _, inverse, counts = np.unique(x, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inverse]


# --- agreement on tiers ------------------------------------------------------------------------------------------


def accuracy(a: Sequence, b: Sequence) -> float:
    """The share of positions where `a` and `b` are equal (any comparable values); nan for no item."""
    _paired(a, b)
    if not len(a):
        return math.nan
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def within_one(a: Sequence[int], b: Sequence[int]) -> float:
    """The share of positions where the tier indices differ by at most 1; nan for no item."""
    _paired(a, b)
    if not len(a):
        return math.nan
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b)) <= 1))


def qwk(a: Sequence[int], b: Sequence[int], n_levels: int = 5) -> float:
    """Quadratic weighted kappa of the tier indices `a` (reference) and `b` (prediction), both in 0..n_levels-1.

    kappa = 1 - sum(W * O) / sum(W * E): O the confusion counts (rows = `a`, columns = `b`), E = outer(row sums, column
    sums) / N the counts expected by chance, W[i][j] = (i - j)^2 / (n_levels - 1)^2. 1 for a perfect agreement, 0 for
    chance level (a constant prediction gives exactly 0 whenever the truth varies), negative below. nan when the
    expected disagreement sum(W * E) is 0 (no item, or both vectors constant on the same tier), where kappa is 0/0.
    Symmetric in `a` and `b`. A value outside 0..n_levels-1 is a ValueError.
    """
    _paired(a, b)
    if not len(a):
        return math.nan
    x, y = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    if min(x.min(), y.min()) < 0 or max(x.max(), y.max()) >= n_levels:
        raise ValueError(f"tier index outside 0..{n_levels - 1}")
    observed = np.zeros((n_levels, n_levels))
    np.add.at(observed, (x, y), 1.0)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(x)
    levels = np.arange(n_levels)
    weights = (levels[:, None] - levels[None, :]) ** 2 / (n_levels - 1) ** 2
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return math.nan
    return 1.0 - float((weights * observed).sum()) / denominator


def confusion(a: Sequence[Hashable], b: Sequence[Hashable], labels: Sequence[Hashable]) -> np.ndarray:
    """The len(labels) x len(labels) matrix whose row i is the distribution of the predictions `b` over the items whose
    truth `a` is labels[i]: rows = truth, columns = prediction, each row normalised to sum 1; a row without any item is
    all zeros. A value of `a` or `b` not in `labels` is a ValueError."""
    _paired(a, b)
    position = {label: i for i, label in enumerate(labels)}
    counts = np.zeros((len(labels), len(labels)))
    for truth, prediction in zip(a, b, strict=True):
        if truth not in position or prediction not in position:
            raise ValueError(f"{truth!r} or {prediction!r} is not in the labels {list(labels)}")
        counts[position[truth], position[prediction]] += 1
    totals = counts.sum(axis=1, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def macro_f1(a: Sequence[Hashable], b: Sequence[Hashable], labels: Sequence[Hashable] | None = None) -> float:
    """The unweighted mean over `labels` of the one-vs-rest F1 of each label, `a` the truth and `b` the prediction.

    F1 of a label = 2 TP / (2 TP + FP + FN), 0.0 when that denominator is 0 (a label given in `labels` that nobody
    has and nobody predicts still counts, as 0). `labels` defaults to the sorted union of the values of `a` and `b`
    (so they must be sortable: the caller drops the nulls); nan for no label.
    """
    _paired(a, b)
    if labels is None:
        labels = sorted(set(a) | set(b))
    if not len(labels):
        return math.nan
    scores = []
    for label in labels:
        tp = sum(x == label and y == label for x, y in zip(a, b, strict=True))
        fp = sum(x != label and y == label for x, y in zip(a, b, strict=True))
        fn = sum(x == label and y != label for x, y in zip(a, b, strict=True))
        scores.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(scores))


def krippendorff_alpha_ordinal(votes: Sequence[Sequence[int | None]]) -> float:
    """Krippendorff's alpha with the ordinal metric (Krippendorff 2011, "Computing Krippendorff's Alpha-Reliability").

    `votes`: one inner list per unit (a prompt), the values given by its coders, None for a missing one. Units with
    fewer than 2 values are ignored. With the coincidence matrix o[c][k] (each unit of m values adds 1/(m-1) for every
    ordered pair of its values), n_c its row sums and N = sum n_c the pairable values:
    alpha = 1 - (N - 1) * sum(o[c][k] d2[c][k]) / sum(n_c n_k d2[c][k]), with the ordinal squared difference between
    the distinct observed values c < k (in increasing order) d2 = (n_c + ... + n_k - (n_c + n_k) / 2)^2. 1 for a
    perfect agreement, 0 for chance level. nan when undefined: no pairable value, or only one distinct value observed.
    """
    units = [[v for v in unit if v is not None] for unit in votes]
    units = [unit for unit in units if len(unit) >= 2]
    values = sorted({v for unit in units for v in unit})
    if len(values) < 2:
        return math.nan
    position = {v: i for i, v in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)))
    for unit in units:
        indices = [position[v] for v in unit]
        for i, c in enumerate(indices):
            for j, k in enumerate(indices):
                if i != j:
                    coincidence[c, k] += 1.0 / (len(unit) - 1)
    n = coincidence.sum(axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(n)])
    index = np.arange(len(values))
    low, high = np.minimum.outer(index, index), np.maximum.outer(index, index)
    delta2 = (cumulative[high + 1] - cumulative[low] - (n[low] + n[high]) / 2.0) ** 2
    expected = float((np.outer(n, n) * delta2).sum())
    return 1.0 - (n.sum() - 1.0) * float((coincidence * delta2).sum()) / expected


# --- binary decisions ---------------------------------------------------------------------------------------------


def _counts(y: Sequence[bool], p: Sequence[bool]) -> tuple[int, int, int, int]:
    _paired(y, p)
    tp = sum(bool(t) and bool(q) for t, q in zip(y, p, strict=True))
    fp = sum(not t and bool(q) for t, q in zip(y, p, strict=True))
    fn = sum(bool(t) and not q for t, q in zip(y, p, strict=True))
    return tp, fp, fn, len(y) - tp - fp - fn


def f1(y: Sequence[bool], p: Sequence[bool]) -> float:
    """F1 of the positive class, `y` the truth and `p` the prediction: 2 TP / (2 TP + FP + FN); 0.0 when that
    denominator is 0 (no positive in the truth nor in the prediction)."""
    tp, fp, fn, _ = _counts(y, p)
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def mcc(y: Sequence[bool], p: Sequence[bool]) -> float:
    """Matthews correlation coefficient, `y` the truth and `p` the prediction:
    (TP TN - FP FN) / sqrt((TP + FP)(TP + FN)(TN + FP)(TN + FN)); 0.0 when any of the four factors is 0 (a constant
    truth or a constant prediction), the usual convention."""
    tp, fp, fn, tn = _counts(y, p)
    root = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(root) if root else 0.0


def auroc(y: Sequence[bool], p: Sequence[float]) -> float:
    """Area under the ROC curve, `y` the truth and `p` the score (higher = more positive): the share of (positive,
    negative) pairs where the positive has the higher score, a tie of scores counting for half (Mann-Whitney U over
    n_pos * n_neg, by average ranks). All scores equal gives exactly 0.5. nan when one class is empty."""
    _paired(y, p)
    positive = np.asarray(y, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = len(positive) - n_pos
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = _average_ranks(np.asarray(p, dtype=float))
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pr_curve(y: Sequence[bool], p: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(precision, recall, thresholds) of the positive class: one point per distinct score, the thresholds descending;
    at threshold t every item with score >= t is predicted positive (so a tie is never split). precision = TP / (TP +
    FP), recall = TP / (number of positives); there is no extra (1, 0) end point. The recall is nan everywhere when the
    truth has no positive."""
    _paired(y, p)
    positive = np.asarray(y, dtype=bool)
    scores = np.asarray(p, dtype=float)
    thresholds = np.unique(scores)[::-1]
    predicted = scores[None, :] >= thresholds[:, None]
    tp = (predicted & positive[None, :]).sum(axis=1)
    precision = tp / predicted.sum(axis=1)
    recall = tp / positive.sum() if positive.any() else np.full(len(thresholds), math.nan)
    return precision, recall, thresholds


# --- scores -------------------------------------------------------------------------------------------------------


def mae(x: Sequence[float], y: Sequence[float]) -> float:
    """Mean absolute error between `x` and `y`; nan for no item."""
    _paired(x, y)
    if not len(x):
        return math.nan
    return float(np.mean(np.abs(np.asarray(x, dtype=float) - np.asarray(y, dtype=float))))


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman's rho: the Pearson correlation of the average ranks (a group of ties gets the mean of the ranks it
    spans). nan when `x` or `y` is constant (or has fewer than 2 items), where the correlation is 0/0."""
    _paired(x, y)
    if len(x) < 2:
        return math.nan
    rx, ry = _average_ranks(np.asarray(x, dtype=float)), _average_ranks(np.asarray(y, dtype=float))
    rx, ry = rx - rx.mean(), ry - ry.mean()
    denominator = math.sqrt(float((rx**2).sum()) * float((ry**2).sum()))
    return float((rx * ry).sum()) / denominator if denominator else math.nan


def reliability(score: Sequence[float], target: Sequence[float], bins: int = 10) -> list[tuple]:
    """The reliability table of `score` against `target`: (bin_low, bin_high, n, mean_score, mean_target) for each
    non-empty bin of [0, 1] cut in `bins` equal bins, in increasing order. A score s goes to bin floor(s * bins), in
    floating point (bin i is [i / bins, (i + 1) / bins)); a score of exactly 1 goes to the last bin, which is closed on
    the right. A score outside [0, 1] (or nan) is a ValueError: the calibration is for probabilities."""
    _paired(score, target)
    s, t = np.asarray(score, dtype=float), np.asarray(target, dtype=float)
    if np.isnan(s).any() or (s < 0).any() or (s > 1).any():
        raise ValueError("a score outside [0, 1]")
    index = np.minimum(np.floor(s * bins).astype(int), bins - 1)
    return [
        (b / bins, (b + 1) / bins, int(mask.sum()), float(s[mask].mean()), float(t[mask].mean()))
        for b in range(bins)
        if (mask := index == b).any()
    ]


def ece(score: Sequence[float], target: Sequence[float], bins: int = 10) -> float:
    """Expected calibration error: sum over the bins of `reliability` of n_b / N * |mean_score_b - mean_target_b|;
    0 for a perfectly calibrated score; nan for no item."""
    table = reliability(score, target, bins)
    total = sum(row[2] for row in table)
    if not total:
        return math.nan
    return sum(n / total * abs(mean_score - mean_target) for _, _, n, mean_score, mean_target in table)


# --- bootstrap ----------------------------------------------------------------------------------------------------


def _draws(keys: Sequence[str], groups: dict[str, list[int]], n: int, seed: int) -> list[list[int]]:
    """n resamples of the keys with replacement, each as the concatenated row indices of the drawn keys (a key drawn
    twice contributes its indices twice). `np.random.default_rng(seed)`, `rng.integers(0, len(keys), len(keys))` for
    each draw in turn."""
    if not keys:
        raise ValueError("no key to resample")
    rng = np.random.default_rng(seed)
    rows = [np.asarray(groups[key], dtype=int) for key in keys]
    return [np.concatenate([rows[i] for i in rng.integers(0, len(keys), len(keys))]).tolist() for _ in range(n)]


def _interval(values: list[float]) -> tuple[float, float]:
    kept = [v for v in values if not math.isnan(v)]
    if not kept:
        return math.nan, math.nan
    low, high = np.percentile(kept, [_LOW, _HIGH])
    return float(low), float(high)


def bootstrap(
    keys: Sequence[str],
    groups: dict[str, list[int]],
    stat: Callable[[list[int]], float],
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float]:
    """The 95 % bootstrap interval of `stat`, resampling the base keys, not the rows: the four versions of a prompt
    always travel together. `groups` maps each key of `keys` to its row indices; each of the `n` draws takes
    len(keys) keys with replacement and calls `stat` on the concatenated indices of the drawn keys (a key drawn twice
    contributes its indices twice, 8 for 4 versions). Returns the 2.5 and 97.5 percentiles (linear interpolation) of
    the non-nan statistics; a nan draw (a degenerate resample) is dropped, all draws nan gives (nan, nan). The same
    `seed` gives the same bounds."""
    return _interval([stat(draw) for draw in _draws(keys, groups, n, seed)])


def paired_difference(
    keys: Sequence[str],
    groups: dict[str, list[int]],
    stat_a: Callable[[list[int]], float],
    stat_b: Callable[[list[int]], float],
    n: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(difference, low, high): difference = stat_a - stat_b on all the rows (the indices of `keys` in order), and
    (low, high) the 95 % percentile bounds of stat_a - stat_b over the SAME `n` resamples of the keys for both
    statistics (as `bootstrap`; a draw where either is nan is dropped, none left gives (nan, nan)). A positive
    difference means `stat_a` is the higher."""
    everything = [i for key in keys for i in groups[key]]
    differences = [stat_a(draw) - stat_b(draw) for draw in _draws(keys, groups, n, seed)]
    return stat_a(everything) - stat_b(everything), *_interval(differences)


# --- the report's helpers -----------------------------------------------------------------------------------------


def one_vs_rest_truth(
    votes_by_judge: dict[str, list[list[str | None]]],
    judge: str,
    median: Callable[[list[str | None]], str | None],
) -> list[str | None]:
    """The truth of the other judges, for the ceiling of `judge`: per base key, `median` of the votes of every judge
    but `judge`, concatenated. `votes_by_judge[j]` has one entry per base key (the same keys in the same order for every
    judge), each the list of j's votes (tier names or None) on the key's 4 versions: with 3 main judges, 8 votes per
    key. `median` is `run.even_floor_median`-compatible."""
    if judge not in votes_by_judge:
        raise ValueError(f"{judge!r} is not among the judges {list(votes_by_judge)}")
    if len({len(rows) for rows in votes_by_judge.values()}) > 1:
        raise ValueError("the judges do not have the same number of keys")
    others = [rows for name, rows in votes_by_judge.items() if name != judge]
    return [median([vote for rows in votes for vote in rows]) for votes in zip(*others, strict=True)]


def rank_only(columns: Collection[str], name: str) -> bool:
    """True when classifier `name` is judged on ranks only: the results table (`columns`, its column names) has
    `<name>.raw_score` but no `<name>.score` (its labels carry no score, its raw output is not a probability): then
    Spearman of the raw score against the truth's tier index and the AUROC of the raw score for "high or expert", and
    no MAE, ECE nor reliability."""
    return f"{name}.raw_score" in columns and f"{name}.score" not in columns


def not_predicted(column) -> bool:
    """True when a `task` or `needs_tools` column (a list or a pyarrow array) is entirely null: the classifier does not
    predict it, and no metric of that field is computed for it. An empty column is True (nothing is predicted). A
    False or 0 is a value, not a null."""
    if hasattr(column, "null_count"):
        return column.null_count == len(column)
    return all(value is None for value in column)
