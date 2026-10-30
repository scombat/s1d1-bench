"""The metrics of the report. Every expected value is worked out by hand in the comment next to it."""

import math

import numpy as np
import pyarrow as pa
import pytest

from s1d1_bench.metrics import (
    _average_ranks,
    accuracy,
    auroc,
    bootstrap,
    confusion,
    ece,
    f1,
    krippendorff_alpha_ordinal,
    macro_f1,
    mae,
    mcc,
    not_predicted,
    one_vs_rest_truth,
    paired_difference,
    pr_curve,
    qwk,
    rank_only,
    reliability,
    spearman,
    within_one,
)
from s1d1_bench.run import even_floor_median

T, F = True, False


# --- agreement on tiers -----------------------------------------------------------------------------------------


def test_qwk_perfect_agreement_is_one():
    assert qwk([0, 1, 2, 3, 4, 2], [0, 1, 2, 3, 4, 2]) == pytest.approx(1.0)


def test_qwk_three_levels_by_hand():
    # n_levels = 3, truth a = [0,0,1,1,2,2], prediction b = [0,1,1,2,2,2]; N = 6.
    # O (rows truth, cols prediction): (0,0)=1 (0,1)=1 (1,1)=1 (1,2)=1 (2,2)=2. W[i][j] = (i-j)^2 / 4: 1/4 off by one.
    # sum(W*O) = 1/4 (the (0,1) cell) + 1/4 (the (1,2) cell) = 0.5.
    # row sums [2,2,2], column sums [1,2,3]; E[i][j] = row_i * col_j / 6.
    # sum(W*E) = (2/6) * sum_i sum_j col_j W[i][j]: i=0: 2*(1/4) + 3*1 = 3.5; i=1: 1*(1/4) + 3*(1/4) = 1.0;
    # i=2: 1*1 + 2*(1/4) = 1.5; total (2/6) * 6.0 = 2.0.   kappa = 1 - 0.5 / 2.0 = 0.75
    assert qwk([0, 0, 1, 1, 2, 2], [0, 1, 1, 2, 2, 2], n_levels=3) == pytest.approx(0.75)


def test_qwk_five_levels_by_hand():
    # a = [0,1,2,3,4], b = [0,1,2,3,3]; N = 5; the only disagreement is truth 4 predicted 3: W = 1/16, so
    # sum(W*O) = 1/16.
    # row sums all 1; column sums [1,1,1,2,0]. sum(W*E) = (1/5) * (1/16) * sum_i sum_j col_j (i-j)^2, with
    # f(i) = i^2 + (i-1)^2 + (i-2)^2 + 2 (i-3)^2: f(0) = 0+1+4+18 = 23, f(1) = 1+0+1+8 = 10, f(2) = 4+1+0+2 = 7,
    # f(3) = 9+4+1+0 = 14, f(4) = 16+9+4+2 = 31; sum = 85. sum(W*E) = 85 / 80 = 1.0625.
    # kappa = 1 - 0.0625 / 1.0625 = 1 - 1/17 = 16/17
    assert qwk([0, 1, 2, 3, 4], [0, 1, 2, 3, 3]) == pytest.approx(16 / 17)


def test_qwk_a_constant_prediction_is_zero_when_the_truth_varies():
    # a = [0..4], b = [2]*5: sum(W*O) = (4+1+0+1+4)/16 = 10/16; column sums [0,0,5,0,0] so
    # sum(W*E) = (1/5) * 5 * (4+1+0+1+4)/16 = 10/16 too: kappa = 1 - 1 = 0
    assert qwk([0, 1, 2, 3, 4], [2] * 5) == pytest.approx(0.0, abs=1e-12)
    # truth constant on 2, prediction constant on 3: O and E are both the single cell (2,3): kappa = 1 - 1 = 0
    assert qwk([2, 2], [3, 3]) == pytest.approx(0.0, abs=1e-12)


def test_qwk_is_nan_when_the_expected_disagreement_is_zero():
    # both constant on the same tier: E has one cell, (2,2), where W = 0 -> 0/0
    assert math.isnan(qwk([2, 2, 2], [2, 2, 2]))
    assert math.isnan(qwk([], []))


def test_qwk_refuses_a_tier_outside_the_levels_and_unequal_lengths():
    with pytest.raises(ValueError):
        qwk([0, 5], [0, 1])
    with pytest.raises(ValueError):
        qwk([0, 1], [-1, 1])
    with pytest.raises(ValueError):
        qwk([0, 1], [0])


def test_within_one_and_accuracy():
    # a = [0,1,2,3,4], b = [1,3,2,0,4]: |diff| = 1,2,0,3,0 -> within one on positions 0,2,4 = 3/5; equal on 2,4 = 2/5
    a, b = [0, 1, 2, 3, 4], [1, 3, 2, 0, 4]
    assert within_one(a, b) == pytest.approx(0.6)
    assert accuracy(a, b) == pytest.approx(0.4)
    assert accuracy(["x", "y"], ["x", "z"]) == 0.5  # not only ints
    assert math.isnan(within_one([], [])) and math.isnan(accuracy([], []))


def test_macro_f1_by_hand():
    # a = [0,0,1,1,2,2], b = [0,1,1,1,2,0]
    # label 0: TP 1 (pos 0), FP 1 (pos 5: truth 2 predicted 0), FN 1 (pos 1) -> 2/(2+1+1) = 1/2
    # label 1: TP 2 (pos 2,3), FP 1 (pos 1), FN 0 -> 4/(4+1+0) = 4/5
    # label 2: TP 1 (pos 4), FP 0, FN 1 (pos 5) -> 2/(2+0+1) = 2/3
    # macro = (1/2 + 4/5 + 2/3) / 3 = (15 + 24 + 20) / 30 / 3 = 59/90
    a, b = [0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0]
    assert macro_f1(a, b) == pytest.approx(59 / 90)
    # a label given but absent from both counts as 0: (59/30 + 0) / 4 = 59/120
    assert macro_f1(a, b, labels=[0, 1, 2, 3]) == pytest.approx(59 / 120)
    # string labels, default labels = sorted union ["x", "y"]: x: TP 1 FP 0 FN 1 -> 2/3; y: TP 1 FP 1 FN 0 -> 2/3
    assert macro_f1(["x", "x", "y"], ["x", "y", "y"]) == pytest.approx(2 / 3)
    assert math.isnan(macro_f1([], []))


def test_confusion_rows_are_the_truth_and_are_normalised():
    # a = [0,0,0,1,2], b = [0,0,1,1,0], labels [0,1,2]
    # truth 0 (3 items): predicted 0,0,1 -> [2/3, 1/3, 0]; truth 1 (1 item): predicted 1 -> [0, 1, 0];
    # truth 2 (1 item): predicted 0 -> [1, 0, 0]
    matrix = confusion([0, 0, 0, 1, 2], [0, 0, 1, 1, 0], [0, 1, 2])
    assert matrix.shape == (3, 3)
    assert matrix == pytest.approx(np.array([[2 / 3, 1 / 3, 0], [0, 1, 0], [1, 0, 0]]))
    # not symmetric: swapping truth and prediction would put the 1/3 below the diagonal
    assert matrix[0, 1] == pytest.approx(1 / 3) and matrix[1, 0] == 0


def test_confusion_a_label_without_item_is_an_all_zero_row_and_labels_set_the_order():
    matrix = confusion(["b", "b", "a"], ["b", "a", "a"], ["b", "a", "c"])
    # truth b (2): predicted b, a -> [1/2, 1/2, 0]; truth a (1): predicted a -> [0, 1, 0]; truth c: nobody -> zeros
    assert matrix == pytest.approx(np.array([[0.5, 0.5, 0], [0, 1, 0], [0, 0, 0]]))
    with pytest.raises(ValueError):
        confusion(["a"], ["z"], ["a", "b"])


# --- Krippendorff's alpha ---------------------------------------------------------------------------------------

# Krippendorff (2011), the 4 coders x 12 units example (None = missing), whose published values are nominal 0.743,
# interval 0.849, ordinal 0.815. Cross-checked beforehand with an independent throwaway script (coincidence matrix and
# the three metrics, not the code of `metrics.py`), which gave 0.7434 / 0.8491 / 0.8154.
CODERS = {
    "A": [1, 2, 3, 3, 2, 1, 4, 1, 2, None, None, None],
    "B": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, None, 3],
    "C": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, None],
    "D": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, None],
}
UNITS = [list(unit) for unit in zip(*CODERS.values(), strict=True)]


def test_alpha_krippendorff_2011_ordinal():
    # Pairable values per unit (unit 12 has a single value and is ignored): coincidence matrix o (m = values of the
    # unit, each ordered pair of values adds 1/(m-1)), values 1..5:
    #   o11 = 7, o22 = 10, o33 = 8, o44 = 4, o55 = 3, o12 = o21 = 4/3, o23 = o32 = 4/3, o13 = o14 = o24 = o34 = 1/3
    #   (and symmetric); the row sums n = [9, 13, 10, 5, 3], N = 40.
    # Ordinal d2(c, k) = (n_c + ... + n_k - (n_c + n_k) / 2)^2:
    #   d12 = (22-11)^2 = 121, d13 = (32-9.5)^2 = 506.25, d14 = (37-7)^2 = 900, d15 = (40-6)^2 = 1156,
    #   d23 = (23-11.5)^2 = 132.25, d24 = (28-9)^2 = 361, d25 = (31-8)^2 = 529, d34 = (15-7.5)^2 = 56.25,
    #   d35 = (18-6.5)^2 = 132.25, d45 = (8-4)^2 = 16.
    # sum(o d2) over ordered pairs = 2 (4/3*121 + 1/3*506.25 + 1/3*900 + 4/3*132.25 + 1/3*361 + 1/3*56.25) = 1891
    # sum(n_c n_k d2) = 2 (9*13*121 + 9*10*506.25 + 9*5*900 + 9*3*1156 + 13*10*132.25 + 13*5*361 + 13*3*529
    #                      + 10*5*56.25 + 10*3*132.25 + 5*3*16) = 2 * 199740 = 399480
    # alpha = 1 - (N-1) * 1891 / 399480 = 1 - 39 * 1891 / 399480 = 1 - 73749 / 399480 = 0.81539 (published: 0.815)
    assert krippendorff_alpha_ordinal(UNITS) == pytest.approx(1 - 73749 / 399480)
    assert krippendorff_alpha_ordinal(UNITS) == pytest.approx(0.815, abs=5e-4)


def test_alpha_two_coders_by_hand_and_missing_values_are_ignored():
    # units [1,2], [2,2], [1,1]: o11 = 2, o12 = o21 = 1, o22 = 2; n = [3, 3], N = 6; d2(1,2) = (6 - 3)^2 = 9
    # sum(o d2) = (1+1) * 9 = 18; sum(n n d2) = 2 * 3 * 3 * 9 = 162; alpha = 1 - 5 * 18 / 162 = 1 - 5/9 = 4/9
    assert krippendorff_alpha_ordinal([[1, 2], [2, 2], [1, 1]]) == pytest.approx(4 / 9)
    # a unit with one value (the other missing) or none is ignored
    assert krippendorff_alpha_ordinal([[1, 2], [None, 1], [2, 2], [3, None, None], [], [1, 1]]) == pytest.approx(4 / 9)


def test_alpha_perfect_agreement_is_one_and_only_the_order_of_the_values_matters():
    assert krippendorff_alpha_ordinal([[1, 1], [2, 2], [3, 3]]) == pytest.approx(1.0)
    # ordinal: relabelling the values 1..4 as 10, 20, 30, 40 keeps the ranks, hence the alpha
    relabelled = [[None if v is None else 10 * v for v in unit] for unit in UNITS]
    assert krippendorff_alpha_ordinal(relabelled) == pytest.approx(krippendorff_alpha_ordinal(UNITS))


def test_alpha_is_nan_when_undefined():
    assert math.isnan(krippendorff_alpha_ordinal([]))
    assert math.isnan(krippendorff_alpha_ordinal([[1], [None, 2]]))  # no unit with 2 values
    assert math.isnan(
        krippendorff_alpha_ordinal([[2, 2], [2, 2]])
    )  # one single value observed: no expected disagreement


# --- binary decisions -------------------------------------------------------------------------------------------


def test_f1_and_mcc_by_hand():
    # TP 3, FP 1, FN 2, TN 4:
    y = [T] * 3 + [F] + [T] * 2 + [F] * 4
    p = [T] * 3 + [T] + [F] * 2 + [F] * 4
    # F1 = 2*3 / (2*3 + 1 + 2) = 6/9 = 2/3
    assert f1(y, p) == pytest.approx(2 / 3)
    # MCC = (3*4 - 1*2) / sqrt((3+1)(3+2)(4+1)(4+2)) = 10 / sqrt(600) = 10 / (10 sqrt 6) = 1/sqrt(6)
    assert mcc(y, p) == pytest.approx(1 / math.sqrt(6))


def test_f1_and_mcc_perfect_inverted_and_degenerate():
    y = [T, T, F, F]
    assert f1(y, y) == 1.0 and mcc(y, y) == pytest.approx(1.0)
    # inverted predictions: TP 0, FP 2, FN 2, TN 0: F1 = 0 / (0+2+2) = 0; MCC = (0 - 4) / sqrt(2*2*2*2) = -1
    assert f1(y, [F, F, T, T]) == 0.0
    assert mcc(y, [F, F, T, T]) == pytest.approx(-1.0)
    # no positive in the truth nor the prediction: F1 denominator 0 -> 0.0 by convention; constant prediction: MCC 0.0
    assert f1([F, F], [F, F]) == 0.0
    assert mcc([T, F, T], [F, F, F]) == 0.0
    assert mcc([T, T], [T, T]) == 0.0


def test_auroc_by_hand():
    # y = [T,T,F,F], p = [0.9, 0.4, 0.6, 0.1]; pairs (pos, neg): 0.9>0.6 1, 0.9>0.1 1, 0.4<0.6 0, 0.4>0.1 1 -> 3/4
    assert auroc([T, T, F, F], [0.9, 0.4, 0.6, 0.1]) == pytest.approx(0.75)
    # perfect separation 1, perfect inversion 0
    assert auroc([T, F], [0.9, 0.1]) == 1.0 and auroc([T, F], [0.1, 0.9]) == 0.0


def test_auroc_counts_a_tie_for_half():
    # every score equal: each of the n_pos * n_neg pairs is a tie -> exactly 0.5
    assert auroc([T, F, T, F, F], [3.0] * 5) == 0.5
    assert auroc([T, F], [0.0, 0.0]) == 0.5
    # y = [T,T,T,F,F], p = [0.8, 0.5, 0.5, 0.5, 0.2]: 6 pairs; the 0.8 beats both negatives: 2;
    # each 0.5 positive ties the 0.5 negative (1/2) and beats the 0.2 one (1): 1.5, twice -> 3; total 5/6
    assert auroc([T, T, T, F, F], [0.8, 0.5, 0.5, 0.5, 0.2]) == pytest.approx(5 / 6)


def test_auroc_is_nan_when_a_class_is_empty():
    assert math.isnan(auroc([T, T], [0.1, 0.9]))
    assert math.isnan(auroc([F, F], [0.1, 0.9]))
    assert math.isnan(auroc([], []))


def test_pr_curve_by_hand():
    # y = [T,F,T,F], p = [0.9, 0.8, 0.8, 0.1]; positives = 2; thresholds (distinct scores, descending) 0.9, 0.8, 0.1
    # t = 0.9: predicted {item 0}: TP 1, FP 0 -> precision 1, recall 1/2
    # t = 0.8: predicted {0, 1, 2} (the tie at 0.8 stays together): TP 2, FP 1 -> precision 2/3, recall 2/2 = 1
    # t = 0.1: everything: TP 2, FP 2 -> precision 1/2, recall 1
    precision, recall, thresholds = pr_curve([T, F, T, F], [0.9, 0.8, 0.8, 0.1])
    assert thresholds == pytest.approx([0.9, 0.8, 0.1])
    assert precision == pytest.approx([1.0, 2 / 3, 0.5])
    assert recall == pytest.approx([0.5, 1.0, 1.0])
    assert np.isnan(pr_curve([F, F], [0.2, 0.1])[1]).all()  # no positive: the recall is undefined


# --- scores -----------------------------------------------------------------------------------------------------


def test_mae():
    # |1-2| + |4-1| + |0-0| = 4, over 3 items
    assert mae([1, 4, 0], [2, 1, 0]) == pytest.approx(4 / 3)
    assert math.isnan(mae([], []))


def test_average_ranks_give_ties_the_mean_rank():
    # sorted 1, 2, 2, 4 spans ranks 1, 2-3, 4 -> 1, 2.5, 2.5, 4 whatever the input order
    assert _average_ranks(np.array([2.0, 4.0, 1.0, 2.0])) == pytest.approx([2.5, 4.0, 1.0, 2.5])
    # three equal values span ranks 1-3 -> 2 each
    assert _average_ranks(np.array([7.0, 7.0, 7.0])) == pytest.approx([2.0, 2.0, 2.0])


def test_spearman_with_ties_by_hand():
    # x = [1,2,2,3] -> ranks [1, 2.5, 2.5, 4]; y = [1,3,2,4] -> ranks [1, 3, 2, 4]; both means 2.5
    # dx = [-1.5, 0, 0, 1.5], dy = [-1.5, 0.5, -0.5, 1.5]: cov = 2.25 + 0 + 0 + 2.25 = 4.5
    # sum dx^2 = 4.5, sum dy^2 = 2.25 + 0.25 + 0.25 + 2.25 = 5: rho = 4.5 / sqrt(4.5 * 5) = sqrt(0.9)
    assert spearman([1, 2, 2, 3], [1, 3, 2, 4]) == pytest.approx(math.sqrt(0.9))
    # many zeros, as the raw LiteLLM scores: x = [0,0,0,1,2] -> ranks [2,2,2,4,5] (mean 3), y ranks [1..5] (mean 3)
    # dx = [-1,-1,-1,1,2], dy = [-2,-1,0,1,2]: cov = 2+1+0+1+4 = 8; sum dx^2 = 8, sum dy^2 = 10:
    # 8 / sqrt(80) = 2/sqrt(5)
    assert spearman([0, 0, 0, 1, 2], [1, 2, 3, 4, 5]) == pytest.approx(2 / math.sqrt(5))


def test_spearman_extremes_and_nan():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert math.isnan(spearman([0, 0, 0], [1, 2, 3]))  # a constant side
    assert math.isnan(spearman([1, 2, 3], [5, 5, 5]))
    assert math.isnan(spearman([1], [1]))


def test_reliability_and_ece_by_hand():
    # bins = 10; scores [0.1, 0.1, 0.9, 1.0] go to bins 1, 1, 9, 9 (1.0 is in the last bin, closed on the right)
    # bin 1 = [0.1, 0.2): n 2, mean score 0.1, mean target (0 + 0.5) / 2 = 0.25
    # bin 9 = [0.9, 1.0]: n 2, mean score (0.9 + 1.0) / 2 = 0.95, mean target (1.0 + 0.5) / 2 = 0.75; empty bins omitted
    score, target = [0.1, 0.1, 0.9, 1.0], [0.0, 0.5, 1.0, 0.5]
    table = reliability(score, target)
    assert [row[2] for row in table] == [2, 2]
    assert table[0] == pytest.approx((0.1, 0.2, 2, 0.1, 0.25))
    assert table[1] == pytest.approx((0.9, 1.0, 2, 0.95, 0.75))
    # ECE = 2/4 * |0.1 - 0.25| + 2/4 * |0.95 - 0.75| = 0.075 + 0.1 = 0.175
    assert ece(score, target) == pytest.approx(0.175)


def test_reliability_bins_argument_and_the_boundary_goes_up():
    # bins = 2: scores [0.1, 0.4, 0.5, 1.0] -> bins [0, .5) twice and [.5, 1] twice (0.5 itself is in the upper bin)
    # bin 0: mean score 0.25, targets 0, 0 -> 0; bin 1: mean score 0.75, targets 1, 1 -> 1; ECE = .5*.25 + .5*.25 = .25
    table = reliability([0.1, 0.4, 0.5, 1.0], [0, 0, 1, 1], bins=2)
    assert table[0] == pytest.approx((0.0, 0.5, 2, 0.25, 0.0))
    assert table[1] == pytest.approx((0.5, 1.0, 2, 0.75, 1.0))
    assert ece([0.1, 0.4, 0.5, 1.0], [0, 0, 1, 1], bins=2) == pytest.approx(0.25)


def test_ece_of_a_perfectly_calibrated_score_is_zero():
    # bin 2 holds four scores 0.2 whose targets average (0+0+0+0.8)/4 = 0.2; bin 8 four scores 0.8, targets
    # (1+1+1+0.2)/4 = 0.8: every bin has mean score = mean target
    score = [0.2] * 4 + [0.8] * 4
    target = [0, 0, 0, 0.8, 1, 1, 1, 0.2]
    assert ece(score, target) == pytest.approx(0.0, abs=1e-12)


def test_reliability_refuses_a_score_outside_zero_one_and_ece_of_nothing_is_nan():
    with pytest.raises(ValueError):
        reliability([0.5, 1.5], [0, 1])
    with pytest.raises(ValueError):
        reliability([-0.1], [0])
    assert math.isnan(ece([], []))


# --- bootstrap --------------------------------------------------------------------------------------------------


def four_versions(n_keys):
    keys = [f"k{i}" for i in range(n_keys)]
    return keys, {key: [4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3] for i, key in enumerate(keys)}


def test_bootstrap_resamples_the_keys_with_their_four_versions():
    keys, groups = four_versions(5)
    seen = []

    def stat(indices):
        seen.append(list(indices))
        return float(len(indices))

    low, high = bootstrap(keys, groups, stat, n=60, seed=3)
    assert len(seen) == 60
    # 5 keys drawn per resample, 4 versions each: always 20 indices, a multiple of the group size
    assert all(len(draw) == 20 and len(draw) % 4 == 0 for draw in seen)
    assert (low, high) == (20.0, 20.0)
    duplicated = 0
    for draw in seen:
        # the indices come in whole groups: each chunk of 4 is exactly one key's four versions
        chunks = [tuple(draw[i : i + 4]) for i in range(0, 20, 4)]
        assert all(chunk in {tuple(g) for g in groups.values()} for chunk in chunks)
        for group in groups.values():
            times = chunks.count(tuple(group))  # how many times the key was drawn
            duplicated += times == 2
            # a key drawn twice contributes each of its indices twice: eight indices for four versions
            assert sum(draw.count(i) for i in group) == 4 * times
    assert duplicated  # with 5 keys, 60 draws without a single repeated key is impossible in practice (0.038 ** 60)


def test_bootstrap_same_seed_same_bounds_and_the_seed_matters():
    keys, groups = four_versions(20)
    values = np.random.default_rng(42).normal(size=80)

    def stat(indices):
        return float(np.mean(values[indices]))

    first = bootstrap(keys, groups, stat, n=200, seed=7)
    assert bootstrap(keys, groups, stat, n=200, seed=7) == first
    assert bootstrap(keys, groups, stat, n=200, seed=8) != first
    low, high = first
    assert low < float(np.mean(values)) < high


def test_bootstrap_bounds_by_hand():
    # two keys A (row 0, value 0) and B (row 1, value 10), stat = the mean of the drawn values. A resample is AA
    # (mean 0, probability 1/4), AB or BA (5, 1/2) or BB (10, 1/4): the 2.5 % and 97.5 % points of 1000 draws are
    # 0 and 10 (each extreme has frequency about 25 %, far above 2.5 %)
    values = np.array([0.0, 10.0])

    def stat(indices):
        return float(np.mean(values[indices]))

    assert bootstrap(["A", "B"], {"A": [0], "B": [1]}, stat) == pytest.approx((0.0, 10.0))
    # one key: every resample is that key
    assert bootstrap(["A"], {"A": [0, 1]}, stat) == pytest.approx((5.0, 5.0))


def test_bootstrap_drops_nan_draws_and_is_nan_when_all_are():
    # the stat is undefined (nan) on the AA resamples (1/4 of the draws); the others all give 7: bounds (7, 7)
    groups = {"A": [0], "B": [1]}
    assert bootstrap(["A", "B"], groups, lambda idx: math.nan if set(idx) == {0} else 7.0) == (7.0, 7.0)
    low, high = bootstrap(["A", "B"], groups, lambda idx: math.nan)
    assert math.isnan(low) and math.isnan(high)
    with pytest.raises(ValueError):
        bootstrap([], {}, lambda idx: 0.0)


def test_paired_difference_by_hand():
    # keys A (row 0) and B (row 1); x = [0, 10], y = [0, 4]; stat_a = mean of x over the drawn rows, stat_b = mean of y
    # all rows: mean x = 5, mean y = 2 -> difference 3
    # resamples: AA -> 0 - 0 = 0 (1/4), AB or BA -> 5 - 2 = 3 (1/2), BB -> 10 - 4 = 6 (1/4): bounds (0, 6)
    x, y = np.array([0.0, 10.0]), np.array([0.0, 4.0])
    groups = {"A": [0], "B": [1]}

    def a(idx):
        return float(np.mean(x[idx]))

    def b(idx):
        return float(np.mean(y[idx]))

    assert paired_difference(["A", "B"], groups, a, b) == pytest.approx((3.0, 0.0, 6.0))
    # swapping the two statistics flips the sign: -3, bounds (-6, 0)
    assert paired_difference(["A", "B"], groups, b, a) == pytest.approx((-3.0, -6.0, 0.0))


def test_paired_difference_uses_the_same_draws_for_both_statistics():
    # stat_b = stat_a - 1 on every resample, so the paired difference is exactly 1 on every draw: bounds (1, 1);
    # with independent draws for the two statistics it would vary (the mean of a resample of {0, 10} is 0, 5 or 10)
    values = np.array([0.0, 10.0])
    groups = {"A": [0], "B": [1]}

    def a(idx):
        return float(np.mean(values[idx]))

    def b(idx):
        return a(idx) - 1.0

    assert paired_difference(["A", "B"], groups, a, b) == pytest.approx((1.0, 1.0, 1.0))


def test_paired_difference_drops_the_draws_where_a_statistic_is_nan():
    # stat_b is nan on AA; stat_a - stat_b is 3 on AB / BA (1/2 of the draws) and 6 on BB (1/4): the kept draws are 2/3
    # of 3 and 1/3 of 6, so the bounds are (3, 6)
    x, y = np.array([0.0, 10.0]), np.array([0.0, 4.0])
    groups = {"A": [0], "B": [1]}

    def a(idx):
        return float(np.mean(x[idx]))

    def b(idx):
        return math.nan if set(idx) == {0} else float(np.mean(y[idx]))

    assert paired_difference(["A", "B"], groups, a, b) == pytest.approx((3.0, 3.0, 6.0))


# --- the report's helpers ---------------------------------------------------------------------------------------

VOTES = {
    "g": [["low"] * 4, ["high"] * 4],
    "q": [["medium"] * 4, ["low"] * 4],
    "o": [["medium"] * 4, ["expert"] * 4],
}


def test_one_vs_rest_truth_takes_the_other_judges_eight_votes():
    # key 0, g left out: q + o = 8 x medium -> medium
    # key 0, q left out: g + o = 4 x low + 4 x medium; sorted [1,1,1,1,2,2,2,2], the two middle (4th and 5th) are 1, 2:
    #   (1 + 2) // 2 = 1 -> low; o left out: g + q is the same votes -> low
    # key 1, g left out: q + o = 4 x low + 4 x expert -> middle indices 1, 4 -> (1 + 4) // 2 = 2 -> medium
    # key 1, q left out: g + o = 4 x high + 4 x expert -> (3 + 4) // 2 = 3 -> high
    # key 1, o left out: g + q = 4 x high + 4 x low -> (1 + 3) // 2 = 2 -> medium
    assert one_vs_rest_truth(VOTES, "g", even_floor_median) == ["medium", "medium"]
    assert one_vs_rest_truth(VOTES, "q", even_floor_median) == ["low", "high"]
    assert one_vs_rest_truth(VOTES, "o", even_floor_median) == ["low", "medium"]


def test_one_vs_rest_truth_passes_the_votes_of_the_others_only_and_keeps_nulls():
    seen = []

    def recording(votes):
        seen.append(list(votes))
        return even_floor_median(votes)

    one_vs_rest_truth(VOTES, "g", recording)
    assert [len(votes) for votes in seen] == [8, 8]  # 4 versions x 2 other judges, per key
    assert seen[0] == ["medium"] * 8 and seen[1] == ["low"] * 4 + ["expert"] * 4
    # a null vote is given to the median as it is: 4 nulls of 8 votes are not more than half -> the 4 lows
    with_null = {"g": [["high"] * 4], "q": [[None] * 4], "o": [["low"] * 4]}
    assert one_vs_rest_truth(with_null, "g", even_floor_median) == ["low"]


def test_one_vs_rest_truth_refuses_an_unknown_judge_and_unequal_key_counts():
    with pytest.raises(ValueError):
        one_vs_rest_truth(VOTES, "nobody", even_floor_median)
    with pytest.raises(ValueError):
        one_vs_rest_truth({"g": [["low"] * 4], "q": [], "o": [["low"] * 4]}, "g", even_floor_median)


def test_rank_only_needs_a_raw_score_and_no_score():
    columns = ["id", "litellm.needs_tools", "litellm.raw_score", "litellm.raw_tier", "s1d1.score", "s1d1.raw_score"]
    assert rank_only(columns, "litellm") is True  # raw_score, no score
    assert rank_only(columns, "s1d1") is False  # it has a score: full metrics
    assert rank_only(columns, "random") is False  # neither
    assert rank_only(["x.score"], "x") is False
    assert rank_only(["other.raw_score"], "x") is False  # another classifier's raw score does not count


def test_not_predicted_is_an_entirely_null_column():
    assert not_predicted([None, None, None]) is True
    assert not_predicted([]) is True  # nothing predicted
    assert not_predicted([None, "coding"]) is False
    assert not_predicted([False, None]) is False  # False is a value, not a null
    assert not_predicted([0, None]) is False
    assert not_predicted(pa.array([None, None], type=pa.string())) is True
    assert not_predicted(pa.array([], type=pa.bool_())) is True
    assert not_predicted(pa.array([False, None], type=pa.bool_())) is False
    assert (
        not_predicted(pa.chunked_array([pa.array([None], type=pa.bool_()), pa.array([None], type=pa.bool_())])) is True
    )
    assert not_predicted(pa.chunked_array([pa.array([None], type=pa.bool_()), pa.array([True])])) is False
