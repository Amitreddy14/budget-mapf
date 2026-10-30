"""Tests for the ablation statistics.

A wrong p-value is worse than no p-value: it converts "we don't know yet" into
a claim. An earlier normal-approximation fallback reported p=0.038 where the
true Student-t value at df=2.3 was 0.159 -- the difference between publishing
a result and not having one.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from eval.stats import (regularised_incomplete_beta, student_t_sf, summarise,
                        welch_ttest)


@pytest.mark.parametrize("df,t_crit", [
    (1, 12.706), (2, 4.303), (3, 3.182), (5, 2.571),
    (10, 2.228), (30, 2.042), (100, 1.984),
])
def test_t_distribution_matches_published_critical_values(df, t_crit):
    """Two-tailed p at the 0.05 critical value must be 0.05."""
    assert 2 * student_t_sf(t_crit, df) == pytest.approx(0.05, abs=5e-4)


def test_incomplete_beta_endpoints_and_symmetry():
    assert regularised_incomplete_beta(2, 3, 0.0) == 0.0
    assert regularised_incomplete_beta(2, 3, 1.0) == 1.0
    # I_x(a,b) = 1 - I_{1-x}(b,a)
    for x in (0.1, 0.35, 0.5, 0.8):
        assert regularised_incomplete_beta(2.5, 3.5, x) == pytest.approx(
            1 - regularised_incomplete_beta(3.5, 2.5, 1 - x), abs=1e-9)


def test_t_is_heavier_tailed_than_normal_at_small_df():
    """The exact reason the normal fallback was dangerous."""
    import math
    normal_p = math.erfc(2.08 / math.sqrt(2))
    t_p = 2 * student_t_sf(2.08, 2.3)
    assert t_p > 4 * normal_p, "small-df t should be far more conservative"


def test_welch_reproduces_the_phase5_collision_comparison():
    none = [0.093, 0.081, 0.115]
    full = [0.072, 0.080, 0.073]
    r = welch_ttest(none, full)
    assert r["t"] == pytest.approx(2.08, abs=0.02)
    assert r["df"] == pytest.approx(2.3, abs=0.1)
    assert r["p"] == pytest.approx(0.159, abs=0.01)
    assert not r["p"] < 0.05, "this comparison must NOT read as significant"
    assert r["n_for_80pc_power"] == 6


def test_identical_arms_are_not_significant():
    r = welch_ttest([0.10, 0.11, 0.09], [0.10, 0.11, 0.09])
    assert r["p"] > 0.9


def test_clearly_separated_arms_are_significant():
    r = welch_ttest([1.00, 1.02, 0.98], [0.10, 0.12, 0.08])
    assert r["p"] < 0.01


def test_summarise_labels_direction_for_lower_is_better():
    line = summarise("collisions", [0.20, 0.21, 0.19], [0.10, 0.11, 0.09],
                     lower_is_better=True)
    assert "better" in line
    line = summarise("success", [0.20, 0.21, 0.19], [0.10, 0.11, 0.09])
    assert "worse" in line


def test_single_seed_returns_nan_rather_than_a_fake_result():
    r = welch_ttest([0.1], [0.2])
    import math
    assert math.isnan(r["p"])


# --- eval_window -----------------------------------------------------------
# Phase 6 was decided on the final evaluation of each run. Consecutive
# evaluations of one converged run swing by 0.20-0.30, so that number is a
# single draw from a wide distribution. These tests pin the averaging that
# replaced it.

def test_eval_window_averages_the_tail():
    from eval.stats import eval_window
    # iter 0 is excluded as a pre-training measurement, so the usable rows
    # here are the four at iters 1-4.
    hist = [{"iter": i, "eval_success": s, "eval_coll": 0.1}
            for i, s in enumerate([0.1, 0.2, 0.9, 0.7, 0.5])]
    assert eval_window(hist, 1)["eval_success"] == 0.5
    assert abs(eval_window(hist, 3)["eval_success"] - (0.9 + 0.7 + 0.5) / 3) < 1e-12
    assert abs(eval_window(hist, 99)["eval_success"]
               - (0.2 + 0.9 + 0.7 + 0.5) / 4) < 1e-12


def test_eval_window_excludes_the_pre_training_evaluation():
    """iter 0 runs before any gradient step, so it measures the warm start.

    At 2M steps a run has 4 evaluations and that row carried a quarter of
    the weight: it reported collisions of 1.96 against 0.26 for the trained
    policy, inflating the collision column 2.6x and giving the `none` arm a
    success rate it earned before training began.
    """
    from eval.stats import eval_window
    hist = [{"iter": 0, "eval_success": 0.90, "eval_coll": 2.00},
            {"iter": 10, "eval_success": 0.70, "eval_coll": 0.30},
            {"iter": 20, "eval_success": 0.72, "eval_coll": 0.26}]
    w = eval_window(hist, 10)
    assert abs(w["eval_success"] - 0.71) < 1e-12
    assert abs(w["eval_coll"] - 0.28) < 1e-12
    assert w["eval_window"] == 2


def test_eval_window_keeps_iter_zero_when_it_is_the_only_row():
    """A single-evaluation history must not come back empty."""
    from eval.stats import eval_window
    w = eval_window([{"iter": 0, "eval_success": 0.4, "eval_coll": 1.0}], 10)
    assert w["eval_success"] == 0.4
    assert w["eval_window"] == 1


def test_eval_window_skips_rows_without_evaluation():
    from eval.stats import eval_window
    hist = [{"iter": 0, "pg_loss": 1.0},
            {"iter": 1, "eval_success": 0.4, "eval_coll": 0.2},
            {"iter": 2, "pg_loss": 0.5},
            {"iter": 3, "eval_success": 0.6, "eval_coll": 0.1}]
    w = eval_window(hist, 2)
    assert abs(w["eval_success"] - 0.5) < 1e-12
    assert w["eval_window"] == 2


def test_eval_window_reduces_spread_of_a_noisy_run():
    """The point of the change: averaging shrinks the sampling spread."""
    import random
    from eval.stats import eval_window
    rng = random.Random(0)
    singles, windows = [], []
    for _ in range(200):
        hist = [{"iter": i, "eval_success": rng.gauss(0.65, 0.10)} for i in range(30)]
        singles.append(eval_window(hist, 1)["eval_success"])
        windows.append(eval_window(hist, 10)["eval_success"])

    def sd(v):
        m = sum(v) / len(v)
        return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

    assert sd(windows) < 0.5 * sd(singles)


def test_eval_window_rejects_a_history_with_no_evaluations():
    from eval.stats import eval_window
    try:
        eval_window([{"iter": 0, "pg_loss": 1.0}], 10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
