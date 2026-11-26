"""The seeded headroom probe: aggregation and the pre-registered decision rule.

Torch-free on purpose. The rule is the part worth testing, because a rule that
says SIGNAL when the offline metric disagrees defeats the reason it was written
down first.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "headroom_seeds", os.path.join(ROOT, "scripts", "headroom_seeds.py"))
hs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hs)


def _history(on_goal, acc=0.99, epochs=12):
    """A 12-epoch history whose last epochs sit at `on_goal`."""
    rows = []
    for e in range(1, epochs + 1):
        rows.append({"epoch": e, "acc": acc - 0.01 * (epochs - e) / epochs,
                     "env_on_goal": on_goal, "env_success": on_goal / 2,
                     "env_coll": 0.1, "val_loss": 0.05})
    return rows


def _write(out, tag, arm, seed, history):
    d = hs.run_dir(str(out), tag, arm, seed)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "history.json"), "w") as fh:
        json.dump(history, fh)


def _fill(tmp_path, tag, spec, n=5, noise=0.01, seed0=0):
    """spec: {arm: (on_goal_mean, acc)}. Writes n seeds per arm."""
    rng = np.random.default_rng(seed0)
    for arm, (og, acc) in spec.items():
        for s in range(n):
            _write(tmp_path, tag, arm, s,
                   _history(float(og + rng.normal(0, noise)), acc=acc))
    return hs.collect(str(tmp_path), tag, range(n), window=4)


# ---------------------------------------------------------------- plumbing

def test_window_mean_takes_the_tail_and_skips_missing():
    h = [{"v": 1.0}, {"v": 2.0}, {"v": None}, {"v": 4.0}, {"v": 6.0}]
    assert hs.window_mean(h, "v", 2) == 5.0
    assert hs.window_mean(h, "v", 99) == pytest.approx((1 + 2 + 4 + 6) / 4)
    with pytest.raises(ValueError):
        hs.window_mean(h, "missing", 2)


def test_offline_error_uses_best_accuracy_not_last():
    h = [{"acc": 0.90, "env_on_goal": 0.5, "env_success": 0.1, "env_coll": 0.2},
         {"acc": 0.99, "env_on_goal": 0.5, "env_success": 0.1, "env_coll": 0.2},
         {"acc": 0.95, "env_on_goal": 0.5, "env_success": 0.1, "env_coll": 0.2}]
    assert hs.summarise_run(h, 3)["err"] == pytest.approx(0.01)


def test_is_complete_needs_all_epochs(tmp_path):
    d = hs.run_dir(str(tmp_path), "t", "full", 0)
    assert not hs.is_complete(d, 12)                       # nothing there
    _write(tmp_path, "t", "full", 0, _history(0.5, epochs=7))
    assert not hs.is_complete(d, 12)                       # interrupted run
    _write(tmp_path, "t", "full", 0, _history(0.5, epochs=12))
    assert hs.is_complete(d, 12)
    with open(os.path.join(d, "history.json"), "w") as fh:
        fh.write("{ not json")
    assert not hs.is_complete(d, 12)                       # corrupt, redo it


# ------------------------------------------------------------ decision rule

def test_identical_arms_give_a_null(tmp_path):
    got = _fill(tmp_path, "t", {"none": (0.5, 0.99), "shuffled": (0.6, 0.99),
                                "full": (0.6, 0.99)})
    assert hs.verdict(got).startswith("NULL")


def test_clear_gap_with_agreeing_offline_metric_is_a_signal(tmp_path):
    got = _fill(tmp_path, "t", {"none": (0.4, 0.985), "shuffled": (0.5, 0.985),
                                "full": (0.6, 0.995)})
    v = hs.verdict(got)
    assert v.startswith("SIGNAL"), v


def test_closed_loop_gap_the_offline_metric_does_not_corroborate(tmp_path):
    """The case that motivated writing the rule down: the maze closed-loop gap
    was ~5pp while offline error moved +1.4%. A clean on-goal gap with a
    flat offline metric must NOT be called a signal."""
    got = _fill(tmp_path, "t", {"none": (0.4, 0.99), "shuffled": (0.5, 0.99),
                                "full": (0.6, 0.99)})
    v = hs.verdict(got)
    assert v.startswith("UNCONFIRMED"), v


def test_shuffled_beating_full_is_reported_as_reversed(tmp_path):
    got = _fill(tmp_path, "t", {"none": (0.4, 0.99), "shuffled": (0.65, 0.99),
                                "full": (0.5, 0.99)})
    assert hs.verdict(got).startswith("REVERSED")


def test_too_few_seeds_is_not_a_result(tmp_path):
    """Three seeds produced d=-1.76 once and d=+1.57 at five. The rule refuses
    to speak below its own minimum, however clean the p-value looks."""
    got = _fill(tmp_path, "t", {"none": (0.4, 0.99), "shuffled": (0.5, 0.99),
                                "full": (0.9, 0.999)}, n=3)
    assert hs.verdict(got).startswith("UNDERPOWERED")


def test_a_single_seed_cannot_be_tested(tmp_path):
    got = _fill(tmp_path, "t", {"none": (0.4, 0.99), "shuffled": (0.5, 0.99),
                                "full": (0.6, 0.99)}, n=1)
    assert hs.verdict(got).startswith("INCOMPLETE")


def test_none_arm_at_the_floor_is_unmeasurable_not_null(tmp_path):
    """fov 3 and fov 5 without a distance map: none arm ~0.05, nothing solved.
    The script printed NULL for both, which read as a finding."""
    got = _fill(tmp_path, "t", {"none": (0.05, 0.99), "shuffled": (0.03, 0.99),
                                "full": (0.035, 0.99)}, noise=0.003)
    assert hs.verdict(got).startswith("UNMEASURABLE")


def test_none_arm_at_the_ceiling_is_unmeasurable(tmp_path):
    got = _fill(tmp_path, "t", {"none": (0.95, 0.99), "shuffled": (0.96, 0.99),
                                "full": (0.96, 0.99)})
    assert hs.verdict(got).startswith("UNMEASURABLE")


def test_gate_does_not_hide_a_measurable_null(tmp_path):
    got = _fill(tmp_path, "t", {"none": (0.2, 0.99), "shuffled": (0.3, 0.99),
                                "full": (0.3, 0.99)})
    assert hs.verdict(got).startswith("NULL")


def test_report_prints_without_error(tmp_path, capsys):
    got = _fill(tmp_path, "t", {"none": (0.4, 0.99), "shuffled": (0.5, 0.99),
                                "full": (0.55, 0.995)})
    hs.report(got, "t", 4)
    out = capsys.readouterr().out
    assert "PRIMARY" in out and "VERDICT" in out
