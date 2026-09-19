import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_managerial_implications import confusion_rates, topq_decision


def test_confusion_rates_per_1000_and_ratios():
    is_pos = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    pred = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 0], bool)[None]           # TP=2, FN=2, FP=1, TN=5
    r = confusion_rates(is_pos, pred)
    assert (r["tp_per_1000"], r["fn_per_1000"], r["fp_per_1000"], r["tn_per_1000"]) == (200, 200, 100, 500)
    assert r["precision"] == 2 / 3 and r["recall"] == 0.5
    assert abs(r["leakage"] - 1 / 3) < 1e-12 and r["undercoverage"] == 0.5 and r["screened_per_hit"] == 1.5


def test_errors_partition_the_population():
    rng = np.random.default_rng(0)
    is_pos = rng.random(500) < 0.2
    pred = rng.random((3, 500)) < 0.25
    r = confusion_rates(is_pos, pred)
    assert abs(r["tp_per_1000"] + r["fp_per_1000"] + r["fn_per_1000"] + r["tn_per_1000"] - 1000) < 1e-9


def test_topq_selects_exactly_the_requested_coverage_and_the_highest_scores():
    score = np.array([0.1, 0.9, 0.4, 0.8, 0.2, 0.7, 0.3, 0.6, 0.5, 0.0])
    pos = topq_decision(score, 0.3)[0]
    assert pos.sum() == 3 and set(np.flatnonzero(pos)) == {1, 3, 5}
