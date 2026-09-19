"""Regression tests for the train-row leakage-reintroduction bug found during audit.

scripts/run_model_measurements.py always computed in-sample train predictions/thresholds.
scripts/build_oof_predictions.py corrected this afterward with proper 5-fold OOF values.
Nothing previously stopped a rerun of run_model_measurements.py from silently overwriting
the OOF-corrected train rows with fresh in-sample (leaky) ones.
"""
from __future__ import annotations
import pandas as pd

from scripts.run_model_measurements import preserve_oof_train_rows


def _frame(partition, oof_fold=None, value=1):
    n = len(partition)
    d = {"UID": [f"U{i}" for i in range(n)], "partition": partition, "value": [value] * n}
    if oof_fold is not None:
        d["oof_fold"] = oof_fold
    return pd.DataFrame(d)


def test_oof_calibrated_train_rows_are_preserved_on_rerun():
    existing = _frame(["train", "train", "validation", "test"], oof_fold=[0, 1, None, None], value=1)
    fresh_in_sample = _frame(["train", "train", "validation", "test"], value=999)  # would-be leaky rerun
    result = preserve_oof_train_rows(existing, fresh_in_sample)
    train_rows = result[result.partition.eq("train")]
    assert (train_rows.value == 1).all()  # OOF-calibrated values kept, not overwritten
    assert set(train_rows.oof_fold) == {0, 1}


def test_validation_and_test_rows_are_still_refreshed():
    existing = _frame(["train", "validation", "test"], oof_fold=[0, None, None], value=1)
    fresh = _frame(["train", "validation", "test"], value=2)
    result = preserve_oof_train_rows(existing, fresh)
    non_train = result[~result.partition.eq("train")]
    assert (non_train.value == 2).all()  # non-train rows come from the fresh fit, as before


def test_first_run_without_prior_oof_uses_fresh_predictions_unchanged():
    existing = _frame(["train", "validation"], value=1)  # no oof_fold column at all yet
    fresh = _frame(["train", "validation"], value=2)
    result = preserve_oof_train_rows(existing, fresh)
    assert (result.value == 2).all()


def test_partially_calibrated_train_rows_do_not_trigger_preservation():
    """If only some train rows carry an oof_fold (e.g. a partial/failed prior OOF run), the
    existing rows must not be trusted as leakage-safe -- fall back to the fresh fit."""
    existing = _frame(["train", "train"], oof_fold=[0, None], value=1)
    fresh = _frame(["train", "train"], value=2)
    result = preserve_oof_train_rows(existing, fresh)
    assert (result.value == 2).all()
