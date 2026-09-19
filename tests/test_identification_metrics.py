"""The vectorised metric suite must agree with scikit-learn."""
import numpy as np
import pytest
from sklearn import metrics as skm
from lib.identification.metrics import (auc_binary, average_precision, classification_metrics, confusion,
                                        summarise_all)


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    logits = rng.normal(size=(n, 4)) + 1.4 * np.eye(4)[y]
    p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
    return y, p


def test_multiclass_metrics_match_sklearn():
    y, p = _data()
    m, pred = classification_metrics(y, p), p.argmax(1)
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(y, pred))
    assert m["balanced_accuracy"] == pytest.approx(skm.balanced_accuracy_score(y, pred))
    for avg, tag in (("macro", "macro"), ("weighted", "weighted")):
        assert m[f"precision_{tag}"] == pytest.approx(skm.precision_score(y, pred, average=avg, zero_division=0))
        assert m[f"recall_{tag}"] == pytest.approx(skm.recall_score(y, pred, average=avg, zero_division=0))
        assert m[f"f1_{tag}"] == pytest.approx(skm.f1_score(y, pred, average=avg, zero_division=0))
    assert m["mcc"] == pytest.approx(skm.matthews_corrcoef(y, pred))
    assert m["cohen_kappa"] == pytest.approx(skm.cohen_kappa_score(y, pred))
    assert m["log_loss"] == pytest.approx(skm.log_loss(y, p, labels=[0, 1, 2, 3]))
    assert m["auc_macro_ovr"] == pytest.approx(skm.roc_auc_score(y, p, multi_class="ovr", average="macro"))
    assert m["pr_auc_double_risk"] == pytest.approx(skm.average_precision_score(y == 3, p[:, 3]))
    assert m["brier"] == pytest.approx(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1)))
    np.testing.assert_array_equal(confusion(y, pred), skm.confusion_matrix(y, pred, labels=[0, 1, 2, 3]))


def test_axis_metrics_match_sklearn_binary():
    y, p = _data(seed=1)
    m, pred = classification_metrics(y, p), p.argmax(1)
    li_true, li_pred = y % 2 == 1, pred % 2 == 1
    assert m["income_axis_precision"] == pytest.approx(skm.precision_score(li_true, li_pred))
    assert m["income_axis_recall"] == pytest.approx(skm.recall_score(li_true, li_pred))
    assert m["income_axis_auc"] == pytest.approx(skm.roc_auc_score(li_true, p[:, 1] + p[:, 3]))
    tn, fp, fn, tp = skm.confusion_matrix(y >= 2, pred >= 2).ravel()
    assert m["cost_axis_specificity"] == pytest.approx(tn / (tn + fp))


def test_auc_and_ap_handle_degenerate_inputs():
    assert np.isnan(auc_binary(np.zeros(10, bool), np.arange(10)))
    assert np.isnan(average_precision(np.zeros(10, bool), np.arange(10)))


def test_bootstrap_summary_has_ci_for_every_metric_and_legacy_names():
    y, p = _data(200, 2)
    s = summarise_all(y, np.stack([p, p]), n_boot=30)
    for key in ("accuracy", "precision_macro", "recall_macro", "auc_macro_ovr", "mcc", "log_loss", "income_axis_auc"):
        assert s[f"{key}_lo"] <= s[key] <= s[f"{key}_hi"] or np.isnan(s[key])
    assert s["macro_f1"] == s["f1_macro"] and "f1_ci_lo" in s and "recall_3" in s
