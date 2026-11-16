"""All evaluation metrics for four-class predictions, by country and pooled, with household-bootstrap intervals."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score, log_loss,
                             matthews_corrcoef, precision_recall_fscore_support, roc_auc_score)
from common import QUAD

METRICS = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "kappa", "mcc", "roc_auc_macro", "log_loss", "brier"]


def all_metrics(y, p) -> dict:
    y = np.asarray(y, int); p = np.clip(np.asarray(p, float), 1e-9, 1); p = p / p.sum(1, keepdims=True)
    yh = p.argmax(1)
    out = {"n": len(y), "accuracy": accuracy_score(y, yh), "balanced_accuracy": balanced_accuracy_score(y, yh),
           "f1_macro": f1_score(y, yh, average="macro", labels=[0, 1, 2, 3], zero_division=0),
           "f1_weighted": f1_score(y, yh, average="weighted", labels=[0, 1, 2, 3], zero_division=0),
           "kappa": cohen_kappa_score(y, yh), "mcc": matthews_corrcoef(y, yh)}
    present = np.unique(y)
    try:
        out["roc_auc_macro"] = roc_auc_score(y, p[:, present] / p[:, present].sum(1, keepdims=True), multi_class="ovr",
                                             labels=present) if len(present) > 2 else np.nan
    except ValueError:
        out["roc_auc_macro"] = np.nan
    out["log_loss"] = log_loss(y, p, labels=[0, 1, 2, 3])
    out["brier"] = float(np.mean(np.sum((p - np.eye(4)[y]) ** 2, axis=1)))
    pr, rc, f1, sup = precision_recall_fscore_support(y, yh, labels=[0, 1, 2, 3], zero_division=0)
    for i, q in QUAD.items():
        out[f"precision_{q}"], out[f"recall_{q}"], out[f"f1_{q}"], out[f"support_{q}"] = pr[i], rc[i], f1[i], sup[i]
    return out


def _ba(y, yh):
    r = [np.mean(yh[y == k] == k) for k in range(4) if np.any(y == k)]
    return np.mean(r)


def _f1m(y, yh):
    return f1_score(y, yh, average="macro", labels=[0, 1, 2, 3], zero_division=0)


def boot_ci(y, yh, fn=_ba, B=500, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    v = [fn(y[i], yh[i]) for i in (rng.integers(0, n, n) for _ in range(B))]
    return np.percentile(v, [2.5, 97.5])


def paired_diff(y, yh_a, yh_b, fn=_ba, B=500, seed=0):
    """Mean difference (a - b) and 95% bootstrap interval, same resampled households for both."""
    rng = np.random.default_rng(seed); n = len(y)
    d = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        d.append(fn(y[i], yh_a[i]) - fn(y[i], yh_b[i]))
    return fn(y, yh_a) - fn(y, yh_b), np.percentile(d, 2.5), np.percentile(d, 97.5)
