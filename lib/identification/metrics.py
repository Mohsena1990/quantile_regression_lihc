"""Complete classification-metric suite for the four-class energy-poverty task.

Every number a reviewer might ask for, not just macro-F1: accuracy, balanced accuracy, per-class and
macro/weighted precision, recall (= sensitivity), specificity, F1, per-class ROC-AUC and PR-AUC (one-vs-rest)
with their macro means, Matthews correlation, Cohen's kappa, log-loss and the multiclass Brier score, plus the
same binary metrics for the two axes the label is built from (income axis, cost axis).

Vectorised numpy so a household bootstrap (hundreds of resamples x repeats) stays cheap. Definitions match
scikit-learn (verified in tests/test_identification_metrics.py); PR-AUC is average precision.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import rankdata

CLASS_KEYS = ("no_risk", "income_risk", "expenditure_risk", "double_risk")
K = 4
EPS = 1e-15


def confusion(y: np.ndarray, pred: np.ndarray, k: int = K) -> np.ndarray:
    """(k, k) counts, rows = true class, columns = predicted class."""
    return np.bincount(np.asarray(y, int) * k + np.asarray(pred, int), minlength=k * k).reshape(k, k)


def _safe(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def auc_binary(target: np.ndarray, score: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney statistic (ties averaged). NaN if only one class present."""
    t = np.asarray(target, bool)
    n_pos = t.sum()
    n_neg = len(t) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(score)
    return float((r[t].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(target: np.ndarray, score: np.ndarray) -> float:
    """PR-AUC as average precision (mean of precision at each true positive, descending score)."""
    t = np.asarray(target, bool)
    if t.sum() == 0:
        return np.nan
    order = np.argsort(-np.asarray(score), kind="stable")
    ts = t[order]
    tp = np.cumsum(ts)
    prec = tp / (np.arange(len(ts)) + 1)
    return float(prec[ts].mean())


def _binary_from_cm(tn, fp, fn, tp) -> dict:
    n = tn + fp + fn + tp
    prec = float(_safe(tp, tp + fp))
    rec = float(_safe(tp, tp + fn))
    spec = float(_safe(tn, tn + fp))
    return {"accuracy": float(_safe(tp + tn, n)), "precision": prec, "recall": rec, "specificity": spec,
            "f1": float(_safe(2 * prec * rec, prec + rec)), "balanced_accuracy": (rec + spec) / 2,
            "npv": float(_safe(tn, tn + fn))}


def axis_metrics(y_axis: np.ndarray, pred_axis: np.ndarray, score: np.ndarray, prefix: str) -> dict:
    y_axis, pred_axis = np.asarray(y_axis, bool), np.asarray(pred_axis, bool)
    tp = int((y_axis & pred_axis).sum()); tn = int((~y_axis & ~pred_axis).sum())
    fp = int((~y_axis & pred_axis).sum()); fn = int((y_axis & ~pred_axis).sum())
    out = {f"{prefix}_{k}": v for k, v in _binary_from_cm(tn, fp, fn, tp).items()}
    out[f"{prefix}_auc"] = auc_binary(y_axis, score)
    out[f"{prefix}_pr_auc"] = average_precision(y_axis, score)
    out[f"{prefix}_prevalence"] = float(y_axis.mean())
    return out


def classification_metrics(y: np.ndarray, proba: np.ndarray) -> dict:
    """All scalar metrics for one set of predictions. `proba` is (n, 4); prediction = argmax."""
    y = np.asarray(y, int)
    proba = np.asarray(proba, float)
    pred = proba.argmax(axis=1)
    n = len(y)
    cm = confusion(y, pred)
    tp = np.diag(cm).astype(float)
    row, col = cm.sum(axis=1).astype(float), cm.sum(axis=0).astype(float)
    fp, fn = col - tp, row - tp
    tn = n - tp - fp - fn
    precision, recall = _safe(tp, col), _safe(tp, row)
    specificity = _safe(tn, tn + fp)
    f1 = _safe(2 * precision * recall, precision + recall)
    support_w = row / n
    c, s = tp.sum(), float(n)
    mcc_den = np.sqrt((s ** 2 - (col ** 2).sum()) * (s ** 2 - (row ** 2).sum()))
    pe = (row * col).sum() / s ** 2
    onehot = np.eye(K)[y]
    out = {
        "n": n, "accuracy": float(c / s), "balanced_accuracy": float(recall[row > 0].mean()),
        "precision_macro": float(precision.mean()), "recall_macro": float(recall.mean()), "f1_macro": float(f1.mean()),
        "precision_weighted": float((precision * support_w).sum()), "recall_weighted": float((recall * support_w).sum()),
        "f1_weighted": float((f1 * support_w).sum()), "specificity_macro": float(specificity.mean()),
        "mcc": float((c * s - (row * col).sum()) / mcc_den) if mcc_den > 0 else 0.0,
        "cohen_kappa": float((c / s - pe) / (1 - pe)) if pe < 1 else 0.0,
        "log_loss": float(-np.log(np.clip(proba[np.arange(n), y], EPS, 1.0)).mean()),
        "brier": float(((proba - onehot) ** 2).sum(axis=1).mean()),
    }
    aucs, aps = [], []
    for k, name in enumerate(CLASS_KEYS):
        a, ap = auc_binary(y == k, proba[:, k]), average_precision(y == k, proba[:, k])
        aucs.append(a); aps.append(ap)
        out.update({f"precision_{name}": float(precision[k]), f"recall_{name}": float(recall[k]),
                    f"specificity_{name}": float(specificity[k]), f"f1_{name}": float(f1[k]),
                    f"auc_{name}": a, f"pr_auc_{name}": ap, f"support_{name}": int(row[k])})
    out["auc_macro_ovr"] = float(np.nanmean(aucs))
    out["pr_auc_macro"] = float(np.nanmean(aps))
    li, hc = (y % 2 == 1), (y >= 2)
    out.update(axis_metrics(li, pred % 2 == 1, proba[:, 1] + proba[:, 3], "income_axis"))
    out.update(axis_metrics(hc, pred >= 2, proba[:, 2] + proba[:, 3], "cost_axis"))
    return out


def average_over_repeats(y: np.ndarray, proba: np.ndarray) -> dict:
    """`proba` (R, n, 4): metrics per repeat, averaged (counts/support are taken from the first repeat)."""
    per = [classification_metrics(y, p) for p in proba]
    return {k: (per[0][k] if k.startswith("support_") or k == "n" else float(np.nanmean([m[k] for m in per])))
            for k in per[0]}


def summarise_all(y: np.ndarray, proba: np.ndarray, *, n_boot: int = 200, seed: int = 3) -> dict:
    """Point estimate + household-bootstrap 95% CI (`<metric>_lo`, `<metric>_hi`) for EVERY scalar metric,
    averaged over repeats. `proba` is (R, n, 4) or (n, 4)."""
    proba = np.asarray(proba, float)
    if proba.ndim == 2:
        proba = proba[None]
    y = np.asarray(y, int)
    point = average_over_repeats(y, proba)
    keys = [k for k in point if not k.startswith("support_") and k != "n"]
    rng = np.random.default_rng(seed)
    draws = np.empty((n_boot, len(keys)))
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        m = average_over_repeats(y[idx], proba[:, idx])
        draws[b] = [m[k] for k in keys]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
    out = dict(point)
    for k, a, b_ in zip(keys, lo, hi):
        out[f"{k}_lo"], out[f"{k}_hi"] = float(a), float(b_)
    # legacy column names used by the benchmark / factor scripts
    out.update({"macro_f1": out["f1_macro"], "f1_ci_lo": out["f1_macro_lo"], "f1_ci_hi": out["f1_macro_hi"],
                "auc_income_axis": out["income_axis_auc"], "auc_cost_axis": out["cost_axis_auc"]})
    out.update({f"recall_{k}": out[f"recall_{name}"] for k, name in enumerate(CLASS_KEYS)})
    return out
