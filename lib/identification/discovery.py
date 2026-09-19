"""Predictor discovery and the universality test (Van Hove's universal-vs-contextual hypothesis).

Per country, per target (four-class `y`, income axis `li`, cost axis `hc`), repeated stratified CV with a
CatBoost model. On every held-out fold (never on training rows -- Van Hove scored importance on unseen data)
we record two importances:
  * mean |SHAP| (pooled over classes for the four-class target)
  * permutation importance: the increase in log-loss and drop in balanced accuracy when a feature's values are
    shuffled (Van Hove found the two methods can disagree, e.g. for heating strategy -- so both are kept and a
    factor must clear the bar on BOTH)
Stability = share of the CV folds in which a feature is in the top-K under a method. A feature is *universal* for a
target if it is stable under both methods in every country (`strict`) or all but one (`relaxed`).

Universality is then TESTED, not just declared: `loco_universality` selects the factors from the other countries
only, trains a factors-only model on them and scores the held-out country (selection and test never share a
country), next to that country's own full model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from .data import NOMINAL
from .protocol import _strata

TARGETS = {"y": "Four-class", "li": "Income axis", "hc": "Cost axis"}
TOP_K = 6
MIN_STABILITY_SHAP = 0.70
MIN_STABILITY_PERM = 0.50      # permutation importance is noisier on small folds
SEED = 20260811


def _prep(train: pd.DataFrame, test: pd.DataFrame, feats: list[str]):
    cat = [i for i, f in enumerate(feats) if f in NOMINAL or not pd.api.types.is_numeric_dtype(train[f])]
    a, b = train[feats].copy(), test[feats].copy()
    for i, f in enumerate(feats):
        if i in cat:
            a[f], b[f] = a[f].astype("object").fillna("NA").astype(str), b[f].astype("object").fillna("NA").astype(str)
        else:
            med = a[f].median()
            a[f], b[f] = a[f].fillna(med), b[f].fillna(med)
    return a, b, cat


def _model(target: str, seed: int) -> CatBoostClassifier:
    loss = "MultiClass" if target == "y" else "Logloss"
    return CatBoostClassifier(iterations=100, depth=4, learning_rate=0.05, l2_leaf_reg=10, loss_function=loss,
                              auto_class_weights="Balanced", verbose=False, random_seed=seed, thread_count=4)


def _shap_abs(model, x: pd.DataFrame, y: pd.Series, cat: list[int], target: str) -> np.ndarray:
    sv = np.asarray(model.get_feature_importance(Pool(x, y, cat_features=cat), type="ShapValues"))
    return (np.abs(sv[:, :, :-1]).mean(axis=(0, 1)) if target == "y" else np.abs(sv[:, :-1]).mean(axis=0))


def _proba(model, x: pd.DataFrame, target: str) -> np.ndarray:
    p = model.predict_proba(x)
    if target == "y":
        out = np.zeros((len(x), 4)); out[:, np.asarray(model.classes_, int)] = p
        return out
    return p[:, 1]


def _score(y: np.ndarray, p: np.ndarray, target: str) -> tuple[float, float]:
    """(log-loss, balanced accuracy) of predictions p against y."""
    if target == "y":
        return log_loss(y, np.clip(p, 1e-12, 1), labels=[0, 1, 2, 3]), balanced_accuracy_score(y, p.argmax(1))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1]), balanced_accuracy_score(y, p > 0.5)


def fold_importance(frame: pd.DataFrame, feats: list[str], target: str, *, n_splits: int = 5, n_repeats: int = 3,
                    n_perm: int = 5, seed: int = 1, collect_shap: bool = False):
    """Long table of held-out-fold importances (one row per fold x feature). With `collect_shap=True` also returns
    the out-of-fold per-household SHAP values (for the dependence scatter; scalar per feature, so binary targets)."""
    frame = frame.reset_index(drop=True)
    y_all = frame[target].astype(int)
    rng = np.random.default_rng(seed)
    rows, oof = [], []
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(cv.split(frame, _strata(frame))):
        train, test = frame.iloc[tr], frame.iloc[te]
        xa, xb, cat = _prep(train, test, feats)
        model = _model(target, SEED + i).fit(xa, y_all.iloc[tr], cat_features=cat)
        yt = y_all.iloc[te].to_numpy()
        shap = _shap_abs(model, xb, y_all.iloc[te], cat, target)
        base_ll, base_ba = _score(yt, _proba(model, xb, target), target)
        for j, f in enumerate(feats):
            d_ll, d_ba = [], []
            for _ in range(n_perm):
                xp = xb.copy()
                xp[f] = rng.permutation(xp[f].to_numpy())
                ll, ba = _score(yt, _proba(model, xp, target), target)
                d_ll.append(ll - base_ll); d_ba.append(base_ba - ba)
            rows.append({"repeat": i // n_splits, "fold": i % n_splits, "feature": f, "shap_abs": float(shap[j]),
                         "perm_logloss": float(np.mean(d_ll)), "perm_bacc": float(np.mean(d_ba))})
        if collect_shap and target != "y":
            sv = np.asarray(model.get_feature_importance(Pool(xb, yt, cat_features=cat), type="ShapValues"))[:, :-1]
            oof.append(pd.DataFrame(sv, columns=feats).assign(_row=te, _repeat=i // n_splits))
    out = pd.DataFrame(rows)
    if collect_shap and target != "y":
        return out, pd.concat(oof)
    return out


def summarise_stability(imp: pd.DataFrame, *, k: int = TOP_K) -> pd.DataFrame:
    """Per feature: mean normalised importance and top-k stability under each method (imp = fold_importance)."""
    d = imp.copy()
    keys = ["repeat", "fold"]
    for m in ("shap_abs", "perm_logloss", "perm_bacc"):
        d[m] = d[m].clip(lower=0)
        d[f"{m}_share"] = d[m] / d.groupby(keys)[m].transform("sum").replace(0, np.nan)
        d[f"{m}_top"] = d.groupby(keys)[m].rank(ascending=False, method="first") <= k
    g = d.groupby("feature")
    out = pd.DataFrame({
        "shap_share": g["shap_abs_share"].mean(), "perm_share": g["perm_logloss_share"].mean(),
        "perm_bacc_share": g["perm_bacc_share"].mean(),
        "shap_stability": g["shap_abs_top"].mean(), "perm_stability": g["perm_logloss_top"].mean()})
    return out.fillna(0).reset_index()


def select_universal(stab: pd.DataFrame, countries: list[str], *, n_top: int = 2) -> pd.DataFrame:
    """`stab` = long table (country, feature, shap_share, perm_share, shap_stability, perm_stability) for ONE
    target/state. Returns one row per feature with the cross-country consensus and the strict/relaxed flags."""
    st = stab[stab.country.isin(countries)].copy()
    st["passes"] = (st.shap_stability >= MIN_STABILITY_SHAP) & (st.perm_stability >= MIN_STABILITY_PERM)
    g = st.groupby("feature")
    out = pd.DataFrame({
        "mean_shap_share": g.shap_share.mean(), "mean_perm_share": g.perm_share.mean(),
        "min_shap_stability": g.shap_stability.min(), "min_perm_stability": g.perm_stability.min(),
        "shap_cv": g.shap_share.std(ddof=0) / g.shap_share.mean().replace(0, np.nan),
        "n_countries_pass": g.passes.sum(), "n_countries": g.size()}).reset_index()
    out["consensus_score"] = (out.mean_shap_share + out.mean_perm_share) / 2
    out["universal_strict"] = out.n_countries_pass == out.n_countries
    out["universal_relaxed"] = out.n_countries_pass >= out.n_countries - 1
    cand = out[out.universal_strict]
    if len(cand) < n_top:                     # fall back to the relaxed set; may still be empty ("no universal factor")
        cand = out[out.universal_relaxed]
    chosen = cand.sort_values("consensus_score", ascending=False).head(n_top).feature.tolist()
    out["selected"] = out.feature.isin(chosen)
    return out.sort_values("consensus_score", ascending=False)


def partial_dependence(frame: pd.DataFrame, feats: list[str], target: str, feature: str, *, max_points: int = 10) -> pd.DataFrame:
    """Model-averaged partial dependence of `feature` (Friedman; Van Hove Fig. 8) from a model fit on `frame`.
    Rows: (value, class or 'axis', mean predicted probability)."""
    frame = frame.reset_index(drop=True)
    xa, _, cat = _prep(frame, frame, feats)
    model = _model(target, SEED).fit(xa, frame[target].astype(int), cat_features=cat)
    values = xa[feature].unique()
    if feature not in [feats[i] for i in cat] and len(values) > max_points:
        values = np.unique(np.quantile(xa[feature], np.linspace(0.05, 0.95, max_points)))
    rows = []
    for v in sorted(values, key=str):
        xv = xa.copy(); xv[feature] = v
        p = _proba(model, xv, target)
        if target == "y":
            for k, name in enumerate(("No Risk", "Income Risk", "Expenditure Risk", "Double Risk")):
                rows.append({"feature": feature, "value": v, "outcome": name, "pd": float(p[:, k].mean())})
        else:
            rows.append({"feature": feature, "value": v, "outcome": TARGETS[target], "pd": float(p.mean())})
    return pd.DataFrame(rows)
