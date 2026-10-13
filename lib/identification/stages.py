"""Feature stages B0 -> B1 -> B2 with nested "attention + SHAP" feature selection (income is a predictor in every stage).

  B0  Van Hove et al. (2022), Table 1, exactly: income bracket, floor area (H3), household size (sum of the S1 composition
      counts), house detachment (H1), house age (H2), birth year of the respondent (S4) and heating strategy (H9: one setting /
      manual adjustment / programmed thermostat / no control / don't know; categorical)
  B1  B0 + the best of ALL candidate variables (B1_CANDIDATES, 82): every survey item answered in all four countries
      (lib/features/survey_items.py), the cleaned welfare items S7 and S8, the main heating fuel and solid-fuel heating, and
      23 engineered household features (lib/features/engineered.py)
  B2  B1 + the best of 18 climate indices (lib/climate/b2_indices.py; NUTS3 / Polish NUTS2 regions)
  B2p placebo control: as B2 with each region's climate moved to another region of the same country

Feature selection -- "Attention Mechanisms as Embedded Selectors + SHAP" -- nested inside every OUTER training fold, so the
households that are scored never influence the choice:
  1. rank: the outer training rows are split 80 / 20. On the 80%, three models are fitted with the base set + all candidates:
     an FT-Transformer, whose [CLS] attention to each feature token is an EMBEDDED importance (the model learns where to look),
     and CatBoost and XGBoost, whose exact TreeSHAP values give each feature's contribution. All three are measured on the 20%.
     Each candidate's score = average of its attention rank and its mean SHAP rank (CatBoost and XGBoost averaged).
  2. curve: inner repeated cross-validation (2 x 3 folds) of CatBoost on base + the k best-ranked candidates, for k in a grid
  3. decide: ONE-STANDARD-ERROR RULE (Breiman et al. 1984; Hastie, Tibshirani & Friedman 2009, Sec. 7.10) -- the smallest k
     whose mean inner macro-F1 is within one standard error of the best k. k = 0 keeps the base set.
B2 selects climate indices on top of the B1 set chosen in that same outer fold.

Placebo B2p: climate is measured per region, so any region-level number lets a model recognise regions (a dry run with random
numbers per region raised macro-F1 by up to 0.13). B2p repeats the B2 selection with climate permuted across regions within each
country; only B2 - B2p can be credited to climate.

Each stage's selected features are then scored by all three models -- CatBoost, XGBoost and FT-Transformer -- on the outer
held-out households, with every metric and each model's own importance (SHAP for the trees, attention for the transformer).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from lib.climate.b2_indices import B2_CLIMATE, B2_NAMES
from lib.features.engineered import ENGINEERED
from lib.features.survey_items import ITEM_NAMES, ITEMS
from .models3 import MODELS3, fit as fit_model
from .protocol import _strata

B0_FEATURES = ["income_bracket", "H3", "household_size", "H1", "H2", "birth_year", "heating_strategy"]
HEATING_ALTERNATIVES = ["primary_heating_fuel", "has_solid_fuel_heating"]
B1_CANDIDATES = list(ITEMS) + ["S7", "S8"] + HEATING_ALTERNATIVES + list(ENGINEERED)
B2_CANDIDATES = list(B2_CLIMATE)
ALL_FEATURES = B0_FEATURES + B1_CANDIDATES + B2_CANDIDATES
FEATURE_NAMES = {"income_bracket": "income bracket", "H3": "floor area", "household_size": "household size",
                 "H1": "house detachment", "H2": "house age", "birth_year": "birth year respondent",
                 "heating_strategy": "heating strategy (H9)", "primary_heating_fuel": "main heating fuel",
                 "has_solid_fuel_heating": "solid-fuel heating", "S7": "S7 energy-bill aid", "S8": "S8 income adequacy",
                 **ITEM_NAMES, **{f: f.replace("_", " ") for f in ENGINEERED}, **B2_NAMES}
K_GRID = {"B1": (0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, len(B1_CANDIDATES)), "B2": (0, 1, 2, 3, 4, 6, 9, 12, len(B2_CANDIDATES))}
STAGES = ("B0", "B1", "B2", "B2p")
INNER_FOLDS, INNER_REPEATS = 3, 2
SEED = 20260811


def _ranks(s: pd.Series) -> pd.Series:
    return s.rank(ascending=False, method="average")


def rank_candidates(train: pd.DataFrame, base: list[str], candidates: list[str], seed: int) -> pd.DataFrame:
    """Attention + SHAP ranking of the candidates on an 80/20 split of `train`. Columns: attention, shap_catboost,
    shap_xgboost, score (lower = better), ordered best first."""
    feats = base + candidates
    tr, va = train_test_split(train, test_size=0.2, stratify=train["y"], random_state=seed)
    att = fit_model("ft_transformer", tr, feats, seed, val=va).importance(va)
    cb = fit_model("catboost", tr, feats, seed).importance(va)
    xg = fit_model("xgboost", tr, feats, seed).importance(va)
    t = pd.DataFrame({"attention": att, "shap_catboost": cb, "shap_xgboost": xg}).loc[candidates]
    t["score"] = (_ranks(t.attention) + (_ranks(t.shap_catboost) + _ranks(t.shap_xgboost)) / 2) / 2
    return t.sort_values("score")


def _inner_folds(train: pd.DataFrame, seed: int):
    y = train["y"].astype(int).to_numpy()
    counts = np.bincount(y, minlength=4)
    n_splits = max(2, min(INNER_FOLDS, int(counts[counts > 0].min())))
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=INNER_REPEATS, random_state=seed).split(train, y))


def one_se_choice(curve: dict[int, tuple[float, float]]) -> int:
    """Smallest k whose mean score is within one standard error of the best k's mean (curve: k -> (mean, se))."""
    best_k = max(curve, key=lambda k: curve[k][0])
    bar = curve[best_k][0] - curve[best_k][1]
    return min(k for k, (m, _) in curve.items() if m >= bar)


def select_features(train: pd.DataFrame, base: list[str], candidates: list[str], grid: tuple[int, ...], *, seed: int) -> dict:
    """Nested attention + SHAP forward selection on `train` only, sized by the one-standard-error rule. Returns
    {'selected', 'k', 'ranking' (DataFrame), 'curve' (k -> (mean inner macro-F1, SE))}."""
    train = train.reset_index(drop=True)
    ranking = rank_candidates(train, base, candidates, seed)
    y = train["y"].astype(int).to_numpy()
    folds = _inner_folds(train, seed)
    curve = {}
    for k in sorted(set(min(k, len(candidates)) for k in grid)):
        f = base + list(ranking.index[:k])
        scores = []
        for i, (tr, te) in enumerate(folds):
            pred = fit_model("catboost", train.iloc[tr], f, seed + i).predict_proba(train.iloc[te]).argmax(1)
            scores.append(f1_score(y[te], pred, average="macro", labels=[0, 1, 2, 3], zero_division=0))
        curve[k] = (float(np.mean(scores)), float(np.std(scores, ddof=1) / np.sqrt(len(scores))))
    k = one_se_choice(curve)
    return {"selected": base + list(ranking.index[:k]), "k": k, "ranking": ranking, "curve": curve}


def placebo_climate(frame: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    """`frame` with the climate columns of each region replaced by those of another region of the same country
    (random permutation of regions with no fixed point where the country has >= 2 regions)."""
    out = frame.copy()
    rng = np.random.default_rng(seed)
    for _, g in frame.groupby("country"):
        regions = g["climate_region"].drop_duplicates().to_numpy()
        if len(regions) < 2:
            continue
        while True:
            perm = rng.permutation(len(regions))
            if not (perm == np.arange(len(regions))).any():
                break
        table = g.drop_duplicates("climate_region").set_index("climate_region")[cols]
        donor = dict(zip(regions, regions[perm]))
        out.loc[g.index, cols] = table.loc[g["climate_region"].map(donor)].to_numpy()
    return out


def run_stages(frame: pd.DataFrame, *, n_splits: int = 5, n_repeats: int = 3, seed: int = 1) -> dict:
    """Outer repeated CV over one country's labelled households. Returns
      proba       {(stage, model): (R, n, 4)} held-out probabilities for CatBoost, XGBoost and FT-Transformer
      shap        {stage: (R, n, 4, len(ALL_FEATURES))} held-out CatBoost SHAP (0 for features not selected in a fold)
      importance  long table: fold x stage x model x feature, held-out importance (SHAP or attention)
      selection   long table: fold x stage x candidate -- selected?, attention, SHAP, combined rank
      curve       long table: fold x stage x k -- inner macro-F1, SE, chosen?"""
    frame = frame.reset_index(drop=True)
    n = len(frame)
    proba = {(s, m): np.zeros((n_repeats, n, 4)) for s in STAGES for m in MODELS3}
    shap = {s: np.zeros((n_repeats, n, 4, len(ALL_FEATURES)), dtype=np.float32) for s in STAGES}
    col = {f: j for j, f in enumerate(ALL_FEATURES)}
    sel_rows, curve_rows, imp_rows = [], [], []
    placebo = {r: placebo_climate(frame, B2_CANDIDATES, SEED + 500 + r) for r in range(n_repeats)}
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(cv.split(frame, _strata(frame))):
        r = i // n_splits
        fp = placebo[r]
        s1 = select_features(frame.iloc[tr], B0_FEATURES, B1_CANDIDATES, K_GRID["B1"], seed=SEED + i)
        s2 = select_features(frame.iloc[tr], s1["selected"], B2_CANDIDATES, K_GRID["B2"], seed=SEED + 1000 + i)
        s2p = select_features(fp.iloc[tr], s1["selected"], B2_CANDIDATES, K_GRID["B2"], seed=SEED + 1000 + i)
        sets = {"B0": B0_FEATURES, "B1": s1["selected"], "B2": s2["selected"], "B2p": s2p["selected"]}
        for stage, sel in (("B1", s1), ("B2", s2), ("B2p", s2p)):
            for kk, (m, se) in sel["curve"].items():
                curve_rows.append({"repeat": r, "fold": i % n_splits, "stage": stage, "k": kk, "inner_f1": m, "inner_se": se,
                                   "chosen": kk == sel["k"]})
            for pos, (c, row) in enumerate(sel["ranking"].iterrows(), start=1):
                sel_rows.append({"repeat": r, "fold": i % n_splits, "stage": stage, "feature": c, "selected": c in sel["selected"],
                                 "k": sel["k"], "rank": pos, "attention": row.attention, "shap_catboost": row.shap_catboost,
                                 "shap_xgboost": row.shap_xgboost})
        for stage, feats in sets.items():
            src = fp if stage == "B2p" else frame
            train, test = src.iloc[tr], src.iloc[te]
            for model in MODELS3:
                m = fit_model(model, train, feats, SEED + i)
                proba[(stage, model)][r, te] = m.predict_proba(test)
                imp_rows += [{"repeat": r, "fold": i % n_splits, "stage": stage, "model": model, "feature": f, "importance": float(v)}
                             for f, v in m.importance(test).items()]
                if model == "catboost":
                    block = np.zeros((len(te), 4, len(ALL_FEATURES)), dtype=np.float32)     # fancy-indexed views are copies
                    block[:, :, [col[f] for f in feats]] = m.shap_values(test)
                    shap[stage][r, te] = block
    return {"proba": proba, "shap": shap, "importance": pd.DataFrame(imp_rows), "selection": pd.DataFrame(sel_rows),
            "curve": pd.DataFrame(curve_rows)}


def paired_delta(y: np.ndarray, proba_a: np.ndarray, proba_b: np.ndarray, metrics=("f1_macro", "balanced_accuracy", "f1_weighted"), *,
                 n_boot: int = 500, seed: int = 11) -> dict:
    """Difference b - a of repeat-averaged metrics with a PAIRED household bootstrap 95% CI (same resampled households
    for both), plus the share of resamples in which b > a."""
    from .metrics import average_over_repeats
    y = np.asarray(y, int)
    pa, pb = average_over_repeats(y, proba_a), average_over_repeats(y, proba_b)
    rng = np.random.default_rng(seed)
    draws = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        a, b = average_over_repeats(y[idx], proba_a[:, idx]), average_over_repeats(y[idx], proba_b[:, idx])
        for m in metrics:
            draws[m].append(b[m] - a[m])
    out = {}
    for m in metrics:
        d = np.asarray(draws[m])
        out.update({f"delta_{m}": pb[m] - pa[m], f"delta_{m}_lo": float(np.percentile(d, 2.5)),
                    f"delta_{m}_hi": float(np.percentile(d, 97.5)), f"p_improve_{m}": float((d > 0).mean())})
    return out
