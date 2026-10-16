"""The study design: one pooled model for the five study countries, read country by country, in three stages, with model sets
"income only", "without income" and "full", and four improvement variants.

Stages (ranking and sizing: lib/identification/selection.py):
  B0   core profile -- income bracket, floor area, household size, house detachment, house age, birth year of the respondent,
       heating strategy (H9) -- + country
  B1   B0 + groups chosen in two levels (which groups, then which variables inside them):
         household_indices  dwelling efficiency, inefficient equipment, heating system type, household vulnerability,
                            wealth (house-value stand-in), income per adult equivalent (EUR)
         enable_top3        S8 income adequacy, floor area per person, dishwasher age
         partial_modules    environmental attitudes (H15), energy-saving behaviour (G), knows the kWh price (E1), bill too
                            complicated (C5C) -- asked in some countries only; missing elsewhere
         region             the region itself (NUTS2 code, one categorical variable)
  B2   B1 + the top 3 of 30 climate candidates (20 regional measures, 10 climate x home interactions)
  B2p  the same with climate swapped between regions of the same country (placebo)
Model sets: income_only (income bracket alone), no_income (the stage's variables without the income-derived ones), full (with
income). Variants of the full B1 model (model set "variant"):
  robust      training weights for borderline and likely mislabelled households (robust.py)
  augmented   + households whose energy cost is missing, each entered twice ("above" with weight 0.5 x P, "not above" with weight
              0.5 x (1 - P)), P from the bill model; scored only on households with an observed bill
  bill        "predict the bill, then classify" (bill_model.py)
  tuned       XGBoost with Optuna-tuned hyper-parameters (tuning.py)
Ensembles, targeting curves and "not sure" are computed afterwards from the stored probabilities (scripts/build_feature_stages.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from lib.climate.b2_indices import B2_CLIMATE
from lib.features.indices import CLIMATE_CANDIDATES, INCOME_DERIVED, climate_features
from lib.features.partial_modules import PARTIAL
from lib.reporting.names import CORE_PROFILE
from .bill_model import BillModel, four_class
from .lines import fold_threshold
from .models import MODELS3, fit as fit_model
from .robust import robust_weight
from .selection import _curve, one_se_choice, paired_gate, placebo_climate, rank_candidates
from .tuning import tune_xgboost

B0_FEATURES = list(CORE_PROFILE)
B1_GROUPS = {"household_indices": ["dwelling_efficiency_index", "inefficient_equipment_index", "heating_system_type",
                                   "vulnerability_index", "wealth_index", "income_eur_equiv"],
             "enable_top3": ["S8", "floor_area_per_person", "H11B"],
             "partial_modules": list(PARTIAL),
             "region": ["region"]}
B1_CANDIDATES = [f for g in B1_GROUPS.values() for f in g]
B2_CANDIDATES = [c for c in CLIMATE_CANDIDATES if c != "need_burden"]       # need_burden divides by income: not a climate measure
BASE = B0_FEATURES + ["country"]
UNIVERSE = BASE + B1_CANDIDATES + B2_CANDIDATES
CLIMATE_BASE = list(B2_CLIMATE)
TOP_CLIMATE = 3
L2_GRID = (1, 2, 3, 4, 6, 8, 12, 17)
B2_GRID = (0, 1, 2, 3, 5, 8, 12, 30)
STAGES = ("B0", "B1", "B2", "B2p")
VARIANTS = (("robust", "catboost"), ("robust", "xgboost"), ("robust", "ft_transformer"), ("augmented", "catboost"),
            ("augmented", "xgboost"), ("augmented", "ft_transformer"), ("bill", "xgboost_regression"), ("tuned", "xgboost"))
IMPUTED_WEIGHT = 0.5
SEED = 20260811


def select_b2(train: pd.DataFrame, base: list[str], seed: int) -> dict:
    """Top-3 climate candidates by attention + SHAP on top of the chosen B1 set; the one-SE choice is recorded too."""
    train = train.reset_index(drop=True)
    ranking = rank_candidates(train, base, B2_CANDIDATES, seed)
    sets = {k: base + list(ranking.index[:k]) for k in B2_GRID}
    curve, per = _curve(train, sets, seed + 700)
    return {"selected": base + list(ranking.index[:TOP_CLIMATE]), "ranking": ranking, "curve": curve,
            "k_one_se": paired_gate(one_se_choice(curve), per)}


def placebo_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Regional climate moved to another region of the same country; climate x dwelling features recomputed."""
    return climate_features(placebo_climate(frame, CLIMATE_BASE, seed))


def without_income(feats: list[str]) -> list[str]:
    return [f for f in feats if f not in INCOME_DERIVED]


def outer_splits(frame: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3, seed: int = 1):
    strata = frame["country"].astype(str) + "_" + frame["y"].astype(str)
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(frame, strata))


def select_b1(train: pd.DataFrame, seed: int) -> dict:
    """Two-level selection: groups first, then the variables inside the chosen groups."""
    train = train.reset_index(drop=True)
    ranking = rank_candidates(train, BASE, B1_CANDIDATES, seed)
    group_score = {g: float(ranking.loc[f, "score"].mean()) for g, f in B1_GROUPS.items()}
    order = sorted(group_score, key=group_score.get)
    sets1 = {k: BASE + [f for g in order[:k] for f in B1_GROUPS[g]] for k in range(len(order) + 1)}
    curve1, per1 = _curve(train, sets1, seed)
    groups = order[:paired_gate(one_se_choice(curve1), per1)]
    if not groups:
        return {"selected": BASE, "groups": [], "ranking": ranking, "group_score": group_score, "curve1": curve1, "curve2": {}}
    cands = [f for f in ranking.index if any(f in B1_GROUPS[g] for g in groups)]
    sets2 = {k: BASE + cands[:k] for k in sorted({min(k, len(cands)) for k in L2_GRID})}
    sets2[0] = BASE
    curve2, per2 = _curve(train, sets2, seed + 500)
    k2 = paired_gate(one_se_choice(curve2), per2)
    return {"selected": BASE + cands[:k2], "groups": groups, "ranking": ranking, "group_score": group_score,
            "curve1": curve1, "curve2": curve2}


def _prior(y: np.ndarray) -> np.ndarray:
    return np.bincount(np.asarray(y, int), minlength=4) / len(y)


def augmented_training(train: pd.DataFrame, unl: pd.DataFrame, bill: BillModel, spec: dict) -> tuple[pd.DataFrame, np.ndarray]:
    """Training households + every cost-missing household twice (above / not above its line), weighted by the bill model."""
    thr = fold_threshold(train, unl, spec["family"], spec["tau"])
    ok = np.isfinite(thr)
    u = unl[ok].reset_index(drop=True)
    p = bill.p_above(u, thr[ok])
    hi, lo = u.assign(hc=1), u.assign(hc=0)
    hi["y"], lo["y"] = hi.li + 2, lo.li
    aug = pd.concat([train, hi, lo], ignore_index=True)
    w = np.concatenate([np.ones(len(train)), IMPUTED_WEIGHT * p, IMPUTED_WEIGHT * (1 - p)])
    return aug, w


def run_fold(frame: pd.DataFrame, unl: pd.DataFrame, cost_only: pd.DataFrame, spec: dict, tr: np.ndarray, te: np.ndarray,
             fold_id: int, repeat: int) -> dict:
    """Everything for one outer fold. `unl`: households with income but no cost (imputation); `cost_only`: households with cost
    but no income (bill model only); `spec`: the label's {family, cutoff, tau}."""
    seed = SEED + fold_id
    train, test = frame.iloc[tr], frame.iloc[te]
    fp = placebo_frame(frame, SEED + 500 + repeat)
    s1 = select_b1(train, seed)
    s2 = select_b2(train, s1["selected"], seed + 1000)
    s2p = select_b2(fp.iloc[tr], s1["selected"], seed + 1000)
    stage_sets = {"B0": BASE, "B1": s1["selected"], "B2": s2["selected"], "B2p": s2p["selected"]}
    model_sets = {("income_only", "all"): ["income_bracket"]}
    for st in ("B0", "B1", "B2"):
        model_sets[("no_income", st)] = without_income(stage_sets[st])
    for st in STAGES:
        model_sets[("full", st)] = stage_sets[st]
    proba, imp_rows, shap = {}, [], {}
    col = {f: j for j, f in enumerate(UNIVERSE)}
    for (mset, st), feats in model_sets.items():
        src = fp if st == "B2p" else frame
        for clf in MODELS3:
            m = fit_model(clf, src.iloc[tr], feats, seed)
            proba[(mset, st, clf)] = m.predict_proba(src.iloc[te])
            imp_rows += [{"model_set": mset, "stage": st, "model": clf, "feature": f, "importance": float(v)}
                         for f, v in m.importance(src.iloc[te]).items()]
            if clf == "catboost" and mset == "full" and st in ("B1", "B2"):
                block = np.zeros((len(te), 4, len(UNIVERSE)), dtype=np.float32)
                block[:, :, [col[f] for f in feats]] = m.shap_values(src.iloc[te])
                shap[st] = block

    # ---- variants of the full B1 model ----
    feats = s1["selected"]
    info = {}
    w, info["robust"] = robust_weight(train, feats, seed + 7)
    for clf in MODELS3:
        proba[("variant", "robust", clf)] = fit_model(clf, train, feats, seed, weight=w).predict_proba(test)
    bill = BillModel(pd.concat([train, cost_only], ignore_index=True), feats, seed + 11)
    aug, wa = augmented_training(train, unl, bill, spec)
    info["augmented"] = {"added_households": int((len(aug) - len(train)) // 2), "train_households": len(train)}
    for clf in MODELS3:
        proba[("variant", "augmented", clf)] = fit_model(clf, aug, feats, seed, weight=wa).predict_proba(test)
    thr_te = fold_threshold(train, test, spec["family"], spec["tau"])
    p_hi = bill.p_above(test, thr_te)
    proba[("variant", "bill", "xgboost_regression")] = four_class(test.li.to_numpy(), p_hi, _prior(train.y))
    resid = np.log(test.cost.clip(lower=1).to_numpy()) - bill.mu(test)
    info["bill"] = {"rmse_log_cost": float(np.sqrt(np.mean(resid ** 2))),
                    "r2_log_cost": float(1 - np.mean(resid ** 2) / np.var(np.log(test.cost.clip(lower=1))))}
    params, inner = tune_xgboost(train, feats, seed + 13)
    proba[("variant", "tuned", "xgboost")] = fit_model("xgboost", train, feats, seed, params=params).predict_proba(test)
    info["tuned"] = {"inner_balanced_accuracy": inner, **params}

    sel = []
    for st, s in (("B1", s1), ("B2", s2), ("B2p", s2p)):
        for pos, (f, r) in enumerate(s["ranking"].iterrows(), start=1):
            sel.append({"stage": st, "feature": f, "selected": f in stage_sets[st], "rank": pos, "score": r.score,
                        "attention": r.attention, "shap_catboost": r.shap_catboost, "shap_xgboost": r.shap_xgboost,
                        "weight_attention": r.weight_attention, "weight_shap_catboost": r.weight_shap_catboost,
                        "weight_shap_xgboost": r.weight_shap_xgboost})
    curves = [{"stage": "B1_groups", "k": k, "score": m, "se": se} for k, (m, se) in s1["curve1"].items()]
    curves += [{"stage": "B1_variables", "k": k, "score": m, "se": se} for k, (m, se) in s1["curve2"].items()]
    curves += [{"stage": st, "k": k, "score": m, "se": se} for st, s in (("B2", s2), ("B2p", s2p)) for k, (m, se) in s["curve"].items()]
    meta = {"groups": s1["groups"], "group_score": s1["group_score"], "b1": s1["selected"], "b2_top3": s2["selected"][-TOP_CLIMATE:],
            "b2_k_one_se": s2["k_one_se"], "b2p_top3": s2p["selected"][-TOP_CLIMATE:], "b2p_k_one_se": s2p["k_one_se"], "variants": info}
    return {"fold_id": fold_id, "repeat": repeat, "te": te, "proba": proba, "importance": imp_rows, "shap": shap,
            "selection": sel, "curves": curves, "meta": meta}
