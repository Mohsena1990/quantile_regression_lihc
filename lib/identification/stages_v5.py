"""Version-5 feature stages: the pooled design of version 4 on five countries (the four + Spain), with the nine improvements.

Stages (as version 4, ranking = "Attention Mechanisms as Embedded Selectors + SHAP", sizing = one-standard-error rule + paired gate):
  B0   Van Hove's seven predictors + country
  B1   B0 + groups chosen in two levels (which groups, then which variables inside them). Groups:
         household_indices  dwelling efficiency, inefficient equipment, heating system type, household vulnerability,
                            wealth (house-value stand-in), income per adult equivalent (EUR)
         enable_top3        S8 income adequacy, floor area per person, dishwasher age
         partial_modules    environmental attitudes (H15), energy-saving behaviour (G), knows the kWh price (E1), bill too
                            complicated (C5C) -- asked in some countries only; missing elsewhere (idea 7)
         region             the region itself (NUTS2 code, one categorical variable) (idea 3). It replaces version 4's regional
                            statistics (GDP, unemployment, poverty, density, ownership, old dwellings), which the B1p placebo showed
                            worked only as region labels.
  B2   B1 + the top 3 climate candidates;  B2p  the same with climate swapped between regions of a country (placebo)
  Models A / B / C as version 4 (A income only, B without income, C with income).

Variants, all on the C-B1 feature set chosen in the fold (model_set "V"):
  robust      training weights for borderline and likely mislabelled households (lib/identification/robust.py; idea 4)
  augmented   + households whose energy cost is missing, with an imputed spending side (idea 2): the bill model (below) trained on
              the fold's training households gives each of them P(above the line); each enters twice -- "above" with weight
              0.5 x P and "not above" with weight 0.5 x (1 - P). They are training households only; scoring is on households with
              an observed bill, so results stay comparable.
  bill        "predict the bill, then classify" (lib/identification/bill_model.py; idea 1)
  tuned       XGBoost with Optuna-tuned hyper-parameters (lib/identification/tuning.py; idea 6)
Ensembles (idea 5), targeting curves (idea 8) and "not sure" (idea 9) are computed afterwards from the stored probabilities
(scripts/build_feature_stages_v5.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from lib.features.partial_modules import PARTIAL
from .bill_model import BillModel, four_class
from .lines_v5 import fold_threshold
from .models3 import MODELS3, fit as fit_model
from .robust import robust_weight
from .stages import B0_FEATURES
from .stages_v4 import (B2_CANDIDATES, B2_GRID, L2_GRID, TOP_CLIMATE, _curve, one_se_choice, outer_splits, paired_gate, placebo_frame,
                        rank_candidates, select_b2, without_income)
from .tuning import tune_xgboost

B1_GROUPS = {"household_indices": ["dwelling_efficiency_index", "inefficient_equipment_index", "heating_system_type",
                                   "vulnerability_index", "wealth_index", "income_eur_equiv"],
             "enable_top3": ["S8", "floor_area_per_person", "H11B"],
             "partial_modules": list(PARTIAL),
             "region": ["region"]}
B1_CANDIDATES = [f for g in B1_GROUPS.values() for f in g]
BASE = B0_FEATURES + ["country"]
UNIVERSE = BASE + B1_CANDIDATES + B2_CANDIDATES
STAGES = ("B0", "B1", "B2", "B2p")
VARIANTS = (("robust", "catboost"), ("robust", "xgboost"), ("robust", "ft_transformer"), ("augmented", "catboost"),
            ("augmented", "xgboost"), ("augmented", "ft_transformer"), ("bill", "xgboost_regression"), ("tuned", "xgboost"))
IMPUTED_WEIGHT = 0.5
SEED = 20260811
__all__ = ["B1_GROUPS", "B1_CANDIDATES", "BASE", "UNIVERSE", "STAGES", "VARIANTS", "outer_splits", "run_fold", "select_b1"]


def select_b1(train: pd.DataFrame, seed: int) -> dict:
    """Two-level selection (version 4's select_b1 with the version-5 groups)."""
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
    model_sets = {("A", "all"): ["income_bracket"]}
    for st in ("B0", "B1", "B2"):
        model_sets[("B", st)] = without_income(stage_sets[st])
    for st in STAGES:
        model_sets[("C", st)] = stage_sets[st]
    proba, imp_rows, shap = {}, [], {}
    col = {f: j for j, f in enumerate(UNIVERSE)}
    for (mset, st), feats in model_sets.items():
        src = fp if st == "B2p" else frame
        for clf in MODELS3:
            m = fit_model(clf, src.iloc[tr], feats, seed)
            proba[(mset, st, clf)] = m.predict_proba(src.iloc[te])
            imp_rows += [{"model_set": mset, "stage": st, "model": clf, "feature": f, "importance": float(v)}
                         for f, v in m.importance(src.iloc[te]).items()]
            if clf == "catboost" and mset == "C" and st in ("B1", "B2"):
                block = np.zeros((len(te), 4, len(UNIVERSE)), dtype=np.float32)
                block[:, :, [col[f] for f in feats]] = m.shap_values(src.iloc[te])
                shap[st] = block

    # ---- version-5 variants on the C-B1 feature set ----
    feats = s1["selected"]
    info = {}
    w, info["robust"] = robust_weight(train, feats, seed + 7)
    for clf in MODELS3:
        proba[("V", "robust", clf)] = fit_model(clf, train, feats, seed, weight=w).predict_proba(test)
    bill = BillModel(pd.concat([train, cost_only], ignore_index=True), feats, seed + 11)
    aug, wa = augmented_training(train, unl, bill, spec)
    info["augmented"] = {"added_households": int((len(aug) - len(train)) // 2), "train_households": len(train)}
    for clf in MODELS3:
        proba[("V", "augmented", clf)] = fit_model(clf, aug, feats, seed, weight=wa).predict_proba(test)
    thr_te = fold_threshold(train, test, spec["family"], spec["tau"])
    p_hi = bill.p_above(test, thr_te)
    proba[("V", "bill", "xgboost_regression")] = four_class(test.li.to_numpy(), p_hi, _prior(train.y))
    resid = np.log(test.cost.clip(lower=1).to_numpy()) - bill.mu(test)
    info["bill"] = {"rmse_log_cost": float(np.sqrt(np.mean(resid ** 2))),
                    "r2_log_cost": float(1 - np.mean(resid ** 2) / np.var(np.log(test.cost.clip(lower=1))))}
    params, inner = tune_xgboost(train, feats, seed + 13)
    proba[("V", "tuned", "xgboost")] = fit_model("xgboost", train, feats, seed, params=params).predict_proba(test)
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
