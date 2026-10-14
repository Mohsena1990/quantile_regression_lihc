"""Version-4 feature stages: one pooled model for the four countries, read country by country; Models A / B / C.

Design (agreed 2026-10-14):
  * POOLED: one model trained on the households of all four countries with a country indicator (Dalla Longa et al. 2021 found
    ~1,000 training households are needed for stable results; one country has 380-650). Scores and SHAP are then read per
    country from that country's held-out households. Per-country B0 models are fitted alongside, as a check of the pooling.
  * Models A / B / C (Dalla Longa et al. 2021): A = income bracket only; B = the stage's features without income-derived ones;
    C = the stage's features with income.
  * B0  Van Hove's seven predictors (+ country)
  * B1  B0 + groups chosen in two levels: (1) which candidate groups, (2) which variables inside them. Groups:
          dalla_longa  population density, home-ownership share (region), wealth index (house-value stand-in)
          enable_top3  S8 income adequacy, floor area per person, dishwasher age (the three most selected before)
          indices      dwelling efficiency, inefficient equipment, heating system type, household vulnerability, income (EUR)
          regional     GDP per head, unemployment rate, poverty rate, share of dwellings built before 1981
  * B2  B1 + the TOP 3 climate candidates (lib/features/indices.py: 20 regional measures + 11 climate x dwelling features);
        the one-standard-error choice is recorded alongside
  * B2p placebo: regional climate permuted across regions within each country, household climate x dwelling features
        recomputed from the permuted values
  * B1p placebo (version 4b): the region-level B1 variables (population density, ownership share, GDP per head, unemployment,
        poverty rate, share of old dwellings) permuted across regions within each country before the B1 selection; only the
        gain of B1 over B1p can be credited to what those variables measure rather than to "which region is this"
  * "heating need per EUR of income" (need_burden) is not a climate candidate in version 4b: it divides by income and the B2p
        placebo showed its strength comes from income, not climate
Ranking everywhere: "Attention Mechanisms as Embedded Selectors + SHAP" with reliability weights (stages.rank_candidates);
sizing: one-standard-error rule + paired gate (stages.one_se_choice, stages.paired_gate). Selection never sees the households
that are scored.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lib.climate.b2_indices import B2_CLIMATE
from lib.features.indices import CLIMATE_CANDIDATES, INCOME_DERIVED, climate_features
from .models3 import MODELS3, fit as fit_model
from .stages import B0_FEATURES, _inner_folds, one_se_choice, paired_gate, rank_candidates

B1_GROUPS = {"dalla_longa": ["pop_density", "owner_share", "wealth_index"],
             "enable_top3": ["S8", "floor_area_per_person", "H11B"],
             "indices": ["dwelling_efficiency_index", "inefficient_equipment_index", "heating_system_type", "vulnerability_index",
                         "income_eur_equiv"],
             "regional": ["gdp_per_capita", "unemployment_rate", "poverty_rate", "pre1981_share"]}
B1_CANDIDATES = [f for g in B1_GROUPS.values() for f in g]
B2_CANDIDATES = [c for c in CLIMATE_CANDIDATES if c != "need_burden"]
REGIONAL_B1 = ["pop_density", "owner_share", "gdp_per_capita", "unemployment_rate", "poverty_rate", "pre1981_share"]
BASE = B0_FEATURES + ["country"]
UNIVERSE = BASE + B1_CANDIDATES + B2_CANDIDATES
CLIMATE_BASE = list(B2_CLIMATE)                      # regional columns the placebo permutes
TOP_CLIMATE = 3
L2_GRID = (1, 2, 3, 4, 6, 8, 12, 17)
B2_GRID = (0, 1, 2, 3, 5, 8, 12, 30)
STAGES = ("B0", "B1", "B1p", "B2", "B2p")
SEED = 20260811


def _curve(train: pd.DataFrame, sets: dict, seed: int) -> tuple[dict, dict]:
    """Inner repeated-CV balanced accuracy of CatBoost for each candidate feature set. Returns (curve, per-fold scores)."""
    y = train["y"].astype(int).to_numpy()
    folds = _inner_folds(train, seed)
    curve, per = {}, {}
    for key, feats in sets.items():
        s = [balanced_accuracy_score(y[te], fit_model("catboost", train.iloc[tr], feats, seed + i).predict_proba(train.iloc[te]).argmax(1))
             for i, (tr, te) in enumerate(folds)]
        per[key] = np.asarray(s)
        curve[key] = (float(np.mean(s)), float(np.std(s, ddof=1) / np.sqrt(len(s))))
    return curve, per


def select_b1(train: pd.DataFrame, seed: int) -> dict:
    """Two-level selection: groups first, then variables inside the chosen groups."""
    train = train.reset_index(drop=True)
    ranking = rank_candidates(train, BASE, B1_CANDIDATES, seed)
    group_score = {g: float(ranking.loc[f, "score"].mean()) for g, f in B1_GROUPS.items()}
    order = sorted(group_score, key=group_score.get)
    sets1 = {k: BASE + [f for g in order[:k] for f in B1_GROUPS[g]] for k in range(len(order) + 1)}
    curve1, per1 = _curve(train, sets1, seed)
    g_k = paired_gate(one_se_choice(curve1), per1)
    groups = order[:g_k]
    if not groups:
        return {"selected": BASE, "groups": [], "ranking": ranking, "group_score": group_score, "curve1": curve1, "curve2": {}}
    cands = [f for f in ranking.index if any(f in B1_GROUPS[g] for g in groups)]
    sets2 = {k: BASE + cands[:k] for k in sorted({min(k, len(cands)) for k in L2_GRID})}
    sets2[0] = BASE
    curve2, per2 = _curve(train, sets2, seed + 500)
    k2 = paired_gate(one_se_choice({k: v for k, v in curve2.items()}), per2)
    return {"selected": BASE + cands[:k2], "groups": groups, "ranking": ranking, "group_score": group_score,
            "curve1": curve1, "curve2": curve2}


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
    from .stages import placebo_climate
    return climate_features(placebo_climate(frame, CLIMATE_BASE, seed))


def regional_placebo_frame(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Region-level B1 variables moved to another region of the same country; climate x home features recomputed."""
    from .stages import placebo_climate
    return climate_features(placebo_climate(frame, REGIONAL_B1, seed))


def without_income(feats: list[str]) -> list[str]:
    return [f for f in feats if f not in INCOME_DERIVED]


def outer_splits(frame: pd.DataFrame, n_splits: int = 5, n_repeats: int = 3, seed: int = 1):
    strata = frame["country"].astype(str) + "_" + frame["y"].astype(str)
    return list(RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed).split(frame, strata))


def run_fold(frame: pd.DataFrame, tr: np.ndarray, te: np.ndarray, fold_id: int, repeat: int) -> dict:
    """Everything for one outer fold of the pooled design."""
    seed = SEED + fold_id
    train, test = frame.iloc[tr], frame.iloc[te]
    fp = placebo_frame(frame, SEED + 500 + repeat)
    fr = regional_placebo_frame(frame, SEED + 800 + repeat)
    s1 = select_b1(train, seed)
    s1p = select_b1(fr.iloc[tr], seed)
    s2 = select_b2(train, s1["selected"], seed + 1000)
    s2p = select_b2(fp.iloc[tr], s1["selected"], seed + 1000)
    stage_sets = {"B0": BASE, "B1": s1["selected"], "B1p": s1p["selected"], "B2": s2["selected"], "B2p": s2p["selected"]}
    model_sets = {("A", "all"): ["income_bracket"]}
    for st in ("B0", "B1", "B2"):
        model_sets[("B", st)] = without_income(stage_sets[st])
    for st in STAGES:
        model_sets[("C", st)] = stage_sets[st]
    proba, imp_rows, shap = {}, [], {}
    col = {f: j for j, f in enumerate(UNIVERSE)}
    for (mset, st), feats in model_sets.items():
        src = fp if st == "B2p" else fr if st == "B1p" else frame
        for clf in MODELS3:
            m = fit_model(clf, src.iloc[tr], feats, seed)
            proba[(mset, st, clf)] = m.predict_proba(src.iloc[te])
            imp_rows += [{"model_set": mset, "stage": st, "model": clf, "feature": f, "importance": float(v)}
                         for f, v in m.importance(src.iloc[te]).items()]
            if clf == "catboost" and mset == "C" and st in ("B1", "B2"):
                block = np.zeros((len(te), 4, len(UNIVERSE)), dtype=np.float32)
                block[:, :, [col[f] for f in feats]] = m.shap_values(src.iloc[te])
                shap[st] = block
    country_b0 = {}
    for c in frame.country.unique():
        trc, tec = tr[frame.country.iloc[tr].to_numpy() == c], te[frame.country.iloc[te].to_numpy() == c]
        for clf in MODELS3:
            m = fit_model(clf, frame.iloc[trc], B0_FEATURES, seed)
            country_b0[(c, clf)] = (tec, m.predict_proba(frame.iloc[tec]))
    sel = []
    for st, s in (("B1", s1), ("B1p", s1p), ("B2", s2), ("B2p", s2p)):
        for pos, (f, r) in enumerate(s["ranking"].iterrows(), start=1):
            sel.append({"stage": st, "feature": f, "selected": f in stage_sets[st], "rank": pos, "score": r.score,
                        "attention": r.attention, "shap_catboost": r.shap_catboost, "shap_xgboost": r.shap_xgboost,
                        "weight_attention": r.weight_attention, "weight_shap_catboost": r.weight_shap_catboost,
                        "weight_shap_xgboost": r.weight_shap_xgboost})
    curves = [{"stage": "B1_groups", "k": k, "score": m, "se": se} for k, (m, se) in s1["curve1"].items()]
    curves += [{"stage": "B1_variables", "k": k, "score": m, "se": se} for k, (m, se) in s1["curve2"].items()]
    curves += [{"stage": st, "k": k, "score": m, "se": se} for st, s in (("B2", s2), ("B2p", s2p)) for k, (m, se) in s["curve"].items()]
    meta = {"groups": s1["groups"], "group_score": s1["group_score"], "b1": s1["selected"], "b1p_groups": s1p["groups"], "b2_top3": s2["selected"][-TOP_CLIMATE:],
            "b2_k_one_se": s2["k_one_se"], "b2p_top3": s2p["selected"][-TOP_CLIMATE:], "b2p_k_one_se": s2p["k_one_se"]}
    return {"fold_id": fold_id, "repeat": repeat, "te": te, "proba": proba, "importance": imp_rows, "shap": shap,
            "country_b0": country_b0, "selection": sel, "curves": curves, "meta": meta}
