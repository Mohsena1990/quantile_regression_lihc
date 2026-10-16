"""Faithful replica of Van Hove, Dalla Longa & van der Zwaan (2022) and its thesis (van Hove 2020, TUM), applied to THIS
study's five countries (Bulgaria, Hungary, Italy, Poland, Spain): the validation stream of the study -- does this study's pipeline
reproduce the reference results when it is run their way?

Their pipeline, as published (paper Sec. 2-3, Table 1; thesis Sec. 5.2, 5.3, 7.4, 7.5), with one change -- the countries:
  * they pooled all 11 ENABLE countries in one model; here the same steps run on the five study countries
  * income bracket 1-10 from EITHER the monthly or the yearly income question (S9MONTH / S9YEAR); 98/99 invalid
  * energy expenditure = annual heating + annual electricity cost; heating = season total, else monthly cost x months
    paid, else monthly cost x the country's median months; electricity = annual bill, else monthly x 12; converted to
    EUR with the thesis' Table 5.1 rates ("first available exchange rate to euros in 2018")
  * labels: income bracket <= 3; expenditure above the country's 80th percentile
  * seven features with "don't know" codes kept as values (Table 1 counts them: floor area 7, house detachment 5,
    house age 9, heating strategy 5 values); household size = sum of the six S1 composition counts; birth year = S4
  * CatBoost, depth 4, learning rate 0.05, up to 500 trees, class weights = inverse class frequency of the training set,
    the iteration with the best validation TotalF1 kept (TotalF1 = support-weighted F1)
  * 20% stratified test split; the rest split 80/20 into training and validation
  * reported: per-class true-positive rates (recall) on test, 60-74%; best validation TotalF1 0.65 (5-fold CV 0.63)

`clean=True` swaps in this project's preprocessing (don't-know codes and impossible values as missing).
Households of the study's locked test partition (config/household_split_v1.csv) are excluded.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from data.enable import COUNTRIES, HOUSEHOLD_SHEET, STUDY_COUNTRY_NAMES

ROOT = Path(__file__).resolve().parents[2]
EUR_PER_LOCAL = {"Bulgaria": 0.511292, "France": 1, "Germany": 1, "Hungary": 0.003226, "Italy": 1, "Norway": 0.102712,
                 "Poland": 0.239406, "Serbia": 0.008446, "Spain": 1, "Ukraine": 0.029633, "United Kingdom": 1.124859}
COMP = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]
COST_SENTINELS = {98, 99, 999, 9999, 99999}
VH_FEATURES = ["income_bracket", "floor_area", "household_size", "house_detachment", "house_age", "birth_year", "heating_strategy"]
CLASS_KEYS = ("no_risk", "income_risk", "expenditure_risk", "double_risk")


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _valid_cost(s):
    v = _num(s)
    return v.where(v.ge(0) & ~v.isin(COST_SENTINELS))


def load_frame(*, clean: bool = False, countries: tuple[str, ...] = STUDY_COUNTRY_NAMES) -> pd.DataFrame:
    cols = ["UID", "Country", "S9MONTH", "S9YEAR", "H1", "H2", "H3", "H9", "S4", "H7A1", "H7A2", "H7AA", "H7A88", "H8A", "H8B"] + COMP
    d = pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx", sheet_name=HOUSEHOLD_SHEET, usecols=cols)
    d["country"] = d["Country"].map(COUNTRIES)
    d = d[d.country.isin(countries)]
    split = pd.read_csv(ROOT / "config/household_split_v1.csv", dtype={"UID": str})
    spent = set(split.UID[split.partition == "test"])
    d = d[~d.UID.astype(str).isin(spent)].copy()
    m, y = _num(d.S9MONTH), _num(d.S9YEAR)
    d["income_bracket"] = m.where(m.between(1, 10)).fillna(y.where(y.between(1, 10)))
    monthly, months, season = _valid_cost(d.H7A1), _num(d.H7AA).where(lambda s: s.between(1, 12)), _valid_cost(d.H7A2)
    med_months = months.groupby(d.country).transform("median")
    heat = season.fillna(monthly * months).fillna(monthly * med_months)
    heat = heat.where(_num(d.H7A88).ne(1), 0.0)                      # explicit "no heating system"
    elec = _valid_cost(d.H8B).fillna(_valid_cost(d.H8A) * 12)
    d["cost"] = (heat + elec) * d.country.map(EUR_PER_LOCAL)
    d = d[d.income_bracket.notna() & d.cost.notna()].copy()
    comp = d[COMP].apply(_num)
    d["household_size"] = comp.sum(axis=1, min_count=1)
    d["floor_area"], d["house_detachment"], d["house_age"] = _num(d.H3), _num(d.H1), _num(d.H2)
    d["birth_year"] = _num(d.S4)
    d["heating_strategy"] = _num(d.H9).astype("Int64").astype(str)
    if clean:
        d["floor_area"] = d.floor_area.where(d.floor_area.between(1, 6))
        d["house_detachment"] = d.house_detachment.where(d.house_detachment.between(1, 4))
        d["house_age"] = d.house_age.where(d.house_age.between(1, 8))
        d["birth_year"] = d.birth_year.where(d.birth_year.between(1900, 2005))
        d["household_size"] = d.household_size.where(d.household_size.between(1, 20))
    d["li"] = (d.income_bracket <= 3).astype(int)
    d["hc"] = (d.cost > d.groupby("country").cost.transform(lambda s: s.quantile(0.8))).astype(int)
    d["y"] = d.li + 2 * d.hc
    return d.reset_index(drop=True)


def _prep(frame: pd.DataFrame, feats: list[str]) -> tuple[pd.DataFrame, list[int]]:
    x = frame[feats].copy()
    cat = [i for i, f in enumerate(feats) if f in ("heating_strategy", "country")]
    for i in cat:
        x[feats[i]] = x[feats[i]].astype(str)
    return x, cat


CONFIGS = {
    # Van Hove: up to 500 trees, best validation TotalF1 iteration, inverse-frequency class weights, CatBoost default L2 = 3
    "vanhove": dict(iterations=500, depth=4, learning_rate=0.05, eval_metric="TotalF1", use_best_model=True),
    # this project's cb4 until now: 100 trees, strong L2, no validation-based stopping
    "project_cb4": dict(iterations=100, depth=4, learning_rate=0.05, l2_leaf_reg=10, use_best_model=False),
}


def fit_predict(train: pd.DataFrame, val: pd.DataFrame | None, test: pd.DataFrame, feats: list[str], config: dict, seed: int):
    xa, cat = _prep(train, feats)
    xt, _ = _prep(test, feats)
    counts = np.bincount(train.y, minlength=4).astype(float)
    weights = (counts.sum() / np.maximum(counts, 1) / 4).tolist()          # inverse class frequency
    cfg = dict(config)
    use_best = cfg.pop("use_best_model", False)
    model = CatBoostClassifier(loss_function="MultiClass", class_weights=weights, random_seed=seed, verbose=False,
                               thread_count=4, **cfg)
    if use_best and val is not None:
        xv, _ = _prep(val, feats)
        model.fit(xa, train.y, cat_features=cat, eval_set=(xv, val.y), use_best_model=True)
    else:
        model.fit(xa, train.y, cat_features=cat)
    proba = np.zeros((len(test), 4))
    proba[:, np.asarray(model.classes_, int)] = model.predict_proba(xt)
    return proba, model


def scores(y, proba) -> dict:
    pred = proba.argmax(1)
    rec = recall_score(y, pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    return {"weighted_f1": f1_score(y, pred, average="weighted", labels=[0, 1, 2, 3], zero_division=0),
            "macro_f1": f1_score(y, pred, average="macro", labels=[0, 1, 2, 3], zero_division=0),
            "balanced_accuracy": balanced_accuracy_score(y, pred), "accuracy": float((pred == y).mean()),
            "min_tpr": float(rec.min()), "max_tpr": float(rec.max()),
            **{f"tpr_{k}": float(r) for k, r in zip(CLASS_KEYS, rec)}}


def vanhove_protocol(frame: pd.DataFrame, feats: list[str], config: dict, seed: int) -> dict:
    """One run of their protocol: stratified 20% test, then 80/20 train/validation, fit, score the test set."""
    rest, test = train_test_split(frame, test_size=0.2, stratify=frame.y, random_state=seed)
    train, val = train_test_split(rest, test_size=0.2, stratify=rest.y, random_state=seed + 1)
    proba, model = fit_predict(train, val, test, feats, config, seed)
    out = scores(test.y.to_numpy(), proba)
    out["trees"] = int(model.tree_count_)
    return out


TUNING_GRID = [dict(depth=d, learning_rate=lr, l2_leaf_reg=l2) for d in (2, 4, 6) for lr in (0.03, 0.1) for l2 in (1, 3, 10)]


def tuned_protocol(frame: pd.DataFrame, feats: list[str], seed: int) -> dict:
    """Their protocol with hyperparameters searched: each grid point (depth x learning rate x L2) is fit on the training
    split with up to 1000 trees and validation early stopping (log-loss, 100 rounds); the grid point with the best
    VALIDATION balanced accuracy is kept and scored once on the test split."""
    rest, test = train_test_split(frame, test_size=0.2, stratify=frame.y, random_state=seed)
    train, val = train_test_split(rest, test_size=0.2, stratify=rest.y, random_state=seed + 1)
    best = None
    for g in TUNING_GRID:
        cfg = dict(iterations=1000, early_stopping_rounds=100, use_best_model=True, **g)
        pv, _ = fit_predict(train, val, val, feats, cfg, seed)
        score = balanced_accuracy_score(val.y, pv.argmax(1))
        if best is None or score > best[0]:
            best = (score, cfg)
    proba, model = fit_predict(train, val, test, feats, best[1], seed)
    out = scores(test.y.to_numpy(), proba)
    out.update({"trees": int(model.tree_count_), "depth": best[1]["depth"], "learning_rate": best[1]["learning_rate"],
                "l2_leaf_reg": best[1]["l2_leaf_reg"]})
    return out
