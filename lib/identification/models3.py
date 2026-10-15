"""The three classifiers of the feature-stage study, behind one interface: fit -> predict_proba(frame) and importance(frame).

  CatBoost        gradient-boosted oblivious trees with native categorical handling (Van Hove et al.'s library); depth 4,
                  learning rate 0.05, 100 trees, L2 10, balanced class weights. importance = mean |SHAP| (exact TreeSHAP).
  XGBoost         gradient-boosted trees (Chen & Guestrin 2016); depth 4, learning rate 0.05, 300 trees, row / column
                  subsampling 0.8, balanced sample weights, native categorical splits, missing values routed natively.
                  importance = mean |SHAP| (exact TreeSHAP, pred_contribs).
  FT-Transformer  Feature Tokenizer + Transformer (lib/identification/ft_transformer.py). importance = [CLS] attention.
SHAP and attention are averaged over the four classes and the households scored.
Version 5: every model accepts per-household training weights (`weight`, multiplied with the balanced class weights) for the
noisy-label and imputation variants, and XGBoost accepts tuned hyper-parameters (`params`, scripts via lib/identification/tuning.py).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import Pool
from .discovery import _model, _prep
from .ft_transformer import FTTransformerClassifier

MODELS3 = ("catboost", "xgboost", "ft_transformer")


def categorical(frame: pd.DataFrame, feats: list[str]) -> list[str]:
    return [f for f in feats if not pd.api.types.is_numeric_dtype(frame[f])]


class _CatBoost:
    def __init__(self, train, feats, seed, weight=None):
        xa, _, self.cat = _prep(train, train, feats)
        self.train, self.feats = train, feats
        self.m = _model("y", seed).fit(xa, train["y"].astype(int), cat_features=self.cat,
                                       sample_weight=None if weight is None else np.asarray(weight, float))

    def _x(self, frame):
        return _prep(self.train, frame, self.feats)[1]

    def predict_proba(self, frame):
        out = np.zeros((len(frame), 4))
        out[:, np.asarray(self.m.classes_, int)] = self.m.predict_proba(self._x(frame))
        return out

    def shap_values(self, frame):
        """(n, 4, n_features); classes absent from training stay 0."""
        sv = np.asarray(self.m.get_feature_importance(Pool(self._x(frame), cat_features=self.cat), type="ShapValues"))[:, :, :-1]
        out = np.zeros((len(frame), 4, sv.shape[2]))
        out[:, np.asarray(self.m.classes_, int), :] = sv
        return out

    def importance(self, frame):
        return pd.Series(np.abs(self.shap_values(frame)).mean(axis=(0, 1)), index=self.feats)


XGB_DEFAULT = {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}


class _XGBoost:
    def __init__(self, train, feats, seed, weight=None, params=None):
        self.feats, self.cat = feats, categorical(train, feats)
        self.levels = {c: sorted(train[c].dropna().astype(str).unique()) for c in self.cat}
        y = train["y"].astype(int).to_numpy()
        self.present = np.unique(y)
        code = {c: i for i, c in enumerate(self.present)}
        counts = np.bincount(y, minlength=4)
        w = (len(y) / (len(self.present) * counts[y]))
        if weight is not None:
            w = w * np.asarray(weight, float)
        self.m = xgb.XGBClassifier(**{**XGB_DEFAULT, **(params or {})}, tree_method="hist", enable_categorical=True,
                                   max_cat_to_onehot=1, random_state=seed, n_jobs=4, verbosity=0)
        self.m.fit(self._x(train), np.vectorize(code.get)(y), sample_weight=w)

    def _x(self, frame):
        x = frame[self.feats].copy()
        for c in self.feats:
            x[c] = (pd.Categorical(x[c].astype("object").where(x[c].notna(), None), categories=self.levels[c]) if c in self.cat
                    else pd.to_numeric(x[c], errors="coerce"))
        return x

    def predict_proba(self, frame):
        out = np.zeros((len(frame), 4))
        out[:, self.present] = self.m.predict_proba(self._x(frame))
        return out

    def importance(self, frame):
        contrib = self.m.get_booster().predict(xgb.DMatrix(self._x(frame), enable_categorical=True), pred_contribs=True)
        contrib = np.asarray(contrib)
        if contrib.ndim == 2:                                  # binary edge case: (n, F+1)
            contrib = contrib[:, None, :]
        return pd.Series(np.abs(contrib[:, :, :-1]).mean(axis=(0, 1)), index=self.feats)


class _FT:
    def __init__(self, train, feats, seed, val=None, weight=None):
        self.feats = feats
        self.m = FTTransformerClassifier(cat_features=categorical(train, feats), seed=seed)
        if val is None:
            self.m.fit(train[feats], train["y"].astype(int), sample_weight=weight)
        else:
            self.m.fit(train[feats], train["y"].astype(int), val[feats], val["y"].astype(int))

    def predict_proba(self, frame):
        return self.m.predict_proba(frame[self.feats])

    def importance(self, frame):
        return self.m.attention_importance(frame[self.feats])


def fit(kind: str, train: pd.DataFrame, feats: list[str], seed: int, val: pd.DataFrame | None = None, *,
        weight=None, params: dict | None = None):
    """`weight`: optional per-household training weights (aligned with `train`); `params`: XGBoost hyper-parameters."""
    if kind == "catboost":
        return _CatBoost(train, feats, seed, weight)
    if kind == "xgboost":
        return _XGBoost(train, feats, seed, weight, params)
    if kind == "ft_transformer":
        return _FT(train, feats, seed, val, weight)
    raise ValueError(kind)
