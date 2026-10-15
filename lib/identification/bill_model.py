"""Idea 1 -- "predict the bill, then classify".

The classifiers learn only "above the line: yes / no". The bill model instead learns each household's energy bill in EUR (the full
number), then asks how likely that bill is to be above the household's own line:

  log(bill) = mu(x) + error,   error ~ Normal(0, sigma_country)
  mu        XGBoost regression of log annual energy cost on the household's features (depth 4, learning rate 0.05, 300 trees,
            row / column subsampling 0.8; missing values and categories handled natively)
  sigma     standard deviation of the out-of-fold residuals (5-fold, training households only), per country
  P(high expenditure) = 1 - Phi((log thr - mu) / sigma)        thr from lines_v5.fold_threshold

The income axis needs no model: income is a predictor and "low income" is the income bracket <= cut-off, so it is known exactly.
The four-class probability puts P(high) on "low income x high spending" and 1 - P(high) on "low income x normal spending" (or the
high-income pair). Before taking the most likely class the probabilities are divided by the training class shares and
renormalised -- the same "balanced" decision the classifiers make through their class weights.

Training households: the labelled training households plus households that reported a bill but no income (lines_v5.cost_only) --
they inform the bill, not the label (idea 2, second option).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import norm
from sklearn.model_selection import KFold
from .models3 import categorical

PARAMS = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
              enable_categorical=True, max_cat_to_onehot=1, n_jobs=4, verbosity=0)


class _Encoder:
    def __init__(self, train: pd.DataFrame, feats: list[str]):
        self.feats, self.cat = feats, categorical(train, feats)
        self.levels = {c: sorted(train[c].dropna().astype(str).unique()) for c in self.cat}

    def __call__(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame[self.feats].copy()
        for c in self.feats:
            x[c] = (pd.Categorical(x[c].astype("object").where(x[c].notna(), None), categories=self.levels[c]) if c in self.cat
                    else pd.to_numeric(x[c], errors="coerce"))
        return x


class BillModel:
    def __init__(self, train: pd.DataFrame, feats: list[str], seed: int):
        train = train[train.cost.notna()].reset_index(drop=True)
        self.enc = _Encoder(train, feats)
        x, y = self.enc(train), np.log(train.cost.clip(lower=1.0).to_numpy())
        resid = np.full(len(y), np.nan)
        for tr, te in KFold(5, shuffle=True, random_state=seed).split(x):
            m = xgb.XGBRegressor(**PARAMS, random_state=seed).fit(x.iloc[tr], y[tr])
            resid[te] = y[te] - m.predict(x.iloc[te])
        pooled = float(np.std(resid, ddof=1))
        self.sigma = {c: float(np.std(resid[g], ddof=1)) if g.sum() >= 30 else pooled
                      for c, g in ((c, (train.country == c).to_numpy()) for c in train.country.unique())}
        self.pooled_sigma = pooled
        self.m = xgb.XGBRegressor(**PARAMS, random_state=seed).fit(x, y)

    def mu(self, frame: pd.DataFrame) -> np.ndarray:
        return self.m.predict(self.enc(frame))

    def p_above(self, frame: pd.DataFrame, thr: np.ndarray) -> np.ndarray:
        s = frame.country.map(self.sigma).fillna(self.pooled_sigma).to_numpy(float)
        return 1.0 - norm.cdf((np.log(np.clip(thr, 1.0, None)) - self.mu(frame)) / s)


def four_class(li: np.ndarray, p_high: np.ndarray, prior: np.ndarray | None = None) -> np.ndarray:
    """(n, 4) probabilities from the known income axis and P(high expenditure); `prior` (4,) applies the balanced decision."""
    li = np.asarray(li, int)
    out = np.zeros((len(li), 4))
    out[np.arange(len(li)), li] = 1 - p_high
    out[np.arange(len(li)), li + 2] = p_high
    if prior is not None:
        out = out / np.clip(prior, 1e-6, None)
        out = out / out.sum(axis=1, keepdims=True)
    return out
