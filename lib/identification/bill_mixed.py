"""Idea A -- a multilevel bill model: tree boosting for the household, a random effect for the region (GPBoost; Sigrist 2022,
"Gaussian process boosting", JMLR 23).

Same use as lib/identification/bill_model.py ("predict the bill, then classify"), but the region enters as a random intercept
instead of a feature: each region's effect is estimated with partial pooling, so a region with few households is pulled towards
the overall level instead of being fitted to its handful of bills -- the classic remedy for small groups. Households without a
region (France, the UK) form one group per country; a region not seen in training gets no region effect.

  log(bill) = F(x) + u_region + e,   F = boosted trees (depth 4, learning rate 0.05, 300 rounds, column subsampling 0.8)
  P(above line) = 1 - Phi((log thr - mean) / sd), mean and sd from GPBoost's predictive distribution (it includes the residual
  variance and the uncertainty of the region effect).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm
from .models3 import categorical

PARAMS = {"learning_rate": 0.05, "max_depth": 4, "num_leaves": 15, "min_data_in_leaf": 20, "feature_fraction": 0.8,
          "verbose": -1}                         # GPBoost does not allow row bagging
ROUNDS = 300


def _groups(frame: pd.DataFrame) -> np.ndarray:
    return frame["climate_region"].astype("object").where(frame["climate_region"].notna(), "country_" + frame["country"].astype(str)).astype(str).to_numpy()


class MixedBillModel:
    def __init__(self, train: pd.DataFrame, feats: list[str], seed: int):
        import gpboost as gpb
        train = train[train.cost.notna()].reset_index(drop=True)
        self.feats = [f for f in feats if f != "region"]               # the region is the random effect, not a feature
        self.cat = categorical(train, self.feats)
        self.levels = {c: {v: i for i, v in enumerate(sorted(train[c].dropna().astype(str).unique()))} for c in self.cat}
        x, y = self._x(train), np.log(train.cost.clip(lower=1.0).to_numpy())
        gp = gpb.GPModel(group_data=_groups(train), likelihood="gaussian")
        data = gpb.Dataset(x, y, categorical_feature=[self.feats.index(c) for c in self.cat] or "auto")
        self.bst = gpb.train({**PARAMS, "seed": seed}, data, gp_model=gp, num_boost_round=ROUNDS)

    def _x(self, frame: pd.DataFrame) -> np.ndarray:
        cols = []
        for c in self.feats:
            if c in self.cat:
                cols.append(frame[c].astype("object").map(lambda v: self.levels[c].get(str(v), np.nan) if pd.notna(v) else np.nan).astype(float))
            else:
                cols.append(pd.to_numeric(frame[c], errors="coerce").astype(float))
        return np.column_stack(cols)

    def predict(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        p = self.bst.predict(data=self._x(frame), group_data_pred=_groups(frame), predict_var=True, pred_latent=False)
        return np.asarray(p["response_mean"]), np.asarray(p["response_var"])

    def mu(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict(frame)[0]

    def p_above(self, frame: pd.DataFrame, thr: np.ndarray) -> np.ndarray:
        m, v = self.predict(frame)
        return 1.0 - norm.cdf((np.log(np.clip(thr, 1.0, None)) - m) / np.sqrt(np.clip(v, 1e-6, None)))
