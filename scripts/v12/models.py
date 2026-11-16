"""Three classifiers: multinomial logistic regression, CatBoost, XGBoost. Balanced class weights in all three."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from common import NOMINAL

MODELS = ("logit", "catboost", "xgboost")


def _split(frame, fs):
    cat = [f for f in fs if f in NOMINAL or not pd.api.types.is_numeric_dtype(frame[f])]
    return cat, [f for f in fs if f not in cat]


def _cats(x, cat):
    x = x.copy()
    for c in cat:
        x[c] = x[c].astype("object").where(x[c].notna(), "NA").astype(str)
    return x


class Logit:
    def __init__(self, train, fs, seed, weight=None):
        self.fs = fs
        cat, num = _split(train, fs)
        self.cat = cat
        tr = [("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)]
        if num:
            tr.append(("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num))
        self.m = make_pipeline(ColumnTransformer(tr),
                               LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000))
        y = train.y.astype(int).to_numpy()
        self.m.fit(_cats(train[fs], cat), y, logisticregression__sample_weight=weight)
        self.classes = self.m.classes_

    def predict_proba(self, frame):
        out = np.zeros((len(frame), 4))
        out[:, self.classes] = self.m.predict_proba(_cats(frame[self.fs], self.cat))
        return out


class CatB:
    def __init__(self, train, fs, seed, weight=None):
        self.fs = fs
        self.cat, num = _split(train, fs)
        self.med = train[num].median()
        self.m = CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=10, loss_function="MultiClass",
                                    auto_class_weights="Balanced" if weight is None else None, verbose=False,
                                    random_seed=seed, thread_count=1)
        self.m.fit(self._x(train), train.y.astype(int), cat_features=self.cat, sample_weight=weight)
        self.classes = np.asarray(self.m.classes_, int)

    def _x(self, frame):
        x = _cats(frame[self.fs], self.cat)
        num = [f for f in self.fs if f not in self.cat]
        x[num] = x[num].apply(pd.to_numeric, errors="coerce").fillna(self.med)
        return x

    def predict_proba(self, frame):
        out = np.zeros((len(frame), 4))
        out[:, self.classes] = self.m.predict_proba(self._x(frame))
        return out

    def shap(self, frame):
        sv = np.asarray(self.m.get_feature_importance(Pool(self._x(frame), cat_features=self.cat), type="ShapValues"))[:, :, :-1]
        out = np.zeros((len(frame), 4, len(self.fs)), dtype=np.float32)
        out[:, self.classes, :] = sv
        return out


class XGB:
    def __init__(self, train, fs, seed, weight=None):
        self.fs = fs
        self.cat, _ = _split(train, fs)
        self.levels = {c: sorted(_cats(train[[c]], [c])[c].unique()) for c in self.cat}
        y = train.y.astype(int).to_numpy()
        self.present = np.unique(y)
        code = {c: i for i, c in enumerate(self.present)}
        cnt = np.bincount(y, minlength=4)
        w = len(y) / (len(self.present) * cnt[y]) if weight is None else np.asarray(weight, float)
        self.m = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                                   tree_method="hist", enable_categorical=True, max_cat_to_onehot=1, random_state=seed,
                                   n_jobs=1, verbosity=0)
        self.m.fit(self._x(train), np.vectorize(code.get)(y), sample_weight=w)

    def _x(self, frame):
        x = _cats(frame[self.fs], self.cat)
        for c in self.fs:
            x[c] = pd.Categorical(x[c], categories=self.levels[c]) if c in self.cat else pd.to_numeric(x[c], errors="coerce")
        return x

    def predict_proba(self, frame):
        out = np.zeros((len(frame), 4))
        out[:, self.present] = self.m.predict_proba(self._x(frame))
        return out


def fit(kind, train, fs, seed, weight=None):
    return {"logit": Logit, "catboost": CatB, "xgboost": XGB}[kind](train, fs, seed, weight)
