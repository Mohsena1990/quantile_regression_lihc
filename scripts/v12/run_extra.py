"""(a) Shapley shares of the six blocks (all 64 subsets; logit + CatBoost; one repeat of 5-fold CV)
(b) class-imbalance handling (CatBoost, full set): balanced weights / no weights / random over- / under-sampling
(c) country-specific training (CatBoost + logit, full set, each country alone)
(d) need lines at tau = 0.80 (policy-order sets, CatBoost + logit)"""
from __future__ import annotations
import sys, time, pickle, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from common import feats, ORDER, labels, COUNTRIES
from models import fit

S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"
OUT = Path(S) / "v12"
L = pd.read_pickle(S + "labels.pkl")
SUBSETS = [tuple(b for b in ORDER if b in c) for k in range(0, 7) for c in itertools.combinations(ORDER, k)]


def shapley_task(r, k, tr, te):
    e = L[r]; res = {"r": r, "k": k, "te": te, "proba": {}}
    for sub in SUBSETS:
        fs = feats(sub)
        for m in ("logit", "catboost"):
            res["proba"][(sub, m)] = fit(m, e.iloc[tr], fs, 11 + k).predict_proba(e.iloc[te]).astype(np.float32)
    return res


def resample(train, how, seed):
    rng = np.random.default_rng(seed)
    g = [train.index[train.y == c].to_numpy() for c in range(4)]
    g = [x for x in g if len(x)]
    n = max(map(len, g)) if how == "over" else min(map(len, g))
    idx = np.concatenate([rng.choice(x, n, replace=len(x) < n or how == "over") for x in g])
    return train.loc[idx].reset_index(drop=True)


def imbalance_task(r, rep, k, tr, te):
    e = L[r]; fs = feats(tuple(ORDER)); res = {"r": r, "rep": rep, "k": k, "te": te, "proba": {}}
    train = e.iloc[tr]
    res["proba"]["no_weights"] = fit("catboost", train, fs, 5, weight=np.ones(len(train))).predict_proba(e.iloc[te])
    res["proba"]["oversample"] = fit("catboost", resample(train, "over", k), fs, 5, weight=None).predict_proba(e.iloc[te])
    res["proba"]["undersample"] = fit("catboost", resample(train, "under", k), fs, 5, weight=None).predict_proba(e.iloc[te])
    return res


def country_task(r, c, rep, k, tr, te):
    e = L[r]; g = e[e.country == c].reset_index()
    fs = [f for f in feats(tuple(ORDER)) if f != "country"]
    res = {"r": r, "c": c, "rep": rep, "k": k, "te": g["index"].to_numpy()[te], "proba": {}}
    for m in ("logit", "catboost"):
        res["proba"][m] = fit(m, g.iloc[tr], fs, 3 + k).predict_proba(g.iloc[te])
    return res


def tau_task(r, e, rep, k, tr, te):
    res = {"r": r, "rep": rep, "k": k, "te": te, "proba": {}}
    for j in range(1, 7):
        sub = tuple(ORDER[:j])
        for m in ("logit", "catboost"):
            res["proba"][(sub, m)] = fit(m, e.iloc[tr], feats(sub), 9 + k).predict_proba(e.iloc[te])
    return res


def splits(e, reps=3, seed=1):
    strata = e.country + "_" + e.y.astype(str)
    return list(RepeatedStratifiedKFold(n_splits=5, n_repeats=reps, random_state=seed).split(e, strata))


if __name__ == "__main__":
    t = time.time()
    jobs = [delayed(shapley_task)(r, i, tr, te) for r, e in L.items() for i, (tr, te) in enumerate(splits(e, 1))]
    pickle.dump(Parallel(n_jobs=2, verbose=2)(jobs), open(OUT / "shapley.pkl", "wb")); print("shapley", time.time() - t)
    jobs = [delayed(imbalance_task)(r, i // 5, i % 5, tr, te) for r, e in L.items() for i, (tr, te) in enumerate(splits(e))]
    pickle.dump(Parallel(n_jobs=2, verbose=2)(jobs), open(OUT / "imbalance.pkl", "wb")); print("imbalance", time.time() - t)
    jobs = []
    for r, e in L.items():
        for c in COUNTRIES:
            g = e[e.country == c].reset_index(drop=True)
            for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(g, g.y)):
                jobs.append(delayed(country_task)(r, c, i // 5, i % 5, tr, te))
    pickle.dump(Parallel(n_jobs=2, verbose=2)(jobs), open(OUT / "country.pkl", "wb")); print("country", time.time() - t)
    import sys as _s
    dev = pd.read_pickle(S + "dev.pkl")
    from lib.identification.lines import make_labels
    T = {}
    for r, fam in (("R2", "needqr_noclim"), ("R3", "needqr")):
        e = make_labels(dev, fam, 3, 0.80); T[r] = e[e.country.isin(COUNTRIES)].reset_index(drop=True)
    pd.to_pickle(T, OUT / "labels_tau80.pkl")
    jobs = [delayed(tau_task)(r, e, i // 5, i % 5, tr, te) for r, e in T.items() for i, (tr, te) in enumerate(splits(e))]
    pickle.dump(Parallel(n_jobs=2, verbose=2)(jobs), open(OUT / "tau80.pkl", "wb")); print("tau", time.time() - t)
