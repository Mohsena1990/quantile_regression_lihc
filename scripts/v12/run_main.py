"""Main experiment: R0-R3 x 13 information sets x 3 models, repeated stratified 5-fold CV (3 repeats), out-of-fold probabilities
plus training-fold scores (overfitting gap) and CatBoost SHAP of the full model."""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score
from common import feats, model_sets, placebo
from models import MODELS, fit

S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"
OUT = Path(S) / "v12"; OUT.mkdir(exist_ok=True)
L = pd.read_pickle(S + "labels.pkl")
SETS = model_sets()


def task(r, rep, k, tr, te):
    e = L[r]
    fp = placebo(e, 1000 + rep)
    res = {"r": r, "rep": rep, "k": k, "te": te, "proba": {}, "train": {}}
    for name, blocks in SETS.items():
        src = fp if name == "ALL_placebo" else e
        fs = feats(blocks)
        for m in MODELS:
            mod = fit(m, src.iloc[tr], fs, 7 + 100 * rep + k)
            res["proba"][(name, m)] = mod.predict_proba(src.iloc[te]).astype(np.float32)
            ptr = mod.predict_proba(src.iloc[tr]).argmax(1)
            ytr = src.y.iloc[tr].to_numpy()
            res["train"][(name, m)] = (balanced_accuracy_score(ytr, ptr), f1_score(ytr, ptr, average="macro"))
            if m == "catboost" and name == "+".join(["I", "H", "D", "S", "R", "C"]):
                res["shap"] = mod.shap(src.iloc[te])
                res["shap_feats"] = fs
    return res


if __name__ == "__main__":
    jobs = []
    for r, e in L.items():
        strata = e.country + "_" + e.y.astype(str)
        for i, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1).split(e, strata)):
            jobs.append((r, i // 5, i % 5, tr, te))
    t = time.time()
    out = Parallel(n_jobs=2, verbose=5)(delayed(task)(*j) for j in jobs)
    pickle.dump(out, open(OUT / "main.pkl", "wb"))
    print("done", time.time() - t)
