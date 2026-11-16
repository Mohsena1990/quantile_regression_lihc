"""Nested selection of the best information set for each country and threshold (CatBoost).
Outer: 5-fold CV. Inner: 3-fold CV on the outer training part, all 32 sets that contain income; the set with the highest inner
balanced accuracy *in that country* is chosen, refitted on the outer training part and scored on the outer test part."""
import sys, time, pickle, itertools
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from common import feats, ORDER, COUNTRIES
from models import fit
from metrics import _ba
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"
L = pd.read_pickle(S + "labels.pkl")
SUBS = [tuple(b for b in ORDER if b in ("I",) + c) for k in range(0, 6) for c in itertools.combinations([b for b in ORDER if b != "I"], k)]

def outer_task(r, k, tr, te):
    e = L[r]; etr = e.iloc[tr].reset_index(drop=True)
    strata = etr.country + "_" + etr.y.astype(str)
    inner = {s: np.zeros((len(etr), 4)) for s in SUBS}
    for itr, ite in StratifiedKFold(3, shuffle=True, random_state=10 + k).split(etr, strata):
        for s in SUBS:
            inner[s][ite] = fit("catboost", etr.iloc[itr], feats(s), 21).predict_proba(etr.iloc[ite])
    y = etr.y.to_numpy(); chosen = {}
    for c in COUNTRIES + ["All five"]:
        ix = np.arange(len(etr)) if c == "All five" else np.where(etr.country == c)[0]
        sc = {s: _ba(y[ix], inner[s][ix].argmax(1)) for s in SUBS}
        chosen[c] = max(sc, key=sc.get)
    out = {}
    for s in set(chosen.values()) | {tuple(ORDER)}:
        out[s] = fit("catboost", etr, feats(s), 31).predict_proba(e.iloc[te])
    return {"r": r, "k": k, "te": te, "chosen": chosen, "proba": out}

if __name__ == "__main__":
    t = time.time(); jobs = []
    for r, e in L.items():
        strata = e.country + "_" + e.y.astype(str)
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=77).split(e, strata)):
            jobs.append(delayed(outer_task)(r, k, tr, te))
    res = Parallel(n_jobs=2, verbose=5)(jobs)
    pickle.dump(res, open(S + "v12/nested.pkl", "wb")); print("done", time.time() - t)
