import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd
from common import *
from metrics import _ba, paired_diff
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; T = "/home/claude/work/v12/tables/"
L = pd.read_pickle(S + "labels.pkl"); sh = pickle.load(open(S + "v12/shapley.pkl", "rb"))
P = {}
for o in sh:
    for key, p in o["proba"].items():
        P.setdefault((o["r"],) + key, np.zeros((len(L[o["r"]]), 4)))[o["te"]] = p
rows = []
for r, e in L.items():
    y = e.y.to_numpy()
    for m in ("catboost", "logit"):
        subs = [k[1] for k in P if k[0] == r and k[2] == m and "I" in k[1]]
        for c in COUNTRIES + ["All five"]:
            ix = np.arange(len(e)) if c == "All five" else np.where(e.country == c)[0]
            v = {s: _ba(y[ix], P[(r, s, m)][ix].argmax(1)) for s in subs}
            best = max(v, key=v.get); full = tuple(ORDER); inc = ("I",)
            d, lo, hi = paired_diff(y[ix], P[(r, best, m)][ix].argmax(1), P[(r, full, m)][ix].argmax(1), _ba, B=400, seed=5)
            rows.append({"R": r, "model": m, "country": c, "ba_income": v[inc], "ba_full": v[full], "best_set": "+".join(best),
                         "ba_best": v[best], "best_minus_full": d, "lo": lo, "hi": hi, "full_rank": sorted(v.values(), reverse=True).index(v[full]) + 1,
                         "n_sets": len(v)})
D = pd.DataFrame(rows); D.to_csv(T + "best_subset.csv", index=False)
pd.set_option("display.width", 250)
x = D[D.model == "catboost"]
print(x[["R", "country", "ba_income", "ba_full", "best_set", "ba_best", "best_minus_full", "lo", "hi", "full_rank"]].round(3).to_string(index=False))
