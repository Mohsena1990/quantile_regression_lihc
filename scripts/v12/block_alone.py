"""Each block on its own next to income (from the 64-combination runs): I + block vs I."""
import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd
from common import *
from metrics import paired_diff, _ba, all_metrics
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; T = "/home/claude/work/v12/tables/"
L = pd.read_pickle(S + "labels.pkl"); sh = pickle.load(open(S + "v12/shapley.pkl", "rb"))
P = {}
for o in sh:
    for key, p in o["proba"].items():
        P.setdefault((o["r"],) + key, np.zeros((len(L[o["r"]]), 4)))[o["te"]] = p
rows = []
for r, e in L.items():
    y = e.y.to_numpy()
    for m in ("logit", "catboost"):
        base = P[(r, ("I",), m)].argmax(1)
        for b in ["H", "D", "S", "R", "C"]:
            sub = tuple(x for x in ORDER if x in ("I", b))
            yh = P[(r, sub, m)].argmax(1)
            for c in COUNTRIES + ["All five"]:
                ix = np.arange(len(e)) if c == "All five" else np.where(e.country == c)[0]
                d, lo, hi = paired_diff(y[ix], yh[ix], base[ix], _ba, B=400, seed=3)
                rows.append({"R": r, "model": m, "block": b, "country": c, "ba_income": _ba(y[ix], base[ix]), "ba_income_plus_block": _ba(y[ix], yh[ix]),
                             "gain": d, "lo": lo, "hi": hi, "sig": (lo > 0) or (hi < 0)})
D = pd.DataFrame(rows); D.to_csv(T + "block_alone.csv", index=False)
x = D[D.model == "catboost"]; x["s"] = x.gain.round(3).astype(str) + x.sig.map({True: "*", False: ""})
pd.set_option("display.width", 250)
print(x.pivot_table(index=["block", "country"], columns="R", values="s", aggfunc="first").to_string())
