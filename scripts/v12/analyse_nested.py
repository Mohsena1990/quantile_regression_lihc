"""Honest (nested) best information set per country and threshold vs the full model and income only."""
import sys; sys.path.insert(0, '.')
import pickle, collections, numpy as np, pandas as pd
from common import *
from metrics import _ba, paired_diff
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; T = "/home/claude/work/v12/tables/"
L = pd.read_pickle(S + "labels.pkl"); nest = pickle.load(open(S + "v12/nested.pkl", "rb"))
sh = pickle.load(open(S + "v12/shapley.pkl", "rb"))
INC = {}
for o in sh:
    p = o["proba"].get((("I",), "catboost"))
    if p is not None:
        INC.setdefault(o["r"], np.zeros((len(L[o["r"]]), 4)))[o["te"]] = p
rows, blk = [], []
for r, e in L.items():
    y = e.y.to_numpy(); n = len(e)
    for c in COUNTRIES + ["All five"]:
        sel = np.zeros((n, 4)); full = np.zeros((n, 4)); picks = []
        for o in (o for o in nest if o["r"] == r):
            s = o["chosen"][c]; picks.append(s)
            sel[o["te"]] = o["proba"][s]; full[o["te"]] = o["proba"][tuple(ORDER)]
        ix = np.arange(n) if c == "All five" else np.where(e.country == c)[0]
        a, f, i = sel[ix].argmax(1), full[ix].argmax(1), INC[r][ix].argmax(1)
        d, lo, hi = paired_diff(y[ix], a, f, _ba, B=500, seed=3)
        di, li, hi_i = paired_diff(y[ix], a, i, _ba, B=500, seed=4)
        cnt = collections.Counter("+".join(p) for p in picks)
        rows.append({"R": r, "country": c, "ba_income": _ba(y[ix], i), "ba_full": _ba(y[ix], f), "ba_selected": _ba(y[ix], a),
                     "sel_minus_full": d, "lo": lo, "hi": hi, "sel_minus_income": di, "lo_i": li, "hi_i": hi_i,
                     "modal_set": cnt.most_common(1)[0][0], "modal_n": cnt.most_common(1)[0][1],
                     "sets": "; ".join(f"{k}×{v}" for k, v in cnt.most_common())})
        for b in ORDER:
            blk.append({"R": r, "country": c, "block": b, "share": np.mean([b in p for p in picks])})
D = pd.DataFrame(rows); D.to_csv(T + "nested_best.csv", index=False)
B = pd.DataFrame(blk).pivot_table(index=["R", "country"], columns="block", values="share")[list(ORDER)]
B.to_csv(T + "nested_block_freq.csv")
pd.set_option("display.width", 260)
print(D.drop(columns=["sets"]).round(3).to_string(index=False)); print(B.round(2).to_string())
print(D[["R", "country", "sets"]].to_string(index=False))
