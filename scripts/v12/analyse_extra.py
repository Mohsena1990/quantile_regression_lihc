import sys; sys.path.insert(0, '.')
import pickle, itertools, math, numpy as np, pandas as pd
from common import *
from metrics import all_metrics, _ba
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; T = "/home/claude/work/v12/tables/"
L = pd.read_pickle(S + "labels.pkl")
def idx_of(e, c): return np.arange(len(e)) if c == "All five" else np.where(e.country == c)[0]
# ---- Shapley
sh = pickle.load(open(S + "v12/shapley.pkl", "rb"))
P = {}
for o in sh:
    for key, p in o["proba"].items():
        P.setdefault((o["r"],) + key, np.zeros((len(L[o["r"]]), 4)))[o["te"]] = p
rows = []
for r, e in L.items():
    y = e.y.to_numpy()
    for m in ("logit", "catboost"):
        for c in COUNTRIES + ["All five"]:
            ix = idx_of(e, c)
            v = {sub: _ba(y[ix], P[(r, sub, m)][ix].argmax(1)) for sub in {k[1] for k in P if k[0] == r and k[2] == m}}
            n = 6
            for b in ORDER:
                phi = 0.0
                others = [x for x in ORDER if x != b]
                for k in range(0, 6):
                    for comb in itertools.combinations(others, k):
                        with_b = tuple(x for x in ORDER if x in comb or x == b); wo = tuple(x for x in ORDER if x in comb)
                        phi += math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n) * (v[with_b] - v[wo])
                rows.append({"R": r, "model": m, "country": c, "block": b, "shapley_ba": phi, "ba_base": v[()], "ba_full": v[tuple(ORDER)]})
pd.DataFrame(rows).to_csv(T + "shapley_blocks.csv", index=False)
# ---- imbalance
im = pickle.load(open(S + "v12/imbalance.pkl", "rb"))
coll = pickle.load(open(S + "v12/main_collected.pkl", "rb"))["P"]
Q = {}
for o in im:
    for key, p in o["proba"].items():
        Q.setdefault((o["r"], key), np.zeros((3, len(L[o["r"]]), 4)))[o["rep"], o["te"]] = p
rows = []
for r, e in L.items():
    Q[(r, "balanced_weights")] = coll[(r, "I+H+D+S+R+C", "catboost")]
    for k in ("balanced_weights", "no_weights", "oversample", "undersample"):
        pm = Q[(r, k)].mean(0)
        for c in COUNTRIES + ["All five"]:
            ix = idx_of(e, c); rows.append({"R": r, "handling": k, "country": c, **all_metrics(e.y.to_numpy()[ix], pm[ix])})
pd.DataFrame(rows).to_csv(T + "imbalance.csv", index=False)
# ---- country-specific training
ct = pickle.load(open(S + "v12/country.pkl", "rb"))
Q = {}
for o in ct:
    for m, p in o["proba"].items():
        Q.setdefault((o["r"], m), np.full((3, len(L[o["r"]]), 4), np.nan))[o["rep"], o["te"]] = p
rows = []
for (r, m), p3 in Q.items():
    e = L[r]; pm = np.nanmean(p3, 0)
    for c in COUNTRIES:
        ix = idx_of(e, c); rows.append({"R": r, "model": m, "country": c, "training": "country only", **all_metrics(e.y.to_numpy()[ix], pm[ix])})
pd.DataFrame(rows).to_csv(T + "country_training.csv", index=False)
# ---- tau 0.80 need lines
tl = pd.read_pickle(S + "v12/labels_tau80.pkl"); tt = pickle.load(open(S + "v12/tau80.pkl", "rb"))
Q = {}
for o in tt:
    for key, p in o["proba"].items():
        Q.setdefault((o["r"],) + key, np.zeros((3, len(tl[o["r"]]), 4)))[o["rep"], o["te"]] = p
rows = []
for (r, sub, m), p3 in Q.items():
    e = tl[r]; pm = p3.mean(0)
    for c in COUNTRIES + ["All five"]:
        ix = idx_of(e, c); rows.append({"R": r, "set": "+".join(sub), "model": m, "country": c, **all_metrics(e.y.to_numpy()[ix], pm[ix])})
pd.DataFrame(rows).to_csv(T + "tau80.csv", index=False)
print("ok")
