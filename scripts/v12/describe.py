"""Descriptive results that need no model: quadrant shares, hardship gap (validity) with bootstrap CI, between-region share of
high cost (eta^2; a model-free measure of how much the region tells about high cost), and threshold summaries."""
import sys; sys.path.insert(0, '.')
import numpy as np, pandas as pd
from common import *
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"
T = "/home/claude/work/v12/tables/"
L = pd.read_pickle(S + "labels.pkl")
rows, gap, eta = [], [], []
rng = np.random.default_rng(0)
for r, e in L.items():
    for c, g in list(e.groupby("country")) + [("All five", e)]:
        sh = g.y.map(QUAD).value_counts(normalize=True)
        rows.append({"R": r, "country": c, "n": len(g), **{q: sh.get(q, 0) for q in QUAD.values()},
                     "high_cost_share": g.hc.mean(), "threshold_median_eur": g.thr.median()})
        lo = g[g.li == 1].dropna(subset=["S8"])
        hard = (lo.S8 >= 3).astype(float).to_numpy(); hc = lo.hc.to_numpy()
        def gp(i): h, k = hard[i], hc[i]; return h[k == 1].mean() - h[k == 0].mean()
        bs = [gp(rng.integers(0, len(lo), len(lo))) for _ in range(1000)]
        gap.append({"R": r, "country": c, "gap": gp(np.arange(len(lo))), "lo": np.nanpercentile(bs, 2.5), "hi": np.nanpercentile(bs, 97.5),
                    "n_low_income": len(lo), "n_LIHC": int(hc.sum())})
        if c != "All five":
            # eta^2 of the high-cost indicator across regions (NUTS2) within the country, and the same corrected for chance
            m = g.groupby("region").hc.agg(["mean", "size"])
            ssb = (m["size"] * (m["mean"] - g.hc.mean()) ** 2).sum(); sst = ((g.hc - g.hc.mean()) ** 2).sum()
            perm = []
            for _ in range(500):
                h = rng.permutation(g.hc.to_numpy()); mm = pd.Series(h).groupby(g.region.to_numpy()).agg(["mean", "size"])
                perm.append((mm["size"] * (mm["mean"] - h.mean()) ** 2).sum() / sst)
            eta.append({"R": r, "country": c, "regions": g.region.nunique(), "eta2": ssb / sst, "eta2_chance": np.mean(perm),
                        "eta2_excess": ssb / sst - np.mean(perm), "p": np.mean(np.array(perm) >= ssb / sst)})
pd.DataFrame(rows).to_csv(T + "quadrant_shares.csv", index=False)
pd.DataFrame(gap).to_csv(T + "hardship_gap.csv", index=False)
pd.DataFrame(eta).to_csv(T + "region_eta2.csv", index=False)
print(pd.DataFrame(gap).pivot(index="country", columns="R", values="gap").round(3))
print(pd.DataFrame(eta).pivot(index="country", columns="R", values="eta2_excess").round(3))
print(pd.DataFrame(eta).pivot(index="country", columns="R", values="p").round(3))
