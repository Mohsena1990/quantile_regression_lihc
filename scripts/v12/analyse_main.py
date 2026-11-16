import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd
from common import *
from metrics import all_metrics, paired_diff, boot_ci, _ba, _f1m
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; T = "/home/claude/work/v12/tables/"
L = pd.read_pickle(S + "labels.pkl"); res = pickle.load(open(S + "v12/main.pkl", "rb"))
SETS = list(model_sets()); MODELS = ("logit", "catboost", "xgboost")
P, TRN, SH = {}, [], {}
for r in L:
    n = len(L[r])
    for key in [(s, m) for s in SETS for m in MODELS]:
        P[(r,) + key] = np.zeros((3, n, 4), np.float32)
for o in res:
    for key, p in o["proba"].items():
        P[(o["r"],) + key][o["rep"], o["te"]] = p
    for key, (ba, f1) in o["train"].items():
        TRN.append({"R": o["r"], "set": key[0], "model": key[1], "train_ba": ba, "train_f1": f1})
    if "shap" in o:
        SH.setdefault(o["r"], {"v": np.zeros((3, len(L[o["r"]]), 4, len(o["shap_feats"])), np.float32), "f": o["shap_feats"]})
        SH[o["r"]]["v"][o["rep"], o["te"]] = o["shap"]
pickle.dump({"P": P, "SH": SH}, open(S + "v12/main_collected.pkl", "wb"))
rows = []
for (r, s, m), p3 in P.items():
    e = L[r]; pm = p3.mean(0)
    for c in COUNTRIES + ["All five"]:
        idx = np.arange(len(e)) if c == "All five" else np.where(e.country == c)[0]
        y = e.y.to_numpy()[idx]
        d = all_metrics(y, pm[idx])
        d["ba_sd_repeats"] = np.std([_ba(y, p3[k][idx].argmax(1)) for k in range(3)])
        lo, hi = boot_ci(y, pm[idx].argmax(1)); d["ba_lo"], d["ba_hi"] = lo, hi
        rows.append({"R": r, "set": s, "model": m, "country": c, **d})
M = pd.DataFrame(rows)
tr = pd.DataFrame(TRN).groupby(["R", "set", "model"]).mean().reset_index()
M = M.merge(tr, on=["R", "set", "model"], how="left")
M.to_csv(T + "metrics_all.csv", index=False)
# paired comparisons (balanced accuracy and macro F1)
CMP = [("I+H", "I", "step: +Household"), ("I+H+D", "I+H", "step: +Dwelling"), ("I+H+D+S", "I+H+D", "step: +Heating system & fuel"),
       ("I+H+D+S+R", "I+H+D+S", "step: +Region"), ("I+H+D+S+R+C", "I+H+D+S+R", "step: +Climate"),
       ("I+H+D+S+R+C", "ALL_placebo", "climate vs placebo")] + \
      [("I+H+D+S+R+C", f"ALL-{b}", f"drop {BLOCK_NAME[b]}") for b in ["I", "H", "D", "S", "R"]] + [("ALL-R", "I+H+D+S", "climate when region unknown")]
crow = []
for r in L:
    e = L[r]; y0 = e.y.to_numpy()
    for m in MODELS:
        for a, b, lab in CMP:
            ya, yb = P[(r, a, m)].mean(0).argmax(1), P[(r, b, m)].mean(0).argmax(1)
            for c in COUNTRIES + ["All five"]:
                idx = np.arange(len(e)) if c == "All five" else np.where(e.country == c)[0]
                d, lo, hi = paired_diff(y0[idx], ya[idx], yb[idx], _ba, B=400, seed=1)
                f, flo, fhi = paired_diff(y0[idx], ya[idx], yb[idx], _f1m, B=400, seed=2)
                crow.append({"R": r, "model": m, "comparison": lab, "a": a, "b": b, "country": c, "d_ba": d, "lo": lo, "hi": hi,
                             "d_f1": f, "f1_lo": flo, "f1_hi": fhi, "sig": (lo > 0) or (hi < 0)})
pd.DataFrame(crow).to_csv(T + "comparisons.csv", index=False)
# SHAP: block importance by country (CatBoost, full model) and direction towards LIHC
brow, drow = [], []
for r, z in SH.items():
    v = z["v"].mean(0); f = z["f"]; e = L[r]
    absv = np.abs(v).mean(1)                   # (n, F) mean over classes
    for c in COUNTRIES + ["All five"]:
        idx = np.arange(len(e)) if c == "All five" else np.where(e.country == c)[0]
        imp = pd.Series(absv[idx].mean(0), index=f)
        tot = imp.sum()
        for b, fs in BLOCKS.items():
            brow.append({"R": r, "country": c, "block": b, "share": imp[[x for x in fs if x in imp]].sum() / tot})
        brow.append({"R": r, "country": c, "block": "country", "share": imp["country"] / tot})
        for j, x in enumerate(f):
            drow.append({"R": r, "country": c, "feature": x, "mean_abs_shap": imp[x], "mean_abs_shap_LIHC": np.abs(v[idx, 3, j]).mean()})
pd.DataFrame(brow).to_csv(T + "block_importance_shap.csv", index=False)
pd.DataFrame(drow).to_csv(T + "feature_importance_shap.csv", index=False)
print(M[(M.country == "All five")].pivot_table(index=["set"], columns=["R", "model"], values="balanced_accuracy").round(3).to_string())
