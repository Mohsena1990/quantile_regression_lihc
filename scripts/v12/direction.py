import sys; sys.path.insert(0, '.')
import pickle, numpy as np, pandas as pd
from common import *
S = "/tmp/claude-0/-home-claude/c08a9517-68ec-5ad8-aa50-30b1cf29b014/scratchpad/"; T = "/home/claude/work/v12/tables/"
L = pd.read_pickle(S + "labels.pkl"); SH = pickle.load(open(S + "v12/main_collected.pkl", "rb"))["SH"]
rows = []
VARS = {"primary_heating_fuel": None, "H1": {1: "detached", 2: "semi-detached/terraced", 3: "apartment", 4: "other"},
        "H3": {1: "<45 m2", 2: "45-64", 3: "65-90", 4: "91-120", 5: "121-200", 6: ">200 m2"}, "area_type": None}
for r, z in SH.items():
    v = z["v"].mean(0); f = z["f"]; e = L[r]
    for var, mp in VARS.items():
        j = f.index(var)
        vals = e[var].map(mp) if mp else e[var].astype(str)
        d = pd.DataFrame({"country": e.country, "value": vals, "shap_LIHC": v[:, 3, j], "lihc": (e.y == 3).astype(int)})
        g = d.groupby(["country", "value"]).agg(n=("shap_LIHC", "size"), mean_shap_LIHC=("shap_LIHC", "mean"), lihc_share=("lihc", "mean")).reset_index()
        g["R"] = r; g["variable"] = var; rows.append(g)
out = pd.concat(rows); out.to_csv(T + "direction_shap.csv", index=False)
x = out[(out.R == "R2") & (out.variable == "primary_heating_fuel") & (out.n >= 15)]
print(x.pivot(index="value", columns="country", values="mean_shap_LIHC").round(3))
print(x.pivot(index="value", columns="country", values="n"))
y = out[(out.R == "R0") & (out.variable == "H3") & (out.n >= 15)]
print(y.pivot(index="value", columns="country", values="mean_shap_LIHC").round(3))
