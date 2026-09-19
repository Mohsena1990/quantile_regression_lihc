"""Policy-anchored identification target: self-reported hardship (S8 >= 3, "finding it difficult /
very difficult on present income"), the survey's own measure of a household struggling to cope.

Why: scripts/build_identification_benchmark.py's external validation shows hardship tracks the
INCOME axis (0.65-0.72 among low-income households vs 0.24-0.27 otherwise) while the cost axis
(spending relative to physical need) adds nothing detectable. A hardship target has no such
construct-validity gap. S8 is the target here, so it is excluded from every predictor set, and
S7 is excluded too (bill-aid receipt is itself a targeting outcome).

Predictor sets: T1 structural | T2 + socio-economic | T2 + income percentile (the deployable case
where a benefits/tax register supplies income; the project's income-excluded design is the
proxy-targeting case). Repeated 5-fold CV ROC-AUC, development partitions only.

Output: outputs/tables/table_hardship_identification.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES, load_development_frame, tier_features
from lib.identification.protocol import binary_axis_auc_cv


def main() -> None:
    dev = load_development_frame()
    dev = dev[dev.S8.notna()].copy()
    dev["hard"] = (dev.S8 >= 3).astype(int)
    t = tier_features()
    strip = {"S7", "S8"}
    sets = {"T1 structural": t["T1"], "T2 + socio-economic": [f for f in t["T2"] if f not in strip],
            "T2 + income percentile": [f for f in t["T2"] if f not in strip] + ["income_percentile"]}
    rows = []
    for scope in COUNTRIES + ["Pooled"]:
        g = dev if scope == "Pooled" else dev[dev.country == scope]
        for name, feats in sets.items():
            gg = g.dropna(subset=["income_percentile"]) if "income" in name else g
            feats = feats + (["country"] if scope == "Pooled" else [])
            rows.append({"scope": scope, "predictors": name, "n": len(gg), "hardship_rate": gg.hard.mean(),
                         "auc": binary_axis_auc_cv(gg, feats, "hard")})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "outputs/tables/table_hardship_identification.csv", index=False)
    print(out.pivot_table(index="predictors", columns="scope", values="auc")[COUNTRIES + ["Pooled"]].round(3).to_string())
    print(out.drop_duplicates("scope").set_index("scope")[["n", "hardship_rate"]].round(3).T.to_string())


if __name__ == "__main__":
    main()
