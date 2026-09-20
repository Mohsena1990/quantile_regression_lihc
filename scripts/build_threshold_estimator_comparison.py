"""How much does the choice of cost-threshold estimator matter?

Compares the three regression families that estimate a household's conditional high-cost threshold -- linear quantile
regression (M2), a spline-basis GAM (M5) and gradient-boosted quantile regression (M7), each at its own OOF-calibrated
quantile, feature set A1 -- on the VALIDATION partition (development data; the test partition is not touched):

  * regression accuracy against observed annual energy cost: MAE, MAPE, RMSE, R-squared, pinball loss, coverage error
  * label agreement: Cohen's kappa of the resulting high-cost flag between families, and against this study's headline
    need-based flag, per country (do different estimators put the same households above their threshold?)

Reads artifacts/predictions/m{2,5,7}_a1.parquet (run_model_measurements.py) and the OOF calibration table.
Outputs: outputs/tables/table_threshold_estimators.csv, table_threshold_label_agreement.csv
"""
from __future__ import annotations
import itertools, sys, warnings
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.evaluation.cost_metrics import cost_score
from lib.identification.data import load_development_frame
from lib.identification.labels import add_labels

FAMILIES = {"M2": "linear quantile regression", "M5": "GAM (spline + Gamma GLM)", "M7": "gradient-boosted quantile regression"}


def main() -> None:
    cal = pd.read_csv(ROOT / "outputs/tables/table_a1_oof_calibration.csv")
    qs = cal[cal.country.str.contains("overall")].set_index("family")["quantile"]   # bracket access: DataFrame.quantile is a method
    need = add_labels(load_development_frame(), cost_kind="need_adjusted")[["UID", "hc"]].rename(columns={"hc": "need_based_high_cost"})
    frames, rows = {}, []
    for fam in FAMILIES:
        p = pd.read_parquet(ROOT / f"artifacts/predictions/{fam.lower()}_a1.parquet")
        v = p[(p.partition == "validation") & p.measurement_eligible].copy()
        v["high_cost"] = v.high_cost.astype(int)
        frames[fam] = v.set_index("UID")
        q = float(qs[fam])
        for c, g in v.groupby("country"):
            rows.append({"family": fam, "method": FAMILIES[fam], "country": c, "n_validation": len(g), "quantile": q,
                         "high_cost_rate": g.high_cost.mean(), **cost_score(g, q)})
        rows.append({"family": fam, "method": FAMILIES[fam], "country": "All four", "n_validation": len(v), "quantile": q,
                     "high_cost_rate": v.high_cost.mean(), **cost_score(v, q)})
    est = pd.DataFrame(rows)
    est.to_csv(ROOT / "outputs/tables/table_threshold_estimators.csv", index=False)
    agree = []
    need = need.set_index("UID").need_based_high_cost
    for a, b in itertools.combinations(FAMILIES, 2):
        j = frames[a][["country", "high_cost"]].join(frames[b].high_cost.rename("hc_b"), how="inner")
        for c, g in list(j.groupby("country")) + [("All four", j)]:
            agree.append({"pair": f"{a} vs {b}", "country": c, "n": len(g), "kappa": cohen_kappa_score(g.high_cost, g.hc_b), "agreement": (g.high_cost == g.hc_b).mean()})
    for a in FAMILIES:
        j = frames[a][["country", "high_cost"]].join(need, how="inner")
        for c, g in list(j.groupby("country")) + [("All four", j)]:
            agree.append({"pair": f"{a} vs need-based (headline)", "country": c, "n": len(g), "kappa": cohen_kappa_score(g.high_cost, g.need_based_high_cost), "agreement": (g.high_cost == g.need_based_high_cost).mean()})
    ag = pd.DataFrame(agree)
    ag.to_csv(ROOT / "outputs/tables/table_threshold_label_agreement.csv", index=False)
    pd.set_option("display.width", 200)
    print(est[est.country == "All four"][["family", "quantile", "n_validation", "mae", "rmse", "r2", "pinball", "empirical_coverage_error", "high_cost_rate"]].round(3).to_string(index=False))
    print(ag[ag.country == "All four"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
