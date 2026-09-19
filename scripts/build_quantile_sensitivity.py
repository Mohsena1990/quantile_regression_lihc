"""Quantile-sensitivity analysis for R1-R3: how do the four-class outcome and
downstream screening quality move as the high-cost quantile varies, relative to
the two named benchmarks this project is grounded in --
Van Hove et al. (2022)'s country Q80 (M0) and the official/Hills-style LIHC
handbook's country-median (~Q50) required-cost threshold?

Each R1-R3 family's champion is now already calibrated at its own continuously-
optimized quantile q* (lib/tuning/quantile_search.py's 5-fold OOF calibration
search over q in [0.50, 0.80]), which typically sits somewhere between the two
named anchors above but is not fixed to either. This script sweeps a wider grid
(QUANTILE_SWEEP, 0.50-0.85) bracketing both anchors AND q*, refitting each of the 3
M2/M5/M7 families at its already-selected champion feature config (A0 or A1, from
outputs/tables/table_family_champions.csv) -- config choice is held fixed so the
sweep isolates the quantile's effect, not a config confound.

Deliberately lighter than the production path: `tune=False` (skip the internal
k-fold hyperparameter search -- this is a trend/diagnostic sweep, not a
calibration-accuracy exercise) and a single classifier (CatBoost) rather than the
full 2-classifier matrix, so the whole sweep is a few dozen fits.

Outputs:
  outputs/tables/table_quantile_sensitivity.csv
  outputs/figures/quantile_sensitivity.png/.pdf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure
from lib.reporting.codes import regression_code, REGRESSION_FAMILY_ORDER
from lib.screening.models import fit_screening_model
from measurement.families.model_based import fit_measurement_strategy
from lib.evaluation.cost_metrics import cost_score
from lib.features.household_features import stack_features

QUANTILE_SWEEP = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85)
VAN_HOVE_Q, OFFICIAL_LIHC_Q = 0.80, 0.50
SURFACE, GRID, INK, INK_MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"
# Fixed categorical order/colors for R1-R3 (same slots the rest of the project uses).
R_COLOR = {"R1": "#2a78d6", "R2": "#eb6834", "R3": "#1baf7a"}


def main() -> None:
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    h_features = list(cfg["features"])
    hc_features = stack_features(h_features, cfg["climate_features"])
    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions.csv")
    champions = champions[champions.status.eq("AVAILABLE") & champions.family.ne("M0")]

    rows = []
    for champ in champions.itertuples():
        family = champ.family
        r_code = regression_code(family)
        src = ROOT / f"artifacts/predictions/{champ.champion_experiment_id}.parquet"
        if not src.exists():
            print(f"SKIP {family}: {src.name} not found")
            continue
        base = pd.read_parquet(src)
        features = hc_features if champ.feature_config == "A1" else h_features
        train = base[base.partition.eq("train")]
        for q in QUANTILE_SWEEP:
            try:
                _, measured = fit_measurement_strategy(family, train, base, features=features,
                                                         target=cfg["target"], quantile=q, tune=False)
            except Exception as e:
                print(f"FIT_FAIL {family} q{q}: {type(e).__name__}: {e}")
                continue
            val_measured = measured[measured.partition.eq("validation") & measured.measurement_eligible]
            reg = cost_score(val_measured, q)
            eligible = measured[measured.measurement_eligible]
            row = {"regression_code": r_code, "family": family, "quantile": q,
                   "n_eligible": int(eligible.measurement_eligible.sum()),
                   "share_high_cost": float(eligible.high_cost.mean()),
                   "share_double_risk": float((eligible.risk_class_code == 3).mean()),
                   **{f"val_{k}": v for k, v in reg.items()}}
            train_m = measured[measured.partition.eq("train") & measured.measurement_eligible]
            val_m = measured[measured.partition.eq("validation") & measured.measurement_eligible]
            if len(val_m) >= 10 and train_m.risk_class_code.nunique() >= 2:
                try:
                    _, _, metrics = fit_screening_model("catboost", train_m, val_m, features=features)
                    row["macro_f1"] = metrics["macro_f1"]
                except Exception as e:
                    row["macro_f1"] = None
                    print(f"SCREEN_FAIL {family} q{q}: {type(e).__name__}: {e}")
            else:
                row["macro_f1"] = None
            rows.append(row)
        print(f"DONE {family}", flush=True)

    table = pd.DataFrame(rows).sort_values(["regression_code", "quantile"])
    table.to_csv(ROOT / "outputs/tables/table_quantile_sensitivity.csv", index=False)

    panels = [("share_double_risk", "Double-Risk share", lambda v: v * 100),
              ("val_pinball", "Validation pinball loss (EUR)", lambda v: v),
              ("macro_f1", "Screening macro-F1 (CatBoost)", lambda v: v)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6))
    for ax, (col, label, tf) in zip(axes, panels):
        for r_code in [regression_code(f) for f in REGRESSION_FAMILY_ORDER if f != "M0"]:
            g = table[table.regression_code.eq(r_code)].dropna(subset=[col])
            if g.empty:
                continue
            # bracket access, not `g.quantile` -- pandas' DataFrame.quantile() method shadows
            # plain attribute access to a column literally named "quantile" (see the
            # pandas-Series-shadowing bug this project has hit before, e.g.
            # scripts/build_b_measurements.py / scripts/build_policy_maps.py).
            ax.plot(g["quantile"], tf(g[col]), marker="o", markersize=3.5, linewidth=1.4,
                    color=R_COLOR[r_code], label=r_code)
        ax.axvline(OFFICIAL_LIHC_Q, color=INK_MUTED, linestyle=":", linewidth=1.1, zorder=0)
        ax.axvline(VAN_HOVE_Q, color=INK_MUTED, linestyle="--", linewidth=1.1, zorder=0)
        ax.set_xlabel("Quantile", fontsize=8.5)
        ax.set_title(label, fontsize=9.5)
        ax.grid(color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5, colors=INK_MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=8)
    fig.suptitle("Quantile-sensitivity analysis, R1-R3\n"
                 "(dotted = official-LIHC median anchor q50; dashed = Van Hove anchor q80)", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    save_figure(fig, ROOT / "outputs/figures/quantile_sensitivity")
    plt.close(fig)
    print(f"Wrote {len(table)} quantile-sensitivity rows -> outputs/tables/table_quantile_sensitivity.csv, "
          f"outputs/figures/quantile_sensitivity.png/.pdf")


if __name__ == "__main__":
    main()
