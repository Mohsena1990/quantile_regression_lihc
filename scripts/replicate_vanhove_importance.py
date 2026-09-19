"""Replication of Van Hove et al. (2022) Fig. 7 -- permutation and mean|SHAP| importance of their seven predictors.

Their design, on this project's five retained countries pooled: four classes from income bracket <= 3 and country
Q80 of raw annual energy cost; predictors income bracket, floor area (H3), household size, house detachment (H1),
house age (H2), respondent age (birth-year equivalent) and heating strategy (primary heating fuel). Importance is
computed on HELD-OUT cross-validation folds (they used an unseen test set), normalised so the largest value = 100,
with fold-to-fold spread as error bars. Their reported ordering (mean|SHAP| and permutation agree on the top 3):
income > floor area > household size; house age and birth year lowest; house detachment and heating inconclusive.

Outputs: outputs/tables/table_vanhove_importance_replication.csv; outputs/figures/vanhove_replication_importance
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import load_development_frame
from lib.identification.discovery import fold_importance
from lib.identification.labels import add_labels
from lib.reporting.figures import save_figure

VH_FEATURES = ["income_bracket", "H3", "household_size", "H1", "H2", "respondent_age", "primary_heating_fuel"]
NAMES = {"income_bracket": "income bracket", "H3": "floor area", "household_size": "household size", "H1": "house detachment",
         "H2": "house age", "respondent_age": "birth year respondent", "primary_heating_fuel": "heating strategy"}
VH_ORDER = ["income_bracket", "H3", "household_size", "H1", "primary_heating_fuel", "respondent_age", "H2"]   # their SHAP ranking, high to low
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"


def main() -> None:
    lab = add_labels(load_development_frame(), cost_kind="raw_q80")          # bracket <= 3, country Q80: Van Hove's labels
    print("pooled households:", len(lab), "| class shares:", lab.y.value_counts(normalize=True).sort_index().round(3).to_dict())
    imp = fold_importance(lab, VH_FEATURES, "y", n_repeats=3, n_perm=5)
    g = imp.assign(shap_abs=imp.shap_abs.clip(lower=0))
    out = []
    for m, label in (("shap_abs", "mean |SHAP|"), ("perm_logloss", "permutation (log-loss)"), ("perm_bacc", "permutation (balanced accuracy)")):
        piv = g.pivot_table(index=["repeat", "fold"], columns="feature", values=m)
        top = piv.mean().max()
        norm = piv / top * 100
        out.append(pd.DataFrame({"method": label, "feature": norm.columns, "importance_norm": norm.mean().values, "fold_sd": norm.std().values}))
    t = pd.concat(out, ignore_index=True)
    t["feature_name"] = t.feature.map(NAMES)
    rows = []
    for m, d in t.groupby("method"):
        d = d.set_index("feature").loc[VH_ORDER]
        rho = stats.spearmanr(np.arange(len(VH_ORDER)), -d.importance_norm.to_numpy())[0]
        rows.append({"method": m, "spearman_vs_vanhove_order": rho, "top3": ", ".join(d.sort_values("importance_norm", ascending=False).head(3).feature_name)})
    t.to_csv(ROOT / "outputs/tables/table_vanhove_importance_replication.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "outputs/tables/table_vanhove_importance_replication_summary.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True); fig.patch.set_facecolor(SURFACE)
    order = VH_ORDER[::-1]
    for ax, (method, col) in zip(axes, (("mean |SHAP|", "#2a78d6"), ("permutation (log-loss)", "#eb6834"))):
        ax.set_facecolor(SURFACE); d = t[t.method == method].set_index("feature").loc[order]
        ax.barh([NAMES[f] for f in order], d.importance_norm, xerr=d.fold_sd, color=col, height=0.6, error_kw={"ecolor": INK2, "elinewidth": 0.9, "capsize": 2})
        ax.axvline(0, color=MUTED, lw=0.9); ax.grid(axis="x", color=GRID, lw=0.7); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8.5); ax.set_title(method, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("normalised importance (largest = 100), held-out folds, error bars = fold SD", fontsize=8, color=INK2)
    fig.suptitle("Replication of Van Hove et al. (2022) Fig. 7 on the five retained countries pooled. Their order, top to bottom: income, floor area, household size, "
                 "detachment, heating, birth year, house age", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save_figure(fig, ROOT / "outputs/figures/vanhove_replication_importance"); plt.close(fig)


if __name__ == "__main__":
    main()
