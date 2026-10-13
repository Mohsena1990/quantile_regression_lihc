"""Why are this project's Van Hove-design numbers lower than Van Hove et al. (2022)? A factor-by-factor check on the four
study countries (Bulgaria, Hungary, Italy, Poland) -- the study's design assumption; the other seven ENABLE countries are not used.

Every scenario uses Van Hove's labels (income bracket <= 3 x country Q80 of energy expenditure), their seven predictors and their
protocol (stratified 20% test split, then 80/20 train/validation), repeated over N_SEEDS random splits so the spread of a single split
-- all a one-split study can report -- is visible. One thing changes at a time:
  A  the four countries pooled in one model, Van Hove's preprocessing and CatBoost settings
  B  as A with this project's earlier CatBoost settings (100 trees, L2 10, no validation stopping)
  C  as A with "don't know" codes and impossible values set to missing
  D  one model per country, Van Hove settings
  E  as D with this project's earlier CatBoost settings
  F  as D with hyperparameters tuned on the validation split (depth, learning rate, L2; up to 1000 trees, early stopping)
Metrics: per-class true-positive rate (recall; the diagonal of their confusion matrix), their TotalF1 (support-weighted F1),
macro-F1, balanced accuracy, accuracy. Households in this project's spent test partition are excluded throughout.

Outputs: outputs/tables/table_vanhove_gap.csv (per run), table_vanhove_gap_summary.csv; outputs/figures/vanhove_gap.
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES
from lib.identification.vanhove_replica import CONFIGS, VH_FEATURES, load_frame, tuned_protocol, vanhove_protocol
from lib.reporting.figures import save_figure

N_SEEDS = 20
PAPER_TPR = (0.60, 0.74)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"


def run(name, frame, config, seed, country="pooled"):
    res = tuned_protocol(frame, VH_FEATURES, seed) if config == "tuned" else vanhove_protocol(frame, VH_FEATURES, CONFIGS[config], seed)
    return {"scenario": name, "country": country, "seed": seed, "n": len(frame), **res}


def main() -> None:
    vh, vh_clean = load_frame(), load_frame(clean=True)
    jobs = []
    for s in range(N_SEEDS):
        jobs += [delayed(run)("A", vh, "vanhove", s), delayed(run)("B", vh, "project_cb4", s), delayed(run)("C", vh_clean, "vanhove", s)]
        for c in COUNTRIES:
            g = vh[vh.country == c].reset_index(drop=True)
            jobs += [delayed(run)("D", g, "vanhove", s, c), delayed(run)("E", g, "project_cb4", s, c), delayed(run)("F", g, "tuned", s, c)]
    t = pd.DataFrame(Parallel(n_jobs=12)(jobs))
    t.to_csv(ROOT / "outputs/tables/table_vanhove_gap.csv", index=False)
    metrics = ["tpr_no_risk", "tpr_income_risk", "tpr_expenditure_risk", "tpr_double_risk", "weighted_f1", "macro_f1",
               "balanced_accuracy", "accuracy", "trees"]
    summ = t.groupby(["scenario", "country"])[metrics].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()
    summ.to_csv(ROOT / "outputs/tables/table_vanhove_gap_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4.8)); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    order = [("A", "pooled"), ("B", "pooled"), ("C", "pooled")] + [(sc, c) for sc in ("D", "E", "F") for c in COUNTRIES]
    names = {"A": "A pooled,\nVan Hove", "B": "B pooled,\nproject", "C": "C pooled,\ncleaned"}
    for i, (sc, c) in enumerate(order):
        ax.boxplot(t[(t.scenario == sc) & (t.country == c)].balanced_accuracy, positions=[i], widths=0.6, patch_artist=True,
                   boxprops={"facecolor": "#cfe0f5", "edgecolor": INK2}, medianprops={"color": INK}, flierprops={"markersize": 2})
    ax.axhspan(*PAPER_TPR, color="#eda100", alpha=0.15, label="Van Hove et al. (2022): per-class recall 0.60-0.74 (one split, 11 countries)")
    ax.set_xticks(range(len(order)), [names.get(sc, f"{sc} {c[:3]}") for sc, c in order], fontsize=7.5)
    ax.set_ylabel("balanced accuracy (= mean per-class recall)", fontsize=8.5, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.7); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"Van Hove's design on the four study countries ({N_SEEDS} random splits each). D-F one model per country: "
                 "D Van Hove settings, E project settings, F tuned", fontsize=9.5, loc="left")
    fig.tight_layout()
    save_figure(fig, ROOT / "outputs/figures/vanhove_gap")
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(summ[["scenario", "country"] + [f"{m}_mean" for m in metrics] + ["balanced_accuracy_std"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
