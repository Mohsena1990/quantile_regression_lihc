"""Confusion-matrix figure for the single best-performing measurement, broken out
by country. Restricted to the overall champion (highest macro-F1 across all M0/A0/A1
measurements, from table_family_champions_final.csv) rather than every measurement --
2026-08-26: this project generates one confusion-matrix figure for its best
configuration, not one per measurement x stage (see scripts/build_b_confusion_matrices.py
for the B-family's per-country-winning-config figure, which is "best configuration"
by construction).

Draws a grid of one confusion matrix per (country x classifier) cell, so both
classifiers' error patterns AND each country's own behaviour are directly
comparable side by side. A pooled "All countries" column is kept as the rightmost
column for the aggregate view.

Reads the per-row prediction parquets that scripts/run_screening.py already
writes (artifacts/predictions/<measurement>_<model>_<stage>.parquet,
columns UID/truth/prediction/prob_0..3/country) -- no re-fitting.

Outputs:
  outputs/figures/confusion_best_configuration.png/.pdf
  outputs/tables/table_confusion_matrices.csv (long form: one row per cell, per country)
"""
from __future__ import annotations
import sys
from itertools import product
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure
from lib.reporting.codes import classifier_code

RISK_ORDER = ["No Risk", "Income Risk", "Expenditure Risk", "Double Risk"]
RISK_SHORT = ["No", "Income", "Expend.", "Double"]
MODELS = ["catboost", "hierarchical_catboost"]
MODEL_LABEL = {name: classifier_code(name) for name in MODELS}
LEGEND_CAPTION = "C0=CatBoost  |  C1=Hierarchical CatBoost"
COUNTRY_ORDER = ["Bulgaria", "Hungary", "Italy", "Poland", "Spain"]
COLUMN_ORDER = COUNTRY_ORDER + ["All countries"]


def _draw_cell(ax, d: pd.DataFrame, title: str, *, show_ylabel: bool) -> list[dict]:
    if d is None or d.empty:
        ax.axis("off")
        ax.set_title(f"{title}\n(unavailable)", fontsize=7.5)
        return []
    cm = confusion_matrix(d.truth, d.prediction, labels=[0, 1, 2, 3])
    row_total = cm.sum(axis=1, keepdims=True)
    cm_pct = np.divide(cm, row_total, out=np.zeros_like(cm, dtype=float), where=row_total != 0) * 100
    macro_f1 = f1_score(d.truth, d.prediction, labels=[0, 1, 2, 3], average="macro", zero_division=0)
    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i, j in product(range(4), range(4)):
        v_pct, v_n = cm_pct[i, j], cm[i, j]
        ax.text(j, i, f"{v_pct:.0f}%\n{v_n}", ha="center", va="center", fontsize=5.8,
                 color="#ffffff" if v_pct > 55 else "#0b0b0b")
    ax.set_xticks(range(4)); ax.set_xticklabels(RISK_SHORT, fontsize=5.8, rotation=30, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(RISK_SHORT if show_ylabel else [], fontsize=5.8)
    ax.set_title(f"{title}\nmacro-F1={macro_f1:.2f}, n={len(d)}", fontsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    rows = []
    for i, j in product(range(4), range(4)):
        rows.append({"truth_class": RISK_ORDER[i], "predicted_class": RISK_ORDER[j],
                      "count": int(cm[i, j]), "row_pct": round(cm_pct[i, j], 2)})
    return rows


def plot_config(measurement: str, stage: str, frames: dict[str, pd.DataFrame]) -> list[dict]:
    ncols = len(COLUMN_ORDER)
    fig, axes = plt.subplots(len(MODELS), ncols, figsize=(1.9 * ncols, 2.5 * len(MODELS)))
    rows = []
    for r, kind in enumerate(MODELS):
        d = frames.get(kind)
        for c, country in enumerate(COLUMN_ORDER):
            ax = axes[r, c]
            slice_d = d if (d is None or country == "All countries") else d[d.country.eq(country)]
            cell_rows = _draw_cell(ax, slice_d, country, show_ylabel=(c == 0))
            for cr in cell_rows:
                rows.append({"measurement": measurement, "stage": stage, "model": kind, "country": country, **cr})
        axes[r, 0].set_ylabel(MODEL_LABEL[kind], fontsize=8)
    fig.suptitle(f"Confusion matrices by country -- best configuration: {measurement} / {stage}\n{LEGEND_CAPTION}",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, ROOT / "outputs/figures/confusion_best_configuration")
    plt.close(fig)
    return rows


def main() -> None:
    champions_path = ROOT / "outputs/tables/table_family_champions_final.csv"
    if not champions_path.exists():
        print("No table_family_champions_final.csv -- run scripts/select_family_champions.py first.")
        return
    champions = pd.read_csv(champions_path).sort_values("macro_f1", ascending=False)
    best = champions.iloc[0]
    measurement = best.champion_experiment_id
    print(f"Best configuration: {measurement} (macro_f1={best.macro_f1:.4f})")

    pred_dir = ROOT / "artifacts/predictions"
    frames = {}
    for kind in MODELS:
        p = pred_dir / f"{measurement}_{kind}_development.parquet"
        if p.exists():
            frames[kind] = pd.read_parquet(p)
    if not frames:
        print(f"No development-stage predictions found for {measurement}.")
        return
    rows = plot_config(measurement, "development", frames)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "outputs/tables/table_confusion_matrices.csv", index=False)
    print(f"Wrote confusion-matrix figure for best configuration -> outputs/figures/confusion_best_configuration.png/.pdf, "
          f"outputs/tables/table_confusion_matrices.csv ({len(table)} cell rows)")


if __name__ == "__main__":
    main()
