"""Confusion-matrix figure for the B-family's single best-performing measurement
family, per-country winning-config classifiers.

Unlike scripts/build_confusion_matrices.py (pooled measurements, evaluated the same
way in every country), each column here used its country's *own* winning feature
config (outputs/tables/table_percountry_winners.csv) -- so a column can differ in
which features fed the model, not just in the data slice. Reads the per-row
validation predictions scripts/run_percountry_feature_selection.py already wrote
(artifacts/predictions/percountry/{am_id}_{country}_{classifier}.parquet), filtered
to each country's winning config -- no re-fitting. A pooled "All countries" column
concatenates the five countries' winning-config predictions for the aggregate view.

Restricted to the single family with the highest average per-country macro-F1
(2026-08-26: one confusion-matrix figure for the best configuration, not one per
family -- see scripts/build_confusion_matrices.py's module docstring).

Outputs:
  outputs/figures/confusion_b_best_family.png/.pdf
  outputs/tables/table_b_confusion_matrices.csv (long form: one row per cell, per country)
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_confusion_matrices import _draw_cell, COLUMN_ORDER, MODELS, MODEL_LABEL, LEGEND_CAPTION
from lib.reporting.codes import regression_code
import matplotlib.pyplot as plt

PRED_DIR = ROOT / "artifacts/predictions/percountry"


def plot_family(family: str, winners: pd.DataFrame) -> list[dict]:
    fam_winners = winners[winners.family.eq(family)]
    if fam_winners.empty:
        return []
    frames: dict[str, dict[str, pd.DataFrame]] = {kind: {} for kind in MODELS}
    for row in fam_winners.itertuples():
        for kind in MODELS:
            p = PRED_DIR / f"{row.am_id}_{row.country}_{kind}.parquet"
            if p.exists():
                frames[kind][row.country] = pd.read_parquet(p)

    ncols = len(COLUMN_ORDER)
    fig, axes = plt.subplots(len(MODELS), ncols, figsize=(1.9 * ncols, 2.5 * len(MODELS)))
    rows = []
    for r, kind in enumerate(MODELS):
        by_country = frames[kind]
        pooled = pd.concat(by_country.values(), ignore_index=True) if by_country else None
        for c, country in enumerate(COLUMN_ORDER):
            ax = axes[r, c]
            slice_d = pooled if country == "All countries" else by_country.get(country)
            config_label = (fam_winners[fam_winners.country.eq(country)].config.iloc[0]
                             if country != "All countries" and (fam_winners.country == country).any() else country)
            title = country if country == "All countries" else f"{country}\n({config_label})"
            cell_rows = _draw_cell(ax, slice_d, title, show_ylabel=(c == 0))
            for cr in cell_rows:
                rows.append({"family": family, "model": kind, "country": country, **cr})
        axes[r, 0].set_ylabel(MODEL_LABEL[kind], fontsize=8)
    r_code = regression_code(family)
    fig.suptitle(f"Confusion matrices by country (own winning feature config) -- "
                 f"best B-family: {r_code} ({family})\n{LEGEND_CAPTION}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    from lib.reporting.figures import save_figure
    save_figure(fig, ROOT / "outputs/figures/confusion_b_best_family")
    plt.close(fig)
    return rows


def main() -> None:
    winners = pd.read_csv(ROOT / "outputs/tables/table_percountry_winners.csv")
    if winners.empty:
        print("No table_percountry_winners.csv rows -- run scripts/run_percountry_feature_selection.py first.")
        return
    best_family = winners.groupby("family").macro_f1.mean().idxmax()
    print(f"Best B-family: {best_family} (avg per-country macro_f1={winners.groupby('family').macro_f1.mean().max():.4f})")
    rows = plot_family(best_family, winners)
    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "outputs/tables/table_b_confusion_matrices.csv", index=False)
    print(f"Wrote confusion-matrix figure for best B-family -> outputs/figures/confusion_b_best_family.png/.pdf, "
          f"outputs/tables/table_b_confusion_matrices.csv ({len(table)} cell rows)")


if __name__ == "__main__":
    main()
