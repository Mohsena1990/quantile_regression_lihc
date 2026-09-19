"""Curated paper vs. appendix/supplementary output split.

Everything upstream keeps writing to outputs/tables/ and outputs/figures/ exactly
as before -- this script builds a copy layer on top, last, once every other output
exists. It does not change any other script's output path.

outputs/paper/    the headline comparison table (best A-config x M-family x
                   classifier per country and overall, from
                   scripts/run_percountry_feature_selection.py's
                   table_best_overall.csv) plus one confusion-matrix figure
                   showing only those winning cells.
outputs/appendix/ a copy of everything else already in outputs/tables/ and
                   outputs/figures/ (full per-config/per-classifier detail,
                   per-country and B-family confusion-matrix grids,
                   policy maps, etc.).

Outputs:
  outputs/paper/table_best_overall.csv
  outputs/paper/confusion_best_overall.png/.pdf
  outputs/paper/table_confusion_best_overall.csv
  outputs/appendix/tables/*, outputs/appendix/figures/*
"""
from __future__ import annotations
import shutil, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_confusion_matrices import _draw_cell
from lib.reporting.figures import save_figure

PAPER_DIR = ROOT / "outputs/paper"
APPENDIX_DIR = ROOT / "outputs/appendix"
PRED_DIR = ROOT / "artifacts/predictions/percountry"


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    (APPENDIX_DIR / "tables").mkdir(parents=True, exist_ok=True)
    (APPENDIX_DIR / "figures").mkdir(parents=True, exist_ok=True)

    best_path = ROOT / "outputs/tables/table_best_overall.csv"
    if not best_path.exists():
        raise SystemExit("Run scripts/run_percountry_feature_selection.py first (table_best_overall.csv missing)")
    best_overall = pd.read_csv(best_path)
    shutil.copy2(best_path, PAPER_DIR / "table_best_overall.csv")

    raw = pd.read_csv(ROOT / "outputs/tables/table_percountry_feature_selection.csv")
    raw = raw[raw.status.eq("AVAILABLE")]

    ncols = len(best_overall)
    fig, axes = plt.subplots(1, ncols, figsize=(1.9 * ncols, 2.7))
    axes = [axes] if ncols == 1 else list(axes)
    cell_rows = []
    for ax, row in zip(axes, best_overall.itertuples()):
        cell = raw[raw.country.eq(row.country) & raw.family.eq(row.family)
                    & raw.config.eq(row.config) & raw.classifier.eq(row.classifier)]
        if cell.empty:
            ax.axis("off")
            ax.set_title(f"{row.country}\n(unavailable)", fontsize=7.5)
            continue
        stem = f"{row.am_id}_{row.country}_{row.classifier}"
        p = PRED_DIR / f"{stem}.parquet"
        if not p.exists():
            ax.axis("off")
            ax.set_title(f"{row.country}\n(missing)", fontsize=7.5)
            continue
        pred = pd.read_parquet(p)
        title = f"{row.country}\n{row.am_id} / {row.classifier}"
        for cr in _draw_cell(ax, pred, title, show_ylabel=(ax is axes[0])):
            cell_rows.append({"country": row.country, "family": row.family, "config": row.config,
                               "am_id": row.am_id, "classifier": row.classifier, **cr})
    fig.suptitle("Best A-config x M-family x classifier per country and overall", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, PAPER_DIR / "confusion_best_overall")
    plt.close(fig)
    pd.DataFrame(cell_rows).to_csv(PAPER_DIR / "table_confusion_best_overall.csv", index=False)

    n_tables = n_figures = 0
    for f in (ROOT / "outputs/tables").iterdir():
        if f.is_file() and f.name != "table_best_overall.csv":
            shutil.copy2(f, APPENDIX_DIR / "tables" / f.name)
            n_tables += 1
    for f in (ROOT / "outputs/figures").iterdir():
        if f.is_file():
            shutil.copy2(f, APPENDIX_DIR / "figures" / f.name)
            n_figures += 1
    print(f"PAPER_APPENDIX_SPLIT_COMPLETE paper_rows={len(best_overall)} "
          f"appendix_tables={n_tables} appendix_figures={n_figures}")


if __name__ == "__main__":
    main()
