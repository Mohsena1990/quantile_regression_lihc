"""Symbol legend for the paper-facing reporting layer: R0-R3 (regression methods),
C0-C1 (screening classifiers), A0-A2 (per-country/whole feature configs), and the
`A{config}R{regression}C{classifier}` combo naming (e.g. "A0R2C1").

Reads scripts/select_family_champions.py's already-computed champion quantile per
M2/M5/M7 family (outputs/tables/table_family_champions.csv) -- does not refit anything.

Outputs:
  outputs/tables/table_symbol_legend.csv
  outputs/figures/symbol_legend.png/pdf   (compact reference table for a paper appendix)
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting import (
    REGRESSION_CODES, CLASSIFIER_CODES, CONFIG_CODES,
    REGRESSION_DESCRIPTIONS, CLASSIFIER_DESCRIPTIONS, CONFIG_DESCRIPTIONS,
    regression_locked_quantile, save_figure,
)

SURFACE, GRID, INK_MUTED, INK = "#fcfcfb", "#e1e0d9", "#898781", "#0b0b0b"


def build_table() -> pd.DataFrame:
    rows = []
    for code, family in REGRESSION_CODES.items():
        quantile = regression_locked_quantile(family)
        rows.append({"code": code, "kind": "regression", "underlying": family,
                     "description": REGRESSION_DESCRIPTIONS[family],
                     "locked_quantile": quantile})
    for code, name in CLASSIFIER_CODES.items():
        rows.append({"code": code, "kind": "classifier", "underlying": name,
                     "description": CLASSIFIER_DESCRIPTIONS[name], "locked_quantile": None})
    for code in CONFIG_CODES:
        rows.append({"code": code, "kind": "config", "underlying": code,
                     "description": CONFIG_DESCRIPTIONS[code], "locked_quantile": None})
    return pd.DataFrame(rows)


def render_figure(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 0.305 * len(table) + 0.55))
    ax.set_axis_off()
    cell_text, colours = [], []
    for row in table.itertuples():
        quantile = "" if pd.isna(row.locked_quantile) else f"q{round(row.locked_quantile * 100):02d}"
        cell_text.append([row.code, row.underlying, row.description, quantile])
        colours.append(SURFACE)
    tbl = ax.table(cellText=cell_text, colLabels=["Code", "Underlying", "Description", "Locked quantile"],
                    colWidths=[0.08, 0.16, 0.62, 0.14], cellLoc="left", loc="upper center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_text_props(weight="bold", color=INK)
            cell.set_facecolor("#eeece4")
        else:
            cell.set_facecolor(SURFACE)
    ax.set_title("Symbol legend -- A{config}R{regression}C{classifier}, e.g. \"A0R3C1\"", fontsize=10.5, pad=12)
    fig.tight_layout()
    save_figure(fig, ROOT / "outputs/figures/symbol_legend")
    plt.close(fig)


def main() -> None:
    table = build_table()
    table.to_csv(ROOT / "outputs/tables/table_symbol_legend.csv", index=False)
    render_figure(table)
    print(f"Wrote {len(table)} symbol-legend rows -> outputs/tables/table_symbol_legend.csv, "
          f"outputs/figures/symbol_legend.png/pdf")


if __name__ == "__main__":
    main()
