"""Regression-threshold interpretation figures: for each R0-R3 regression method
(at its locked champion quantile, see lib/reporting/codes.py), show how the fitted
high-cost threshold actually sits against the observed cost distribution -- not
just the resulting class counts (already covered by scripts/build_four_class_figures.py).

Households are sorted by their own conditional_threshold per country panel, the
threshold is drawn as a line, and each household's observed cost is scattered on
top, coloured by whether it fell above (HighCost) or below (LowCost) its own
threshold -- so a reader can see both how the threshold varies across households
(flat for R0's per-country Q80 benchmark, sloped/conditional for R1-R3) and how
much of the cost distribution it separates.

Reuses each family's already-selected champion (outputs/tables/table_family_champions.csv)
and its prediction parquet -- no re-fitting.

Outputs:
  outputs/figures/threshold_R{0-7}.png/.pdf
  outputs/tables/table_threshold_summary.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure
from lib.reporting.codes import REGRESSION_CODES, REGRESSION_DESCRIPTIONS

from data.enable import RETAINED_COUNTRY_NAMES
COUNTRY_ORDER = list(RETAINED_COUNTRY_NAMES)
# Same LowCost/HighCost status colors as scripts/build_four_class_figures.py:COST_COLOR,
# reused here rather than re-derived so the two-state cost split reads identically
# across every figure in the paper.
COST_COLOR = {"LowCost": "#0ca30c", "HighCost": "#d03b3b"}
THRESHOLD_LINE_COLOR = "#0b0b0b"
SURFACE, GRID, INK_MUTED = "#fcfcfb", "#e1e0d9", "#898781"


def plot_threshold(code: str, family: str, champion_row: pd.Series, d: pd.DataFrame) -> list[dict]:
    d = d.dropna(subset=["conditional_threshold", "observed_energy_cost_eur_reference", "high_cost", "country"])
    countries = [c for c in COUNTRY_ORDER if c in d.country.unique()]
    fig, axes = plt.subplots(1, len(countries), figsize=(3.6 * len(countries), 4.2), sharey=True)
    axes = np.atleast_1d(axes)
    summary_rows = []
    for ax, country in zip(axes, countries):
        g = d[d.country.eq(country)].sort_values("conditional_threshold").reset_index(drop=True)
        x = np.arange(len(g))
        ax.plot(x, g.conditional_threshold, color=THRESHOLD_LINE_COLOR, linewidth=1.6, zorder=3,
                label="Conditional threshold")
        for cls, color in (("LowCost", COST_COLOR["LowCost"]), ("HighCost", COST_COLOR["HighCost"])):
            mask = (g.high_cost == (cls == "HighCost")).to_numpy()
            ax.scatter(x[mask], g.observed_energy_cost_eur_reference.to_numpy()[mask], s=7, color=color,
                       alpha=0.55, linewidths=0, zorder=2, label=cls)
        ax.set_title(f"{country}\nn={len(g)}", fontsize=9)
        ax.set_xlabel("Households (sorted by threshold)", fontsize=7.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_MUTED)
        share_high = float(g.high_cost.mean())
        summary_rows.append({"regression_code": code, "family": family, "country": country, "n": len(g),
                              "threshold_mean_eur": float(g.conditional_threshold.mean()),
                              "threshold_median_eur": float(g.conditional_threshold.median()),
                              "share_high_cost": share_high})
    axes[0].set_ylabel("Energy cost (EUR)", fontsize=8.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06), frameon=False, fontsize=8.5)
    quantile = champion_row["quantile"]
    quantile_note = "" if pd.isna(quantile) else f" @ q{round(quantile * 100):02d}"
    fig.suptitle(f"{code} ({family}{quantile_note}) -- {REGRESSION_DESCRIPTIONS[family]}\n"
                 f"How the fitted high-cost threshold sits against observed cost, by country", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    save_figure(fig, ROOT / "outputs/figures" / f"threshold_{code}")
    plt.close(fig)
    return summary_rows


def main() -> None:
    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions.csv")
    all_rows: list[dict] = []
    n_figures = 0
    for code, family in REGRESSION_CODES.items():
        row = champions.loc[champions.family.eq(family)]
        if row.empty or not row.iloc[0]["status"] == "AVAILABLE":
            continue
        champion_row = row.iloc[0]
        p = ROOT / f"artifacts/predictions/{champion_row['champion_experiment_id']}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        rows = plot_threshold(code, family, champion_row, d)
        if rows:
            all_rows += rows
            n_figures += 1
    table = pd.DataFrame(all_rows)
    table.to_csv(ROOT / "outputs/tables/table_threshold_summary.csv", index=False)
    print(f"Wrote {n_figures} regression-threshold figures -> outputs/figures/threshold_R*.png/.pdf, "
          f"outputs/tables/table_threshold_summary.csv ({len(table)} rows)")


if __name__ == "__main__":
    main()
