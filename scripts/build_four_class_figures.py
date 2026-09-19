"""Per-country risk-class distribution figures for every available measurement step.

One stacked bar chart per measurement (outputs/figures/four_class_<measurement>.*),
plus a cross-measurement overview heatmap of highest-severity-class prevalence, and a
long-form summary table (outputs/tables/table_four_class_by_country.csv).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure
FAMILIES = json.loads((ROOT / "config/pipeline.json").read_text())["families"]

# Status palette (good/warning/serious/critical) -- the four risk classes are an
# ordered severity scale, not an unordered category set, so they take the fixed
# status roles rather than the categorical hue ramp.
RISK_ORDER = ["No Risk", "Income Risk", "Expenditure Risk", "Double Risk"]
RISK_COLOR = {"No Risk": "#0ca30c", "Income Risk": "#fab219", "Expenditure Risk": "#ec835a", "Double Risk": "#d03b3b"}
COUNTRY_ORDER = ["Bulgaria", "Hungary", "Italy", "Poland", "Spain"]
SURFACE, GRID, INK, INK_MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"

# Every measurement step that can carry a four-class label.
CORE_STEMS = ["m0_descriptive", "m0_predictive"] + [f"m{f[1:]}_h" for f in FAMILIES]
HC_STEMS = [f"m{f[1:]}_hc" for f in FAMILIES]
BASE_STEMS = ([f"m{f[1:]}_h_base" for f in FAMILIES]
              + [f"m{f[1:]}_hc_base" for f in FAMILIES])


def class_shares(d: pd.DataFrame, class_col: str, order: list[str]) -> pd.DataFrame:
    d = d.dropna(subset=[class_col, "country"])
    tab = pd.crosstab(d["country"], d[class_col], normalize="index").reindex(columns=order, fill_value=0.0)
    return tab.reindex(index=[c for c in COUNTRY_ORDER if c in tab.index])


def plot_stacked(tab: pd.DataFrame, colors: dict, order: list[str], title: str, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.3))
    bottoms = np.zeros(len(tab))
    for cls in order:
        vals = tab[cls].to_numpy() * 100
        bars = ax.bar(tab.index, vals, bottom=bottoms, color=colors[cls], label=cls, width=0.6,
                       edgecolor=SURFACE, linewidth=1.5, zorder=3)
        for rect, v in zip(bars, vals):
            if v >= 6:  # direct label -- redundant with color so severity reads without hue alone
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + v / 2, f"{v:.0f}%",
                         ha="center", va="center", fontsize=8, color="#ffffff" if cls in ("Expenditure Risk", "Double Risk") else "#0b0b0b")
        bottoms += vals
    ax.set_ylabel("Share of retained cohort (%)")
    ax.set_ylim(0, 100)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=len(order), frameon=False, fontsize=8)
    fig.tight_layout()
    save_figure(ax, ROOT / "outputs/figures" / stem)
    plt.close(fig)


def main() -> None:
    pred_dir = ROOT / "artifacts/predictions"
    rows = []

    for stem in CORE_STEMS + HC_STEMS + BASE_STEMS:
        p = pred_dir / f"{stem}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if "risk_class_name" not in d or "country" not in d:
            continue
        tab = class_shares(d, "risk_class_name", RISK_ORDER)
        if tab.empty:
            continue
        plot_stacked(tab, RISK_COLOR, RISK_ORDER, f"Four-class distribution by country -- {stem}", f"four_class_{stem}")
        for country, r in tab.iterrows():
            rows += [{"measurement": stem, "country": country, "class": cls, "share": r[cls]} for cls in RISK_ORDER]

    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "outputs/tables/table_four_class_by_country.csv", index=False)

    severe = summary[summary["class"] == "Double Risk"]
    if len(severe):
        piv = severe.pivot(index="measurement", columns="country", values="share")
        piv = piv.reindex(columns=[c for c in COUNTRY_ORDER if c in piv.columns])
        vals_pct = piv.to_numpy(dtype=float) * 100
        fig, ax = plt.subplots(figsize=(1.15 * len(piv.columns) + 2.5, 0.32 * len(piv) + 2))
        vmax = max(1.0, np.nanmax(vals_pct))
        im = ax.imshow(vals_pct, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(piv.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(piv.index, fontsize=7)
        for i in range(vals_pct.shape[0]):
            for j in range(vals_pct.shape[1]):
                v = vals_pct[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                            color="#ffffff" if v > 0.55 * vmax else "#0b0b0b")
        for spine in ax.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7)
        cbar.set_label("Double-Risk / HighCost share (%)", fontsize=8)
        ax.set_title("Highest-severity class prevalence by country -- every measurement step", fontsize=10)
        fig.tight_layout()
        save_figure(ax, ROOT / "outputs/figures/four_class_overview_heatmap")
        plt.close(fig)

    print(f"Wrote {summary.measurement.nunique() if len(summary) else 0} per-measurement figures "
          f"({len(rows)} country-class rows) -> outputs/figures/four_class_*.png/.pdf, "
          f"outputs/tables/table_four_class_by_country.csv")


if __name__ == "__main__":
    main()
