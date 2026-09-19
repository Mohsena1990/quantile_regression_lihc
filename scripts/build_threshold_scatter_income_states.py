"""Threshold scatter plots, four-class coloured, in two states: income excluded
from the predictor set (this project's production design) vs. income included
(the Van-Hove-style arm, see scripts/compare_income_ablation.py). Two rows of
panels (top = income-excluded, bottom = income-included), one panel per country,
using the project's single best-overall champion family/quantile
(outputs/tables/table_family_champions_final.csv) so the comparison isolates the
income-inclusion effect rather than a family confound.

Households in each panel are sorted by their own conditional_threshold and plotted
as a line; each household's observed cost is scattered on top, coloured by its
resulting four-class outcome -- so a reader can see directly how adding income as a
predictor reshapes which households cross into Double Risk / Expenditure Risk /
Income Risk / No Risk, not just the aggregate class-prevalence shift already
reported in table_income_ablation.csv.

Outputs:
  outputs/figures/threshold_scatter_income_states.png/.pdf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import HOUSEHOLD_SHEET, retained_cohort
from lib.features.household_features import BASE_RAW_COLUMNS, RAW_COLUMNS as FEATURE_RAW_COLUMNS, engineer
from lib.labels.income_harmonization import harmonized_income_percentile
from lib.reporting.figures import save_figure
from measurement.families.model_based import fit_measurement_strategy

from data.enable import RETAINED_COUNTRY_NAMES
COUNTRY_ORDER = list(RETAINED_COUNTRY_NAMES)
COMP = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]
RISK_ORDER = ["No Risk", "Income Risk", "Expenditure Risk", "Double Risk"]
RISK_COLOR = {"No Risk": "#0ca30c", "Income Risk": "#fab219", "Expenditure Risk": "#ec835a", "Double Risk": "#d03b3b"}
THRESHOLD_LINE_COLOR = "#0b0b0b"
SURFACE, GRID, INK, INK_MUTED = "#fcfcfb", "#e1e0d9", "#0b0b0b", "#898781"


def load_frame(cfg: dict) -> pd.DataFrame:
    base = pd.read_parquet(ROOT / "artifacts/data/harmonised_observed_cost.parquet")
    cols = ["UID", "Country", "NUTS1"] + BASE_RAW_COLUMNS + FEATURE_RAW_COLUMNS + COMP
    hh = retained_cohort(pd.read_excel(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx",
                                        sheet_name=HOUSEHOLD_SHEET, usecols=list(dict.fromkeys(cols))))
    hh["household_size"] = hh[COMP].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(COMP))
    hh = engineer(hh)
    d = base.merge(hh.drop(columns=["Country", "NUTS1", "country"]), on="UID", how="left", validate="one_to_one")
    d["harmonized_income_percentile"] = harmonized_income_percentile(d.income_bracket, d.country)
    split = json.loads((ROOT / cfg["split"]).read_text())["assignments"]
    part = {str(x["household_id"]): x["partition"] for x in split}
    d["partition"] = d.UID.astype(str).map(part)
    return d


def plot_row(axes, d: pd.DataFrame, title_prefix: str) -> None:
    d = d.dropna(subset=["conditional_threshold", "observed_energy_cost_eur_reference", "risk_class_name", "country"])
    for ax, country in zip(axes, COUNTRY_ORDER):
        g = d[d.country.eq(country)].sort_values("conditional_threshold").reset_index(drop=True)
        if g.empty:
            ax.axis("off")
            continue
        x = np.arange(len(g))
        ax.plot(x, g.conditional_threshold, color=THRESHOLD_LINE_COLOR, linewidth=1.5, zorder=3)
        for cls in RISK_ORDER:
            mask = (g.risk_class_name == cls).to_numpy()
            ax.scatter(x[mask], g.observed_energy_cost_eur_reference.to_numpy()[mask], s=8,
                       color=RISK_COLOR[cls], alpha=0.65, linewidths=0, zorder=2, label=cls)
        ax.set_title(f"{title_prefix}\n{country} (n={len(g)})", fontsize=9)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7, colors=INK_MUTED)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_MUTED)


def main() -> None:
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions_final.csv")
    champions = champions[champions.family.ne("M0")].sort_values("macro_f1", ascending=False)
    if champions.empty:
        print("No non-M0 champions in table_family_champions_final.csv -- run scripts/select_family_champions.py first.")
        return
    champ = champions.iloc[0]
    family, quantile = champ["family"], float(champ["quantile"])   # bracket access: Series.quantile is a method, not the column
    print(f"Using champion: {family} (quantile={quantile:.3f})")

    d = load_frame(cfg)
    h_features = list(cfg["features"])
    train = d[d.partition.eq("train")]

    fig, axes = plt.subplots(2, len(COUNTRY_ORDER), figsize=(3.6 * len(COUNTRY_ORDER), 8.4), sharey="row")
    fig.patch.set_facecolor(SURFACE)
    for row, (arm, features) in zip(axes, (("Income excluded", h_features),
                                            ("Income included (Van-Hove-style)",
                                             h_features + ["harmonized_income_percentile"]))):
        _, measured = fit_measurement_strategy(family, train, d, features=features, target=cfg["target"],
                                                 quantile=quantile, tune=False)
        plot_row(row, measured, arm)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=RISK_COLOR[c], label=c) for c in RISK_ORDER]
    handles.append(plt.Line2D([0], [0], color=THRESHOLD_LINE_COLOR, linewidth=1.5, label="Conditional threshold"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), labelcolor=INK_MUTED)
    fig.suptitle(f"Threshold scatter by four-class outcome -- {family} champion, income excluded vs. included",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    save_figure(fig, ROOT / "outputs/figures/threshold_scatter_income_states")
    plt.close(fig)
    print("Wrote outputs/figures/threshold_scatter_income_states.png/.pdf")


if __name__ == "__main__":
    main()
