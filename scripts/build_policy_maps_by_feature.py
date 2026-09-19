"""Policy maps segmented by household/demographic feature: not just "where is risk
concentrated overall" (scripts/build_policy_maps.py) but "does risk concentration
differ by dwelling type, heating fuel, respondent gender, or settlement type, and
where." This is the project's actual purpose stated by the user directly: how does
each observable factor relate to energy-poverty risk, per country/region -- found
by reading the already-fitted champion's predictions back against the household's
own raw feature values, not by fitting anything new.

One dashboard figure: one row per segmenting feature, one Double-Risk-share
NUTS1 choropleth panel per category of that feature (2 contrasting categories per
feature, so the grid stays legible). Uses the single best-overall champion family
(outputs/tables/table_family_champions_final.csv).

Outputs:
  outputs/figures/policy_map_by_feature.png/.pdf
  outputs/tables/table_policy_map_by_feature.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data.enable import HOUSEHOLD_SHEET, retained_cohort
from lib.features.household_features import BASE_RAW_COLUMNS, RAW_COLUMNS as FEATURE_RAW_COLUMNS, engineer
from lib.reporting.codes import regression_code
from lib.reporting.figures import save_figure
from scripts.build_policy_maps import load_geometries, country_label_points, _draw_choropleth

COMP = [f"S1{g}c{i}" for g in ("A", "B") for i in (1, 2, 3)]
# (column, {category shown: raw value(s) it matches}) -- two contrasting categories per
# feature, chosen for a clear policy contrast (e.g. detached vs apartment, not all 4 H1 codes).
SEGMENTS: dict[str, dict[str, object]] = {
    "Dwelling type (H1)": {"Detached house": [1], "Apartment (6+ flats)": [4]},
    "Primary heating fuel": {"Solid fuel (wood/coal/biomass)": ["wood", "coal_coke", "pellets", "biomass"],
                              "Gas / district / electric": ["gas_lpg", "district_heating", "electricity"]},
    "Respondent gender": {"Male": ["Male"], "Female": ["Female"]},
    "Settlement type": {"Rural (<10,000)": [1, 2], "Urban (50,000+ / capital)": [4, 5]},
}


def load_household_frame() -> pd.DataFrame:
    hh = retained_cohort(pd.read_excel(
        ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx", sheet_name=HOUSEHOLD_SHEET,
        usecols=list(dict.fromkeys(["UID", "Country", "NUTS1"] + BASE_RAW_COLUMNS + FEATURE_RAW_COLUMNS + COMP))))
    hh["household_size"] = hh[COMP].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=len(COMP))
    hh = engineer(hh)
    return hh


def main() -> None:
    final = ROOT / "outputs/tables/table_family_champions_final.csv"
    if not final.exists():
        print("No table_family_champions_final.csv -- run scripts/select_family_champions.py first.")
        return
    champ = pd.read_csv(final).sort_values("macro_f1", ascending=False).iloc[0]
    src = ROOT / f"artifacts/predictions/{champ.champion_experiment_id}.parquet"
    if not src.exists():
        print(f"Champion prediction file {src.name} not found.")
        return
    pred = pd.read_parquet(src)[["UID", "NUTS1", "risk_class_name"]]
    hh = load_household_frame().drop(columns=["NUTS1", "Country"])
    d = pred.merge(hh, on="UID", how="inner", validate="one_to_one")
    d["H1"] = pd.to_numeric(d["H1"], errors="coerce")
    d["S6"] = pd.to_numeric(d["S6"], errors="coerce")

    geo = load_geometries()
    label_points = country_label_points(geo)
    ncols = 2
    nrows = len(SEGMENTS)
    # The 4-country bounding box (Italy-Poland) is much wider than tall in lat/lon --
    # match panel aspect to that so geopandas' equal-aspect plot doesn't leave large
    # empty vertical margins inside each cell.
    minx, miny, maxx, maxy = geo.total_bounds
    panel_h = 4.6 * (maxy - miny) / (maxx - minx) + 0.9  # +title/colorbar allowance
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, panel_h * nrows))
    fig.patch.set_facecolor("#fcfcfb")
    rows = []
    for r, (feature_label, categories) in enumerate(SEGMENTS.items()):
        source_col = "S6" if feature_label == "Settlement type" else \
                     "H1" if feature_label == "Dwelling type (H1)" else \
                     "primary_heating_fuel" if feature_label == "Primary heating fuel" else "gender"
        for c, (cat_label, values) in enumerate(categories.items()):
            ax = axes[r, c]
            sub = d[d[source_col].isin(values)]
            counts = sub.groupby("NUTS1").size().rename("n")
            share = (sub[sub.risk_class_name.eq("Double Risk")].groupby("NUTS1").size() / counts * 100).fillna(0.0)
            _draw_choropleth(ax, geo, share, counts, cmap="Reds", vmin=0, vmax=60,
                              label="Double Risk share (%)", label_points=label_points)
            ax.set_title(f"{feature_label}\n{cat_label} (n={len(sub)})", fontsize=9)
            for nuts_id in counts.index:
                rows.append({"feature": feature_label, "category": cat_label, "NUTS1": nuts_id,
                             "n": int(counts[nuts_id]), "double_risk_share": float(share.get(nuts_id, 0.0))})
    r_code = regression_code(champ.family)
    fig.suptitle(f"Double-Risk prevalence by NUTS1 region, segmented by household factor\n"
                 f"({r_code} / {champ.family}, {champ.champion_experiment_id}, overall best configuration)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.6, w_pad=0.6)
    save_figure(fig, ROOT / "outputs/figures/policy_map_by_feature")
    plt.close(fig)

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "outputs/tables/table_policy_map_by_feature.csv", index=False)
    print(f"Wrote policy_map_by_feature ({nrows} features x 2 categories) -> outputs/figures/, "
          f"{len(table)} rows -> outputs/tables/table_policy_map_by_feature.csv")


if __name__ == "__main__":
    main()
