"""NUTS1 choropleth policy maps: where is which kind of energy-poverty risk concentrated.

Reuses each measurement family's already-selected champion
(outputs/tables/table_family_champions.csv) rather than refitting anything.
NUTS1 regions with fewer than MIN_N households are shown hatched/grey ("insufficient
sample") rather than colour-coded, so a single-household region (ES7 / Canarias, n=1
in this cohort) can never visually imply a policy conclusion.

Outputs:
  outputs/figures/policy_map_double_risk_by_family.png/.pdf   (small multiples, one per family)
  outputs/figures/policy_map_dominant_need.png/.pdf           (categorical, overall-best family champion)
  outputs/tables/table_policy_map_by_nuts1.csv
"""
from __future__ import annotations
import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure
from lib.reporting.codes import regression_code

MIN_N = 20  # below this, a NUTS1 region is shown as "insufficient sample", never colour-coded
RISK_ORDER = ["No Risk", "Income Risk", "Expenditure Risk", "Double Risk"]
RISK_COLOR = {"No Risk": "#0ca30c", "Income Risk": "#fab219", "Expenditure Risk": "#ec835a", "Double Risk": "#d03b3b"}
INSUFFICIENT_COLOR = "#e1e0d9"
COUNTRY_CODES = {"Bulgaria": "BG", "Hungary": "HU", "Italy": "IT", "Poland": "PL"}


def load_geometries() -> gpd.GeoDataFrame:
    g = gpd.read_file(ROOT / "artifacts/reference/NUTS_RG_01M_2016_4326_LEVL_1.geojson")
    return g[g.CNTR_CODE.isin(COUNTRY_CODES.values())][["NUTS_ID", "NAME_LATN", "CNTR_CODE", "geometry"]]


def country_label_points(geo: gpd.GeoDataFrame) -> dict[str, tuple[float, float]]:
    """One label point per country (ISO2 code), placed inside the country's own
    dissolved geometry via representative_point() -- safe for concave/multi-part
    shapes (e.g. Italy with Sicily and Sardinia) where a bounding-box or
    convex-hull centroid can fall outside the landmass or over the sea."""
    dissolved = geo.dissolve(by="CNTR_CODE")
    points = dissolved.geometry.representative_point()
    return {code: (pt.x, pt.y) for code, pt in points.items()}


def _label_countries(ax, label_points: dict[str, tuple[float, float]]) -> None:
    for code, (x, y) in label_points.items():
        ax.annotate(code, (x, y), ha="center", va="center", fontsize=8, fontweight="bold",
                    color="#0b0b0b", path_effects=[pe.withStroke(linewidth=2.2, foreground="#fcfcfb")])


def nuts1_shares(measurement: str, class_col: str, order: list[str]) -> pd.DataFrame:
    d = pd.read_parquet(ROOT / f"artifacts/predictions/{measurement}.parquet")
    d = d.dropna(subset=[class_col, "NUTS1"])
    counts = d.groupby("NUTS1").size().rename("n")
    tab = pd.crosstab(d["NUTS1"], d[class_col], normalize="index").reindex(columns=order, fill_value=0.0)
    return tab.join(counts)


def _draw_choropleth(ax, geo: gpd.GeoDataFrame, values: pd.Series, n: pd.Series, *, cmap, vmin, vmax, label,
                      label_points: dict[str, tuple[float, float]] | None = None) -> None:
    merged = geo.merge(values.rename("value"), left_on="NUTS_ID", right_index=True, how="left")
    merged = merged.merge(n.rename("n"), left_on="NUTS_ID", right_index=True, how="left")
    sufficient = merged.n.fillna(0) >= MIN_N
    merged[sufficient].plot(ax=ax, column="value", cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="#fcfcfb", linewidth=0.4)
    if (~sufficient).any():
        merged[~sufficient].plot(ax=ax, color=INSUFFICIENT_COLOR, edgecolor="#fcfcfb", linewidth=0.4, hatch="///")
    if label_points:
        _label_countries(ax, label_points)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
    cbar = ax.figure.colorbar(sm, ax=ax, shrink=0.65, pad=0.01)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def build_double_risk_grid(geo: gpd.GeoDataFrame, champions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = champions[champions.status.eq("AVAILABLE")]
    label_points = country_label_points(geo)
    ncols = 4
    nrows = int(np.ceil(len(families) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, families.itertuples()):
        tab = nuts1_shares(row.champion_experiment_id, "risk_class_name", RISK_ORDER)
        _draw_choropleth(ax, geo, tab["Double Risk"] * 100, tab["n"], cmap="Reds", vmin=0, vmax=60,
                          label="Double Risk share (%)", label_points=label_points)
        r_code = regression_code(row.family)
        quantile_note = "" if pd.isna(row.quantile) else f" q{round(row.quantile * 100):02d}"
        ax.set_title(f"{r_code} ({row.family}{quantile_note})", fontsize=9)
        for nuts_id, r in tab.iterrows():
            rows.append({"family": row.family, "measurement": row.champion_experiment_id, "NUTS1": nuts_id,
                         "n": int(r["n"]), **{cls: float(r[cls]) for cls in RISK_ORDER}})
    for ax in axes[len(families):]:
        ax.axis("off")
    fig.suptitle("Double-Risk (low income + high cost) prevalence by NUTS1 region -- every screening-eligible family", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, ROOT / "outputs/figures/policy_map_double_risk_by_family")
    plt.close(fig)
    return pd.DataFrame(rows)


def build_dominant_need_map(geo: gpd.GeoDataFrame, champion_row) -> None:
    tab = nuts1_shares(champion_row.champion_experiment_id, "risk_class_name", RISK_ORDER)
    dominant = tab[RISK_ORDER].idxmax(axis=1)
    merged = geo.merge(dominant.rename("dominant"), left_on="NUTS_ID", right_index=True, how="left")
    merged = merged.merge(tab["n"], left_on="NUTS_ID", right_index=True, how="left")
    fig, ax = plt.subplots(figsize=(8, 7))
    sufficient = merged.n.fillna(0) >= MIN_N
    for cls in RISK_ORDER:
        sel = merged[sufficient & merged.dominant.eq(cls)]
        if len(sel):
            sel.plot(ax=ax, color=RISK_COLOR[cls], edgecolor="#fcfcfb", linewidth=0.5, label=cls)
    if (~sufficient).any():
        merged[~sufficient].plot(ax=ax, color=INSUFFICIENT_COLOR, edgecolor="#fcfcfb", linewidth=0.5, hatch="///",
                                  label=f"Insufficient sample (n<{MIN_N})")
    _label_countries(ax, country_label_points(geo))
    ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=RISK_COLOR[c]) for c in RISK_ORDER]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=INSUFFICIENT_COLOR, hatch="///"))
    ax.legend(handles, RISK_ORDER + [f"Insufficient sample (n<{MIN_N})"], loc="lower left", fontsize=8, frameon=False)
    # champion_row is a `.iloc[0]` Series here -- use bracket access for "quantile" to
    # avoid pandas' Series.quantile() method shadowing plain attribute access (see
    # AUDIT_AND_REPAIR_REPORT.md / scripts/build_b_measurements.py for the same bug class).
    family = champion_row["family"]
    r_code = regression_code(family)
    quantile = champion_row["quantile"]
    quantile_note = "" if pd.isna(quantile) else f" q{round(quantile * 100):02d}"
    ax.set_title(f"Dominant risk class by NUTS1 region -- {r_code} ({family}{quantile_note}, "
                 f"{champion_row['champion_experiment_id']}, overall macro-F1 champion)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, ROOT / "outputs/figures/policy_map_dominant_need")
    plt.close(fig)


def main() -> None:
    geo = load_geometries()
    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions.csv")
    table = build_double_risk_grid(geo, champions)
    table.to_csv(ROOT / "outputs/tables/table_policy_map_by_nuts1.csv", index=False)

    final = ROOT / "outputs/tables/table_family_champions_final.csv"
    if final.exists():
        finals = pd.read_csv(final).sort_values("macro_f1", ascending=False)
        overall = champions[champions.family.eq(finals.iloc[0].family)].iloc[0]
        build_dominant_need_map(geo, overall)

    print(f"Wrote policy_map_double_risk_by_family, policy_map_dominant_need -> "
          f"outputs/figures/; {len(table)} NUTS1 x family rows -> outputs/tables/table_policy_map_by_nuts1.csv "
          f"(MIN_N={MIN_N} households required to colour-code a region)")


if __name__ == "__main__":
    main()
