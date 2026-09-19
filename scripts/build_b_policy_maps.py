"""NUTS1 choropleth policy maps for the B-family (B2/B5/B7): each household measured using
its own country's winning feature config (outputs/tables/table_percountry_winners.csv),
stitched by scripts/build_b_measurements.py.

Mirrors scripts/build_policy_maps.py's double-risk small-multiples and dominant-need map,
pointed at artifacts/predictions/b{family}.parquet instead of the pooled champion
predictions -- no refitting.

Outputs:
  outputs/figures/policy_map_b_double_risk_by_family.png/.pdf
  outputs/figures/policy_map_b_dominant_need.png/.pdf
  outputs/tables/table_b_policy_map_by_nuts1.csv
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure
from lib.reporting.codes import regression_code
from scripts.build_policy_maps import (RISK_ORDER, RISK_COLOR, INSUFFICIENT_COLOR, MIN_N,
                                        load_geometries, nuts1_shares, _draw_choropleth,
                                        country_label_points, _label_countries)

FAMILIES = [f"B{f[1:]}" for f in json.loads((ROOT / "config/pipeline.json").read_text())["families"]]


def build_double_risk_grid(geo) -> pd.DataFrame:
    rows = []
    available = [(fam, ROOT / f"artifacts/predictions/b{fam[1:]}.parquet") for fam in FAMILIES]
    available = [(fam, p) for fam, p in available if p.exists()]
    label_points = country_label_points(geo)
    ncols = 4
    nrows = int(np.ceil(len(available) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (fam, p) in zip(axes, available):
        tab = nuts1_shares(p.stem, "risk_class_name", RISK_ORDER)
        _draw_choropleth(ax, geo, tab["Double Risk"] * 100, tab["n"], cmap="Reds", vmin=0, vmax=60,
                          label="Double Risk share (%)", label_points=label_points)
        r_code = regression_code(f"M{fam[1:]}")
        ax.set_title(f"{r_code} ({fam}, per-country winning config)", fontsize=9)
        for nuts_id, r in tab.iterrows():
            rows.append({"family": fam, "measurement": p.stem, "NUTS1": nuts_id,
                         "n": int(r["n"]), **{cls: float(r[cls]) for cls in RISK_ORDER}})
    for ax in axes[len(available):]:
        ax.axis("off")
    fig.suptitle("B-family: Double-Risk (low income + high cost) prevalence by NUTS1 region -- "
                 "own country's winning feature config per household", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, ROOT / "outputs/figures/policy_map_b_double_risk_by_family")
    plt.close(fig)
    return pd.DataFrame(rows)


def build_dominant_need_map(geo, table: pd.DataFrame) -> None:
    if table.empty:
        return
    b_screening = ROOT / "outputs/tables/table_b_screening_by_country.csv"
    if not b_screening.exists():
        return
    scores = pd.read_csv(b_screening)
    scores = scores[scores.status.eq("AVAILABLE")]
    ranked = scores.groupby("family").macro_f1.mean().sort_values(ascending=False)
    if ranked.empty:
        return
    top_family = ranked.index[0]
    measurement = f"b{top_family[1:]}"
    tab = nuts1_shares(measurement, "risk_class_name", RISK_ORDER)
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
    r_code = regression_code(f"M{top_family[1:]}")
    ax.set_title(f"B-family dominant risk class by NUTS1 region -- {r_code} ({top_family}) "
                 f"(overall macro-F1 champion among B-families, own per-country winning config)", fontsize=11)
    fig.tight_layout()
    save_figure(fig, ROOT / "outputs/figures/policy_map_b_dominant_need")
    plt.close(fig)


def main() -> None:
    geo = load_geometries()
    table = build_double_risk_grid(geo)
    table.to_csv(ROOT / "outputs/tables/table_b_policy_map_by_nuts1.csv", index=False)
    build_dominant_need_map(geo, table)
    print(f"Wrote policy_map_b_double_risk_by_family, policy_map_b_dominant_need -> outputs/figures/; "
          f"{len(table)} NUTS1 x family rows -> outputs/tables/table_b_policy_map_by_nuts1.csv (MIN_N={MIN_N})")


if __name__ == "__main__":
    main()
