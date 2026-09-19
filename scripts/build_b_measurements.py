"""B0-B8: regenerate each measurement family's four-class labels using, for every
household, its own country's winning feature config from
scripts/run_percountry_feature_selection.py (outputs/tables/table_percountry_winners.csv).

B0 = M0 unchanged (Van Hove has no per-household feature model to select over).
B2-B8 = for each country, refit that family's measurement using only that country's
winning config's features (cheap -- the expensive SHAP/LassoNet selection and 3x2
classifier fits were already done per config in the per-country experiment; this just
re-fits the one already-identified winning measurement per family/country and
concatenates across the 5 countries into one stitched dataset). The winning config's
classifier results are *not* refit here -- they are the exact rows already written by
run_percountry_feature_selection.py, just filtered to the winning config per
family/country, so the same country's own model that fed the selection also produces
the B-family headline numbers.

Outputs:
  artifacts/predictions/b{family_number}.parquet   (stitched four-class labels, all countries)
  outputs/tables/table_b_screening_by_country.csv  (winning-config classifier results, per family x country)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from measurement.families.model_based import fit_measurement_strategy
from lib.features.country_context import add_country_context


def main() -> None:
    winners_path = ROOT / "outputs/tables/table_percountry_winners.csv"
    if not winners_path.exists():
        raise SystemExit("Run scripts/run_percountry_feature_selection.py first (table_percountry_winners.csv missing)")
    winners = pd.read_csv(winners_path)
    selected_features = pd.read_csv(ROOT / "outputs/tables/table_percountry_selected_features.csv")
    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions.csv")
    champions = champions[champions.status.eq("AVAILABLE")]
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())

    stitched_summary = []
    for family, champ_group in winners.groupby("family"):
        champ_row = champions[champions.family.eq(family)].iloc[0]
        src = ROOT / f"artifacts/predictions/m{family[1:]}_a1.parquet"
        full = add_country_context(pd.read_parquet(src))
        country_frames = []
        for row in champ_group.itertuples():
            feat_row = selected_features[selected_features.family.eq(family) & selected_features.country.eq(row.country)
                                          & selected_features.config.eq(row.config)]
            if feat_row.empty:
                print(f"MISSING_FEATURES {family} {row.country} {row.config}")
                continue
            feature_list = feat_row.iloc[0].features.split(";")
            d = full[full.country.eq(row.country)]
            train = d[d.partition.eq("train")]
            _, measured = fit_measurement_strategy(family, train, d, features=feature_list,
                                                     target=cfg["target"], quantile=champ_row["quantile"])
            measured["winning_config"] = row.config
            measured["n_features"] = len(feature_list)
            country_frames.append(measured)
            stitched_summary.append({"family": family, "country": row.country, "winning_config": row.config,
                                      "n_features": len(feature_list), "macro_f1": row.macro_f1})
        if not country_frames:
            continue
        stitched = pd.concat(country_frames, ignore_index=True)
        stitched.to_parquet(ROOT / f"artifacts/predictions/b{family[1:]}.parquet", index=False)
        print(f"Wrote b{family[1:]}.parquet ({len(stitched)} rows, {stitched.country.nunique()} countries)")

    # B0 = M0 unchanged.
    m0 = ROOT / "artifacts/predictions/m0_predictive.parquet"
    if m0.exists():
        pd.read_parquet(m0).to_parquet(ROOT / "artifacts/predictions/b0.parquet", index=False)
        print("Wrote b0.parquet (= m0_predictive, unchanged -- no per-household feature model)")

    summary = pd.DataFrame(stitched_summary)
    summary.to_csv(ROOT / "outputs/tables/table_b_stitch_summary.csv", index=False)

    # The winning-config classifier results are already computed; just filter to them.
    all_metrics = pd.read_csv(ROOT / "outputs/tables/table_percountry_feature_selection.csv")
    all_metrics = all_metrics[all_metrics.status.eq("AVAILABLE")]
    key = winners[["family", "country", "config"]]
    b_screening = all_metrics.merge(key, on=["family", "country", "config"])
    b_screening.to_csv(ROOT / "outputs/tables/table_b_screening_by_country.csv", index=False)
    print(f"B_MEASUREMENTS_COMPLETE families={summary.family.nunique() if len(summary) else 0} "
          f"screening_rows={len(b_screening)}")


if __name__ == "__main__":
    main()
