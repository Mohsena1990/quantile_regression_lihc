"""
Restriction-flag search: does *under*-consumption relative to structural
need track hardship better than *over*-consumption does (HQRTM/LIHC's
"high cost" flag)?

Motivation
----------
hqrtm_specification_search.py showed HQRTM's "high cost" flag (actual
expenditure ABOVE a conditional quantile of similar households' spending)
has no positive relationship with independent hardship markers, and a
consistently negative one with income_difficulty -- see README's Central
Finding. The diagnosed mechanism: a household that spends more than
structurally-similar households isn't necessarily worse off (could just
be comfortable/inefficient), while a household that RATIONS -- heats one
room, self-disconnects, goes without -- spends LESS than its structural
profile would predict, and the high-cost flag never catches it.

This script tests the mirror-image flag directly: actual expenditure
BELOW a conditional quantile of similar households' spending ("restricting
relative to need"), using the exact same pooled quantile regression
machinery, just evaluated at low quantiles instead of high ones. If the
mechanism we diagnosed is right, this flag should show a POSITIVE,
correctly-signed relationship with hardship markers where HQRTM's flag
showed none (or a negative one) -- particularly with C1A (cold home),
the most mechanistically direct marker for heating restriction.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from external_validity_check import build_markers, MARKERS, MARKER_LABELS, CONTROLS

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation" / "restriction_flag_search"

QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "household_size",
]
EXP_COL = "total_expenditure"
COUNTRY_COL = "Country"

LOW_QUANTILES = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
FEATURE_SETS = {
    "baseline": QR_FEATURES,
    "+SettlementSize": QR_FEATURES + ["settlement_size_clean"],
}


def clean_settlement(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["settlement_size_clean"] = out["SettlementSize"].where(out["SettlementSize"] != "No answer")
    return out


def fit_restricting_flag(df: pd.DataFrame, features: list, quantile: float):
    model_cols = [EXP_COL] + features + [COUNTRY_COL]
    fit_data = df[model_cols].replace([np.inf, -np.inf], np.nan).dropna()
    formula = f"{EXP_COL} ~ " + " + ".join(features + [f"C({COUNTRY_COL})"])
    res = smf.quantreg(formula, data=fit_data).fit(q=quantile, max_iter=5000, disp=False)

    pred_data = df[features + [COUNTRY_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    expected = pd.Series(index=df.index, dtype=float)
    expected.loc[pred_data.index] = res.predict(pred_data)

    flag = (df[EXP_COL] < expected).where(expected.notna())
    return flag.astype("boolean"), res.prsquared, len(fit_data)


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lihc_df = pd.read_csv(REPO_ROOT / "df_lihc.csv", low_memory=False)
    lihc_df = clean_settlement(lihc_df)
    markers = build_markers(lihc_df)

    lihc_high_cost = lihc_df["high_exp_flag"].astype(int)
    hqrtm_65 = pd.read_csv(REPO_ROOT / "df_hqrtm_65.csv", low_memory=False)
    hqrtm_high_cost = hqrtm_65["high_exp_flag"].astype(int)

    grid_rows = []
    for feature_label, features in FEATURE_SETS.items():
        for quantile in LOW_QUANTILES:
            flag, pseudo_r2, n_fit = fit_restricting_flag(lihc_df, features, quantile)
            restrict_rate = flag.mean()

            frame = pd.DataFrame(
                {
                    "Country": lihc_df["Country"].values,
                    "equivalized_income": lihc_df["equivalized_income"].values,
                    "household_size": lihc_df["household_size"].values,
                    "restricting": flag,  # nullable boolean -- cast to int only after dropna below
                    "hqrtm_high_cost": hqrtm_high_cost.values,
                    "lihc_high_cost": lihc_high_cost.values,
                }
            )
            frame = pd.concat([frame, markers.reset_index(drop=True)], axis=1)

            for marker in MARKERS:
                cols = ["restricting", "hqrtm_high_cost", "lihc_high_cost", "Country", marker] + CONTROLS
                sub = frame[cols].dropna().copy()
                sub["restricting"] = sub["restricting"].astype(int)
                controls_rhs = " + ".join(CONTROLS + ["C(Country)"])

                solo = smf.logit(f"{marker} ~ restricting + {controls_rhs}", data=sub).fit(disp=False)
                solo_ci = solo.conf_int().loc["restricting"]
                try:
                    combined = smf.logit(
                        f"{marker} ~ restricting + hqrtm_high_cost + lihc_high_cost + {controls_rhs}", data=sub
                    ).fit(disp=False)
                    combined_or = np.exp(combined.params["restricting"])
                    combined_p = combined.pvalues["restricting"]
                except Exception:
                    combined_or, combined_p = np.nan, np.nan

                grid_rows.append(
                    {
                        "feature_set": feature_label,
                        "quantile": quantile,
                        "restrict_rate": restrict_rate,
                        "pseudo_r2_qr": pseudo_r2,
                        "n_fit": n_fit,
                        "marker": marker,
                        "solo_odds_ratio": np.exp(solo.params["restricting"]),
                        "solo_or_ci95_low": np.exp(solo_ci[0]),
                        "solo_or_ci95_high": np.exp(solo_ci[1]),
                        "solo_p_value": solo.pvalues["restricting"],
                        "combined_odds_ratio": combined_or,
                        "combined_p_value": combined_p,
                        "n": int(solo.nobs),
                    }
                )
            print(
                f"{feature_label:16s} q={quantile:.2f}: restrict_rate={restrict_rate:.2%}, "
                f"QR pseudo_R2={pseudo_r2:.3f}, n_fit={n_fit}"
            )

    results = pd.DataFrame(grid_rows)
    results.to_csv(OUTPUT_DIR / "restriction_flag_search_results.csv", index=False)

    print("\n=== Significant (p<0.05), correctly-signed (OR>1) restricting associations (solo model) ===")
    hits = results[(results["solo_p_value"] < 0.05) & (results["solo_odds_ratio"] > 1)]
    if hits.empty:
        print("None found across the grid.")
    else:
        print(hits.sort_values("solo_p_value").to_string(index=False))

    print("\n=== By marker, baseline feature set, solo odds ratios across the low-quantile grid ===")
    base = results[results["feature_set"] == "baseline"]
    for marker in MARKERS:
        sub = base[base["marker"] == marker].sort_values("quantile")
        print(f"\n{MARKER_LABELS[marker]}")
        for _, row in sub.iterrows():
            print(
                f"  q={row['quantile']:.2f}  restrict_rate={row['restrict_rate']:.1%}  "
                f"solo OR={row['solo_odds_ratio']:.2f} (p={row['solo_p_value']:.4f})  "
                f"combined OR={row['combined_odds_ratio']:.2f} (p={row['combined_p_value']:.4f})"
            )


if __name__ == "__main__":
    run_all()
