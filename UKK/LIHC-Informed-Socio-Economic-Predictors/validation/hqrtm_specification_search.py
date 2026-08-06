"""
Specification search: is HQRTM's lack of external-validity advantage over
traditional LIHC (see external_validity_check.py, rerun after the
total_expenditure fix in preprocessing.py) specific to the one
quantile/margin combination used so far, or does it hold across the
plausible specification space?

Motivation
----------
HQRTM has two design choices that were never independently justified:
  - quantile: which conditional quantile counts as "expected" expenditure
    (0.60/0.65/0.70 were used throughout, but that range itself was an
    unexplained restriction baked into assign_hqrtm's validation check).
  - margin_scale: an additional buffer added on top of the quantile
    threshold (margin = margin_scale * std(fitted expenditure)), with no
    stated rationale anywhere, that pushes the empirical high-cost rate
    well below the nominal (1 - quantile) rate.

This sweeps both, refits the pooled HQRTM threshold at each combination,
and reruns the horse-race regression against the four external markers
(S7/C5B/C1A/S8) for every combination, so a genuine advantage at some
untried combination isn't mistaken for "HQRTM doesn't work."
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from external_validity_check import build_markers, MARKERS, CONTROLS

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation" / "hqrtm_specification_search"

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

QUANTILES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
MARGIN_SCALES = [0.0, 0.05, 0.10, 0.20]


def fit_hqrtm_flag(df: pd.DataFrame, quantile: float, margin_scale: float) -> pd.Series:
    model_cols = [EXP_COL] + QR_FEATURES + [COUNTRY_COL]
    fit_data = df[model_cols].replace([np.inf, -np.inf], np.nan).dropna()
    formula = f"{EXP_COL} ~ " + " + ".join(QR_FEATURES + [f"C({COUNTRY_COL})"])
    res = smf.quantreg(formula, data=fit_data).fit(q=quantile, max_iter=5000, disp=False)

    pred_data = df[QR_FEATURES + [COUNTRY_COL]].replace([np.inf, -np.inf], np.nan).dropna()
    expected = pd.Series(index=df.index, dtype=float)
    expected.loc[pred_data.index] = res.predict(pred_data)

    margin = np.nanstd(fit_data[EXP_COL]) * margin_scale
    flag = (df[EXP_COL] > (expected + margin)).where(expected.notna())
    return flag.astype("boolean"), res.prsquared


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lihc_df = pd.read_csv(REPO_ROOT / "df_lihc.csv", low_memory=False)
    markers = build_markers(lihc_df)

    rows = []
    for quantile in QUANTILES:
        for margin_scale in MARGIN_SCALES:
            flag, pseudo_r2 = fit_hqrtm_flag(lihc_df, quantile, margin_scale)
            high_cost_rate = flag.mean()

            frame = pd.DataFrame(
                {
                    "Country": lihc_df["Country"].values,
                    "equivalized_income": lihc_df["equivalized_income"].values,
                    "household_size": lihc_df["household_size"].values,
                    "hqrtm_high_cost": flag.astype("boolean"),
                }
            )
            frame = pd.concat([frame, markers.reset_index(drop=True)], axis=1)

            for marker in MARKERS:
                cols = ["hqrtm_high_cost", "Country", marker] + CONTROLS
                sub = frame[cols].dropna().copy()
                if sub["hqrtm_high_cost"].nunique() < 2:
                    continue
                # patsy/statsmodels can't interpret pandas' nullable
                # "boolean" dtype directly, and treats plain `bool` as
                # categorical (producing a `[T.True]`-style param name) --
                # cast to int now that NAs have already been dropped above.
                sub["hqrtm_high_cost"] = sub["hqrtm_high_cost"].astype(int)
                controls_rhs = " + ".join(CONTROLS + ["C(Country)"])
                try:
                    fit = smf.logit(f"{marker} ~ hqrtm_high_cost + {controls_rhs}", data=sub).fit(disp=False)
                except Exception as exc:
                    print(f"  q={quantile} margin={margin_scale} {marker}: failed ({exc})")
                    continue
                coef = fit.params.get("hqrtm_high_cost", np.nan)
                p = fit.pvalues.get("hqrtm_high_cost", np.nan)
                rows.append(
                    {
                        "quantile": quantile,
                        "margin_scale": margin_scale,
                        "high_cost_rate": high_cost_rate,
                        "pseudo_r2_qr": pseudo_r2,
                        "marker": marker,
                        "odds_ratio": np.exp(coef),
                        "p_value": p,
                        "n": int(fit.nobs),
                    }
                )
            print(f"q={quantile:.2f} margin_scale={margin_scale:.2f}: high_cost_rate={high_cost_rate:.2%}, QR pseudo_R2={pseudo_r2:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_DIR / "specification_search_results.csv", index=False)

    print("\n=== Significant (p<0.05), correctly-signed (OR>1) hqrtm_high_cost associations ===")
    hits = results[(results["p_value"] < 0.05) & (results["odds_ratio"] > 1)]
    if hits.empty:
        print("None found across the full grid.")
    else:
        print(hits.sort_values("p_value").to_string(index=False))

    print("\n=== income_difficulty across the full grid ===")
    inc = results[results["marker"] == "income_difficulty"].sort_values(["quantile", "margin_scale"])
    print(inc.to_string(index=False))


if __name__ == "__main__":
    run_all()
