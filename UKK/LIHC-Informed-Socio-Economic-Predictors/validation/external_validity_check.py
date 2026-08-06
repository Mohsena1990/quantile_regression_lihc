"""
External (criterion) validity check: HQRTM vs traditional LIHC.

Motivation
----------
An Energy Economics handling editor rejected the manuscript because the
CatBoost exercise predicted `risk_category` from household characteristics
that were themselves used to *construct* `risk_category` (via the HQRTM
quantile-regression threshold). That shows internal coherence, not
measurement validity: no amount of predictive accuracy on a self-referential
label can demonstrate the label tracks real-world energy hardship.

This script instead tests HQRTM's "high energy cost" flag against four
ENABLE.EU survey items that were NEVER used anywhere in constructing either
the HQRTM or the traditional LIHC label:

    S7  - received public financial aid / social tariff to pay energy
          bills in the last 12 months (energy_aid)
    C5B - "I don't pay much for heating; paying the bill is not a problem
          for me" (1=strongly disagree .. 5=strongly agree), reverse-coded
          into bill_burden (bill_burden_bin, bill_burden_score)
    C1A - self-reported winter indoor temperature (1=24C+ .. 5=17C or
          below), recoded into cold_home
    S8  - subjective difficulty living on present household income
          (1=comfortable .. 4=very difficult), recoded into income_difficulty

Two analyses are run for each marker, for each HQRTM quantile spec
(0.60 / 0.65 / 0.70):

1. Reclassification (discordant-groups) table: split households into
   Both / HQRTM_only / LIHC_only / Neither flagged as high-cost, and report
   each group's prevalence (with 95% Wilson CIs) on each marker. If HQRTM
   is tracking real hardship, HQRTM_only households should look worse than
   Neither; if LIHC_only households look no different from Neither, the
   conventional fixed threshold was flagging cases HQRTM correctly excludes.

2. Horse-race logistic regressions: marker ~ hqrtm_high_cost +
   lihc_high_cost + equivalized_income + household_size + C(Country),
   plus the two single-flag nested models, with likelihood-ratio tests for
   whether each flag adds explanatory power on top of the other. This is
   the actual criterion-validity evidence the editor's comment was asking
   for: does the flag correlate with independent hardship markers, beyond
   what the other method's flag and basic controls already explain.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation" / "external_validity_results"

HQRTM_SPECS = [
    ("df_hqrtm_60.csv", "HQRTM (q=0.60)"),
    ("df_hqrtm_65.csv", "HQRTM (q=0.65)"),
    ("df_hqrtm_70.csv", "HQRTM (q=0.70)"),
]

MARKERS = ["energy_aid", "bill_burden_bin", "cold_home_bin", "income_difficulty"]
MARKER_LABELS = {
    "energy_aid": "Received public aid / social tariff to pay energy bills (S7)",
    "bill_burden_bin": "Paying the heating bill is a problem (reverse of C5B)",
    "cold_home_bin": "Winter indoor temperature <=19C (C1A)",
    "income_difficulty": "Finding it difficult/very difficult on present income (S8)",
}

CONTROLS = ["equivalized_income", "household_size"]


def build_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Recode raw survey items into hardship markers, excluding DK/NA codes."""
    out = pd.DataFrame(index=df.index)

    s7 = df["S7"].where(df["S7"].isin([1, 2]))
    out["energy_aid"] = s7.map({1: 1, 2: 0})

    c5b = df["C5B"].where(~df["C5B"].isin([6, 99]))
    out["bill_burden_score"] = 6 - c5b  # higher = more burden
    out["bill_burden_bin"] = np.select([c5b <= 2, c5b >= 4], [1, 0], default=np.nan)

    c1a = df["C1A"].where(df["C1A"] != 99)
    out["winter_temp_score"] = c1a  # higher = colder
    out["cold_home_bin"] = np.select([c1a >= 4, c1a <= 3], [1, 0], default=np.nan)

    s8 = df["S8"].where(~df["S8"].isin([5, 99]))
    out["income_difficulty"] = np.select([s8 >= 3, s8 <= 2], [1, 0], default=np.nan)

    return out


def build_analysis_frame(hqrtm_df: pd.DataFrame, lihc_df: pd.DataFrame) -> pd.DataFrame:
    """Combine flags from both labeling methods with markers and controls, row-aligned."""
    if not (hqrtm_df["total_expenditure"].values == lihc_df["total_expenditure"].values).all():
        raise ValueError("hqrtm_df and lihc_df are not row-aligned; cannot merge by position.")

    markers = build_markers(lihc_df)

    # Cross-fitted labels can leave a handful of households with no valid
    # high_exp_flag (e.g. a held-out fold containing a main_heating_source
    # category never seen during that fold's fit -- see
    # cross_fitted_labels.py). Drop them explicitly rather than let
    # `.astype(int)` crash on NA or, worse, silently coerce NA to False and
    # misclassify them as "Neither" / "not double risk".
    valid = hqrtm_df["high_exp_flag"].notna().values & lihc_df["high_exp_flag"].notna().values
    if not valid.all():
        print(f"  Dropping {(~valid).sum()} households with no valid high_exp_flag under one or both methods.")
    hqrtm_df = hqrtm_df.loc[valid]
    lihc_df = lihc_df.loc[valid]
    markers = markers.loc[valid]

    frame = pd.DataFrame(
        {
            "Country": lihc_df["Country"].values,
            "equivalized_income": lihc_df["equivalized_income"].values,
            "household_size": lihc_df["household_size"].values,
            "hqrtm_high_cost": hqrtm_df["high_exp_flag"].astype(int).values,
            "hqrtm_double_risk": (hqrtm_df["risk_category"] == "Double risk").astype(int).values,
            "lihc_high_cost": lihc_df["high_exp_flag"].astype(int).values,
            "lihc_double_risk": (lihc_df["risk_category"] == "Double risk").astype(int).values,
        }
    )
    frame = pd.concat([frame, markers.reset_index(drop=True)], axis=1)

    frame["group"] = np.select(
        [
            (frame["hqrtm_high_cost"] == 1) & (frame["lihc_high_cost"] == 1),
            (frame["hqrtm_high_cost"] == 1) & (frame["lihc_high_cost"] == 0),
            (frame["hqrtm_high_cost"] == 0) & (frame["lihc_high_cost"] == 1),
        ],
        ["Both", "HQRTM_only", "LIHC_only"],
        default="Neither",
    )

    return frame


def reclassification_table(frame: pd.DataFrame, quantile_label: str) -> pd.DataFrame:
    rows = []
    for group in ["Neither", "LIHC_only", "HQRTM_only", "Both"]:
        sub = frame[frame["group"] == group]
        for marker in MARKERS:
            valid = sub[marker].dropna()
            n_valid = len(valid)
            n_positive = int(valid.sum())
            prevalence = n_positive / n_valid if n_valid else np.nan
            if n_valid:
                ci_low, ci_high = proportion_confint(n_positive, n_valid, method="wilson")
            else:
                ci_low, ci_high = (np.nan, np.nan)
            rows.append(
                {
                    "quantile_spec": quantile_label,
                    "group": group,
                    "n_group": len(sub),
                    "marker": marker,
                    "n_valid": n_valid,
                    "prevalence_pct": 100 * prevalence,
                    "ci95_low_pct": 100 * ci_low,
                    "ci95_high_pct": 100 * ci_high,
                }
            )
    return pd.DataFrame(rows)


def horse_race_regressions(frame: pd.DataFrame, quantile_label: str) -> pd.DataFrame:
    rows = []
    for marker in MARKERS:
        cols = ["hqrtm_high_cost", "lihc_high_cost", "Country", marker] + CONTROLS
        sub = frame[cols].dropna()

        controls_rhs = " + ".join(CONTROLS + ["C(Country)"])

        specs = {
            "controls_only": f"{marker} ~ {controls_rhs}",
            "lihc_only": f"{marker} ~ lihc_high_cost + {controls_rhs}",
            "hqrtm_only": f"{marker} ~ hqrtm_high_cost + {controls_rhs}",
            "both": f"{marker} ~ hqrtm_high_cost + lihc_high_cost + {controls_rhs}",
        }

        fitted = {}
        for name, formula in specs.items():
            try:
                fitted[name] = smf.logit(formula, data=sub).fit(disp=False)
            except Exception as exc:  # perfect separation / non-convergence
                print(f"  [{quantile_label}] {marker}/{name} failed: {exc}")
                fitted[name] = None

        both = fitted["both"]
        lihc_only = fitted["lihc_only"]
        hqrtm_only = fitted["hqrtm_only"]

        def lr_test(restricted, full):
            if restricted is None or full is None:
                return np.nan
            stat = 2 * (full.llf - restricted.llf)
            df_diff = full.df_model - restricted.df_model
            if df_diff <= 0:
                return np.nan
            return stats.chi2.sf(stat, df_diff)

        hqrtm_adds_over_lihc_p = lr_test(lihc_only, both)
        lihc_adds_over_hqrtm_p = lr_test(hqrtm_only, both)

        for flag in ["hqrtm_high_cost", "lihc_high_cost"]:
            if both is None or flag not in both.params.index:
                continue
            coef = both.params[flag]
            se = both.bse[flag]
            p = both.pvalues[flag]
            odds_ratio = np.exp(coef)
            or_low = np.exp(coef - 1.96 * se)
            or_high = np.exp(coef + 1.96 * se)
            rows.append(
                {
                    "quantile_spec": quantile_label,
                    "marker": marker,
                    "n_obs": int(both.nobs),
                    "flag": flag,
                    "odds_ratio": odds_ratio,
                    "or_ci95_low": or_low,
                    "or_ci95_high": or_high,
                    "p_value": p,
                    "pseudo_r2_controls_only": fitted["controls_only"].prsquared if fitted["controls_only"] else np.nan,
                    "pseudo_r2_this_flag_only": (
                        fitted["hqrtm_only"].prsquared if flag == "hqrtm_high_cost" else fitted["lihc_only"].prsquared
                    ),
                    "pseudo_r2_both_flags": both.prsquared,
                    "pseudo_r2_increment_over_other_flag": (
                        both.prsquared - fitted["lihc_only"].prsquared
                        if flag == "hqrtm_high_cost" and fitted["lihc_only"] is not None
                        else both.prsquared - fitted["hqrtm_only"].prsquared
                        if flag == "lihc_high_cost" and fitted["hqrtm_only"] is not None
                        else np.nan
                    ),
                    "lr_p_this_flag_adds_over_other": (
                        hqrtm_adds_over_lihc_p if flag == "hqrtm_high_cost" else lihc_adds_over_hqrtm_p
                    ),
                }
            )
    return pd.DataFrame(rows)


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    lihc_df = pd.read_csv(REPO_ROOT / "df_lihc.csv", low_memory=False)

    reclass_tables = []
    regression_tables = []

    for csv_name, label in HQRTM_SPECS:
        hqrtm_df = pd.read_csv(REPO_ROOT / csv_name, low_memory=False)
        frame = build_analysis_frame(hqrtm_df, lihc_df)

        print(f"\n=== {label} ===")
        print("Reclassification group sizes:")
        print(frame["group"].value_counts().to_string())

        reclass = reclassification_table(frame, label)
        reclass_tables.append(reclass)

        regression = horse_race_regressions(frame, label)
        regression_tables.append(regression)

    reclass_all = pd.concat(reclass_tables, ignore_index=True)
    regression_all = pd.concat(regression_tables, ignore_index=True)

    reclass_path = OUTPUT_DIR / "reclassification_marker_prevalence.csv"
    regression_path = OUTPUT_DIR / "horse_race_regressions.csv"
    reclass_all.to_csv(reclass_path, index=False)
    regression_all.to_csv(regression_path, index=False)

    print(f"\nSaved: {reclass_path}")
    print(f"Saved: {regression_path}")

    print("\n=== Summary: HQRTM (q=0.65) reclassification prevalence by marker ===")
    main = reclass_all[reclass_all["quantile_spec"] == "HQRTM (q=0.65)"]
    for marker in MARKERS:
        sub = main[main["marker"] == marker].set_index("group")
        print(f"\n{MARKER_LABELS[marker]}")
        for group in ["Neither", "LIHC_only", "HQRTM_only", "Both"]:
            if group not in sub.index:
                continue
            row = sub.loc[group]
            print(
                f"  {group:11s} n={row['n_valid']:5.0f}  "
                f"prevalence={row['prevalence_pct']:5.1f}%  "
                f"[{row['ci95_low_pct']:5.1f}, {row['ci95_high_pct']:5.1f}]"
            )

    print("\n=== Summary: HQRTM (q=0.65) horse-race odds ratios ===")
    main_reg = regression_all[regression_all["quantile_spec"] == "HQRTM (q=0.65)"]
    for marker in MARKERS:
        sub = main_reg[main_reg["marker"] == marker]
        print(f"\n{MARKER_LABELS[marker]}")
        for _, row in sub.iterrows():
            print(
                f"  {row['flag']:16s} OR={row['odds_ratio']:.2f} "
                f"[{row['or_ci95_low']:.2f}, {row['or_ci95_high']:.2f}]  "
                f"p={row['p_value']:.4f}  "
                f"LR-p (adds over other flag)={row['lr_p_this_flag_adds_over_other']:.4f}  "
                f"| effect size: pseudo-R2 {row['pseudo_r2_controls_only']:.4f} -> "
                f"{row['pseudo_r2_both_flags']:.4f} "
                f"(+{row['pseudo_r2_increment_over_other_flag']:.4f} over the other flag+controls) "
                f"-- a significant p-value here is not itself evidence of a large effect; read both."
            )


if __name__ == "__main__":
    run_all()
