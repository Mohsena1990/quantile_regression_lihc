"""
Country policy context for interpreting risk_category results.

Motivation
----------
Risk-category prevalence (Double risk / Income risk / Expenditure risk)
varies a lot by country -- but that variation reflects both the
underlying deprivation AND each country's existing policy response
(social tariffs, subsidies) and public attitudes toward energy policy.
Interpreting a country's Double-risk rate without that context risks
reading policy success/failure as if it were purely a deprivation
signal.

Only S7 (received public financial aid / social tariff to pay energy
bills in the last 12 months) has complete coverage across all 7
countries in this pipeline -- it's a revealed measure of policy reach,
not an attitude survey, which is also why it's used as an external
validity marker elsewhere in this suite (see external_validity_check.py)
and is NOT used here as anything other than descriptive country context.

The G-series policy-attitude items (G2A4/G2A5: which policy priorities
respondents spontaneously mentioned; G6A: perceived success of
low-income energy support policy) add richer context but weren't
administered everywhere: G2A4/G6A are missing for Spain and Italy, and
G2A5 is missing for Spain, Italy, *and* Germany specifically (a finer,
item-level gap, not just the module-level one -- checked directly
against the data rather than assumed). Each country's average below is
computed from whatever valid responses that country actually has; a
column already comes out NaN for a country with none, so no country
needs to be hardcoded here as "has this module" or not.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "outputs" / "validation" / "country_policy_context"

COUNTRY_NAMES = {1: "Bulgaria", 3: "Germany", 4: "Hungary", 5: "Italy", 8: "Serbia", 9: "Spain", 10: "Ukraine"}


def aid_receipt_rate(s7: pd.Series) -> tuple:
    valid = s7.isin([1, 2])
    n_valid = int(valid.sum())
    n_yes = int((s7 == 1).sum())
    if n_valid == 0:
        return np.nan, np.nan, np.nan, 0
    rate = n_yes / n_valid
    lo, hi = proportion_confint(n_yes, n_valid, method="wilson")
    return rate, lo, hi, n_valid


def run_all() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    hqrtm_df = pd.read_csv(REPO_ROOT / "df_hqrtm_65.csv", low_memory=False)
    lihc_df = pd.read_csv(REPO_ROOT / "df_lihc.csv", low_memory=False)

    rows = []
    for country, name in COUNTRY_NAMES.items():
        h_mask = hqrtm_df["Country"] == country
        l_mask = lihc_df["Country"] == country
        n = int(h_mask.sum())

        hqrtm_double_rate = (hqrtm_df.loc[h_mask, "risk_category"] == "Double risk").mean()
        lihc_double_rate = (lihc_df.loc[l_mask, "risk_category"] == "Double risk").mean()

        aid_rate, aid_lo, aid_hi, aid_n = aid_receipt_rate(hqrtm_df.loc[h_mask, "S7"])

        row = {
            "Country": name,
            "n": n,
            "hqrtm_double_risk_pct": 100 * hqrtm_double_rate,
            "lihc_double_risk_pct": 100 * lihc_double_rate,
            "energy_aid_received_pct": 100 * aid_rate,
            "energy_aid_ci95_low_pct": 100 * aid_lo,
            "energy_aid_ci95_high_pct": 100 * aid_hi,
        }

        g = hqrtm_df.loc[h_mask]
        row["pct_mentioned_price_regulation_priority"] = 100 * pd.to_numeric(g["G2A4"], errors="coerce").mean()
        row["pct_mentioned_market_liberalization_priority"] = 100 * pd.to_numeric(g["G2A5"], errors="coerce").mean()
        g6a = pd.to_numeric(g["G6A"], errors="coerce").where(lambda x: x.between(1, 5))
        row["low_income_support_perceived_success_1to5"] = g6a.mean()

        rows.append(row)

    table = pd.DataFrame(rows).sort_values("hqrtm_double_risk_pct", ascending=False)
    table.to_csv(OUTPUT_DIR / "country_policy_context.csv", index=False)

    pd.set_option("display.width", 160)
    print("=== Country policy context (G2A4/G2A5/G6A are NaN for Spain and Italy -- module not administered) ===")
    print(table.round(1).to_string(index=False))
    print(
        "\nNote: G6A is 'perceived success of low-income energy support policy', 1=very successful, "
        "5=very unsuccessful -- lower is more positive."
    )
    print(f"\nSaved: {OUTPUT_DIR / 'country_policy_context.csv'}")


if __name__ == "__main__":
    run_all()
