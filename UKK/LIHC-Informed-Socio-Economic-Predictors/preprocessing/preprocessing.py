from pathlib import Path

import pandas as pd
import numpy as np

from missing_values import drop_feature
from risk_category import assign_hqrtm, assign_traditional_lihc


# =========================
# Config
# =========================
REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "ENABLE.EU_dataset_survey of households.xlsx"
OUTPUT_CLEAN = "preprocessed_data_clean.csv"
OUTPUT_LIHC = "df_lihc.csv"
OUTPUT_HQRTM_60 = "df_hqrtm_60.csv"
OUTPUT_HQRTM_65 = "df_hqrtm_65.csv"
OUTPUT_HQRTM_70 = "df_hqrtm_70.csv"

CURRENT_YEAR = 2018
REDUNDANT_FEATURES = ["UID", "T1", "T1cluster", "T3", "T4", "T5", "NUTS1", "NUTS2", "NUTS3"]
SPECIAL_MISSING_CODES = [99999, 99998, 9999, 999, 99, 98]

COUNTRY_MAP = {
    1: "BG", 2: "FR", 3: "DE", 4: "HU", 5: "IT",
    6: "NO", 7: "PL", 8: "RS", 9: "ES", 10: "UA", 11: "UK"
}

INCOME_MIDPOINTS = {
    "BG": [1700, 2400, 3000, 3600, 4200, 5000, 6000, 7500, 9500, 14000],
    "FR": [8500, 12000, 15000, 18000, 21500, 25000, 29000, 34000, 42000, 58000],
    "DE": [9200, 13000, 16000, 19000, 23000, 27000, 32000, 38000, 47000, 65000],
    "HU": [2900, 4100, 5100, 6100, 7200, 8600, 10300, 12900, 16200, 24000],
    "IT": [7200, 10200, 12800, 15300, 18000, 21000, 24500, 29000, 36000, 50000],
    "NO": [12800, 18100, 22600, 27100, 32000, 37300, 43400, 51400, 64000, 90000],
    "PL": [3400, 4800, 6000, 7200, 8500, 10200, 12200, 15300, 19200, 28500],
    "RS": [1800, 2600, 3300, 4000, 4500, 5400, 6500, 8100, 10200, 15000],
    "ES": [7400, 10500, 13100, 15700, 18500, 21600, 25200, 29800, 37000, 52000],
    "UA": [1100, 1600, 2000, 2400, 2800, 3400, 4100, 5100, 6400, 9500],
    "UK": [9600, 13600, 17000, 20400, 24000, 28000, 32600, 38600, 48000, 67000]
}

EXCHANGE_RATES = {
    1: 0.5113,
    2: 1.0,
    3: 1.0,
    4: 0.00314,
    5: 1.0,
    6: 0.1045,
    7: 0.2346,
    8: 0.00849,
    9: 1.0,
    10: 0.0314,
    11: 1.130
}

H2_TO_YEAR = {1: 1945, 2: 1955, 3: 1965, 4: 1975, 5: 1985, 6: 1995, 7: 2005, 8: 2013, 99: np.nan}
H3_TO_AREA = {1: 21, 2: 54, 3: 78, 4: 105.5, 5: 160.5, 6: 200, 7: np.nan}
AGE_GROUPS = {"S1Ac1": 2.5, "S1Ac2": 11.5, "S1Ac3": 41, "S1Bc1": 70, "S1Bc2": 41, "S1Bc3": 70}
COUNTRY_MEDIAN_BIRTH = {
    "DE": 1965, "FR": 1967, "UK": 1963, "IT": 1964, "ES": 1966,
    "PL": 1968, "HU": 1967, "BG": 1965, "RS": 1966, "UA": 1967, "NO": 1964
}
HOUSING_TYPE_MAP = {1: 0, 2: 0, 3: 1, 4: 2}
SETTLEMENT_MAP = {1.0: 0, 2.0: 1, 3.0: 1, 4.0: 2, 5.0: 2}
HEATING_FEATURES = ["H6A1", "H6A2", "H6A3", "H6A4", "H6A5", "H6A6", "H6A7", "H6A8", "H6A9", "H6A10", "H6A11"]
# QR_FEATURES = [
#     "floor_area",
#     "house_age",
#     "household_size",
#     "house_detachment",
#     "has_insulation",
#     "heating_strategy",
#     "birth_year_respondent",
# ]

# QR_FEATURES = [
#     "floor_area",
#     "house_age",
#     "household_size",
#     "dwelling_type",
#     "insulation_count",
#     "main_heating_source",
#     "children_present",
#     "elderly_present",
# ]


QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    # "heating_control",
    "household_size",
    # "Country"
    # "children_present",
    # "elderly_present",
    # "C2",
    "SettlementSize",
    # "C1A",
    # "C1B",
    "C3"
]

# =========================
# Helpers
# =========================
def cap_right_tail(df: pd.DataFrame, cols: list, q: float = 0.99) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            upper = out[col].quantile(q)
            if pd.notna(upper):
                out[col] = out[col].clip(upper=upper)
    return out


def decile_to_income(row: pd.Series) -> float:
    dec = row["income_bracket"]
    country = row["Country_name"]
    if pd.isna(dec) or pd.isna(country):
        return np.nan
    dec = int(dec)
    if dec < 1 or dec > 10:
        return np.nan
    return INCOME_MIDPOINTS[country][dec - 1]


def print_country_summary(df: pd.DataFrame, label_col: str, title: str) -> None:
    country_col = "Country_name" if "Country_name" in df.columns else "Country"
    counts = df.groupby([country_col, label_col]).size().unstack(fill_value=0).sort_index()
    pct = (counts.div(counts.sum(axis=1), axis=0) * 100).round(2)

    print(f"\n=== {title}: counts ===")
    print(counts)
    print(f"\n=== {title}: percentages ===")
    print(pct)


# =========================
# Load
# =========================
df = pd.read_excel(DATA_PATH, sheet_name=0).copy()

# Drop countries with structurally incomplete survey coverage: France (2),
# Norway (6), Poland (7), and the UK (11) have 0% valid SettlementSize
# responses (the question was never recorded for these countries, not
# scattered individual non-response -- no per-country imputation can
# recover it). The remaining 7 countries have 0% missing.
EXCLUDED_COUNTRIES = [2, 6, 7, 11]
print(f"Dropping countries with incomplete survey coverage: {EXCLUDED_COUNTRIES}")
print(df.loc[df['Country'].isin(EXCLUDED_COUNTRIES), 'Country'].value_counts().sort_index())
df = df[~df["Country"].isin(EXCLUDED_COUNTRIES)].reset_index(drop=True)

df_raw = df.copy()

print("Initial shape (after country filter):", df.shape)
print(df["Country"].value_counts(dropna=False).sort_index())

# =========================
# Drop redundant columns
# =========================
df = drop_feature(df, REDUNDANT_FEATURES)
print("After dropping redundant features:", df.shape)
print("Missing values after dropping redundant features:", df.isnull().sum().sum())

# =========================
# Basic cleaning
# =========================
df["S9MONTH"] = pd.to_numeric(df["S9MONTH"], errors="coerce").replace([98, 99], np.nan)
df.loc[~df["S9MONTH"].between(1, 12, inclusive="both") & df["S9MONTH"].notna(), "S9MONTH"] = np.nan
df["S9MONTH_CAT"] = df["S9MONTH"].fillna(0).astype(int)

df["S9YEAR"] = pd.to_numeric(df["S9YEAR"], errors="coerce").replace([98, 99], np.nan)
df["S9YEAR_CAT"] = df["S9YEAR"].fillna(0).astype(int)


df["C3"] = (
    df["C3"]
    .map({
        1: "All rooms heated",
        2: "Partial heating",
        99: "Do not know"
    })
    .fillna("Missing")
)

# Impute missing SettlementSize with the per-country mode *before* mapping
# to strings -- doing this after mapping (as before) is a no-op, because by
# then every missing value has already been replaced by the literal string
# "No answer" and groupby(...).transform(fillna) has no NaN left to fill.
df["SettlementSize"] = pd.to_numeric(df["SettlementSize"], errors="coerce")
df["SettlementSize"] = df.groupby("Country")["SettlementSize"].transform(
    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x.median())
)

df["SettlementSize"] = (
    df["SettlementSize"]
    .map({1: "1", 2: "2", 3: "3", 4: "4", 5: "5"})
    .fillna("No answer")
)

df["H9"] = df["H9"].fillna(df["H9"].mode().iloc[0] if not df["H9"].mode().empty else 1)

df["H11D"] = df["H11D"].replace({1: "Yes", 2: "No", 99: np.nan}).fillna("No")
df["H11E"] = df["H11E"].replace({1: "Yes", 2: "No", 99: np.nan}).fillna("No")
df["H11F"] = df["H11F"].replace({1: "Yes", 2: "No", 99: np.nan}).fillna("No")

df["C2"] = pd.to_numeric(df["C2"], errors="coerce").replace([99], np.nan).fillna(0).astype(float)

print("Missing values after basic cleaning:", df.isnull().sum().sum())

# =========================
# Energy bill cleaning
# =========================
for col in ["H8A", "H8B", "H7A1", "H7A2"]:
    df[col] = pd.to_numeric(df[col], errors="coerce").replace(SPECIAL_MISSING_CODES, np.nan)

df["H7AA"] = pd.to_numeric(df["H7AA"], errors="coerce").replace([99, 98], np.nan)

for col in ["H7A1", "H7A2"]:
    df[col] = df.groupby("Country")[col].transform(lambda x: x.fillna(x.median()))

print("Missing values after bill cleaning:", df.isnull().sum().sum())

# =========================
# Structural features
# =========================
df["house_age"] = df["H2"].map(
    lambda x: CURRENT_YEAR - H2_TO_YEAR.get(x, np.nan) if pd.notna(x) else np.nan
)
df["house_age"] = df.groupby("Country")["house_age"].transform(lambda x: x.fillna(x.median()))

df["floor_area"] = df["H3"].map(lambda x: H3_TO_AREA.get(x, np.nan) if pd.notna(x) else np.nan)
df["floor_area"] = df.groupby("Country")["floor_area"].transform(lambda x: x.fillna(x.median()))
df["floor_area"] = df["floor_area"].clip(upper=df["floor_area"].quantile(0.99))

df["Country_name"] = df["Country"].map(COUNTRY_MAP)

df["income_bracket"] = df.groupby("Country")["S9MONTH"].transform(
    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 5)
)

# df["income_bracket1"] = pd.to_numeric(df["S9MONTH"], errors="coerce")

# # remove invalid codes
# df.loc[df["income_bracket1"].isin([98, 99]), "income_bracket"] = np.nan


df["approx_income"] = df.apply(decile_to_income, axis=1)

# Needed before total_expenditure below: whether a household's main
# heating source is electric (H6A1) or an electric heat pump (H6A10)
# determines whether its heating cost is already inside its electricity
# bill (moved up from later in this file, where it was originally
# computed only for use as a model feature).
for col in HEATING_FEATURES:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
df["main_heating_source"] = df[HEATING_FEATURES].idxmax(axis=1)

df["H8A_EUR"] = df.apply(
    lambda row: row["H8A"] * EXCHANGE_RATES.get(row["Country"], 1.0) if pd.notna(row["H8A"]) else np.nan,
    axis=1
)
df["H8B_EUR"] = df.apply(
    lambda row: row["H8B"] * EXCHANGE_RATES.get(row["Country"], 1.0) if pd.notna(row["H8B"]) else np.nan,
    axis=1
)
df["H7A1_EUR"] = df.apply(
    lambda row: row["H7A1"] * EXCHANGE_RATES.get(row["Country"], 1.0) if pd.notna(row["H7A1"]) else np.nan,
    axis=1
)
df["H7A2_EUR"] = df.apply(
    lambda row: row["H7A2"] * EXCHANGE_RATES.get(row["Country"], 1.0) if pd.notna(row["H7A2"]) else np.nan,
    axis=1
)

df["num_children"] = df["S1Ac1"].fillna(0) + df["S1Bc1"].fillna(0)
df["num_adults"] = df["S1Ac2"].fillna(0) + df["S1Bc2"].fillna(0)
df["num_elderly"] = df["S1Ac3"].fillna(0) + df["S1Bc3"].fillna(0)

df["children_present"] = (df["num_children"] > 0).astype(int)
df["elderly_present"] = (df["num_elderly"] > 0).astype(int)

# ---------------------------------------------------------------------
# total_expenditure = annual electricity cost + annual heating fuel cost.
#
# H8A is a MONTHLY electricity bill, H8B an ANNUAL one. These were
# previously summed together and the sum multiplied by 12 regardless,
# which is only correct when H8A alone is present; it corrupted ~11% of
# households (mixed units when both are present, 12x inflation when only
# H8B is present). Coalesce instead: prefer the already-annual H8B, fall
# back to H8A x 12.
#
# H7A1/H7A2 (heating fuel cost) were previously excluded from
# total_expenditure entirely, even though for any household not heating
# with electricity, heating fuel is typically the dominant energy cost.
# H7A2 is already a full heating-season total; H7A1 is per-month and is
# annualized using H7AA (number of months heating was actually paid for
# -- heating is seasonal, not a flat 12 months), defaulting to a 6-month
# heating season when H7AA wasn't answered.
#
# Heating cost is only ADDED separately when it isn't already inside the
# electricity bill: households whose main heating source is electricity
# (H6A1) or an electric heat pump (H6A10) have heating billed through
# electricity already, so adding H7 on top would double-count it.
# ---------------------------------------------------------------------
electricity_annual_cost = df["H8B_EUR"].combine_first(df["H8A_EUR"] * 12)

months_heated = pd.to_numeric(df["H7AA"], errors="coerce").clip(lower=1, upper=12)
heating_annual_from_monthly = df["H7A1_EUR"] * months_heated.fillna(6)
heating_annual_cost = df["H7A2_EUR"].combine_first(heating_annual_from_monthly)

heating_billed_via_electricity = df["main_heating_source"].isin(["H6A1", "H6A10"])
both_missing = electricity_annual_cost.isna() & heating_annual_cost.isna()
electricity_plus_heating = electricity_annual_cost.add(heating_annual_cost, fill_value=0).where(~both_missing)

df["total_expenditure"] = np.where(
    heating_billed_via_electricity,
    electricity_annual_cost,
    electricity_plus_heating,
)
df["exp_observed_flag"] = df["total_expenditure"].notna().astype(int)
df["total_expenditure"] = df.groupby("Country")["total_expenditure"].transform(lambda x: x.fillna(x.median()))
df["log_expenditure"] = np.log1p(df["total_expenditure"])

df["household_size"] = df[["S1Ac1", "S1Ac2", "S1Ac3", "S1Bc1", "S1Bc2", "S1Bc3"]].sum(axis=1)
df["household_size"] = df["household_size"].replace(0, np.nan).fillna(1)

ins_cols = ["H5A1", "H5A2", "H5A3"]
for col in ins_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

df["insulation_count"] = df[ins_cols].sum(axis=1)
df["has_any_insulation"] = (df["insulation_count"] > 0).astype(int)
df["no_additional_insulation"] = (df["H5A4"]).fillna(0).astype(int)


# df["has_insulation"] = df[["H5A1", "H5A2", "H5A3", "H5A4"]].fillna(0).any(axis=1).astype(int)

df["equiv_scale"] = 1 + 0.5 * (df["household_size"] - 1).clip(lower=0)
df["equivalized_income"] = df["approx_income"] / df["equiv_scale"]

# heating_map = {col: i for i, col in enumerate(HEATING_FEATURES, start=1)}
# df["main_heating_source_code"] = df["main_heating_source"].map(heating_map)

# Heating control from H9
df["heating_control"] = pd.to_numeric(df["H9"], errors="coerce")
df["heating_control"] = df["heating_control"].replace([9, 99], np.nan)

df["heating_control"] = df.groupby("Country")["heating_control"].transform(
    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x.median())
)

# main_heating_source is computed earlier now (needed for total_expenditure).

# X_heating = StandardScaler().fit_transform(df[HEATING_FEATURES])
# df["heating_strategy"] = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_heating)

# df["num_children"] = df["S1Ac1"].fillna(0) + df["S1Ac2"].fillna(0)
# df["num_elderly"] = df["S1Bc1"].fillna(0) + df["S1Bc3"].fillna(0)

# df["children_present"] = (df["num_children"] > 0).astype(int)
# df["elderly_present"] = (df["num_elderly"] > 0).astype(int)


df["num_children"] = df["S1Ac1"].fillna(0) + df["S1Bc1"].fillna(0)
df["num_adults"] = df["S1Ac2"].fillna(0) + df["S1Bc2"].fillna(0)
df["num_elderly"] = df["S1Ac3"].fillna(0) + df["S1Bc3"].fillna(0)

df["children_present"] = (df["num_children"] > 0).astype(int)
df["elderly_present"] = (df["num_elderly"] > 0).astype(int)
df["household_size"] = df["num_children"] + df["num_adults"] + df["num_elderly"]



df["age_distribution"] = df[["S1Ac1", "S1Ac2", "S1Ac3", "S1Bc1", "S1Bc2", "S1Bc3"]].apply(
    lambda row: [AGE_GROUPS[col] for col in row.index if row[col] > 0 and col in AGE_GROUPS],
    axis=1
)
df["birth_year_respondent"] = df["age_distribution"].apply(
    lambda ages: CURRENT_YEAR - max([a for a in ages if a >= 41]) if any(a >= 41 for a in ages) else np.nan
)
df["birth_year_respondent"] = df.apply(
    lambda row: row["birth_year_respondent"]
    if pd.notna(row["birth_year_respondent"])
    else COUNTRY_MEDIAN_BIRTH.get(row["Country_name"], 1965),
    axis=1,
).clip(lower=1940, upper=2005).astype(int)
df = df.drop(columns=["age_distribution"], errors="ignore")

df["dwelling_type"] = df["H1"].replace({9: np.nan})
df["dwelling_type"] = df.groupby("Country")["dwelling_type"].transform(
    lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else x.median())
).astype(int)

# df["house_detachment"] = df["H1"].map(HOUSING_TYPE_MAP)
# if df["house_detachment"].isnull().any():
#     df["house_detachment"] = df["house_detachment"].fillna(df["SettlementSize"].map(SETTLEMENT_MAP))
# df["house_detachment"] = df["house_detachment"].fillna(1).astype(int)

# Optional derived descriptive variables
df["energy_intensity"] = df["total_expenditure"] / df["floor_area"]
df["seasonal_energy_cost"] = df["H7A1"] * (df["H7AA"].fillna(6) / 12 + df["C2"] * 0.1)

print("Missing values after feature engineering:", df.isnull().sum().sum())

# =========================
# Controlled filling
# =========================
fill_by_country_median = [
    "approx_income",
    "equivalized_income",
    "house_age",
    "floor_area",
    "seasonal_energy_cost",
    "energy_intensity",
]
for col in fill_by_country_median:
    if col in df.columns:
        df[col] = df.groupby("Country")[col].transform(lambda x: x.fillna(x.median()))

df["equivalized_income"] = df["equivalized_income"].fillna(df["equivalized_income"].median())
df["seasonal_energy_cost"] = df["seasonal_energy_cost"].fillna(df["seasonal_energy_cost"].median())

print("Missing values after controlled filling:", df.isnull().sum().sum())

# =========================
# Conservative outlier capping
# =========================
df = cap_right_tail(
    df,
    cols=["total_expenditure", "log_expenditure", "floor_area", "equivalized_income", "seasonal_energy_cost"],
    q=0.99,
)
print("Shape after conservative tail capping:", df.shape)

# =========================
# Build labels
# =========================
# df_lihc = assign_traditional_lihc(
#     df,
#     income_col="equivalized_income",
#     exp_col="total_expenditure",
#     country_col="Country",
#     income_rule="country_median_60",
#     exp_quantile=0.80,
# )
df_lihc = assign_traditional_lihc(
    df,
    exp_col="total_expenditure",
    country_col="Country",
    income_rule="country_median_60",
    income_bracket_col="income_bracket",
    exp_quantile=0.80,
)

df_hqrtm_60 = assign_hqrtm(
    df,
    qr_features=QR_FEATURES,
    income_col="equivalized_income",
    exp_col="total_expenditure",
    country_col="Country",
    income_rule="bracket_lt4",
    quantile=0.60,
    add_country_effects= True,
    margin_scale=0.10,
)

df_hqrtm_65 = assign_hqrtm(
    df,
    qr_features=QR_FEATURES,
    income_col="equivalized_income",
    exp_col="total_expenditure",
    country_col="Country",
    income_rule="bracket_lt4",
    quantile=0.65,
    add_country_effects= True,
    margin_scale=0.10,
)

df_hqrtm_70 = assign_hqrtm(
    df,
    qr_features=QR_FEATURES,
    income_col="equivalized_income",
    exp_col="total_expenditure",
    country_col="Country",
    income_rule="bracket_lt4",
    quantile=0.70,
    add_country_effects= True,
    margin_scale=0.10,
)

# =========================
# Diagnostics
# =========================
print("\nTraditional LIHC distribution:")
print(df_lihc["risk_category"].value_counts(dropna=False))
print_country_summary(df_lihc, "risk_category", "Traditional LIHC")

print("\nHQRTM q=0.60 distribution:")
print(df_hqrtm_60["risk_category"].value_counts(dropna=False))
print_country_summary(df_hqrtm_60, "risk_category", "HQRTM q=0.60")

print("\nHQRTM q=0.65 distribution:")
print(df_hqrtm_65["risk_category"].value_counts(dropna=False))
print_country_summary(df_hqrtm_65, "risk_category", "HQRTM q=0.65")

print("\nHQRTM q=0.70 distribution:")
print(df_hqrtm_70["risk_category"].value_counts(dropna=False))
print_country_summary(df_hqrtm_70, "risk_category", "HQRTM q=0.70")

print("\nBefore:", df_raw.shape, "After:", df.shape)
loss = ((df_raw.shape[0] - df.shape[0]) / df_raw.shape[0]) * 100
print(f"Overall row loss: {loss:.2f}%")

for col in ["income_bracket", "log_expenditure", "floor_area"]:
    if col in df.columns:
        print(f"\n{col} quantiles:")
        print(df[col].quantile([0.95, 0.99, 1.0]))

missing_before = df.isnull().sum()
missing_with_nan = missing_before[missing_before > 0]
print("\n=== Missing values before export ===")
if not missing_with_nan.empty:
    print(missing_with_nan.sort_values(ascending=False))
    print(f"Total features with missing values: {len(missing_with_nan)}")
    print(f"Total missing values: {missing_with_nan.sum()}")
else:
    print("No remaining missing values.")

# =========================
# Save
# =========================
df.to_csv(OUTPUT_CLEAN, index=False)
df_lihc.to_csv(OUTPUT_LIHC, index=False)
df_hqrtm_60.to_csv(OUTPUT_HQRTM_60, index=False)
df_hqrtm_65.to_csv(OUTPUT_HQRTM_65, index=False)
df_hqrtm_70.to_csv(OUTPUT_HQRTM_70, index=False)

print(f"\nSaved: {OUTPUT_CLEAN}")
print(f"Saved: {OUTPUT_LIHC}")
print(f"Saved: {OUTPUT_HQRTM_60}")
print(f"Saved: {OUTPUT_HQRTM_65}")
print(f"Saved: {OUTPUT_HQRTM_70}")