
# import the reuqired packages (libraries and modules)

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
dir = REPO_ROOT / "df_hqrtm_60.csv"
df = pd.read_csv(dir)

# df['SettlementSize'].isnull().sum()
# df['C3'].isnull().sum()


print("SettlementSize unique values:")
print(df['SettlementSize'].unique())

print("\nC3 unique values:")
print(df['C3'].unique())

print("SettlementSize value counts:")
print(df['SettlementSize'].value_counts(dropna=False))

print("\nC3 value counts:")
print(df['C3'].value_counts(dropna=False))


print("SettlementSize types:")
print(df['SettlementSize'].apply(type).value_counts())

print("\nC3 types:")
print(df['C3'].apply(type).value_counts())

# For SettlementSize
non_numeric_ss = df[pd.to_numeric(df['SettlementSize'], errors='coerce').isna()]
print(non_numeric_ss['SettlementSize'].unique())

# For C3
non_numeric_c3 = df[pd.to_numeric(df['C3'], errors='coerce').isna()]
print(non_numeric_c3['C3'].unique())


print("SettlementSize null rows:")
print(df[df['SettlementSize'].isnull()])

print("\nC3 null rows:")
print(df[df['C3'].isnull()])


print("========================================")

import pandas as pd

dir = REPO_ROOT / "df_hqrtm_60.csv"
df = pd.read_csv(dir, low_memory=False)

# Map SettlementSize
settlement_map = {
    1.0: "1",
    2.0: "2",
    3.0: "3",
    4.0: "4",
    5.0: "5"
}

# Map C3
c3_map = {
    1.0: "All rooms same temperature",
    2.0: "Only warm rooms used",
    99.0: "Do not know"
}

df["SettlementSize_label"] = df["SettlementSize"].map(settlement_map)
df["C3_label"] = df["C3"].map(c3_map)

print("SettlementSize with labels:")
print(df[["SettlementSize", "SettlementSize_label"]].drop_duplicates().sort_values("SettlementSize"))

print("\nC3 with labels:")
print(df[["C3", "C3_label"]].drop_duplicates().sort_values("C3"))

