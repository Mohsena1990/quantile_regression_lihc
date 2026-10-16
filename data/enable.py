"""Authoritative ENABLE workbook ingestion and cohort rules."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HOUSEHOLD_SHEET = "ENABLE.EU_dataset_HH"
QUESTION_SHEET = "variables (questions) labels"
VALUE_SHEET = "variables value (answers) label"
COUNTRIES = {1: "Bulgaria", 2: "France", 3: "Germany", 4: "Hungary", 5: "Italy", 6: "Norway", 7: "Poland", 8: "Serbia", 9: "Spain", 10: "Ukraine", 11: "United Kingdom"}
# The five study countries: Bulgaria, Hungary, Italy, Poland and Spain. Every score of the study is computed on their households.
STUDY_COUNTRY_CODES = frozenset({1, 4, 5, 7, 9})
STUDY_COUNTRY_NAMES = tuple(COUNTRIES[c] for c in sorted(STUDY_COUNTRY_CODES))
# The other six ENABLE countries join ONLY as extra training households of pooled models. They are not in the frozen split (it
# was drawn for the five), so all their households can be used.
ALL_COUNTRY_CODES = frozenset(COUNTRIES)
POOL_EXTRA_CODES = ALL_COUNTRY_CODES - STUDY_COUNTRY_CODES
POOL_EXTRA_NAMES = tuple(COUNTRIES[c] for c in sorted(POOL_EXTRA_CODES))
MISSING_CODES = frozenset({98, 99, 999, 9999, 99999, "#NULL!"})

def read_dictionary(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    questions = pd.read_excel(path, sheet_name=QUESTION_SHEET, header=1, names=["variable", "column", "label"])
    values = pd.read_excel(path, sheet_name=VALUE_SHEET, header=1, names=["variable", "value", "value_label"])
    questions = questions.dropna(subset=["variable"]).copy()
    values["variable"] = values["variable"].ffill()
    values = values.dropna(subset=["variable", "value"])
    return questions, values

def governed_dictionary(path: str | Path) -> pd.DataFrame:
    questions, values = read_dictionary(path)
    out = questions.merge(values, on="variable", how="left")
    text = out["value_label"].fillna("").str.lower()
    out["is_missing_code"] = text.str.contains("don't know|do not know|refus|no answer|did not answer", regex=True)
    return out[["variable", "column", "label", "value", "value_label", "is_missing_code"]]

def retained_cohort(frame: pd.DataFrame, codes: frozenset = STUDY_COUNTRY_CODES, *, require_nuts1: bool = True) -> pd.DataFrame:
    """`require_nuts1=False` only for the extra training pool (France and the UK carry no region codes)."""
    required = {"UID", "Country", "NUTS1"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing cohort fields: {sorted(missing)}")
    out = frame.loc[frame["Country"].isin(codes)].copy()
    if require_nuts1 and out["NUTS1"].isna().any():
        raise ValueError("Retained cohort has missing NUTS1")
    if out["UID"].duplicated().any():
        raise ValueError("Retained cohort has duplicate UID")
    out["country"] = out["Country"].map(COUNTRIES)
    return out
