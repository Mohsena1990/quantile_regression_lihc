"""Authoritative ENABLE workbook ingestion and cohort rules."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

HOUSEHOLD_SHEET = "ENABLE.EU_dataset_HH"
QUESTION_SHEET = "variables (questions) labels"
VALUE_SHEET = "variables value (answers) label"
COUNTRIES = {1: "Bulgaria", 2: "France", 3: "Germany", 4: "Hungary", 5: "Italy", 6: "Norway", 7: "Poland", 8: "Serbia", 9: "Spain", 10: "Ukraine", 11: "United Kingdom"}
# Bulgaria, Hungary, Italy, Poland. Spain (code 9) was screened and removed: it fails the per-country support and
# estimability gates (21 Double-Risk households, 4 solid-fuel households) -- see docs/country_screening.md. A country
# that does not carry every stage of the analysis is not in the analysis, so this one constant defines the cohort for
# every downstream stage.
RETAINED_COUNTRY_CODES = frozenset({1, 4, 5, 7})
RETAINED_COUNTRY_NAMES = tuple(COUNTRIES[c] for c in sorted(RETAINED_COUNTRY_CODES))
# Version-5 study cohort: the four countries + Spain. Spain could not carry the earlier per-country four-class models (fewer than
# 30 Double-Risk households) but can enter the pooled model (>= 15 per class and country). The earlier analyses keep the four.
EXTENDED_COUNTRY_CODES = RETAINED_COUNTRY_CODES | {9}
EXTENDED_COUNTRY_NAMES = tuple(COUNTRIES[c] for c in sorted(EXTENDED_COUNTRY_CODES))
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

def retained_cohort(frame: pd.DataFrame, codes: frozenset = RETAINED_COUNTRY_CODES) -> pd.DataFrame:
    required = {"UID", "Country", "NUTS1"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing cohort fields: {sorted(missing)}")
    out = frame.loc[frame["Country"].isin(codes)].copy()
    if out["NUTS1"].isna().any():
        raise ValueError("Retained cohort has missing NUTS1")
    if out["UID"].duplicated().any():
        raise ValueError("Retained cohort has duplicate UID")
    out["country"] = out["Country"].map(COUNTRIES)
    return out
