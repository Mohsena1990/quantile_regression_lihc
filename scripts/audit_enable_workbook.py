"""Generate governed Phase-2 ENABLE audit artifacts from workbook dictionaries."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.enable import COUNTRIES, HOUSEHOLD_SHEET, governed_dictionary, retained_cohort

CORE = ["UID","Country","NUTS1","T3","w","H1","H2","H3","H7A1","H7A2","H7A88","H7AA","H8A","H8B","S9MONTH","S9YEAR","S7","S8"] + [f"H5A{i}" for i in range(1,5)] + ["H5A99"] + [f"H6A{i}" for i in range(1,12)] + ["H6A99"]

def publish(frame: pd.DataFrame, stem: str) -> None:
    target = ROOT / "outputs" / "tables" / stem
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target.with_suffix(".csv"), index=False)
    frame.to_excel(target.with_suffix(".xlsx"), index=False)
    target.with_suffix(".tex").write_text(frame.to_latex(index=False), encoding="utf-8")

def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("workbook", nargs="?", default=str(ROOT / "data/raw/ENABLE.EU_dataset_survey of households.xlsx")); args=parser.parse_args()
    dictionary=governed_dictionary(args.workbook)
    manifest=ROOT / "artifacts" / "manifests" / "variable_dictionary.parquet"; manifest.parent.mkdir(parents=True, exist_ok=True); dictionary.to_parquet(manifest,index=False)
    publish(dictionary, "table02_variable_definitions")
    data=pd.read_excel(args.workbook,sheet_name=HOUSEHOLD_SHEET,usecols=lambda c:c in CORE)
    data["country"]=data.Country.map(COUNTRIES); cohort=retained_cohort(data)
    country=(data.groupby(["Country","country"],dropna=False).agg(raw_n=("UID","size"),nuts1_nonmissing=("NUTS1","count"),nuts1_regions=("NUTS1","nunique"),fieldwork_start=("T3","min"),fieldwork_end=("T3","max")).reset_index())
    country["retained"]=country.Country.isin({1,4,5,7,9}); publish(country,"table01_country_cohort")
    coverage=(cohort.groupby("country")[CORE[4:]].agg(lambda s:s.notna().mean()).reset_index().melt(id_vars="country",var_name="variable",value_name="nonmissing_fraction")); publish(coverage,"table03_missingness")
    h6=[f"H6A{i}" for i in range(1,12)]; sums=cohort[h6].apply(pd.to_numeric,errors="coerce").sum(axis=1,min_count=1)
    heating=pd.DataFrame([{"retained_n":len(cohort),"valid_share_rows":int(sums.notna().sum()),"exact_100_n":int(sums.eq(100).sum()),"within_99_101_n":int(sums.between(99,101).sum()),"inconsistent_n":int((sums.notna() & ~sums.between(99,101)).sum()),"missing_n":int(sums.isna().sum())}]); publish(heating,"table_heating_share_audit")
    weights=cohort.groupby("country").w.agg(["count","nunique","min","max","mean"]).reset_index(); weights["status"]="candidate survey weight; provenance/estimand requires documentation"; publish(weights,"table08_lihc_weight_audit")
    income=cohort.groupby("country").agg(monthly_nonmissing=("S9MONTH","count"),monthly_categories=("S9MONTH","nunique"),annual_nonmissing=("S9YEAR","count"),annual_categories=("S9YEAR","nunique")).reset_index(); income["resolution"]="country-specific brackets; not continuous income"; publish(income,"table09_lihc_income_bracket_availability_audit")
    housing=pd.DataFrame([{"workbook_search_terms":"rent|mortgage|housing cost|housing payment|after housing","household_specific_field_found":False,"exact_AHC_status":"blocked","adapted_status":"requires external model or labelled no-housing-cost sensitivity"}]); publish(housing,"table10_lihc_housing_cost_audit")
    requirements=pd.DataFrame([
      {"requirement":"continuous full household income","available":False,"exact_status":"blocked","adapted_status":"bracket mapping required"},
      {"requirement":"household housing cost / AHC income","available":False,"exact_status":"blocked","adapted_status":"external model or sensitivity required"},
      {"requirement":"survey weight w","available":cohort.w.notna().all(),"exact_status":"provenance validation required","adapted_status":"candidate weighted median"},
      {"requirement":"household composition","available":True,"exact_status":"age bands do not implement under-14 exactly","adapted_status":"mapping required"},
      {"requirement":"engineering coefficients","available":False,"exact_status":"blocked","adapted_status":"external sourced coefficients required"},
      {"requirement":"historical fuel prices","available":False,"exact_status":"blocked","adapted_status":"external country-period prices required"},
      {"requirement":"NUTS1 climate","available":False,"exact_status":"not official-English input","adapted_status":"ERA5-Land acquisition required"},
    ]); publish(requirements,"table07_lihc_data_requirements")
    print(f"retained_n={len(cohort)} nuts1={cohort.NUTS1.nunique()} dictionary_rows={len(dictionary)}")

if __name__ == "__main__": main()
