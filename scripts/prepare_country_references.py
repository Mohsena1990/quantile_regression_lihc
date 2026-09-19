"""Download and govern official Eurostat country-level reference data used as
regression features (see lib/features/country_context.py: ctx_income_d5_eur,
ctx_rent_burden_pct, ctx_owner_occupied_pct) and by scripts/build_expenditure_m0.py
and scripts/prepare_energy_prices.py."""
from __future__ import annotations
import argparse,itertools,json,sys
from datetime import date
from pathlib import Path
from urllib.request import urlopen
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
BASE="https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
GEOS={"BG":"Bulgaria","HU":"Hungary","IT":"Italy","PL":"Poland","ES":"Spain"}

def parse_jsonstat(payload:dict)->pd.DataFrame:
    dims=payload["id"]; categories=[]
    for dim in dims:
        idx=payload["dimension"][dim]["category"]["index"]
        categories.append(sorted(idx,key=idx.get) if isinstance(idx,dict) else idx)
    values=payload.get("value",{}); rows=[]
    for flat,coordinate in enumerate(itertools.product(*categories)):
        value=values.get(str(flat)) if isinstance(values,dict) else values[flat]
        if value is not None: rows.append(dict(zip(dims,coordinate),value=float(value)))
    return pd.DataFrame(rows)

def fetch(dataset:str,year:int,semester:bool=False)->tuple[pd.DataFrame,str]:
    geo="&".join(f"geo={x}" for x in GEOS)
    period=f"sinceTimePeriod={year}-S1&untilTimePeriod={year}-S2" if semester else f"time={year}"
    url=f"{BASE}/{dataset}?lang=EN&{period}&{geo}"
    with urlopen(url) as response: payload=json.load(response)
    return parse_jsonstat(payload),url

def publish(frame:pd.DataFrame,stem:str)->None:
    p=ROOT/'outputs'/'tables'/stem; frame.to_csv(p.with_suffix('.csv'),index=False); frame.to_excel(p.with_suffix('.xlsx'),index=False); p.with_suffix('.tex').write_text(frame.to_latex(index=False),encoding='utf-8')

def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--year',type=int,default=2018); args=ap.parse_args(); today=date.today().isoformat()
    income,url_income=fetch('ilc_di01',args.year)
    income=income.loc[income.quant_inc.isin([f"D{i}" for i in range(1,10)]) & income.statinfo.eq("TC") & income.unit.eq("EUR")].copy()
    income['country']=income.geo.map(GEOS); income_ref=income.rename(columns={'quant_inc':'quantile','value':'cutoff_eur'})[['country','geo','time','quantile','cutoff_eur']]
    income_ref['currency']='EUR'; income_ref['source']='Eurostat ilc_di01'; income_ref['source_url']=url_income; income_ref['retrieval_date']=today
    income_ref.to_csv(ROOT/'artifacts/reference/income_quantiles_2018.csv',index=False); publish(income_ref,'table_lihc_income_quantiles')
    rent,url_rent=fetch('ilc_mded02',args.year); rent=rent.loc[rent.rskpovth.isin(["B_60","A_60"]) & rent.unit.eq("PC")].copy(); rent['country']=rent.geo.map(GEOS); rent['component']='current rent HH060'; rent['excludes_energy_services']=True; rent['source']='Eurostat ilc_mded02'; rent['source_url']=url_rent; rent['retrieval_date']=today
    rent.to_csv(ROOT/'artifacts/reference/eu_silc_rent_burden_2018.csv',index=False)
    tenure,url_tenure=fetch('ilc_lvho02',args.year); tenure=tenure.loc[tenure.rskpovth.isin(["B_60","A_60"]) & tenure.tenure.isin(["OWN_L","OWN_NL","RENT_MKT","RENT_FR"]) & tenure.unit.eq("PC")].copy(); tenure['country']=tenure.geo.map(GEOS); tenure['source']='Eurostat ilc_lvho02'; tenure['source_url']=url_tenure; tenure['retrieval_date']=today
    tenure.to_csv(ROOT/'artifacts/reference/eu_silc_tenure_distribution_2018.csv',index=False)
    climate=pd.read_csv(ROOT/'config/climate_reference_periods.csv'); publish(climate,'table_climate_reference_periods')
    print(f"income_cutoffs={len(income_ref)} rent_rows={len(rent)} tenure_rows={len(tenure)}")
if __name__=='__main__': main()
