"""Prepare 2018 Eurostat household electricity/gas prices and fuel coverage."""
from __future__ import annotations
import sys
from datetime import date
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from data.enable import HOUSEHOLD_SHEET,retained_cohort
from scripts.prepare_country_references import fetch,publish
GEOS={'BG':'Bulgaria','HU':'Hungary','IT':'Italy','PL':'Poland'}
FUEL_COLUMNS={'electricity':'H6A1','district_heating':'H6A2','gas':'H6A3','wood':'H6A4','coal_coke':'H6A5','pellets':'H6A6','heating_oil':'H6A7','waste':'H6A8','biomass':'H6A9','heat_pump':'H6A10','other':'H6A11'}

def main()->None:
    frames=[]
    for dataset,fuel,band in [('nrg_pc_204','electricity','KWH2500-4999'),('nrg_pc_202','gas','GJ20-199')]:
        raw,url=fetch(dataset,2018,semester=True)
        selected=raw.loc[raw.nrg_cons.eq(band)&raw.unit.eq('KWH')&raw.tax.eq('I_TAX')&raw.currency.eq('EUR')].copy()
        selected['country']=selected.geo.map(GEOS); selected['fuel']=fuel; selected['price_eur_per_kwh']=selected.value; selected['consumption_band']=band; selected['tax_inclusion']='all taxes and levies included'; selected['source_dataset']=dataset; selected['source_url']=url; selected['retrieval_date']=date.today().isoformat()
        frames.append(selected[['country','time','fuel','consumption_band','price_eur_per_kwh','tax_inclusion','source_dataset','source_url','retrieval_date']])
    prices=pd.concat(frames,ignore_index=True); prices.to_csv(ROOT/'artifacts/reference/historical_energy_prices.csv',index=False)
    data=pd.read_excel(ROOT/'data/raw/ENABLE.EU_dataset_survey of households.xlsx',sheet_name=HOUSEHOLD_SHEET,usecols=['UID','Country','NUTS1']+list(FUEL_COLUMNS.values())); data=retained_cohort(data)
    rows=[]
    for country,g in data.groupby('country'):
        for fuel,col in FUEL_COLUMNS.items():
            share=pd.to_numeric(g[col],errors='coerce'); share=share.where(share.between(0,100)); available=not prices.loc[(prices.country.eq(country))&(prices.fuel.eq(fuel))].empty
            composite_caveat='H6A3 combines natural gas and LPG/propane; this row is pure-natural-gas Eurostat price availability only and must NOT be read as resolved price coverage for the composite field.' if col=='H6A3' else ''
            rows.append({'country':country,'fuel':fuel,'enable_column':col,'households_using':int(share.gt(0).sum()),'reported_heat_share_sum':float(share.sum()),'price_2018_available':available,'price_status':'official Eurostat 2018' if available else 'missing; must not be assigned zero','composite_field_caveat':composite_caveat})
    coverage=pd.DataFrame(rows); publish(coverage,'table_lihc_fuel_price_coverage'); publish(prices,'table_lihc_historical_energy_prices')
    print(coverage.groupby('country').price_2018_available.sum().to_string())
if __name__=='__main__':main()
