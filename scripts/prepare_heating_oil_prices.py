"""Extract 2018 heating-oil prices from the official EC Weekly Oil Bulletin."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];src=ROOT/'artifacts/reference/weekly_oil_bulletin_history.xlsx'
CODES={'Bulgaria':'BG','Hungary':'HU','Italy':'IT','Poland':'PL'};KWH_PER_LITRE=10.0
raw=pd.read_excel(src,sheet_name='Prices with taxes',header=0);dates=pd.to_datetime(raw.iloc[:,0],errors='coerce',format='mixed');rows=[]
for country,code in CODES.items():
 col=f'{code}_price_with_tax_heating_oil';values=pd.to_numeric(raw[col],errors='coerce');v=values[dates.dt.year.eq(2018)]
 if v.notna().any(): rows.append({'country':country,'time':'2018 annual weekly mean','fuel':'fuel_oil','consumption_band':'household heating oil delivery','price_eur_per_kwh':v.mean()/1000/KWH_PER_LITRE,'tax_inclusion':'duties and taxes included','source_dataset':'European Commission Weekly Oil Bulletin','source_url':'https://energy.ec.europa.eu/data-and-analysis/weekly-oil-bulletin_en','retrieval_date':pd.Timestamp.utcnow().date().isoformat(),'original_unit':'EUR/1000 litres','unit_conversion':'divide by 1000 litres and 10.0 net kWh/litre','conversion_source':'UK Forest Research typical calorific values of fuels','uncertainty':'weekly 2018 price range plus calorific-value sensitivity','observations':int(v.notna().sum())})
out=pd.DataFrame(rows);dest=ROOT/'artifacts/reference/historical_energy_prices.csv';base=pd.read_csv(dest);base=base[~base.fuel.eq('fuel_oil')];pd.concat([base,out],ignore_index=True).to_csv(dest,index=False);out.to_csv(ROOT/'outputs/tables/table_lihc_heating_oil_prices.csv',index=False);print(out.to_string(index=False))
