"""Validate retained ENABLE regions against official Eurostat GISCO NUTS 2016."""
from __future__ import annotations
import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.enable import HOUSEHOLD_SHEET,retained_cohort
from lib.geography.validation import official_nuts_ids,validate_nuts_coverage

def main():
 d=retained_cohort(pd.read_excel(ROOT/'data/raw/ENABLE.EU_dataset_survey of households.xlsx',sheet_name=HOUSEHOLD_SHEET,usecols=['UID','Country','NUTS1']))
 official=official_nuts_ids(ROOT/'artifacts/reference/NUTS_RG_01M_2016_4326_LEVL_1.geojson'); result=validate_nuts_coverage(d.NUTS1.unique(),official)
 result.update({'nuts_version':2016,'geometry_source':'Eurostat GISCO NUTS_RG_01M_2016_4326_LEVL_1.geojson','households':len(d)})
 (ROOT/'artifacts/manifests/nuts1_validation.json').write_text(json.dumps(result,indent=2)+'\n')
 table=d.groupby('country').agg(households=('UID','size'),survey_nuts1=('NUTS1','nunique')).reset_index(); table['matched_nuts1']=table.apply(lambda r:d.loc[d.country.eq(r.country),'NUTS1'].isin(official).groupby(d.loc[d.country.eq(r.country),'NUTS1']).all().sum(),axis=1); table['complete']=table.survey_nuts1.eq(table.matched_nuts1)
 p=ROOT/'outputs/tables/table_nuts1_validation';table.to_csv(p.with_suffix('.csv'),index=False);table.to_excel(p.with_suffix('.xlsx'),index=False);p.with_suffix('.tex').write_text(table.to_latex(index=False))
 if not result['complete']: raise RuntimeError(f"Unmatched NUTS1: {result['missing_regions']}")
 print(json.dumps(result))
if __name__=='__main__':main()
