"""Continuous calibration search + deterministic five-fold OOF training labels,
for all M2/M5/M7 (family x A0/A1) configs. Runs BEFORE run_model_measurements.py:
quantile is discovered here (lib/tuning/quantile_search.py), not read from a
pre-existing manifest.
"""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.enable import HOUSEHOLD_SHEET,retained_cohort
from lib.features.household_features import BASE_RAW_COLUMNS,RAW_COLUMNS as FEATURE_RAW_COLUMNS,engineer,stack_features
from lib.tuning.quantile_search import find_optimal_quantile,oof_fold
COMP=[f'S1{g}c{i}' for g in ('A','B') for i in range(1,4)]
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--feature',choices=['A0','A1'],default='A0');a=parser.parse_args()
 climate_mode=a.feature=='A1'
 cfg=json.loads((ROOT/'config/pipeline.json').read_text());base=pd.read_parquet(ROOT/'artifacts/data/harmonised_observed_cost.parquet');cols=['UID','Country','NUTS1']+BASE_RAW_COLUMNS+FEATURE_RAW_COLUMNS+COMP;hh=retained_cohort(pd.read_excel(ROOT/'data/raw/ENABLE.EU_dataset_survey of households.xlsx',sheet_name=HOUSEHOLD_SHEET,usecols=list(dict.fromkeys(cols))));hh['household_size']=hh[COMP].apply(pd.to_numeric,errors='coerce').sum(axis=1,min_count=len(COMP));hh=engineer(hh);d=base.merge(hh.drop(columns=['Country','NUTS1','country']),on='UID',how='left',validate='one_to_one')
 features=list(cfg['features'])
 if climate_mode:
  climate=ROOT/'artifacts/climate/nuts1_annual_2017.parquet'
  if not climate.exists():raise SystemExit('GATE_3_OPEN')
  c=pd.read_parquet(climate);d=d.merge(c[['NUTS1']+cfg['climate_features']],on='NUTS1',how='left',validate='many_to_one');features=stack_features(features,cfg['climate_features'])
  if d[cfg['climate_features']].isna().any().any():raise SystemExit('HC_CLIMATE_MISSINGNESS')
 split=json.loads((ROOT/cfg['split']).read_text())['assignments'];part={str(x['household_id']):x['partition'] for x in split};d['partition']=d.UID.astype(str).map(part);training=d[d.partition.eq('train')].copy();training['oof_fold']=training.UID.map(oof_fold);suffix=a.feature.lower();bounds=tuple(cfg['quantile_search_bounds']);xatol=cfg['quantile_search_xatol'];audit=[];evaluations=[]
 for family in cfg['families']:
  eid=f'{family.lower()}_{suffix}'
  result=find_optimal_quantile(family,training,features=features,target=cfg['target'],bounds=bounds,xatol=xatol)
  q=result['quantile'];oof=result['oof_predictions']
  path=ROOT/f'artifacts/predictions/{eid}.parquet';oof.to_parquet(path,index=False)  # train-only rows; run_model_measurements.py splices in validation/test next
  for country,g in list(oof.groupby('country'))+[('Five-country overall',oof)]: audit.append({'experiment_id':eid,'family':family,'quantile':q,'country':country,'n':int(g.measurement_eligible.sum()),'observed_high_cost_rate':g.loc[g.measurement_eligible,'high_cost'].mean(),'nominal_high_cost_rate':1-q,'absolute_calibration_error':abs(g.loc[g.measurement_eligible,'high_cost'].mean()-(1-q)),'prediction_scope':'five_fold_oof_training'})
  evaluations.append({'family':family,'quantile':q,'n_evaluations':result['n_evaluations']})
  print(f'{eid} q*={q:.4f} n_evaluations={result["n_evaluations"]}',flush=True)
 out=pd.DataFrame(audit);out.to_csv(ROOT/f'outputs/tables/table_{suffix}_oof_calibration.csv',index=False);status={'status':'AVAILABLE','feature_configuration':a.feature,'folds':5,'assignment':'sha256(oof-v1|UID) modulo 5','quantile_search_bounds':list(bounds),'quantile_search_xatol':xatol,'experiments':len(cfg['families']),'rows':len(out),'max_absolute_calibration_error':float(out.absolute_calibration_error.max()),'evaluations':evaluations};(ROOT/f'artifacts/manifests/{suffix}_oof_calibration.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
if __name__=='__main__':main()
