"""Run M2/M5/M7 measurement strategies on the immutable household split."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.enable import HOUSEHOLD_SHEET,retained_cohort
from lib.features.household_features import BASE_RAW_COLUMNS,RAW_COLUMNS as FEATURE_RAW_COLUMNS,engineer,stack_features
from measurement.families.model_based import fit_measurement_strategy
COMP=[f'S1{g}c{i}' for g in ('A','B') for i in range(1,4)]
def preserve_oof_train_rows(existing:pd.DataFrame,fresh:pd.DataFrame)->pd.DataFrame:
 """If `existing` already carries 5-fold-OOF-calibrated train rows (see build_oof_predictions.py),
 keep them and only take validation/test rows from `fresh`. This is what stops a plain rerun of
 this script from silently reintroducing in-sample train-on-itself threshold leakage after the
 OOF step has already run. If `existing` has no OOF-calibrated train rows yet, `fresh` is used as-is.
 """
 train_mask=existing.partition.eq('train')
 if 'oof_fold' in existing.columns and train_mask.any() and existing.loc[train_mask,'oof_fold'].notna().all():
  return pd.concat([existing[train_mask],fresh[~fresh.partition.eq('train')]],ignore_index=True)
 return fresh
def main():
 p=argparse.ArgumentParser();p.add_argument('--feature',choices=['A0','A1'],default='A0');a=p.parse_args()
 # A0 = this project's standard predictor set (scripts/build_a0_predictor_selection.py);
 # A1 = A0 + climate. No more H_BASE/HC_BASE ablation axis (see AUDIT/README 2026-08-26):
 # the recursive search that builds A0 already measures each feature's marginal value directly.
 climate_mode=a.feature=='A1'
 cfg=json.loads((ROOT/'config/pipeline.json').read_text());base=pd.read_parquet(ROOT/'artifacts/data/harmonised_observed_cost.parquet')
 cols=['UID','Country','NUTS1']+BASE_RAW_COLUMNS+FEATURE_RAW_COLUMNS+COMP
 hh=retained_cohort(pd.read_excel(ROOT/'data/raw/ENABLE.EU_dataset_survey of households.xlsx',sheet_name=HOUSEHOLD_SHEET,usecols=list(dict.fromkeys(cols))))
 hh['household_size']=hh[COMP].apply(pd.to_numeric,errors='coerce').sum(axis=1,min_count=len(COMP));hh=engineer(hh);d=base.merge(hh.drop(columns=['Country','NUTS1','country']),on='UID',how='left',validate='one_to_one')
 features=list(cfg['features']);blocked=None
 if climate_mode:
  cp=ROOT/'artifacts/climate/nuts1_annual_2017.parquet'
  if not cp.exists(): blocked='ERA5_LAND_PRIMARY_MISSING'
  else:
   # Select only NUTS1+climate_features -- the climate parquet also carries its own 'country'
   # column, which would otherwise collide with d's and silently rename both to
   # country_x/country_y, breaking every non-native family's groupby(country_col) offset step.
   climate=pd.read_parquet(cp)[['NUTS1']+cfg['climate_features']]
   d=d.merge(climate,on='NUTS1',how='left',validate='many_to_one');features=stack_features(features,cfg['climate_features'])
   if d[cfg['climate_features']].isna().any().any(): blocked='HC_CLIMATE_MISSINGNESS_OR_POPULATION_MISMATCH'
 if blocked:
  print(blocked);return
 split=json.loads((ROOT/cfg['split']).read_text())['assignments'];part={str(x['household_id']):x['partition'] for x in split};d['partition']=d.UID.astype(str).map(part)
 train=d[d.partition.eq('train')];apply=d;suffix=a.feature.lower()
 calib_path=ROOT/f'outputs/tables/table_{suffix}_oof_calibration.csv'
 if not calib_path.exists(): raise SystemExit(f'Run scripts/build_oof_predictions.py --feature {a.feature} first (missing {calib_path.name})')
 overall=pd.read_csv(calib_path);overall=overall[overall.country.eq('Five-country overall')].set_index('family')
 manifest=[]
 for family in cfg['families']:
  eid=f'{family.lower()}_{suffix}';print(eid,flush=True)
  if family not in overall.index:
   manifest.append({'experiment_id':eid,'status':'BLOCKED','reason':'MISSING_OOF_CALIBRATION_ROW'});continue
  q=float(overall.loc[family,'quantile'])
  try:
   _,out=fit_measurement_strategy(family,train,apply,features=features,target=cfg['target'],quantile=q)
   path=ROOT/f'artifacts/predictions/{eid}.parquet'
   if path.exists(): out=preserve_oof_train_rows(pd.read_parquet(path),out)
   out.to_parquet(path,index=False)
   manifest.append({'experiment_id':eid,'status':'AVAILABLE','eligible_n':int(out.measurement_eligible.sum()),'path':str(path.relative_to(ROOT)),'quantile':q})
  except Exception as e: manifest.append({'experiment_id':eid,'status':'BLOCKED','reason':f'{type(e).__name__}: {e}'})
 mp=ROOT/f'artifacts/manifests/model_measurements_{suffix}.json';expected=len(cfg['families']);assert len(manifest)==expected;mp.write_text(json.dumps(manifest,indent=2)+'\n');print(mp)
if __name__=='__main__':main()
