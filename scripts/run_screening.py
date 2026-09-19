"""Run all eligible four-class measurement labels through Level-3 screening."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,f1_score,log_loss,recall_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.enable import HOUSEHOLD_SHEET,retained_cohort
from lib.features.household_features import BASE_RAW_COLUMNS,RAW_COLUMNS as FEATURE_RAW_COLUMNS,engineer,stack_features
from lib.features.country_context import CONTEXT_COLUMNS
from lib.screening import fit_screening_model

def features_for(measurement,cfg):
 """A0 = cfg['features']; A1 = A0 + climate; A2 = A1 + country context. M0 (no config
 suffix) screens with A0, the same predictor set as every other family."""
 a0=list(cfg['features'])
 if measurement.endswith('_a2'): return stack_features(a0,cfg['climate_features'],CONTEXT_COLUMNS)
 if measurement.endswith('_a1'): return stack_features(a0,cfg['climate_features'])
 return a0

def main():
 p=argparse.ArgumentParser();p.add_argument('--measurement',required=True);p.add_argument('--model',choices=['catboost','hierarchical_catboost']);p.add_argument('--stage',choices=['development','final_test'],default='development');a=p.parse_args();src=ROOT/f'artifacts/predictions/{a.measurement}.parquet'
 d=pd.read_parquet(src);cfg=json.loads((ROOT/'config/pipeline.json').read_text())
 features=features_for(a.measurement,cfg)
 missing=[x for x in features if x not in d.columns]
 if missing:
  comp=[f'S1{g}c{i}' for g in ('A','B') for i in range(1,4)];cols=list(dict.fromkeys(['UID','Country','NUTS1']+BASE_RAW_COLUMNS+FEATURE_RAW_COLUMNS+comp));hh=retained_cohort(pd.read_excel(ROOT/'data/raw/ENABLE.EU_dataset_survey of households.xlsx',sheet_name=HOUSEHOLD_SHEET,usecols=cols))
  hh['household_size']=hh[comp].apply(pd.to_numeric,errors='coerce').sum(axis=1,min_count=len(comp));hh=engineer(hh)
  add=['UID']+[x for x in missing if x in hh.columns];d=d.merge(hh[add],on='UID',how='left',validate='one_to_one')
 if 'partition' not in d and 'split' in d: d=d.rename(columns={'split':'partition'})
 if 'measurement_eligible' not in d: d['measurement_eligible']=d.risk_class_code.notna()
 d=d[d.measurement_eligible]
 if a.stage=='development': train=d[d.partition.eq('train')];test=d[d.partition.eq('validation')]
 else:
  decision=ROOT/'artifacts/manifests/final_selection_frozen.json'
  if not decision.exists(): raise SystemExit('FINAL_TEST_LOCKED: freeze validation selections first')
  state=json.loads(decision.read_text())
  if not state.get('choices_frozen') or not state.get('test_unlock_authorized'): raise SystemExit('FINAL_TEST_LOCKED: explicit frozen-decision authorization required')
  train=d[d.partition.isin(['train','validation'])];test=d[d.partition.eq('test')]
 rows=[];country_rows=[]
 for kind in ([a.model] if a.model else ['catboost','hierarchical_catboost']):
  try:
   _,pred,metrics=fit_screening_model(kind,train,test,features=features);stem=f'{a.measurement}_{kind}_{a.stage}';pred=pred.merge(test[['UID','country']],on='UID',validate='one_to_one');pred.to_parquet(ROOT/f'artifacts/predictions/{stem}.parquet',index=False)
   for country,g in pred.groupby('country'):
    probs=g[[f'prob_{k}' for k in range(4)]].to_numpy();probs=probs/probs.sum(axis=1,keepdims=True);country_rows.append({'measurement':a.measurement,'model':kind,'evaluation_stage':a.stage,'country':country,'n':len(g),'macro_f1':f1_score(g.truth,g.prediction,average='macro',zero_division=0),'double_risk_f1':f1_score(g.truth,g.prediction,labels=[3],average='macro',zero_division=0),'double_risk_recall':recall_score(g.truth,g.prediction,labels=[3],average='macro',zero_division=0),'balanced_accuracy':balanced_accuracy_score(g.truth,g.prediction),'log_loss':log_loss(g.truth,probs,labels=[0,1,2,3])})
   rows.append({'measurement':a.measurement,'model':kind,'evaluation_stage':a.stage,'evaluation_partition':'validation' if a.stage=='development' else 'test','status':'AVAILABLE',**metrics})
  except Exception as e: rows.append({'measurement':a.measurement,'model':kind,'evaluation_stage':a.stage,'evaluation_partition':'validation' if a.stage=='development' else 'test','status':'BLOCKED','reason':f'{type(e).__name__}: {e}'})
 out=pd.DataFrame(rows);dest=ROOT/f'outputs/tables/screening_{a.measurement}.csv' if a.stage=='development' else ROOT/f'outputs/tables/final_test_screening_{a.measurement}.csv'
 existing=pd.read_csv(dest) if dest.exists() else pd.DataFrame()
 if a.model and len(existing): existing=existing[existing.model.ne(a.model)];out=pd.concat([existing,out],ignore_index=True)
 out.to_csv(dest,index=False)
 country_out=pd.DataFrame(country_rows);country_dest=ROOT/f'outputs/tables/screening_country_{a.measurement}.csv' if a.stage=='development' else ROOT/f'outputs/tables/final_test_screening_country_{a.measurement}.csv'
 if a.model and country_dest.exists():
  old=pd.read_csv(country_dest);old=old[old.model.ne(a.model)];country_out=pd.concat([old,country_out],ignore_index=True)
 country_out.to_csv(country_dest,index=False);print(out.to_string(index=False))
if __name__=='__main__':main()
