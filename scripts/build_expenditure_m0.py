"""Build observed-cost target, immutable splits, and M0 labels."""
from __future__ import annotations
import hashlib,json,sys
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.enable import HOUSEHOLD_SHEET,retained_cohort
from lib.evaluation.split_registry import SplitAssignment,read_split_registry,write_split_registry
from lib.expenditure.currency import convert_local_to_eur
from lib.expenditure.reconstruct import annual_electricity_cost,annual_heating_cost,combine_energy_cost
from measurement.families.van_hove import country_thresholds,label_with_thresholds
from scripts.prepare_country_references import BASE,parse_jsonstat,publish
CURRENCY={'Bulgaria':'BGN','Hungary':'HUF','Italy':'EUR','Poland':'PLN','Spain':'EUR'}
def currency_reference()->pd.DataFrame:
 import urllib.request
 url=f'{BASE}/ert_bil_eur_a?lang=EN&time=2017'; payload=json.load(urllib.request.urlopen(url)); raw=parse_jsonstat(payload); selected=raw.loc[raw.statinfo.eq('AVG')&raw.currency.isin(['HUF','PLN'])]
 values=selected.set_index('currency').value.to_dict();rows=[]
 for country,currency in CURRENCY.items():
  if currency=='EUR': rate=1.;source='Euro-area currency'
  elif currency=='BGN':rate=1.95583;source='EU irrevocable/fixed EUR-BGN rate applicable in 2017'
  else:rate=values[currency];source='Eurostat ert_bil_eur_a annual average'
  rows.append({'country':country,'currency':currency,'reference_year':2017,'local_currency_per_eur':rate,'conversion_equation':'EUR = local currency / local_currency_per_eur','source':source,'source_url':url,'retrieval_date':date.today().isoformat(),'notes':'Primary expenditure reference year'})
 return pd.DataFrame(rows)
def _uid_unit_interval(uid:str,*,seed:int,salt:str)->float:
 """Deterministic pseudo-uniform value in [0,1) for one UID; independent of any other row."""
 digest=hashlib.sha256(f'{salt}|{seed}|{uid}'.encode()).digest();return int.from_bytes(digest[:8],'big')/2**64

def deterministic_splits(d:pd.DataFrame,seed:int=42)->pd.DataFrame:
 """Assign train/validation/test per household from a per-UID hash (stratified by country).

 Unlike a positional np.random.permutation over each country group, the assignment for a
 given UID depends only on (seed, country, UID) -- never on the presence, order, or count
 of any other row -- so the split is stable under any upstream addition/removal/reorder of
 unrelated households. This is required for the split to be genuinely immutable once frozen.
 """
 x=d[['UID','country','NUTS1']].copy()
 h=x.apply(lambda r:_uid_unit_interval(str(r.UID),seed=seed,salt=f'split-v1|{r.country}'),axis=1)
 x['split']=np.where(h<.6,'train',np.where(h<.8,'validation','test'))
 return x

def build_split_registry(d:pd.DataFrame,path:Path,*,seed:int=42)->pd.DataFrame:
 """Return the frozen UID->partition split, generating it once and never silently regenerating it.

 If `path` already exists, every UID in the current cohort must already be present in it; the
 existing frozen assignments are reused verbatim. This is what actually makes the split
 immutable in practice, not just in a docstring: a later rerun of this script (e.g. after an
 upstream data fix) cannot silently reshuffle households across train/validation/test.
 """
 uid_lookup=dict(zip(d.UID.astype(str),d.UID))  # str(UID) -> original-dtype UID, so the merge below stays dtype-consistent
 current_uids=set(uid_lookup)
 if path.exists():
  _,existing=read_split_registry(path);existing_by_uid={a.household_id:a for a in existing}
  missing=current_uids-set(existing_by_uid)
  if missing:
   raise SystemExit(f'SPLIT_REGISTRY_IMMUTABLE: {len(missing)} household UID(s) in the current cohort '
                     f'are absent from the frozen split at {path}. The split registry must not be '
                     'silently regenerated once frozen -- create an explicitly versioned new split '
                     '(e.g. v2) instead.')
  return pd.DataFrame([{'UID':uid_lookup[uid],'country':a.country,'NUTS1':a.nuts1,'split':a.partition}
                        for uid,a in existing_by_uid.items() if uid in current_uids])
 split=deterministic_splits(d,seed=seed)
 assignments=[SplitAssignment(str(r.UID),r.split,country=r.country,nuts1=r.NUTS1,split_version='v1_country_60_20_20',seed=seed)
              for r in split.itertuples()]
 write_split_registry(assignments,path,metadata={'seed':seed,'strategy':'per-UID deterministic hash 60/20/20, stratified by country','split_version':'v1_country_60_20_20'})
 return split
def main():
 cols=['UID','Country','NUTS1','H7A1','H7A2','H7A88','H7AA','H8A','H8B','S9MONTH','w','H6A1'];d=retained_cohort(pd.read_excel(ROOT/'data/raw/ENABLE.EU_dataset_survey of households.xlsx',sheet_name=HOUSEHOLD_SHEET,usecols=cols));d['income_bracket']=pd.to_numeric(d.S9MONTH,errors='coerce').where(lambda x:x.between(1,10))
 heat=d.apply(lambda r:annual_heating_cost(r.H7A2,r.H7A1,r.H7AA,no_heating=(r.H7A88==1)),axis=1);electric=d.apply(lambda r:annual_electricity_cost(r.H8B,r.H8A),axis=1);d['heating_cost_local']=[x[0] for x in heat];d['heating_cost_source']=[x[1] for x in heat];d['electricity_cost_local']=[x[0] for x in electric];d['electricity_cost_source']=[x[1] for x in electric];both=d.apply(lambda r:combine_energy_cost(r.heating_cost_local,r.electricity_cost_local),axis=1);d['observed_energy_cost_local']=[x[0] for x in both];d['energy_cost_completeness']=[x[1] for x in both];d['electric_heating_double_count_risk']=pd.to_numeric(d.H6A1,errors='coerce').gt(0)&d.observed_energy_cost_local.notna()
 rates=currency_reference();rates.to_csv(ROOT/'artifacts/reference/currency_conversion.csv',index=False);d['energy_cost_currency']=d.country.map(CURRENCY);d['observed_energy_cost_eur_reference']=convert_local_to_eur(d,rates);d['cost_outlier_flag']=d.groupby('country').observed_energy_cost_eur_reference.transform(lambda x:x.gt(x.quantile(.995)))
 d.to_parquet(ROOT/'artifacts/data/harmonised_observed_cost.parquet',index=False)
 summary=d.groupby('country').agg(cohort_n=('UID','size'),heating_eligible_n=('heating_cost_local','count'),electricity_eligible_n=('electricity_cost_local','count'),observed_cost_eligible_n=('observed_energy_cost_eur_reference','count'),income_eligible_n=('income_bracket','count'),m0_eligible_n=('UID',lambda s:int((d.loc[s.index,'observed_energy_cost_eur_reference'].notna()&d.loc[s.index,'income_bracket'].notna()).sum())),electric_double_count_risk_n=('electric_heating_double_count_risk','sum')).reset_index();publish(summary,'table04_energy_expenditure_summary');publish(rates,'table_currency_conversion')
 split=build_split_registry(d,ROOT/'artifacts/splits/household_split_v1.json',seed=42);d=d.merge(split,on=['UID','country','NUTS1'],validate='one_to_one')
 eligible=d.dropna(subset=['observed_energy_cost_eur_reference','income_bracket']).copy();full_t=country_thresholds(eligible);desc=label_with_thresholds(eligible,full_t);desc['label_scope']='descriptive_full_cohort';desc['threshold_basis']='full eligible country cohort'
 train=eligible[eligible.split.eq('train')];dev=eligible[eligible.split.isin(['train','validation'])];train_t=country_thresholds(train);dev_t=country_thresholds(dev);parts=[]
 for part,g in eligible.groupby('split'):
  thresholds=dev_t if part=='test' else train_t;x=label_with_thresholds(g,thresholds);x['label_scope']='predictive_evaluation';x['threshold_basis']='train+validation' if part=='test' else 'training';parts.append(x)
 pred=pd.concat(parts);meta={'measurement_family':'M0','measurement_name':'van_hove_country_q80','feature_configuration':None,'threshold_configuration':'country_Q80'}
 for frame,path in [(desc,ROOT/'artifacts/predictions/m0_descriptive.parquet'),(pred,ROOT/'artifacts/predictions/m0_predictive.parquet')]:
  for k,v in meta.items():frame[k]=v
  frame.to_parquet(path,index=False)
 thresholds=pd.DataFrame([{'scope':'descriptive_full_cohort','country':c,'threshold_eur':v} for c,v in full_t.items()]+[{'scope':'training','country':c,'threshold_eur':v} for c,v in train_t.items()]+[{'scope':'train_validation_for_test','country':c,'threshold_eur':v} for c,v in dev_t.items()]);publish(thresholds,'table_m0_thresholds')
 distribution=desc.groupby(['country','risk_class_code','risk_class_name']).size().rename('n').reset_index();distribution['share']=distribution.n/distribution.groupby('country').n.transform('sum');overall=desc.groupby(['risk_class_code','risk_class_name']).size().rename('n').reset_index();overall.insert(0,'country','Five-country overall');overall['share']=overall.n/overall.n.sum();distribution=pd.concat([distribution,overall],ignore_index=True);publish(distribution,'table_m0_class_distribution')
 print(summary.to_string(index=False));print(distribution.to_string(index=False))
if __name__=='__main__':main()
