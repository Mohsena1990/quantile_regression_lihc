"""Rank validation-only screening and freeze choices without unlocking test."""
from pathlib import Path
import hashlib,json,sys
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));rows=[]
for f in sorted((ROOT/'outputs/tables').glob('screening_country_m*.csv')):
 d=pd.read_csv(f);assert set(d.evaluation_stage)=={'development'}
 for keys,g in d.groupby(['measurement','model']):
  rows.append(dict(zip(['measurement','model'],keys))|{'macro_f1':g.macro_f1.mean(),'double_risk_f1':g.double_risk_f1.mean(),'double_risk_recall':g.double_risk_recall.mean(),'balanced_accuracy':g.balanced_accuracy.mean(),'log_loss':g.log_loss.mean(),'countries':g.country.nunique()})
r=pd.DataFrame(rows).sort_values(['measurement','macro_f1','double_risk_f1','double_risk_recall','balanced_accuracy','log_loss'],ascending=[True,False,False,False,False,True]);chosen=r.groupby('measurement',as_index=False).head(1)
n_families=len(json.loads((ROOT/'config/pipeline.json').read_text())['families'])
expected=1+n_families*2  # m0_predictive + (families x A0/A1)
assert chosen.measurement.nunique()==expected,(chosen.measurement.nunique(),expected)
p=ROOT/'outputs/tables/table_validation_model_selection.csv';r.to_csv(p,index=False)
manifest={'choices_frozen':True,'test_unlock_authorized':False,'selection_partition':'validation','ranking_hierarchy':['equal_country_macro_f1','equal_country_double_risk_f1','equal_country_double_risk_recall','equal_country_balanced_accuracy','equal_country_log_loss'],'choices':chosen.to_dict('records'),'ranking_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(ROOT/'artifacts/manifests/final_selection_frozen.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest,indent=2))
