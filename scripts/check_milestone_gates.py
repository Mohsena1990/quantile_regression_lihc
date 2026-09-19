"""Evaluate sequential measurement gates from materialized artifacts."""
from pathlib import Path
import json,sys,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from data.enable import RETAINED_COUNTRY_CODES
def exists(x):return (ROOT/x).exists()
N_HCONFIGS=len(json.loads((ROOT/'config/pipeline.json').read_text())['families'])  # quantile is now a per-family searched fit output, not a grid axis
thresholds=pd.read_csv(ROOT/'outputs/tables/table_m0_thresholds.csv');dist=pd.read_csv(ROOT/'outputs/tables/table_m0_class_distribution.csv');hm=json.loads((ROOT/'artifacts/manifests/model_measurements_a0.json').read_text());qc_path=ROOT/'artifacts/climate/nuts1_annual_2017.qc.json';qc=json.loads(qc_path.read_text()) if qc_path.exists() else {}
gates={
 'gate_1_m0':{'status':'CLOSED' if thresholds.query("scope=='descriptive_full_cohort'").country.nunique()==len(RETAINED_COUNTRY_CODES) and dist.country.nunique()==len(RETAINED_COUNTRY_CODES)+1 else 'OPEN'},
 'gate_2_a0_models':{'status':'CLOSED' if len(hm)==N_HCONFIGS and all(x['status']=='AVAILABLE' for x in hm) and exists('artifacts/manifests/a0_oof_calibration.json') else 'IN_PROGRESS','models_available':sum(x['status']=='AVAILABLE' for x in hm),'oof_calibration_available':exists('artifacts/manifests/a0_oof_calibration.json')},
 'gate_3_era5':{'status':'CLOSED' if qc.get('coverage')==1 and qc.get('fused_regions')==qc.get('expected_regions') and qc.get('qc_all_clear') else 'IN_PROGRESS','qc':qc},
 'gate_4_a1_models':{'status':'CLOSED' if exists('artifacts/manifests/model_measurements_a1.json') and len(json.loads((ROOT/'artifacts/manifests/model_measurements_a1.json').read_text()))==N_HCONFIGS else 'BLOCKED_GATE_3'}}
# Enforce sequential interpretation without deleting already materialized debugging outputs.
order=list(gates);prior=True
for key in order:
 gates[key]['prior_gates_closed']=prior
 if not prior and gates[key]['status']=='CLOSED':gates[key]['execution_status']='MATERIALIZED_EARLY_NOT_GATE_ENABLED'
 prior=prior and gates[key]['status']=='CLOSED'
out=ROOT/'artifacts/manifests/milestone_gates.json';out.write_text(json.dumps(gates,indent=2)+'\n');print(json.dumps(gates,indent=2))
