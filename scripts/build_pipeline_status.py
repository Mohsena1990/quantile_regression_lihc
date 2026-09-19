"""Write a machine-readable status manifest without claiming blocked work complete."""
from pathlib import Path
import json,sys,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
h_manifest=ROOT/'artifacts/manifests/model_measurements_a0.json';hc_manifest=ROOT/'artifacts/manifests/model_measurements_a1.json'
climate=ROOT/'artifacts/climate/nuts1_annual_2017.parquet'
# Prefer the current authoritative combined screening table; table_screening_results.csv
# is a legacy snapshot from before HC screening existed and would silently under-report.
screen=ROOT/'outputs/tables/screening_development_all.csv'
if not screen.exists(): screen=ROOT/'outputs/tables/table_screening_results.csv'
h=json.loads(h_manifest.read_text()) if h_manifest.exists() else [];hc=json.loads(hc_manifest.read_text()) if hc_manifest.exists() else []
s=pd.read_csv(screen) if screen.exists() else pd.DataFrame()
champions_path=ROOT/'outputs/tables/table_family_champions.csv'
champions=pd.read_csv(champions_path) if champions_path.exists() else pd.DataFrame()
final_selection=ROOT/'artifacts/manifests/final_selection_frozen.json'
final_state=json.loads(final_selection.read_text()) if final_selection.exists() else {}
n_families=len(json.loads((ROOT/'config/pipeline.json').read_text())['families'])  # families (quantile is now a per-family searched fit output, not a grid axis)
n_screening_expected=(1+n_families)*2+n_families*2  # (M0 + A0 configs) + A1 configs, x 2 classifiers
status={'M0':'AVAILABLE',
 'A0_available':sum(x.get('status')=='AVAILABLE' for x in h),'A0_expected':n_families,
 'ERA5_2017_fused':climate.exists(),'A1':'AVAILABLE' if climate.exists() else 'BLOCKED_ERA5_FUSION',
 'A1_available':sum(x.get('status')=='AVAILABLE' for x in hc),'A1_expected':n_families,
 'screening_runs_available':int(s.status.eq('AVAILABLE').sum()) if len(s) else 0,'screening_runs_expected':n_screening_expected,
 'non_gas_electric_prices':'PARTIAL_SOURCE_GATED','four_class_families':['M0','M2','M5','M7'],
 'family_champions_selected':int((champions.status=='AVAILABLE').sum()) if len(champions) else 0,
 'validation_selection_frozen':final_state.get('choices_frozen',False),'test_unlock_authorized':final_state.get('test_unlock_authorized',False)}
(ROOT/'artifacts/manifests/pipeline_status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
