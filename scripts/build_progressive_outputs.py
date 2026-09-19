"""Build progressive distributions and comparison figures from available outputs."""
import json
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from lib.reporting.figures import save_figure
# Core M0/M2/M5/M7 measurement stems only (not the per-architecture screening prediction
# files that also live in artifacts/predictions/, e.g. m0_predictive_catboost_*.parquet,
# which glob('m[02-8]_*.parquet') used to match too -- see AUDIT_AND_REPAIR_REPORT.md).
families=json.loads((ROOT/'config/pipeline.json').read_text())['families']
CORE_STEMS=['m0_descriptive','m0_predictive']+[f'm{f[1:]}_h' for f in families]
files=[p for p in (ROOT/'artifacts/predictions').glob('*.parquet') if p.stem in CORE_STEMS];rows=[]
for p in sorted(files):
 d=pd.read_parquet(p);cost='RequiredCost' if 'RequiredCost' in d else 'predicted_cost' if 'predicted_cost' in d else 'observed_energy_cost_eur_reference'
 high='HighCost' if 'HighCost' in d else 'high_cost' if 'high_cost' in d else None
 rows.append({'strategy':p.stem,'n':len(d),'cost_n':int(d[cost].notna().sum()),'cost_mean':d[cost].mean(),'cost_median':d[cost].median(),'high_cost_prevalence':d[high].mean() if high else None})
out=pd.DataFrame(rows);out.to_csv(ROOT/'outputs/tables/table_measurement_progress.csv',index=False)
if len(out):
 ax=out.plot(x='strategy',y='high_cost_prevalence',kind='bar',legend=False,figsize=(12,5));ax.set_ylabel('High-cost prevalence');plt.tight_layout();save_figure(ax,ROOT/'outputs/figures/high_cost_prevalence');plt.close()
print(out.to_string(index=False))
