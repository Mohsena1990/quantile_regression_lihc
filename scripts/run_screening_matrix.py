"""Run and assert the full validation-only M0/A0/A1 screening matrix (one script,
replacing the old run_h_screening.py/run_hc_screening.py pair -- there is no more
separate H/HC axis, see config/pipeline.json and scripts/build_a0_predictor_selection.py).
"""
import json
from pathlib import Path
import subprocess, sys
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

families = json.loads((ROOT / 'config/pipeline.json').read_text())['families']
measurements = ['m0_predictive'] + [f'm{f[1:]}_a0' for f in families] + [f'm{f[1:]}_a1' for f in families]
for m in measurements:
    subprocess.run([sys.executable, str(ROOT / 'scripts/run_screening.py'), '--measurement', m, '--stage', 'development'],
                    check=True, cwd=ROOT)
d = pd.concat([pd.read_csv(ROOT / f'outputs/tables/screening_{m}.csv') for m in measurements], ignore_index=True)
expected = len(measurements) * 2  # 2 classifiers, no imbalance-strategy axis
assert len(d) == expected and set(d.evaluation_partition) == {'validation'} and set(d.evaluation_stage) == {'development'}
d.to_csv(ROOT / 'outputs/tables/screening_development_all.csv', index=False)
print('SCREENING_MATRIX_COMPLETE', len(d))
