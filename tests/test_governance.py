import json
from pathlib import Path

import pandas as pd
import pytest
from lib.geography.validation import validate_nuts_coverage

ROOT = Path(__file__).resolve().parents[1]

def test_nuts_validation_never_truncates_codes():
 result=validate_nuts_coverage(['BG3','HU1'],['BG3','HU1','ITC']); assert result['complete']
 assert not validate_nuts_coverage(['BG33'],['BG3'])['complete']

# --- Screening governance: feature leakage, imbalance strategy count, run counts ---

FORBIDDEN_SCREENING_COLUMNS = {
    'observed_energy_cost_eur_reference', 'observed_energy_cost_local', 'predicted_cost',
    'required_cost', 'conditional_threshold', 'high_cost', 'low_income', 'risk_class_code',
    'risk_class_name', 'income_bracket', 'S9MONTH', 'S9YEAR', 'cost_outlier_flag',
}

def test_primary_screening_feature_whitelist_excludes_target_derived_columns():
    """config/pipeline.json's 'features' list is the whitelist screening/models.py draws from
    (screening/models.py: used=list(features)). It must never contain a target-derived or
    income column, since that is the only thing preventing leakage into the classifiers."""
    cfg = json.loads((ROOT / 'config/pipeline.json').read_text())
    leaked = FORBIDDEN_SCREENING_COLUMNS & set(cfg['features'])
    assert not leaked, f'leaky columns present in the screening feature whitelist: {leaked}'

def test_screening_development_run_count_and_two_classifiers():
    path = ROOT / 'outputs/tables/screening_development_all.csv'
    if not path.exists():
        pytest.skip('screening_development_all.csv not generated in this checkout')
    d = pd.read_csv(path)
    n_measurements = 1 + 3 + 3  # 1 M0 + 3 families x {A0, A1}
    assert len(d) == n_measurements * 2  # 2 classifiers, no imbalance-strategy axis
    assert set(d['model']) == {'catboost', 'hierarchical_catboost'}
    assert d['measurement'].nunique() == n_measurements
    counts = d.groupby('model').size()
    assert (counts == n_measurements).all()

def test_no_stale_duplicate_screening_artifacts_outside_current_naming():
    """Regression guard for the archived legacy artifacts found during audit (ISS-005): files
    written by an old script revision (no evaluation-stage suffix) must not silently reappear
    in artifacts/predictions/ where they could be globbed instead of the governed screening matrix."""
    pred_dir = ROOT / 'artifacts/predictions'
    if not pred_dir.exists():
        pytest.skip('artifacts/predictions not present in this checkout')
    legacy = [f for f in pred_dir.glob('*_class_weight.parquet')] + [f for f in pred_dir.glob('*_oversample.parquet')]
    assert not legacy, f'legacy pre-stage-naming screening artifacts found: {[f.name for f in legacy]}'

