import json
from pathlib import Path

from lib.geography.validation import validate_nuts_coverage

ROOT = Path(__file__).resolve().parents[1]

def test_nuts_validation_never_truncates_codes():
 result=validate_nuts_coverage(['BG3','HU1'],['BG3','HU1','ITC']); assert result['complete']
 assert not validate_nuts_coverage(['BG33'],['BG3'])['complete']

# --- Governance: feature leakage ---

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


def test_identification_tiers_never_contain_target_derived_columns():
    """Tiers T1-T3 draw on config/pipeline.json and the study's own lists. Income excluded: no forbidden column at all.
    Income included (Van Hove's design): the income bracket is the ONLY allowed exception."""
    from lib.identification.data import tier_features
    for tier, feats in tier_features().items():
        assert not (FORBIDDEN_SCREENING_COLUMNS & set(feats)), (tier, FORBIDDEN_SCREENING_COLUMNS & set(feats))
    for tier, feats in tier_features(include_income=True).items():
        assert (FORBIDDEN_SCREENING_COLUMNS & set(feats)) == {'income_bracket'}, tier
