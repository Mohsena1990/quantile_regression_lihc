import itertools
import pytest
from lib.reporting.codes import (
    CONFIG_CODES, REGRESSION_CODES, CLASSIFIER_CODES,
    combo_id, parse_combo_id, regression_code, regression_family,
    classifier_code, classifier_name,
)


def test_regression_codes_cover_m0_m2_m5_m7():
    assert REGRESSION_CODES == {"R0": "M0", "R1": "M2", "R2": "M5", "R3": "M7"}


def test_unknown_family_has_no_regression_code():
    with pytest.raises(ValueError, match="M9"):
        regression_code("M9")


def test_regression_code_round_trip():
    for code, family in REGRESSION_CODES.items():
        assert regression_code(family) == code
        assert regression_family(code) == family


def test_classifier_code_round_trip():
    for code, name in CLASSIFIER_CODES.items():
        assert classifier_code(name) == code
        assert classifier_name(code) == name


def test_combo_id_all_well_formed():
    seen = set()
    for a_config, family, classifier in itertools.product(CONFIG_CODES, REGRESSION_CODES.values(), CLASSIFIER_CODES.values()):
        combo = combo_id(a_config, family, classifier)
        parsed = parse_combo_id(combo)
        assert parsed["a_config"] == a_config
        assert parsed["family"] == family
        assert parsed["classifier"] == classifier
        seen.add(combo)
    assert len(seen) == 3 * 4 * 2


def test_combo_id_example():
    assert combo_id("A0", "M5", "hierarchical_catboost") == "A0R2C1"


def test_parse_combo_id_rejects_malformed():
    with pytest.raises(ValueError):
        parse_combo_id("A0M5C1")
