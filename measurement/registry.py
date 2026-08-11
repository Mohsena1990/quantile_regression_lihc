"""Measurement registry and programmatic asserts for the redesign.

This module defines measurement families M0..M8, feature configurations H/HC,
and functions to assert the expected counts of configured measurement
combinations (core and sensitivity). Use this at CI time to ensure the
implementation matches the agreed experimental design.
"""
from dataclasses import dataclass
from typing import List


FAMILIES = {
    "M0": "Van Hove observed-expenditure country-Q80 benchmark",
    "M1": "historical_official_lihc_benchmark",
    "M2": "Linear Quantile Regression",
    "M3": "Gamma GLM",
    "M4": "Tweedie Regression",
    "M5": "GAM",
    "M6": "Gradient-Boosted Mean Regression",
    "M7": "Gradient-Boosted Quantile Regression",
    "M8": "Multilevel Gamma/Lognormal",
}

FEATURE_CONFIGS = ["H", "HC"]  # H: household-only, HC: household + climate

CORE_QUANTILES = [0.60, 0.65, 0.70]
SENSITIVITY_QUANTILES = [0.80]


@dataclass
class RegistrySummary:
    family: str
    feature: str
    quantile: float | None


def build_registry(families: List[str], features: List[str], quantiles: List[float]) -> List[RegistrySummary]:
    rows = []
    for f in families:
        if f in ("M0", "M1"):
            # Benchmarks: no quantile axis (M0 uses country 80th percentile, M1 is rule-based)
            rows.append(RegistrySummary(family=f, feature="-", quantile=None))
            continue
        for feat in features:
            for q in quantiles:
                rows.append(RegistrySummary(family=f, feature=feat, quantile=q))
    return rows


def assert_expected_counts(core_only: bool = True) -> None:
    """Assert the expected counts described in the project specification.

    Raises AssertionError if counts do not match.
    """
    model_families = [k for k in FAMILIES.keys() if k not in ("M0", "M1")]
    # core: 7 models x 2 features x 3 quantiles = 42; + 2 benchmarks = 44
    core_expected = len(model_families) * len(FEATURE_CONFIGS) * len(CORE_QUANTILES) + 2
    registry_core = build_registry(list(FAMILIES.keys()), FEATURE_CONFIGS, CORE_QUANTILES)
    assert len(registry_core) == core_expected, f"Core registry length {len(registry_core)} != expected {core_expected}"

    if not core_only:
        # sensitivity: add Q80 for the 7 models
        registry_sens = build_registry(list(FAMILIES.keys()), FEATURE_CONFIGS, CORE_QUANTILES + SENSITIVITY_QUANTILES)
        sens_expected = len(model_families) * len(FEATURE_CONFIGS) * (len(CORE_QUANTILES) + len(SENSITIVITY_QUANTILES)) + 2
        assert len(registry_sens) == sens_expected, f"Sensitivity registry length {len(registry_sens)} != expected {sens_expected}"


if __name__ == "__main__":
    # quick smoke check
    assert_expected_counts(core_only=True)
    print("Registry core counts OK")
