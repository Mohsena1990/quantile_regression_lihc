"""Canonical measurement specifications for the LIHC redesign.

Quantile is no longer a registry axis: for M1/M2/M3 it is a per-(family, feature) FIT
OUTPUT discovered by lib/tuning/quantile_search.py's continuous calibration search
(5-fold OOF, minimising absolute calibration error over q in [0.50, 0.80]), not a
design-time choice enumerated here. M0 (Van Hove, country Q80) keeps its quantile
fixed by its own external methodological definition and was never part of this
registry's quantile axis.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence

FAMILIES = {"M0": "Van Hove observed-expenditure country-Q80 benchmark", "M1": "Linear quantile regression", "M2": "GAM", "M3": "Gradient-boosted quantile regression"}
FEATURE_CONFIGS = ("H", "HC")
BENCHMARK_FAMILIES = frozenset({"M0"})

@dataclass(frozen=True, slots=True)
class FamilyStage:
    """Scientific availability of a measurement family and its downstream uses."""
    status: str
    required_cost: str
    four_class: str
    screening_eligible: bool
    blocker: str | None = None

FAMILY_STAGES: dict[str, FamilyStage] = {}

@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    family: str
    feature: str | None = None
    def __post_init__(self) -> None:
        if self.family not in FAMILIES:
            raise ValueError(f"Unknown measurement family: {self.family}")
        benchmark = self.family in BENCHMARK_FAMILIES
        if benchmark and self.feature is not None:
            raise ValueError(f"{self.family} is a benchmark and has no feature axis")
        if not benchmark and self.feature not in FEATURE_CONFIGS:
            raise ValueError(f"Unknown feature configuration: {self.feature}")
    @property
    def experiment_id(self) -> str:
        if self.family in BENCHMARK_FAMILIES:
            return self.family.lower()
        return f"{self.family.lower()}_{self.feature.lower()}"

RegistrySummary = MeasurementSpec

def build_registry(families: Iterable[str] = FAMILIES, features: Sequence[str] = FEATURE_CONFIGS) -> list[MeasurementSpec]:
    rows: list[MeasurementSpec] = []
    for family in families:
        if family in BENCHMARK_FAMILIES:
            rows.append(MeasurementSpec(family))
        else:
            rows.extend(MeasurementSpec(family, feature) for feature in features)
    ids = [row.experiment_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Registry contains duplicate experiment IDs")
    return rows

def core_registry() -> list[MeasurementSpec]:
    return build_registry()

def assert_expected_counts() -> None:
    assert len(core_registry()) == 7  # M0 + 3 families x {H, HC}

if __name__ == "__main__":
    assert_expected_counts()
    print("Registry counts OK (7 = M0 + 3 families x {H, HC}; "
          "quantile is now a per-family fit output, not a registry axis)")
