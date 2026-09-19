import pytest
from measurement.registry import MeasurementSpec, core_registry

def test_registry_counts_and_unique_ids():
    assert len(core_registry()) == 7
    assert len({row.experiment_id for row in core_registry()}) == 7

def test_benchmark_axes_are_rejected():
    with pytest.raises(ValueError, match="benchmark"):
        MeasurementSpec("M0", "H")

def test_stable_experiment_id():
    assert MeasurementSpec("M7", "HC").experiment_id == "m7_hc"
