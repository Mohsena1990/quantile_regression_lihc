"""Eligibility cohorts for native analyses and fair cross-family comparisons."""
from __future__ import annotations
from collections.abc import Mapping,Set

def method_native_uids(frame,eligible_col='measurement_eligible',uid_col='UID')->frozenset[str]:
    return frozenset(frame.loc[frame[eligible_col].fillna(False),uid_col].astype(str))
def common_comparison_uids(eligible_by_family: Mapping[str,Set[str]], required_families=tuple(f'M{i}' for i in range(9))) -> frozenset[str]:
    missing=set(required_families)-set(eligible_by_family)
    if missing: raise ValueError(f'Missing family eligibility sets: {sorted(missing)}')
    return frozenset.intersection(*(frozenset(eligible_by_family[f]) for f in required_families))
