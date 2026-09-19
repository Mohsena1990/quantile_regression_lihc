import json
from pathlib import Path

import pandas as pd
import pytest

from lib.evaluation.split_registry import SplitAssignment, read_split_registry, write_split_registry
from scripts.build_expenditure_m0 import build_split_registry, deterministic_splits

ROOT = Path(__file__).resolve().parents[1]

def test_split_registry_round_trip(tmp_path):
    rows = [SplitAssignment("1", "train", 0), SplitAssignment("2", "test")]
    path = write_split_registry(rows, tmp_path / "split.json", metadata={"seed": 42, "strategy": "country-stratified"})
    metadata, restored = read_split_registry(path)
    assert metadata["seed"] == 42
    assert restored == rows

def test_duplicate_households_are_rejected(tmp_path):
    rows = [SplitAssignment("1", "train"), SplitAssignment("1", "test")]
    with pytest.raises(ValueError, match="exactly once"):
        write_split_registry(rows, tmp_path / "split.json", metadata={"seed": 1, "strategy": "test"})

def _synthetic_cohort(n=300, seed=7, uid_dtype="str"):
    import numpy as np
    rng = np.random.default_rng(seed)
    countries = rng.choice(["Bulgaria", "Hungary", "Italy"], size=n)
    uids = [f"UID{i:05d}" for i in range(n)] if uid_dtype == "str" else [1_000_000_000_000 + i for i in range(n)]
    return pd.DataFrame({
        "UID": uids,
        "country": countries,
        "NUTS1": [f"{c[:2].upper()}1" for c in countries],
    })

def test_deterministic_splits_are_uid_stable_under_row_removal():
    """A household's assigned partition must not depend on which other households are present.

    This is a regression test for the positional-permutation bug where
    np.random.default_rng(seed).permutation(len(g)) reassigned the majority of a
    country's households whenever the group size or row order changed upstream.
    """
    full = _synthetic_cohort(n=400)
    reduced = full.drop(index=range(50, 80)).reset_index(drop=True)  # drop 30 unrelated rows
    full_split = deterministic_splits(full).set_index("UID")["split"]
    reduced_split = deterministic_splits(reduced).set_index("UID")["split"]
    shared = reduced_split.index
    assert (full_split.loc[shared] == reduced_split.loc[shared]).all()

def test_deterministic_splits_stratified_ratio_is_reasonable():
    d = _synthetic_cohort(n=1000)
    split = deterministic_splits(d)
    shares = split["split"].value_counts(normalize=True)
    assert 0.5 < shares.get("train", 0) < 0.7
    assert 0.1 < shares.get("validation", 0) < 0.3
    assert 0.1 < shares.get("test", 0) < 0.3

def test_build_split_registry_reuses_existing_frozen_split(tmp_path):
    d = _synthetic_cohort(n=50)
    path = tmp_path / "split.json"
    first = build_split_registry(d, path, seed=1)
    raw_before = path.read_text()
    again = build_split_registry(d, path, seed=1)
    assert path.read_text() == raw_before  # never rewritten once frozen
    assert first.set_index("UID")["split"].equals(again.set_index("UID")["split"])

def test_build_split_registry_refuses_to_silently_regenerate_on_cohort_change(tmp_path):
    d = _synthetic_cohort(n=50)
    path = tmp_path / "split.json"
    build_split_registry(d, path, seed=1)
    grown = pd.concat([d, _synthetic_cohort(n=5, seed=99).assign(UID=lambda x: "NEW" + x.UID)], ignore_index=True)
    with pytest.raises(SystemExit, match="SPLIT_REGISTRY_IMMUTABLE"):
        build_split_registry(grown, path, seed=1)

def test_build_split_registry_preserves_uid_dtype_on_reuse(tmp_path):
    """Regression test: ENABLE's real UID column is int64 (e.g. 1000000000016), not str.
    On the 'reuse an existing frozen split' path, build_split_registry used to reconstruct
    UID from the JSON's string household_id without casting back, so
    d.merge(split, on=['UID', ...]) raised "You are trying to merge on int64 and object
    columns" the second time this script was run against an already-frozen split -- exactly
    the failure a user hit running `python scripts/build_expenditure_m0.py` twice."""
    d = _synthetic_cohort(n=30, uid_dtype="int")
    path = tmp_path / "split.json"
    build_split_registry(d, path, seed=1)  # first run: creates the frozen file
    reused = build_split_registry(d, path, seed=1)  # second run: must reuse it without a dtype mismatch
    assert reused["UID"].dtype == d["UID"].dtype
    merged = d.merge(reused, on=["UID", "country", "NUTS1"], validate="one_to_one")
    assert len(merged) == len(d)

def test_real_household_split_v1_has_unique_uids_and_reasonable_country_ratios():
    path = ROOT / "artifacts/splits/household_split_v1.json"
    if not path.exists():
        pytest.skip("household_split_v1.json not generated in this checkout")
    payload = json.loads(path.read_text())
    rows = payload["assignments"]
    uids = [r["household_id"] for r in rows]
    assert len(uids) == len(set(uids))
    by_country = {}
    for r in rows:
        by_country.setdefault(r["country"], []).append(r["partition"])
    for country, partitions in by_country.items():
        n = len(partitions)
        train = partitions.count("train") / n
        validation = partitions.count("validation") / n
        test = partitions.count("test") / n
        assert 0.5 < train < 0.7, country
        assert 0.1 < validation < 0.3, country
        assert 0.1 < test < 0.3, country
