"""Short symbolic codes for the paper-facing reporting layer.

The pipeline's internal names (M0/M2/M5/M7 measurement families, full classifier
names, the per-family quantile found by a continuous OOF calibration search rather
than a fixed grid, lib/tuning/quantile_search.py) are precise but verbose for
figures/tables meant for a paper. This module defines the short-code layer built on
top of them:

- R0-R3: one code per regression method. R0 is the M0 Van Hove benchmark; R1-R3 are
  M2/M5/M7 (the pruned, non-redundant linear/GAM/GBM-quantile trio -- see
  measurement/families/model_based/estimators.py's module docstring for why M3/M4/
  M6/M8 were removed rather than kept), each implicitly meaning "that family at its
  single locked quantile" -- see `regression_locked_quantile`, which reads the
  quantile already chosen by scripts/select_family_champions.py rather than
  refitting anything.
- C0-C1: the 2 screening classifiers (lib/screening/models.py; ARM-Net removed --
  it was statistically indistinguishable from CatBoost/Hierarchical CatBoost on
  macro-F1, worse-calibrated on log-loss, and offered no SHAP-native interpretation
  path).
- A0-A2: the 3 per-country/whole feature configs (scripts/run_percountry_feature_selection.py,
  lib/features/country_context.py) -- already short, included here only for the
  combo-id helper and the legend table. A3/A4 (automatic SHAP-top-12/LassoNet-top-12
  selection) were removed: they picked a different feature subset per country, which
  is incompatible with interpreting one factor's effect consistently across countries
  -- this project's stated goal.

Combo naming for the paper: `A{config}R{regression}C{classifier}`, e.g. "A0R2C1"
(config A0, regression method R2 = M5, classifier C1 = hierarchical_catboost).

Underlying `family`/`model` columns in existing tables are never renamed -- only
figure/table *labels* use these codes, so raw data stays traceable to the
internal names.
"""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

REGRESSION_FAMILY_ORDER = ["M0", "M2", "M5", "M7"]
REGRESSION_CODES = {f"R{i}": family for i, family in enumerate(REGRESSION_FAMILY_ORDER)}
FAMILY_TO_REGRESSION_CODE = {family: code for code, family in REGRESSION_CODES.items()}

CLASSIFIER_ORDER = ["catboost", "hierarchical_catboost"]  # matches build_classifier_comparison.py:MODEL_ORDER
CLASSIFIER_CODES = {f"C{i}": name for i, name in enumerate(CLASSIFIER_ORDER)}
CLASSIFIER_TO_CODE = {name: code for code, name in CLASSIFIER_CODES.items()}

CONFIG_CODES = ("A0", "A1", "A2")

REGRESSION_DESCRIPTIONS = {
    "M0": "Van Hove observed-expenditure country-Q80 benchmark",
    "M2": "Linear quantile regression",
    "M5": "GAM (spline + Gamma GLM)",
    "M7": "Gradient-boosted quantile regression",
}
CLASSIFIER_DESCRIPTIONS = {
    "catboost": "CatBoost (pooled multiclass gradient boosting)",
    "hierarchical_catboost": "Hierarchical CatBoost (global + country-expert shrinkage blend)",
}
CONFIG_DESCRIPTIONS = {
    "A0": "Household-only features (social-science / dwelling / demographic core)",
    "A1": "A0 + climate (HDD/CDD/temperature/frost days)",
    "A2": "A1 + country economic context (Whole scope only: price/income/rent/tenure)",
}


def regression_code(family: str) -> str:
    """M-family -> its R-code, e.g. "M4" -> "R3"."""
    if family not in FAMILY_TO_REGRESSION_CODE:
        raise ValueError(f"No R-code for family {family!r}")
    return FAMILY_TO_REGRESSION_CODE[family]


def regression_family(code: str) -> str:
    """R-code -> its M-family, e.g. "R3" -> "M4"."""
    if code not in REGRESSION_CODES:
        raise ValueError(f"Unknown regression code {code!r}")
    return REGRESSION_CODES[code]


def classifier_code(name: str) -> str:
    if name not in CLASSIFIER_TO_CODE:
        raise ValueError(f"Unknown classifier {name!r}")
    return CLASSIFIER_TO_CODE[name]


def classifier_name(code: str) -> str:
    if code not in CLASSIFIER_CODES:
        raise ValueError(f"Unknown classifier code {code!r}")
    return CLASSIFIER_CODES[code]


def combo_id(a_config: str, family: str, classifier: str) -> str:
    """(A-config, M-family, classifier name or C-code) -> "A0R3C1"."""
    if a_config not in CONFIG_CODES:
        raise ValueError(f"Unknown config {a_config!r}")
    c_code = classifier if classifier in CLASSIFIER_CODES else classifier_code(classifier)
    return f"{a_config}{regression_code(family)}{c_code}"


_COMBO_RE = re.compile(r"^(A[0-2])(R[0-3])(C[0-1])$")


def parse_combo_id(combo: str) -> dict[str, str]:
    """"A0R2C1" -> {"a_config": "A0", "regression_code": "R2", "family": "M5",
    "classifier_code": "C1", "classifier": "hierarchical_catboost"}."""
    m = _COMBO_RE.match(combo)
    if not m:
        raise ValueError(f"Malformed combo id {combo!r}, expected A{{0-2}}R{{0-3}}C{{0-1}}")
    a_config, r_code, c_code = m.groups()
    return {"a_config": a_config, "regression_code": r_code, "family": regression_family(r_code),
            "classifier_code": c_code, "classifier": classifier_name(c_code)}


def regression_locked_quantile(family: str, champions_path: Path | str | None = None) -> float | None:
    """The single quantile locked in for `family`'s R-code, per
    scripts/select_family_champions.py's OOF-calibration selection. Returns None for
    M0 (benchmark, no quantile axis). Does not refit anything -- reads the existing
    outputs/tables/table_family_champions.csv.
    """
    if family == "M0":
        return None
    path = Path(champions_path) if champions_path else ROOT / "outputs/tables/table_family_champions.csv"
    champions = pd.read_csv(path)
    row = champions.loc[champions.family.eq(family)]
    if row.empty:
        raise ValueError(f"No champion row for family {family!r} in {path}")
    quantile = row.iloc[0]["quantile"]
    return None if pd.isna(quantile) else float(quantile)
