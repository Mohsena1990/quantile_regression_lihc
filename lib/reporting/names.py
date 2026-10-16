"""Names used in every table and figure of the study (one place, so the study's own baseline names are used consistently).

The study is inspired by two works (README, "The two studies we build on"), but its streams and stages carry their own names:
  * the P80 line -- low income (bracket <= 3) x spending above the country's 80th percentile -- is the study's BASELINE line. It is
    the definition Van Hove et al. (2022) used; that is acknowledged in the text and checked in the validation stream
    (scripts/validate_van_hove_replica.py), not in the names.
  * the core profile (stage B0) -- the seven household variables of the baseline model (the predictor set of Van Hove et al.).
  * model sets "income only", "without income" and "full" -- the comparison Dalla Longa et al. (2021) called Models A, B and C.
Internal codes (CSV columns, dictionary keys) are short; the display names below are what readers see.
"""
from __future__ import annotations

LINES = {"p80": "P80 spending-percentile line (baseline)", "needqr": "Need line (quantile regression)",
         "needqr_noclim": "Need line without climate", "burden": "Energy-burden line", "pct": "Spending-percentile line (optimised)"}
LINES_SHORT = {"p80": "P80 line", "needqr": "Need line", "needqr_noclim": "Need line, no climate", "burden": "Burden line"}
STAGES = {"B0": "B0 core profile", "B1": "B1 + household & region", "B2": "B2 + climate", "B2p": "B2p placebo climate",
          "all": "income only"}
MODEL_SETS = {"income_only": "Income only", "no_income": "Without income", "full": "Full model", "variant": "Full-model variant"}
MODELS = {"catboost": "CatBoost", "xgboost": "XGBoost", "ft_transformer": "FT-Transformer", "ensemble": "Ensemble (3 models)",
          "xgboost_regression": "Bill model", "bill": "Bill model", "bill_mixed": "Multilevel bill model", "all": "Ensemble+",
          "ensemble_plus": "Ensemble+", "ensemble_mixed": "Ensemble+ multilevel"}
CORE_PROFILE = ["income_bracket", "H3", "household_size", "H1", "H2", "birth_year", "heating_strategy"]
CORE_PROFILE_NAMES = {"income_bracket": "income bracket", "H3": "floor area", "household_size": "household size",
                      "H1": "house detachment", "H2": "house age", "birth_year": "birth year respondent",
                      "heating_strategy": "heating strategy (H9)"}


def feature_names() -> dict:
    """Readable name of every candidate variable."""
    from lib.climate.b2_indices import B2_NAMES
    from lib.features.engineered import ENGINEERED
    from lib.features.indices import NAMES as INDEX_NAMES
    from lib.features.partial_modules import PARTIAL_NAMES
    from lib.features.survey_items import ITEM_NAMES
    return {**CORE_PROFILE_NAMES, "primary_heating_fuel": "main heating fuel", "has_solid_fuel_heating": "solid-fuel heating",
            "S7": "S7 energy-bill aid", "S8": "S8 income adequacy", "H11B": "dishwasher age", "country": "country",
            "region": "region (NUTS2)", **ITEM_NAMES, **{f: f.replace("_", " ") for f in ENGINEERED}, **B2_NAMES, **INDEX_NAMES,
            **PARTIAL_NAMES}
