"""Feature-config selection for the 3 screening-eligible M2/M5/M7 family champions,
evaluated across 6 scopes: each of the 5 countries independently, plus "Whole"
(the pooled 5-country frame). Country scopes feed the B-family (genuinely separate
per-country models, see scripts/build_b_measurements.py); the Whole scope answers
the pooled question and is where country-level economic context can add signal.

Three feature configs, combined with a measurement family into an `am_id` such as
"A1M2" (config A1, family M2) -- a fixed, staged social-science predictor set, the
same set offered in every country (no per-scope automatic selection: the removed
A3/A4 SHAP-top-12/LassoNet-top-12 configs each picked a different feature subset per
country, which is incompatible with interpreting one factor's effect consistently
across countries -- this project's stated goal):

  A0   household-only social-science/dwelling/demographic core
       (config/pipeline.json's `features`)
  A1   A0 + climate (`climate_features`)
  A2   A1 + country-level economic context (lib/features/country_context.py) --
       WHOLE SCOPE ONLY. These columns are constant within a country, so they
       carry zero information for a model trained and evaluated on one country
       alone; only the pooled fit sees them vary.

For each (family, scope, config): refit the measurement (conditional threshold,
hence the four-class label) using ONLY that config's features on that scope's own
train partition, then fit both classifiers on that same scope's train/validation
split -- a genuinely separate model per scope, not a pooled model sliced
afterwards. Cost-sensitive class weighting throughout (this project's single fixed
imbalance strategy, see lib/imbalance/strategies.py).

Outputs:
  artifacts/predictions/percountry/{am_id}_{scope}_{classifier}.parquet
  outputs/tables/table_percountry_feature_selection.csv  (one row per family x scope x config x classifier; scope="Whole" for the pooled fit)
  outputs/tables/table_percountry_selected_features.csv  (one row per family x scope x config: which features)
  outputs/tables/table_config_comparison.csv             (one row per family x scope x config x classifier, macro_f1)
  outputs/tables/table_percountry_winners.csv            (one row per family x country: winning (config, classifier) -- country scopes only, feeds the B-family)
  outputs/tables/table_whole_winners.csv                 (one row per family: winning (config, classifier), Whole scope only)
  outputs/tables/table_best_overall.csv                  (one row per scope (5 countries + Whole): single best family/config/classifier across all families)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.screening import fit_screening_model
from lib.features.country_context import CONTEXT_COLUMNS, add_country_context
from lib.features.household_features import stack_features
from measurement.families.model_based import fit_measurement_strategy

from data.enable import RETAINED_COUNTRY_NAMES
COUNTRIES = list(RETAINED_COUNTRY_NAMES)
SCOPES = COUNTRIES + ["Whole"]
CLASSIFIERS = ["catboost", "hierarchical_catboost"]
PRED_DIR = ROOT / "artifacts/predictions/percountry"


def build_configs(h_features: list[str], hc_features: list[str],
                   *, include_a2: bool) -> dict[str, list[str]]:
    """Fixed, staged social-science predictor set: A0 (household/dwelling/demographic
    core) -> A1 (+ climate) -> A2 (+ country economic context, Whole scope only). No
    per-scope automatic selection (the removed A3/A4 SHAP-top-12/LassoNet-top-12
    configs): those picked a different feature subset per country, which is
    incompatible with interpreting one factor's effect consistently across countries
    -- this project's stated goal (see lib/reporting/codes.py's CONFIG_DESCRIPTIONS)."""
    configs = {"A0": h_features, "A1": hc_features}
    if include_a2:
        configs["A2"] = stack_features(hc_features, CONTEXT_COLUMNS)
    return configs


def main() -> None:
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    h_features = list(cfg["features"])
    hc_features = stack_features(h_features, cfg["climate_features"])
    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions.csv")
    champions = champions[champions.status.eq("AVAILABLE")]

    metric_rows, feature_rows = [], []
    for champ in champions.itertuples():
        family, quantile = champ.family, champ.quantile
        if family == "M0":
            continue  # Van Hove has no per-household feature model to select over
        # Load the A1 (A0+climate) parquet for this family/quantile: it already carries
        # every column any config needs, regardless of which config the pooled champion
        # used for calibration selection. Join country context once here so every scope's
        # slice already carries it (constant per country).
        src = ROOT / f"artifacts/predictions/m{family[1:]}_a1.parquet"
        full = add_country_context(pd.read_parquet(src))
        for scope in SCOPES:
            d = full if scope == "Whole" else full[full.country.eq(scope)]
            scope_train_full = d[d.partition.eq("train")]
            if len(scope_train_full) < 30 or scope_train_full.risk_class_code.notna().sum() < 30:
                print(f"SKIP {family} {scope}: insufficient rows ({len(scope_train_full)})")
                continue
            configs = build_configs(h_features, hc_features, include_a2=(scope == "Whole"))
            for config_name, feature_list in configs.items():
                am_id = f"{config_name}{family}"
                feature_rows.append({"family": family, "country": scope, "config": config_name, "am_id": am_id,
                                      "n_features": len(feature_list), "features": ";".join(feature_list)})
                try:
                    _, measured = fit_measurement_strategy(family, scope_train_full, d, features=feature_list,
                                                             target=cfg["target"], quantile=quantile)
                except Exception as e:
                    print(f"MEASUREMENT_FAIL {family} {scope} {config_name}: {type(e).__name__}: {e}")
                    continue
                measured = measured[measured.measurement_eligible]
                m_train = measured[measured.partition.eq("train")]
                m_val = measured[measured.partition.eq("validation")]
                if len(m_val) < 10 or m_train.risk_class_code.nunique() < 2:
                    print(f"SKIP_SCREEN {family} {scope} {config_name}: insufficient eligible rows")
                    continue
                for kind in CLASSIFIERS:
                    try:
                        _, pred, metrics = fit_screening_model(kind, m_train, m_val, features=feature_list)
                        stem = f"{am_id}_{scope}_{kind}"
                        pred.to_parquet(PRED_DIR / f"{stem}.parquet", index=False)
                        metric_rows.append({"family": family, "country": scope, "config": config_name,
                                             "am_id": am_id, "classifier": kind,
                                             "status": "AVAILABLE", **metrics})
                    except Exception as e:
                        metric_rows.append({"family": family, "country": scope, "config": config_name,
                                             "am_id": am_id, "classifier": kind,
                                             "status": "BLOCKED", "reason": f"{type(e).__name__}: {e}"})
            print(f"DONE {family} {scope}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(ROOT / "outputs/tables/table_percountry_feature_selection.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(ROOT / "outputs/tables/table_percountry_selected_features.csv", index=False)

    available = metrics[metrics.status.eq("AVAILABLE")]
    if len(available):
        config_comparison = available.groupby(["country", "family", "config", "am_id", "classifier"],
                                                as_index=False).macro_f1.mean()
        config_comparison.to_csv(ROOT / "outputs/tables/table_config_comparison.csv", index=False)

        country_only = config_comparison[config_comparison.country.ne("Whole")]
        winners = (country_only.sort_values("macro_f1", ascending=False)
                                .groupby(["family", "country"], as_index=False).head(1)
                                .sort_values(["family", "country"]))
        winners.to_csv(ROOT / "outputs/tables/table_percountry_winners.csv", index=False)
        print(winners.to_string(index=False))

        whole_only = config_comparison[config_comparison.country.eq("Whole")]
        whole_winners = (whole_only.sort_values("macro_f1", ascending=False)
                                    .groupby(["family"], as_index=False).head(1)
                                    .sort_values(["family"]))
        whole_winners.to_csv(ROOT / "outputs/tables/table_whole_winners.csv", index=False)

        best_overall = (config_comparison.sort_values("macro_f1", ascending=False)
                                          .groupby(["country"], as_index=False).head(1))
        scope_order = {s: i for i, s in enumerate(SCOPES)}
        best_overall = best_overall.assign(_order=best_overall.country.map(scope_order)).sort_values("_order").drop(columns="_order")
        best_overall.to_csv(ROOT / "outputs/tables/table_best_overall.csv", index=False)
        print(best_overall.to_string(index=False))
    print(f"PERCOUNTRY_FEATURE_SELECTION_COMPLETE rows={len(metrics)}")


if __name__ == "__main__":
    main()
