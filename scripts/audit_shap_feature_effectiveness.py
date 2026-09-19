"""SHAP feature-effectiveness audit across every A0-A2 config -- which features are
actually pulling their weight against the four-class target, per country. This is
the project's central interpretation tool: with a fixed predictor set offered
identically in every country (A0 social-science core -> A1 + climate -> A2 +
country economic context; the old per-country-selected A3/A4 configs were removed
specifically because they broke this comparability), a feature's SHAP share is
directly comparable across countries.

Diagnostic only: this script flags low-importance features, it does not remove
them. Dropping a flagged feature from the fixed A0/A1 lists (config/pipeline.json)
would ripple through the frozen registry and require re-running downstream
screening/B-family stages, so that is a deliberate follow-up gated on reviewing
this table with the user, not something this script does automatically.

Reuses lib.tuning.feature_selection.rank_features_shap (one CatBoost+SHAP fit per
(family, scope)) and the feature lists scripts/run_percountry_feature_selection.py
already selected (outputs/tables/table_percountry_selected_features.csv) -- no
re-running the per-country classifier matrix.

Outputs:
  outputs/tables/table_shap_feature_effectiveness.csv   (family x scope x config x feature x importance share x flagged)
  outputs/figures/shap_feature_effectiveness_<scope>.png/.pdf  (one bar chart per scope,
    mean importance share per feature averaged across the 3 M2/M5/M7 families)
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.reporting.figures import save_figure
from lib.tuning.feature_selection import rank_features_shap
from lib.features.country_context import CONTEXT_COLUMNS, add_country_context
from lib.features.household_features import stack_features

from data.enable import RETAINED_COUNTRY_NAMES
COUNTRIES = list(RETAINED_COUNTRY_NAMES)
SCOPES = COUNTRIES + ["Whole"]
CONFIGS = ["A0", "A1", "A2"]
MATERIALITY_THRESHOLD = 0.01  # a feature's share of the full context-pool |SHAP| mass below
                               # this is flagged as low-effectiveness (diagnostic, not enforced)
SURFACE, GRID, INK_MUTED, INK = "#fcfcfb", "#e1e0d9", "#898781", "#0b0b0b"
BAR_COLOR, WEAK_COLOR = "#2a78d6", "#d03b3b"  # sequential magnitude hue; status "serious" for flagged bars


def scope_frame(full: pd.DataFrame, scope: str) -> pd.DataFrame:
    d = full if scope == "Whole" else full[full.country.eq(scope)]
    return d[d.partition.eq("train")]


def main() -> None:
    cfg = json.loads((ROOT / "config/pipeline.json").read_text())
    h_features = list(cfg["features"])
    hc_features = stack_features(h_features, cfg["climate_features"])
    context_pool = stack_features(hc_features, CONTEXT_COLUMNS)

    champions = pd.read_csv(ROOT / "outputs/tables/table_family_champions.csv")
    champions = champions[champions.status.eq("AVAILABLE") & champions.family.ne("M0")]
    selected = pd.read_csv(ROOT / "outputs/tables/table_percountry_selected_features.csv")

    rows: list[dict] = []
    pool_shares: dict[tuple[str, str], pd.Series] = {}  # (family, scope) -> full context-pool share
    for champ in champions.itertuples():
        family = champ.family
        src = ROOT / f"artifacts/predictions/m{family[1:]}_a1.parquet"
        if not src.exists():
            print(f"SKIP {family}: {src.name} not found")
            continue
        full = add_country_context(pd.read_parquet(src))
        for scope in SCOPES:
            train = scope_frame(full, scope)
            if len(train) < 30 or train.risk_class_code.notna().sum() < 30:
                continue
            ranked = rank_features_shap(train, context_pool)
            total = ranked.sum()
            share = ranked / total if total > 0 else ranked  # all-zero stays all-zero, not NaN
            pool_shares[(family, scope)] = share

            configs_here = selected[(selected.family.eq(family)) & (selected.country.eq(scope))]
            for config_row in configs_here.itertuples():
                if config_row.config not in CONFIGS:
                    continue
                feature_list = config_row.features.split(";") if config_row.features else []
                for feature in feature_list:
                    feature_share = float(share.get(feature, 0.0))
                    rows.append({"family": family, "scope": scope, "config": config_row.config,
                                 "feature": feature, "importance_share": feature_share,
                                 "flagged_low_effectiveness": feature_share < MATERIALITY_THRESHOLD})
        print(f"DONE {family}", flush=True)

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "outputs/tables/table_shap_feature_effectiveness.csv", index=False)

    # Per-scope figure: mean importance share per feature, averaged across the 3 families,
    # off the full context pool (not config-specific) -- one consistent ruler per scope.
    n_figures = 0
    for scope in SCOPES:
        series_by_family = [s for (fam, sc), s in pool_shares.items() if sc == scope]
        if not series_by_family:
            continue
        mean_share = pd.concat(series_by_family, axis=1).mean(axis=1).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(9, 0.28 * len(mean_share) + 1.2))
        colors = [WEAK_COLOR if v < MATERIALITY_THRESHOLD else BAR_COLOR for v in mean_share]
        ax.barh(mean_share.index[::-1], mean_share.to_numpy()[::-1] * 100, color=colors[::-1], height=0.65)
        ax.axvline(MATERIALITY_THRESHOLD * 100, color=INK_MUTED, linewidth=1, linestyle="--", zorder=0)
        ax.set_xlabel("Mean share of total |SHAP| importance (%), averaged across R1-R3", fontsize=8.5)
        ax.set_title(f"SHAP feature-effectiveness audit -- {scope}\n"
                     f"(red = below {MATERIALITY_THRESHOLD*100:.0f}% materiality threshold)", fontsize=10)
        ax.tick_params(labelsize=7.5, colors=INK_MUTED)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(INK_MUTED)
        fig.tight_layout()
        save_figure(fig, ROOT / "outputs/figures" / f"shap_feature_effectiveness_{scope}")
        plt.close(fig)
        n_figures += 1

    n_flagged = int(table.flagged_low_effectiveness.sum()) if len(table) else 0
    print(f"Wrote {len(table)} feature-effectiveness rows ({n_flagged} flagged <{MATERIALITY_THRESHOLD:.0%}) -> "
          f"outputs/tables/table_shap_feature_effectiveness.csv; {n_figures} figures -> "
          f"outputs/figures/shap_feature_effectiveness_*.png/.pdf")


if __name__ == "__main__":
    main()
