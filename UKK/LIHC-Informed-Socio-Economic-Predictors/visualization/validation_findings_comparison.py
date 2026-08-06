"""
Comparison figures for the validation suite's findings (see README's
Central Finding and Promising Alternative sections). Each figure compares
results ACROSS methods/specifications/countries, rather than reporting one
result in isolation -- that comparison is the point: whether HQRTM beats
traditional LIHC, whether a specification choice matters, whether a
country's rate should be read against its policy context.

Run from the repo root:
    python "UKK/LIHC-Informed-Socio-Economic-Predictors/visualization/validation_findings_comparison.py"

Reads the CSVs already produced by the validation suite -- run those
first if outputs/validation/ is empty (see README's Validation &
Robustness Suite section for the run order).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from viz_style import (
    METHOD_COLORS,
    CATEGORICAL,
    DIVERGING_POSITIVE,
    DIVERGING_NEGATIVE,
    DIVERGING_NEUTRAL,
    SURFACE,
    PRIMARY_INK,
    SECONDARY_INK,
    MUTED_INK,
    GRIDLINE,
    BASELINE,
    apply_chart_style,
    save_figure,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_DIR = REPO_ROOT / "outputs" / "validation"
OUTPUT_DIR = REPO_ROOT / "outputs" / "figures"

COUNTRY_NAMES = {1: "Bulgaria", 3: "Germany", 4: "Hungary", 5: "Italy", 8: "Serbia", 9: "Spain", 10: "Ukraine"}

MARKER_LABELS = {
    "energy_aid": "Received energy-bill aid (S7)",
    "bill_burden_bin": "Bill is a financial problem (C5B)",
    "cold_home_bin": "Cold home in winter (C1A)",
    "income_difficulty": "Difficulty on present income (S8)",
}
MARKER_ORDER = ["energy_aid", "bill_burden_bin", "cold_home_bin", "income_difficulty"]


# ---------------------------------------------------------------------
# Figure 1: headline method comparison -- odds ratios for LIHC, HQRTM,
# and the restriction flag, across all four hardship markers.
# ---------------------------------------------------------------------
def fig_method_comparison() -> None:
    hr = pd.read_csv(VALIDATION_DIR / "external_validity_results" / "horse_race_regressions.csv")
    hr_main = hr[hr["quantile_spec"] == "HQRTM (q=0.65)"]

    rf = pd.read_csv(VALIDATION_DIR / "restriction_flag_search" / "restriction_flag_search_results.csv")
    rf_main = rf[(rf["feature_set"] == "baseline") & (rf["quantile"] == 0.20)]

    rows = []
    for marker in MARKER_ORDER:
        lihc_row = hr_main[(hr_main["marker"] == marker) & (hr_main["flag"] == "lihc_high_cost")].iloc[0]
        hqrtm_row = hr_main[(hr_main["marker"] == marker) & (hr_main["flag"] == "hqrtm_high_cost")].iloc[0]
        rf_row = rf_main[rf_main["marker"] == marker].iloc[0]

        rows.append(("Traditional LIHC", marker, lihc_row["odds_ratio"], lihc_row["or_ci95_low"], lihc_row["or_ci95_high"], lihc_row["p_value"]))
        rows.append(("HQRTM", marker, hqrtm_row["odds_ratio"], hqrtm_row["or_ci95_low"], hqrtm_row["or_ci95_high"], hqrtm_row["p_value"]))
        rows.append(("Restriction flag", marker, rf_row["solo_odds_ratio"], rf_row["solo_or_ci95_low"], rf_row["solo_or_ci95_high"], rf_row["solo_p_value"]))

    df = pd.DataFrame(rows, columns=["method", "marker", "or_", "ci_low", "ci_high", "p"])

    methods = ["Traditional LIHC", "HQRTM", "Restriction flag"]
    n_markers = len(MARKER_ORDER)
    group_height = len(methods) + 1.4
    fig, ax = plt.subplots(figsize=(8.5, 1.15 * n_markers * group_height / 1.6))

    y_positions = {}
    y = 0
    for marker in reversed(MARKER_ORDER):
        for method in methods:
            y_positions[(marker, method)] = y
            y += 1
        y += 1.4

    for marker in MARKER_ORDER:
        for method in methods:
            row = df[(df["marker"] == marker) & (df["method"] == method)].iloc[0]
            y_pos = y_positions[(marker, method)]
            color = METHOD_COLORS[method]
            significant = row["p"] < 0.05

            ax.plot([row["ci_low"], row["ci_high"]], [y_pos, y_pos], color=color, linewidth=2, zorder=3, solid_capstyle="round")
            ax.scatter(
                [row["or_"]], [y_pos],
                s=70 if significant else 46,
                color=color,
                edgecolor=PRIMARY_INK if significant else color,
                linewidth=1.1 if significant else 0,
                zorder=4,
            )

    ax.axvline(1.0, color=BASELINE, linewidth=1.2, linestyle="--", zorder=1)

    marker_centers = {
        marker: np.mean([y_positions[(marker, m)] for m in methods])
        for marker in MARKER_ORDER
    }
    for marker in MARKER_ORDER:
        ax.text(
            -0.02, marker_centers[marker], MARKER_LABELS[marker],
            transform=ax.get_yaxis_transform(), ha="right", va="center",
            fontsize=10.5, color=PRIMARY_INK, fontweight="bold",
        )

    ax.set_yticks([])
    ax.set_ylim(-1, y)
    ax.set_xlabel("Odds ratio (95% CI) -- \">1\" means flagged households more likely to report hardship")
    ax.set_xlim(0.35, 2.3)

    handles = [
        plt.Line2D([0], [0], marker="o", color=METHOD_COLORS[m], linestyle="", markersize=8, label=m)
        for m in methods
    ]
    legend = ax.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0), frameon=True,
        fontsize=10, title="Method (larger, outlined marker = p<0.05)", title_fontsize=9.5,
    )
    legend.get_frame().set_edgecolor(GRIDLINE)
    legend.get_frame().set_facecolor(SURFACE)

    ax.set_title(
        "Does the \"high cost\" flag track independent hardship markers?\nTraditional LIHC and HQRTM mostly don't; the restriction flag (low cost vs. structural need) does",
        fontsize=12.5, pad=14, loc="left",
    )
    apply_chart_style(fig, ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "method_comparison_odds_ratios")


# ---------------------------------------------------------------------
# Figure 2: restriction flag odds ratio across the low-quantile grid,
# small multiples per marker.
# ---------------------------------------------------------------------
def fig_restriction_quantile_sweep() -> None:
    rf = pd.read_csv(VALIDATION_DIR / "restriction_flag_search" / "restriction_flag_search_results.csv")
    rf = rf[rf["feature_set"] == "baseline"].sort_values("quantile")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
    color = METHOD_COLORS["Restriction flag"]

    for ax, marker in zip(axes, MARKER_ORDER):
        sub = rf[rf["marker"] == marker]
        significant = sub["solo_p_value"] < 0.05

        ax.fill_between(sub["quantile"], sub["solo_or_ci95_low"], sub["solo_or_ci95_high"], color=color, alpha=0.15, zorder=2)
        ax.plot(sub["quantile"], sub["solo_odds_ratio"], color=color, linewidth=2, zorder=3)
        ax.scatter(
            sub.loc[significant, "quantile"], sub.loc[significant, "solo_odds_ratio"],
            color=color, s=45, zorder=4, edgecolor=PRIMARY_INK, linewidth=1,
        )
        ax.scatter(
            sub.loc[~significant, "quantile"], sub.loc[~significant, "solo_odds_ratio"],
            facecolor=SURFACE, edgecolor=color, s=45, zorder=4, linewidth=1.4,
        )
        ax.axhline(1.0, color=BASELINE, linewidth=1, linestyle="--", zorder=1)
        ax.set_title(MARKER_LABELS[marker], fontsize=10, color=PRIMARY_INK)
        ax.set_xlabel("Low quantile flagged", fontsize=9.5)
        ax.set_xticks([0.10, 0.20, 0.30, 0.40])

    axes[0].set_ylabel("Odds ratio (solo model)")
    apply_chart_style(fig, axes)
    fig.suptitle(
        "Restriction-flag odds ratio across quantile cutoffs -- filled dot = p<0.05\n"
        "energy_aid is strong everywhere; cold_home needs the most extreme cutoff (q=0.10)",
        fontsize=12, y=1.08,
    )
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "restriction_flag_quantile_sweep")


# ---------------------------------------------------------------------
# Figure 3: HQRTM specification search -- odds ratio heatmap across
# quantile x margin_scale, for income_difficulty (the marker with the
# clearest, most consistent signal in the search).
# ---------------------------------------------------------------------
def fig_hqrtm_search_heatmap() -> None:
    hs = pd.read_csv(VALIDATION_DIR / "hqrtm_specification_search" / "specification_search_results.csv")
    sub = hs[hs["marker"] == "income_difficulty"]

    quantiles = sorted(sub["quantile"].unique())
    margins = sorted(sub["margin_scale"].unique())
    grid = sub.pivot(index="quantile", columns="margin_scale", values="odds_ratio").loc[quantiles, margins]
    p_grid = sub.pivot(index="quantile", columns="margin_scale", values="p_value").loc[quantiles, margins]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    vmax = max(abs(grid.values.max() - 1), abs(1 - grid.values.min())) or 0.1
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("diverging", [DIVERGING_NEGATIVE, DIVERGING_NEUTRAL, DIVERGING_POSITIVE])
    im = ax.imshow(grid.values, cmap=cmap, vmin=1 - vmax, vmax=1 + vmax, aspect="auto")

    for i, q in enumerate(quantiles):
        for j, m in enumerate(margins):
            or_val = grid.loc[q, m]
            p_val = p_grid.loc[q, m]
            marker = "*" if p_val < 0.05 else ""
            text_color = PRIMARY_INK if 0.85 < or_val < 1.15 else "#ffffff"
            ax.text(j, i, f"{or_val:.2f}{marker}", ha="center", va="center", fontsize=9.5, color=text_color)

    ax.set_xticks(range(len(margins)))
    ax.set_xticklabels([f"{m:.2f}" for m in margins])
    ax.set_yticks(range(len(quantiles)))
    ax.set_yticklabels([f"{q:.2f}" for q in quantiles])
    ax.set_xlabel("margin_scale")
    ax.set_ylabel("quantile")
    ax.set_title(
        "HQRTM odds ratio for income_difficulty, across every tested spec\n"
        "* = p<0.05. Every cell is <=1 -- wrong direction everywhere.",
        fontsize=11.5, loc="left",
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label("Odds ratio", color=SECONDARY_INK, fontsize=9.5)
    cbar.ax.tick_params(colors=MUTED_INK, labelsize=8.5)

    apply_chart_style(fig, ax)
    ax.grid(visible=False)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "hqrtm_specification_search_heatmap")


# ---------------------------------------------------------------------
# Figure 4: specification-stability range by country -- min/max Double
# risk prevalence across quantile x feature-set x pooling-method specs.
# ---------------------------------------------------------------------
def fig_specification_stability_range() -> None:
    band = pd.read_csv(VALIDATION_DIR / "specification_stability_results" / "specification_stability_by_country.csv")
    band["Country"] = band["Country"].map(COUNTRY_NAMES).fillna(band["Country"].astype(str))
    band = band.sort_values("range_pp")

    fig, ax = plt.subplots(figsize=(7.5, 0.5 * len(band) + 1.5))
    color = CATEGORICAL[0]

    for i, (_, row) in enumerate(band.iterrows()):
        ax.plot([row["min"], row["max"]], [i, i], color=color, linewidth=4, alpha=0.35, zorder=2, solid_capstyle="round")
        ax.scatter([row["min"]], [i], color=color, s=55, zorder=3)
        ax.scatter([row["max"]], [i], color=color, s=55, zorder=3)
        ax.text(
            row["max"] + band["max"].max() * 0.03, i, f"{row['range_pp']:.1f} pp",
            va="center", fontsize=9, color=SECONDARY_INK,
        )

    ax.set_yticks(range(len(band)))
    ax.set_yticklabels(band["Country"])
    ax.set_xlabel("Double-risk prevalence (%) across all tested specifications")
    ax.set_title(
        "How much does the specification choice change a country's Double-risk rate?\n"
        "Dots = min/max across quantile x feature-set x pooling-method combinations",
        fontsize=11.5, loc="left",
    )
    apply_chart_style(fig, ax)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "specification_stability_range_by_country")


# ---------------------------------------------------------------------
# Figure 5: country policy context -- energy-aid receipt rate vs.
# Double-risk rate, one point per country.
# ---------------------------------------------------------------------
def fig_country_policy_context() -> None:
    ctx = pd.read_csv(VALIDATION_DIR / "country_policy_context" / "country_policy_context.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    color = CATEGORICAL[0]

    ax.errorbar(
        ctx["energy_aid_received_pct"], ctx["hqrtm_double_risk_pct"],
        xerr=[
            ctx["energy_aid_received_pct"] - ctx["energy_aid_ci95_low_pct"],
            ctx["energy_aid_ci95_high_pct"] - ctx["energy_aid_received_pct"],
        ],
        fmt="o", color=color, ecolor=color, elinewidth=1.4, capsize=3, markersize=9, zorder=3,
    )

    # Italy/Bulgaria/Serbia cluster tightly (aid rate 1-3%, Double-risk 13.5-14%)
    # -- default offsets collide there, so stagger those three explicitly.
    label_offsets = {
        "Italy": (-8, 14),
        "Bulgaria": (10, 10),
        "Serbia": (10, -16),
    }
    for _, row in ctx.iterrows():
        dx, dy = label_offsets.get(row["Country"], (8, 4))
        ax.annotate(
            row["Country"], (row["energy_aid_received_pct"], row["hqrtm_double_risk_pct"]),
            xytext=(dx, dy), textcoords="offset points", fontsize=10, color=PRIMARY_INK, fontweight="bold",
        )

    ax.set_xlabel("Households that received energy-bill aid / social tariff (%, 95% CI)")
    ax.set_ylabel("HQRTM Double-risk rate (%)")
    ax.set_title(
        "Country policy context: aid receipt vs. Double-risk rate\n"
        "Ukraine's far higher aid coverage reflects existing policy reach, not a different risk level",
        fontsize=11.5, loc="left",
    )
    ax.set_xlim(-3, max(60, ctx["energy_aid_ci95_high_pct"].max() + 5))
    apply_chart_style(fig, ax)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "country_policy_context_scatter")


if __name__ == "__main__":
    fig_method_comparison()
    fig_restriction_quantile_sweep()
    fig_hqrtm_search_heatmap()
    fig_specification_stability_range()
    fig_country_policy_context()
