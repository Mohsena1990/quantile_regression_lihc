"""Two/three-factor interpretation across countries, tied to identification accuracy.

Question: which policy factors (solid-fuel heating, dwelling age, elderly respondent, household
size; climate as a moderator) are associated with each axis of energy poverty in each country, do
those associations differ between countries, and does that difference show up as a real cost
in identification accuracy (transfer / pooling) -- i.e. can one model serve all countries?

Associational, not causal: survey cross-section, modest n, no instruments. Every effect is an
average marginal effect (percentage points) with a robust 95% CI; cells under the minimum support
are suppressed rather than shown with a misleading estimate. Development partitions only.

Outputs (tables in outputs/tables/, figures in outputs/figures/):
  table_factor_effects.csv               AME per country x axis x factor
  table_factor_heterogeneity.csv         does a factor's effect differ across countries? (LR test, sign agreement)
  table_factor_pairs.csv / _triples.csv  observed rates in 2x2 / 2x2x2 factor cells with Wilson CIs
  table_factor_accuracy_contribution.csv how much identification AUC each factor block buys, per country
  table_country_link.csv                 factor distinctiveness vs transfer gap (needs the benchmark table)
  factor_effects_forest / factor_pair_heatmaps / factor_accuracy_contribution / country_transfer_link
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES, load_development_frame, tier_features
from lib.identification.factors import FACTORS, FACTOR_LABELS, FEATURE_GROUPS, PAIRS, TRIPLE, add_factors
from lib.identification.labels import add_labels
from lib.identification.protocol import binary_axis_auc_cv
from lib.reporting.figures import save_figure

MIN_EXPOSURE_N = 15      # per exposure arm (exposed AND unexposed) to estimate an effect
MIN_CELL_N = 25          # per cell to show an observed rate
CONTROLS = ["H3_z", "hdd_z"]  # household size is itself a factor (large_hh); a linear size control would be collinear with it
AXES = {"li": "Low income", "hc": "Over-spends vs comparable homes"}
# categorical slots 1-5 of the reference palette, fixed order; marker shape = second, non-colour encoding
COUNTRY_STYLE = {"Bulgaria": ("#2a78d6", "o"), "Hungary": ("#eb6834", "s"), "Italy": ("#1baf7a", "^"),
                 "Poland": ("#eda100", "D")}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]


def _ax_style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def _ame(d: pd.DataFrame, axis: str, factors: list[str]):
    x = sm.add_constant(d[factors + CONTROLS].astype(float))
    fit = sm.Logit(d[axis].astype(int), x).fit(disp=0, cov_type="HC1", maxiter=300)
    fr = fit.get_margeff(at="overall", method="dydx").summary_frame()
    return fr.loc[factors].iloc[:, [0, 3, 4, 5]].to_numpy() * np.array([100, 1, 100, 100])  # ame_pp, p, lo_pp, hi_pp


def effects(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in COUNTRIES:
        g = lab[lab.country == c]
        for axis in AXES:
            ok = []
            for f in FACTORS:
                s = g[f].dropna()
                exposed = int((s == 1).sum()), int((s == 0).sum())
                if min(exposed) < MIN_EXPOSURE_N:
                    rows.append({"country": c, "axis": axis, "factor": f, "n_exposed": exposed[0], "n_unexposed": exposed[1],
                                 "ame_pp": np.nan, "ci_lo_pp": np.nan, "ci_hi_pp": np.nan, "p": np.nan, "note": "suppressed: arm n<15"})
                else:
                    ok.append(f)
            d = g.dropna(subset=ok + CONTROLS + [axis])
            try:
                est = _ame(d, axis, ok)
            except Exception as ex:  # separation / singular design
                est = np.full((len(ok), 4), np.nan)
                print(f"WARN {c} {axis}: {type(ex).__name__}")
            for f, (ame, p, lo, hi) in zip(ok, est):
                rows.append({"country": c, "axis": axis, "factor": f, "n_exposed": int((d[f] == 1).sum()),
                             "n_unexposed": int((d[f] == 0).sum()), "ame_pp": ame, "ci_lo_pp": lo, "ci_hi_pp": hi, "p": p, "note": ""})
    return pd.DataFrame(rows)


def heterogeneity(lab: pd.DataFrame, eff: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for axis, f in itertools.product(AXES, FACTORS):
        e = eff[(eff.axis == axis) & (eff.factor == f) & eff.ame_pp.notna()]
        cs = list(e.country)
        if len(cs) < 2:
            continue
        d = lab[lab.country.isin(cs)].dropna(subset=[f, axis] + CONTROLS)
        dummies = pd.get_dummies(d.country, drop_first=True).astype(float)
        base = pd.concat([d[[f] + CONTROLS].astype(float), dummies], axis=1)
        inter = dummies.mul(d[f].astype(float), axis=0).add_prefix(f"{f}_x_")
        y = d[axis].astype(int)
        m0 = sm.Logit(y, sm.add_constant(base)).fit(disp=0, maxiter=300)
        m1 = sm.Logit(y, sm.add_constant(pd.concat([base, inter], axis=1))).fit(disp=0, maxiter=300)
        lr, df = 2 * (m1.llf - m0.llf), inter.shape[1]
        signs = np.sign(e.ame_pp)
        rows.append({"axis": axis, "factor": f, "countries": ";".join(cs), "lr_stat": lr, "df": df, "p_heterogeneity": float(stats.chi2.sf(lr, df)),
                     "sign_agreement": float(max((signs > 0).mean(), (signs < 0).mean())),
                     "opposite_signs": bool((signs > 0).any() and (signs < 0).any())})
    return pd.DataFrame(rows)


def _wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    z, p = 1.96, k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def cells(lab: pd.DataFrame, factors: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    d = lab.assign(double=(lab.y == 3).astype(int))
    for c in COUNTRIES:
        g = d[d.country == c].dropna(subset=list(factors))
        for combo, sub in g.groupby(list(factors)):
            combo = combo if isinstance(combo, tuple) else (combo,)
            for outcome in ("li", "hc", "double"):
                k, n = int(sub[outcome].sum()), len(sub)
                lo, hi = _wilson(k, n)
                rows.append({"country": c, **{f: int(v) for f, v in zip(factors, combo)}, "outcome": outcome, "n": n,
                             "rate": k / n if n else np.nan, "ci_lo": lo, "ci_hi": hi, "suppressed": n < MIN_CELL_N})
    return pd.DataFrame(rows)


def accuracy_contribution(lab: pd.DataFrame) -> pd.DataFrame:
    full = tier_features()["T3"]
    rows = []
    for c in COUNTRIES:
        g = lab[lab.country == c]
        for axis in AXES:
            base = binary_axis_auc_cv(g, full, axis, per_fold=True)
            if not len(base):
                continue
            for name, cols in FEATURE_GROUPS.items():
                keep = [f for f in full if f not in cols]
                without = binary_axis_auc_cv(g, keep, axis, per_fold=True)
                diff = base - without
                rows.append({"country": c, "axis": axis, "group": name, "auc_full": base.mean(), "auc_without": without.mean(),
                             "delta_auc": diff.mean(), "se": diff.std(ddof=1) / np.sqrt(len(diff))})
    return pd.DataFrame(rows)


def country_link(eff: pd.DataFrame) -> pd.DataFrame | None:
    path = ROOT / "outputs/tables/table_identification_benchmark.csv"
    if not path.exists():
        print("SKIP link: run scripts/build_identification_benchmark.py first")
        return None
    b = pd.read_csv(path)
    b = b[(b.cost_kind == "need_adjusted") & (b.tier == "T2") & (b.model == "axis2")]
    rows = []
    for axis, col in (("li", "auc_income_axis"), ("hc", "auc_cost_axis")):
        for c in COUNTRIES:
            e = eff[(eff.axis == axis) & eff.ame_pp.notna()]
            mine = e[e.country == c].set_index("factor").ame_pp
            dist = []
            for f, v in mine.items():
                others = e[(e.factor == f) & (e.country != c)].ame_pp
                if len(others) >= 2:
                    dist.append(abs(v - others.mean()))
            get = lambda scope: float(b[(b.scope == scope) & (b.country == c)][col].iloc[0])
            rows.append({"country": c, "axis": axis, "distinctiveness_pp": float(np.mean(dist)) if dist else np.nan, "n_factors": len(dist),
                         "auc_local": get("local"), "auc_pooled": get("pooled"), "auc_loco": get("loco"),
                         "transfer_gap": get("local") - get("loco"), "pooling_gain": get("pooled") - get("local")})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_forest(eff: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (axis, title) in zip(axes, AXES.items()):
        _ax_style(ax)
        for j, f in enumerate(FACTORS):
            for i, c in enumerate(COUNTRIES):
                r = eff[(eff.axis == axis) & (eff.factor == f) & (eff.country == c)]
                if r.empty or np.isnan(r.ame_pp.iloc[0]):
                    continue
                r = r.iloc[0]
                y = j + (i - 2) * 0.14
                col, mk = COUNTRY_STYLE[c]
                ax.plot([r.ci_lo_pp, r.ci_hi_pp], [y, y], color=col, lw=1.6, zorder=2, solid_capstyle="round")
                ax.plot(r.ame_pp, y, mk, color=col, ms=6.5, mec=SURFACE, mew=1.2, zorder=3)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(len(FACTORS)), [FACTOR_LABELS[f] for f in FACTORS], color=INK2, fontsize=9)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Change in probability, percentage points (95% CI)", fontsize=8.5, color=INK2)
    axes[0].invert_yaxis()   # shared y: invert once so factors read top-to-bottom in FACTORS order
    handles = [plt.Line2D([], [], marker=m, color=col, ls="", ms=6.5, label=c) for c, (col, m) in COUNTRY_STYLE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Same factor, different meaning by country: associations with each energy-poverty axis\n"
                 "(logit average marginal effects, net of floor area and climate; suppressed where an arm has <15 households)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    save_figure(fig, ROOT / "outputs/figures/factor_effects_forest")
    plt.close(fig)


def fig_pairs(pairs: pd.DataFrame) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(12.5, 10.2))
    fig.patch.set_facecolor(SURFACE)
    for r, ((fa, fb), outcome) in enumerate(itertools.product(PAIRS, ("li", "hc"))):
        for k, c in enumerate(COUNTRIES):
            ax = axes[r, k]
            ax.set_facecolor(SURFACE)
            sub = pairs[(pairs.country == c) & (pairs.outcome == outcome) & (pairs.pair == f"{fa}|{fb}")]
            for _, row in sub.iterrows():
                x, y = int(row[fb]), int(row[fa])
                if row.suppressed:
                    ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc="#f0efec", ec=SURFACE, lw=2, hatch="///"))
                    ax.text(x, y, f"n={row.n}", ha="center", va="center", fontsize=7, color=MUTED)
                else:
                    shade = SEQ[min(int(row.rate / 0.8 * (len(SEQ) - 1) + 0.5), len(SEQ) - 1)]
                    ax.add_patch(plt.Rectangle((x - .5, y - .5), 1, 1, fc=shade, ec=SURFACE, lw=2))
                    dark = row.rate / 0.8 * (len(SEQ) - 1) > 2.5
                    ax.text(x, y, f"{row.rate * 100:.0f}%\n(n={row.n})", ha="center", va="center", fontsize=7.5,
                            color="#ffffff" if dark else INK)
            ax.set_xlim(-.5, 1.5); ax.set_ylim(1.5, -.5)
            ax.set_xticks([0, 1], [f"no", f"yes"], fontsize=7.5, color=MUTED)
            ax.set_yticks([0, 1], ["no", "yes"], fontsize=7.5, color=MUTED)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
            if r == 0:
                ax.set_title(c, fontsize=10, color=INK)
            if k == 0:
                ax.set_ylabel(f"{FACTOR_LABELS[fa]}\n[{AXES[outcome]}]", fontsize=8, color=INK2)
            ax.set_xlabel(FACTOR_LABELS[fb], fontsize=7.5, color=INK2)
    fig.suptitle("Two-factor cells: observed share of households that are low-income / over-spend (darker = higher, scale 0-80%)\n"
                 "hatched = fewer than 25 households, not interpretable", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, ROOT / "outputs/figures/factor_pair_heatmaps")
    plt.close(fig)


def fig_contribution(contrib: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    groups = list(FEATURE_GROUPS)
    for ax, (axis, title) in zip(axes, AXES.items()):
        m = contrib[contrib.axis == axis].pivot(index="group", columns="country", values="delta_auc").reindex(index=groups, columns=COUNTRIES)
        ax.set_facecolor(SURFACE)
        for i, gname in enumerate(groups):
            for j, c in enumerate(COUNTRIES):
                v = m.loc[gname, c]
                if np.isnan(v):
                    continue
                step = int(np.clip(v / 0.05, 0, 1) * (len(SEQ) - 1) + 0.5)
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc=SEQ[step] if v > 0 else "#f0efec", ec=SURFACE, lw=2))
                ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=7.5, color="#ffffff" if step >= 4 else INK)
        ax.set_xlim(-.5, len(COUNTRIES) - .5); ax.set_ylim(len(groups) - .5, -.5)
        ax.set_xticks(range(len(COUNTRIES)), COUNTRIES, fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(groups)), groups, fontsize=8.5, color=INK2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle("What each information block is worth for identification: drop in cross-validated AUC when the block is removed\n"
                 "(full model = all tiers; darker = the block matters more in that country)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_figure(fig, ROOT / "outputs/figures/factor_accuracy_contribution")
    plt.close(fig)


def fig_link(link: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    fig.patch.set_facecolor(SURFACE)
    for ax, (axis, title) in zip(axes, AXES.items()):
        _ax_style(ax)
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        d = link[(link.axis == axis)].dropna(subset=["distinctiveness_pp", "transfer_gap"])
        for _, r in d.iterrows():
            col, mk = COUNTRY_STYLE[r.country]
            ax.plot(r.distinctiveness_pp, r.transfer_gap, mk, color=col, ms=8, mec=SURFACE, mew=1.2)
            ax.annotate(r.country, (r.distinctiveness_pp, r.transfer_gap), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xlabel("Factor distinctiveness: mean |country effect - other countries' mean| (pp)", fontsize=8.5, color=INK2)
        ax.set_ylabel("Transfer gap: local AUC - leave-one-country-out AUC", fontsize=8.5, color=INK2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.suptitle("Do countries whose factor effects differ lose more when identified by a model trained elsewhere?\n"
                 "(5 countries per panel: descriptive, not a test)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    save_figure(fig, ROOT / "outputs/figures/country_transfer_link")
    plt.close(fig)


def main() -> None:
    dev = add_factors(load_development_frame())
    lab = add_factors(add_labels(dev, cost_kind="need_adjusted"))
    tab = ROOT / "outputs/tables"
    eff = effects(lab)
    eff.to_csv(tab / "table_factor_effects.csv", index=False)
    het = heterogeneity(lab, eff)
    het.to_csv(tab / "table_factor_heterogeneity.csv", index=False)
    pairs = pd.concat([cells(lab, p).assign(pair="|".join(p)) for p in PAIRS])
    pairs.to_csv(tab / "table_factor_pairs.csv", index=False)
    cells(lab, TRIPLE).to_csv(tab / "table_factor_triples.csv", index=False)
    contrib = accuracy_contribution(lab)
    contrib.to_csv(tab / "table_factor_accuracy_contribution.csv", index=False)
    link = country_link(eff)
    fig_forest(eff)
    fig_pairs(pairs)
    fig_contribution(contrib)
    if link is not None:
        link.to_csv(tab / "table_country_link.csv", index=False)
        fig_link(link)
        for axis in AXES:
            d = link[link.axis == axis].dropna(subset=["distinctiveness_pp", "transfer_gap"])
            if len(d) >= 4:
                rho, p = stats.spearmanr(d.distinctiveness_pp, d.transfer_gap)
                print(f"link[{axis}]: Spearman rho={rho:+.2f} (n={len(d)}, p={p:.2f})")
    pd.set_option("display.width", 220)
    print(het.round(3).to_string(index=False))
    print(eff[eff.ame_pp.notna()].assign(sig=lambda x: x.p < .05).pivot_table(index=["axis", "factor"], columns="country", values="ame_pp").round(1))


if __name__ == "__main__":
    main()
