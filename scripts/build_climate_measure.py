"""Climate and the measure itself: who is reclassified when degree days enter the need line, and how much each expenditure line
drifts with the weather.

(1) Reclassification. The need line with and without degree days (same income cut-off and tau) label the same households;
    the 4x4 transition table and, per climate region, the net movement into high expenditure (into minus out of high
    expenditure, share of the region's households) against the region's winter severity relative to its country
    (log HDD minus the country mean).
(2) Weather drift. Each household's heating bill is rescaled to a different winter with the within-country heating-bill
    elasticity of the multilevel model (table_climate_elasticity.csv, five countries): heating' = heating x (HDD'/HDD)^beta;
    electricity is unchanged. Scenarios:
      milder   HDD x 0.9 in every region (a uniformly 10% milder winter)
      colder   HDD x 1.1 in every region
      regional each region's 2016/17 anomaly against its 1991-2020 normal is removed relative to its country's mean anomaly
               (HDD' = HDD x exp(-(a_r - mean_c a)), a_r = log HDD_2016/17 / HDD_normal): the same national winter, without
               the regional pattern of 2016/17. Only the within-country pattern is used because the normals come from
               monthly means of another reanalysis product, so their level is not comparable with the survey-year HDD.
    Expenditure rules, applied to the rescaled bills (income axis unchanged):
      P80 fixed in EUR       the 2016/17 country 80th percentile, kept in euros
      P80 recomputed         the country 80th percentile of the rescaled bills
      need, no degree days   the fitted climate-free need line (coefficients fixed; its threshold cannot move)
      need with degree days  the fitted need line (coefficients fixed) evaluated at the new HDD
    Reported: share of households whose group changes and share of the rule's Double-Risk group that changes, per country and
    pooled.

Outputs: outputs/tables/table_label_climate_regions.csv, table_weather_drift.csv;
         outputs/figures/label_reclassification, weather_drift
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from lib.identification.data import load_development_frame
from lib.identification.lines import CLASS_NAMES, NEED_CLIM, NEED_NOCLIM, eligible, line_specs, make_labels, quantile_line
from lib.reporting.figures import save_figure

TABLES, FIGS = ROOT / "outputs/tables", ROOT / "outputs/figures"
BLUE, ORANGE, AQUA, YELLOW, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#b5b3ad", "#2b2a27", "#6b6a66"
COUNTRY_ORDER = ["Bulgaria", "Hungary", "Italy", "Poland", "Spain"]
GROUPS = [CLASS_NAMES[k] for k in range(4)]
COUNTRY_MARKERS = {"Bulgaria": "o", "Hungary": "s", "Italy": "^", "Poland": "D", "Spain": "v"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})


MIN_REGION_HH = 10   # regions shown in the reclassification scatter (smaller ones swing by +-100% on one household)


def weighted_corr(g: pd.DataFrame) -> float:
    """Household-weighted correlation of relative log HDD and net move into high expenditure across regions."""
    w, x, y = g.households.to_numpy(float), g.rel_log_hdd.to_numpy(), g.net_into_high_cost.to_numpy()
    cov = np.cov(x, y, aweights=w)
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def rel_log_hdd(e: pd.DataFrame) -> pd.Series:
    lh = np.log(e.hdd_hs1617)
    return lh - lh.groupby(e.country).transform("mean")


# ------------------------------------------------------------------------------------------------ (1) reclassification
def reclassification(dev: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = line_specs()["needqr"]
    a = make_labels(dev, "needqr", s["cutoff"], s["tau"])
    b = make_labels(dev, "needqr_noclim", s["cutoff"], s["tau"])
    a["y0"], a["hc0"], a["rel"] = b.y.to_numpy(), b.hc.to_numpy(), rel_log_hdd(a)
    trans = pd.crosstab(a.y0.map(CLASS_NAMES), a.y.map(CLASS_NAMES)).reindex(index=GROUPS, columns=GROUPS, fill_value=0)
    a["into"] = ((a.hc == 1) & (a.hc0 == 0)).astype(int)
    a["out"] = ((a.hc == 0) & (a.hc0 == 1)).astype(int)
    reg = (a.groupby(["country", "climate_region"])
             .agg(households=("y", "size"), into=("into", "mean"), out=("out", "mean"), rel_log_hdd=("rel", "mean"))
             .reset_index())
    reg["net_into_high_cost"] = reg.into - reg.out
    return trans, reg


def plot_reclassification(trans: pd.DataFrame, reg: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1, 1.25], "wspace": 0.45})
    m = trans.to_numpy()
    off = np.where(np.eye(4, dtype=bool), np.nan, m)
    ax1.imshow(np.where(np.eye(4, dtype=bool), 0, off), cmap="Oranges", vmin=0, vmax=max(np.nanmax(off) * 1.3, 1))
    for i in range(4):
        for j in range(4):
            diag = i == j
            if diag or m[i, j]:
                ax1.text(j, i, f"{m[i, j]:,}", ha="center", va="center", fontsize=8, color=MUTED if diag else INK,
                         fontweight="normal" if diag else "bold")
    short = ["No", "Income", "Expend.", "Double"]
    ax1.set_xticks(range(4), short); ax1.set_yticks(range(4), short)
    ax1.set_xlabel("Group, need line with degree days"); ax1.set_ylabel("Group, need line without degree days")
    ax1.tick_params(length=0); [s.set_visible(False) for s in ax1.spines.values()]
    ax1.set_title("a", loc="left", fontweight="bold", color=INK)
    big = reg[reg.households >= MIN_REGION_HH]
    ax2.axhline(0, color=GREY, lw=0.8); ax2.axvline(0, color=GREY, lw=0.8)
    for c, col in (("Italy", BLUE), ("Spain", ORANGE)):
        g = big[big.country == c]
        ax2.scatter(g.rel_log_hdd, g.net_into_high_cost * 100, s=g.households * 1.2, color=col, edgecolor="white", lw=0.6,
                    label=f"{c} (r = {weighted_corr(g):+.2f})", zorder=3)
        k = np.polyfit(g.rel_log_hdd, g.net_into_high_cost * 100, 1, w=np.sqrt(g.households))
        xs = np.linspace(g.rel_log_hdd.min(), g.rel_log_hdd.max(), 20)
        ax2.plot(xs, np.polyval(k, xs), color=col, lw=1.6, zorder=2)
    ax2.set_xlabel("Region's log HDD minus country mean")
    ax2.set_ylabel("Net move into high expenditure\n(% of region's households)")
    leg = ax2.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, handletextpad=0.2)
    for h in leg.legend_handles:
        h.set_sizes([30])
    ax2.set_title("b", loc="left", fontweight="bold", color=INK)
    save_figure(fig, FIGS / "label_reclassification")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ (2) weather drift
def need_thresholds(e: pd.DataFrame, family: str, tau: float, hdd_new: pd.Series, folds: int = 5, seed: int = 7):
    """Out-of-fold need-line thresholds at the observed HDD and, with the same fitted lines, at `hdd_new`.
    Same folds as lines.expenditure_threshold, so the first output reproduces the study labels."""
    feats = NEED_CLIM if family == "needqr" else NEED_NOCLIM
    t0, t1 = pd.Series(np.nan, index=e.index), pd.Series(np.nan, index=e.index)
    for _, g in e.groupby("country"):
        g = g.reset_index()
        g1 = g.copy(); g1["hdd_hs1617"] = hdd_new.loc[g["index"]].to_numpy()
        a, b = np.full(len(g), np.nan), np.full(len(g), np.nan)
        for tr, te in KFold(folds, shuffle=True, random_state=seed).split(g):
            a[te] = quantile_line(g.iloc[tr], g.iloc[te], feats, tau)
            b[te] = quantile_line(g.iloc[tr], g1.iloc[te], feats, tau) if family == "needqr" else a[te]
        t0[g["index"].to_numpy()], t1[g["index"].to_numpy()] = a, b
    return t0, t1


def weather_drift(dev: pd.DataFrame, beta: float) -> pd.DataFrame:
    s = line_specs()["needqr"]
    e = eligible(dev, "needqr")
    heat = e.heating_cost_eur.clip(lower=0).fillna(0).clip(upper=e.cost)
    a = np.log(e.hdd_hs1617 / e.hdd_normal)
    scen = {"10% milder winter": e.hdd_hs1617 * 0.9, "10% colder winter": e.hdd_hs1617 * 1.1,
            "2016/17 regional pattern removed": e.hdd_hs1617 * np.exp(-(a - a.groupby(e.country).transform("mean")))}
    p80_0 = e.groupby("country").cost.transform(lambda x: x.quantile(0.80))
    nq0, _ = need_thresholds(e, "needqr_noclim", s["tau"], e.hdd_hs1617)
    rows = []
    for name, hdd_new in scen.items():
        cost1 = e.cost - heat + heat * (hdd_new / e.hdd_hs1617) ** beta
        p80_1 = cost1.groupby(e.country).transform(lambda x: x.quantile(0.80))
        nc0, nc1 = need_thresholds(e, "needqr", s["tau"], hdd_new)
        rules = {"P80 fixed in EUR": (p80_0, p80_0), "P80 recomputed": (p80_0, p80_1),
                 "Need line, no degree days": (nq0, nq0), "Need line with degree days": (nc0, nc1)}
        for rule, (t_before, t_after) in rules.items():
            cut = 3 if rule.startswith("P80") else s["cutoff"]
            li_r = (e.income_bracket <= cut).astype(int)
            y0 = li_r + 2 * (e.cost > t_before).astype(int)
            y1 = li_r + 2 * (cost1 > t_after).astype(int)
            for scope, m in [("All five", pd.Series(True, index=e.index))] + [(c, e.country == c) for c in COUNTRY_ORDER]:
                dr0, dr1 = (y0[m] == 3), (y1[m] == 3)
                rows.append({"scenario": name, "rule": rule, "scope": scope, "households": int(m.sum()),
                             "changed_group": float((y0[m] != y1[m]).mean()),
                             "left_high_cost": float(((y0[m] >= 2) & (y1[m] < 2)).mean()),
                             "entered_high_cost": float(((y0[m] < 2) & (y1[m] >= 2)).mean()),
                             "double_risk_before": int(dr0.sum()), "double_risk_after": int(dr1.sum()),
                             "double_risk_changed": float((dr0 != dr1).sum() / max(dr0.sum(), 1))})
    return pd.DataFrame(rows)


def plot_drift(t: pd.DataFrame) -> None:
    scen = list(dict.fromkeys(t.scenario))
    rules = ["P80 fixed in EUR", "P80 recomputed", "Need line, no degree days", "Need line with degree days"]
    fig, axes = plt.subplots(1, len(scen), figsize=(7.2, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, sc, letter in zip(axes, scen, "abc"):
        panel_max = t[t.scenario == sc].changed_group.max() * 100
        for i, r in enumerate(rules):
            g = t[(t.scenario == sc) & (t.rule == r)]
            pooled = g[g.scope == "All five"].changed_group.iloc[0] * 100
            ax.barh(i, pooled, height=0.55, color=BLUE if "degree days" in r and "no" not in r else GREY, zorder=2)
            for c, mk in COUNTRY_MARKERS.items():
                v = g[g.scope == c].changed_group.iloc[0] * 100
                ax.scatter(v, i, s=14, marker=mk, facecolor="white", edgecolor=INK, lw=0.8, zorder=3,
                           label=c if (i == 0 and letter == "a") else None)
            small = pooled < 0.2 * panel_max
            ax.text(pooled if not small else pooled + 0.02 * panel_max, i - 0.42, f"{pooled:.1f}%", fontsize=7, color=INK, va="center",
                    ha="left" if small else "right")
        ax.set_yticks(range(len(rules)), rules); ax.invert_yaxis()
        ax.set_title(f"{letter}  {sc}", loc="left", fontsize=8, color=INK)
        ax.grid(axis="x", color="#e9e8e4", lw=0.6, zorder=0); ax.tick_params(axis="y", length=0)
    axes[1].set_xlabel("Households whose group changes (%); bar: five countries")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.55, -0.16), ncol=5,
               frameon=False, fontsize=7, handletextpad=0.2)
    save_figure(fig, FIGS / "weather_drift")
    plt.close(fig)


def main() -> None:
    dev = load_development_frame()
    heat = pd.read_parquet(ROOT / "artifacts/data/energy_cost_study.parquet")[["UID", "heating_cost_eur"]]
    dev = dev.merge(heat, on="UID", how="left")
    trans, reg = reclassification(dev)
    reg.to_csv(TABLES / "table_label_climate_regions.csv", index=False)
    print(trans, flush=True)
    for c in COUNTRY_ORDER:
        g = reg[(reg.country == c) & (reg.households >= MIN_REGION_HH)]
        print(f"{c}: {len(g)} regions with >= {MIN_REGION_HH} households, household-weighted corr(rel log HDD, net into "
              f"high cost) = {weighted_corr(g):+.2f}", flush=True)
    plot_reclassification(trans, reg)
    el = pd.read_csv(TABLES / "table_climate_elasticity.csv")
    beta = float(el[(el["sample"] == "five") & (el.outcome == "heating_cost_eur") & (el.term == "log_hdd")].coef.iloc[0])
    print(f"heating-bill elasticity used: {beta:.3f}", flush=True)
    t = weather_drift(dev, beta)
    t.to_csv(TABLES / "table_weather_drift.csv", index=False)
    print(t[t.scope == "All five"][["scenario", "rule", "changed_group", "left_high_cost", "entered_high_cost",
                                     "double_risk_changed"]].round(3).to_string(index=False), flush=True)
    plot_drift(t)


if __name__ == "__main__":
    main()
