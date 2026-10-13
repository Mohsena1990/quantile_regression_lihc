"""Build and explain the four expenditure lines (lib/identification/expenditure_lines.py).

Step 1  Hansen's Threshold Regression: per country (and the four pooled with country fixed effects), among low-income
        households, regress hardship (S8 >= 3) on the household's spending position q; find the tipping point gamma, test it
        (bootstrap p-value) and give its 95% confidence set. Two positions: q = percentile of expenditure in the country, and
        q = conditional rank among homes with the same physical need. Saved to artifacts/identification/hansen_tipping_points.json.
Step 2  Lines and labels: Van Hove (country 80th percentile), Quantile Regression (conditional 80th percentile), Hansen
        Threshold (country percentile at gamma_u), Hansen + Quantile Regression (conditional percentile tau = gamma_c).
Step 3  How each line behaves: share of households above it, class sizes, agreement between lines (Cohen's kappa), pinball loss
        of the quantile lines, and SHAP of the Hansen + Quantile Regression line (exact Shapley values; which need variables move it).

Development households only. Outputs (outputs/tables): table_hansen_tipping_points.csv, table_expenditure_lines.csv,
table_expenditure_line_agreement.csv, table_expenditure_line_shap.csv; (outputs/figures): hansen_hardship_jump, hansen_lr_curves,
expenditure_lines, expenditure_line_shap.
"""
from __future__ import annotations
import itertools, json, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import cohen_kappa_score
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES, load_development_frame
from lib.identification.expenditure_lines import (NEED_V2, QR_TAU, TIPPING_PATH, chosen_gamma, conditional_rank_oof, fit_qr_line)
from lib.identification.hansen import LR_CRITICAL_95, hansen_threshold
from lib.identification.labels import CLASS_NAMES, add_labels, support_table, viability
from lib.reporting.figures import save_figure

LINES = {"raw_q80": "Van Hove (80th percentile)", "qr80": "Quantile Regression (80th)", "hansen": "Hansen Threshold",
         "hansen_qr": "Hansen + Quantile Regression"}
COLOR = {"raw_q80": "#0b0b0b", "qr80": "#2a78d6", "hansen": "#eb6834", "hansen_qr": "#1baf7a"}
NEED_NAMES = {"H3": "floor area", "household_size": "household size", "H1": "dwelling type", "H2": "dwelling age",
              "insulation_count": "insulation types", "hdd_hs1617": "heating degree days", "cdd_s2017": "cooling degree days"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"
FLOOR_MID = {1: 35, 2: 54, 3: 78, 4: 105, 5: 160, 6: 250}


def style(ax, grid="both"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis=grid, color=GRID, lw=0.6); ax.set_axisbelow(True)


def positions(e: pd.DataFrame) -> pd.DataFrame:
    """Add q_u (expenditure percentile in the country) and q_c (conditional rank given physical need, out of fold)."""
    parts = []
    for c in COUNTRIES:
        g = e[e.country == c].reset_index(drop=True)
        g["q_u"] = g.cost.rank(pct=True)
        g["q_c"] = conditional_rank_oof(g)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def hansen_inputs(d: pd.DataFrame, q: str, pooled: bool):
    s = d[(d.li == 1) & d.S8.notna()].copy()
    y = (s.S8 >= 3).astype(float).to_numpy()
    Xs = np.column_stack([np.ones(len(s)), s[q].to_numpy()])
    controls = [s.income_bracket.to_numpy(), s.household_size.fillna(s.household_size.median()).to_numpy()]
    if pooled:
        controls += [(s.country == c).astype(float).to_numpy() for c in COUNTRIES[1:]]
    return y, s[q].to_numpy(), Xs, np.column_stack(controls), s


def main() -> None:
    dev = load_development_frame()
    e = positions(add_labels(dev, cost_kind="raw_q80"))
    tp, rows, curves = {}, [], {}
    for kind, q in (("unconditional", "q_u"), ("conditional", "q_c")):
        tp[kind] = {"countries": {}}
        for scope in COUNTRIES + ["pooled"]:
            d = e if scope == "pooled" else e[e.country == scope]
            y, qq, Xs, Xc, s = hansen_inputs(d, q, scope == "pooled")
            h = hansen_threshold(y, qq, Xs, Xc, seed=11)
            curves[(kind, scope)] = (h, s, y)
            rec = {k: h[k] for k in ("gamma", "F", "p_value", "ci_low", "ci_high", "n")}
            rec.update({"hardship_below": float(y[qq <= h["gamma"]].mean()), "hardship_above": float(y[qq > h["gamma"]].mean())})
            rows.append({"position": kind, "scope": scope, **rec})
            if scope == "pooled":
                tp[kind]["pooled"] = rec
            else:
                tp[kind]["countries"][scope] = rec
            print(f"Hansen {kind} {scope}: gamma={h['gamma']:.3f} [{h['ci_low']:.2f}, {h['ci_high']:.2f}] p={h['p_value']:.3f} n={h['n']}", flush=True)
    for kind in tp:
        for c in COUNTRIES:
            tp[kind]["countries"][c]["used"] = chosen_gamma(tp, c, kind)
    TIPPING_PATH.parent.mkdir(parents=True, exist_ok=True)
    TIPPING_PATH.write_text(json.dumps(tp, indent=2) + "\n")
    tpt = pd.DataFrame(rows)
    tpt["used_for_label"] = [tp[r.position]["countries"][r.scope]["used"] if r.scope != "pooled" else np.nan for r in tpt.itertuples()]
    tpt.to_csv(TAB / "table_hansen_tipping_points.csv", index=False)

    labs = {k: add_labels(dev, cost_kind=k) for k in LINES}
    line_rows, agree = [], []
    for k, lab in labs.items():
        sup = viability(support_table(lab)).set_index("country")
        for c in COUNTRIES:
            g = lab[lab.country == c]
            tau = {"raw_q80": 0.80, "qr80": QR_TAU, "hansen": tp["unconditional"]["countries"][c]["used"],
                   "hansen_qr": tp["conditional"]["countries"][c]["used"]}[k]
            pin = float(np.mean(np.maximum(tau * (g.cost - g.threshold), (tau - 1) * (g.cost - g.threshold))))
            line_rows.append({"line": LINES[k], "cost_kind": k, "country": c, "tau": tau, "median_line_eur": float(g.threshold.median()),
                              "share_above": float(g.hc.mean()), "pinball_loss_eur": pin,
                              **{n: int(sup.loc[c, n]) for n in CLASS_NAMES.values()}, "four_class_ok": bool(sup.loc[c, "four_class_ok"]),
                              "hardship_high_cost_low_income": float((g[(g.li == 1) & (g.hc == 1)].S8 >= 3).mean()),
                              "hardship_low_cost_low_income": float((g[(g.li == 1) & (g.hc == 0)].S8 >= 3).mean())})
    for a, b in itertools.combinations(LINES, 2):
        j = labs[a].set_index("UID").hc.rename("a").to_frame().join(labs[b].set_index("UID").hc.rename("b"), how="inner").join(
            labs[a].set_index("UID").country)
        for c, g in j.groupby("country"):
            agree.append({"pair": f"{LINES[a]} vs {LINES[b]}", "country": c, "kappa": cohen_kappa_score(g.a, g.b)})
    lt, at = pd.DataFrame(line_rows), pd.DataFrame(agree)
    lt.to_csv(TAB / "table_expenditure_lines.csv", index=False)
    at.to_csv(TAB / "table_expenditure_line_agreement.csv", index=False)

    shap_rows = []
    for c in COUNTRIES:
        g = labs["hansen_qr"][labs["hansen_qr"].country == c].reset_index(drop=True)
        x = g[NEED_V2].astype(float)
        predict = fit_qr_line(g, tp["conditional"]["countries"][c]["used"])
        f = lambda arr: predict(pd.DataFrame(arr, columns=NEED_V2).assign(cost=np.nan, country=c))
        sv = shap.explainers.Exact(f, shap.maskers.Independent(x.sample(min(100, len(x)), random_state=1), max_samples=100))(x).values
        for j, feat in enumerate(NEED_V2):
            shap_rows.append({"country": c, "feature": feat, "feature_name": NEED_NAMES[feat], "mean_abs_shap_eur": float(np.abs(sv[:, j]).mean())})
    st = pd.DataFrame(shap_rows)
    st.to_csv(TAB / "table_expenditure_line_shap.csv", index=False)

    # Figure 1: hardship by spending position, with the tipping point
    fig, axes = plt.subplots(2, len(COUNTRIES), figsize=(16, 7), sharey=True); fig.patch.set_facecolor(SURFACE)
    for r, (kind, q, xl) in enumerate((("unconditional", "q_u", "position of expenditure in the country (percentile)"),
                                       ("conditional", "q_c", "position among homes with the same need (conditional rank)"))):
        for ax, c in zip(axes[r], COUNTRIES):
            style(ax, "y")
            h, s, y = curves[(kind, c)]
            bins = pd.cut(s[q], np.linspace(0, 1, 11), include_lowest=True)
            m = pd.Series(y, index=s.index).groupby(bins).mean()
            ax.bar(np.arange(0.05, 1, 0.1), m.values, width=0.09, color="#cfe0f5", edgecolor=INK2, lw=0.5)
            ax.axvspan(h["ci_low"], h["ci_high"], color="#eda100", alpha=0.18)
            ax.axvline(h["gamma"], color="#d1342f", lw=1.6)
            ax.set_title(f"{c}: tipping point {h['gamma']:.2f} (p = {h['p_value']:.2f})", fontsize=9, loc="left")
            ax.set_xlabel(xl, fontsize=7.5, color=INK2)
        axes[r, 0].set_ylabel("share finding it (very) difficult\non present income (low-income households)", fontsize=8, color=INK2)
    fig.suptitle("Hansen's Threshold Regression: where does hardship jump? Red line = tipping point, shaded = 95% confidence set",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save_figure(fig, FIG / "hansen_hardship_jump"); plt.close(fig)

    # Figure 2: likelihood-ratio curves (Hansen 2000)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4)); fig.patch.set_facecolor(SURFACE)
    for ax, kind in zip(axes, ("unconditional", "conditional")):
        style(ax)
        for c, col in zip(COUNTRIES + ["pooled"], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", INK]):
            h = curves[(kind, c)][0]
            ax.plot(h["grid"], h["lr"], color=col, lw=1.3, label=c)
        ax.axhline(LR_CRITICAL_95, color=MUTED, ls="--", lw=1)
        ax.set_ylim(0, 30); ax.set_xlabel(f"candidate tipping point ({kind} position)", fontsize=8)
        ax.set_ylabel("likelihood ratio LR(gamma)", fontsize=8)
        ax.set_title(f"{kind} position: values below the dashed line form the 95% confidence set", fontsize=9, loc="left")
    axes[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); save_figure(fig, FIG / "hansen_lr_curves"); plt.close(fig)

    # Figure 3: the four lines over the observed costs, for a typical home by floor area
    fig, axes = plt.subplots(1, len(COUNTRIES), figsize=(17, 4.4)); fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, c in zip(axes, COUNTRIES):
        style(ax)
        g = labs["qr80"][labs["qr80"].country == c].reset_index(drop=True)
        ax.scatter(g.H3.map(FLOOR_MID) * np.exp(rng.normal(0, 0.06, len(g))), g.cost, s=4, color=MUTED, alpha=0.35, lw=0)
        typical = {f: (float(g[f].mode().iloc[0]) if f == "H1" else float(g[f].median())) for f in NEED_V2 if f != "H3"}
        grid = pd.DataFrame({"H3": list(FLOOR_MID), **typical}).assign(cost=np.nan, country=c)
        xs = [FLOOR_MID[k] for k in FLOOR_MID]
        ax.plot(xs, fit_qr_line(g, QR_TAU)(grid), color=COLOR["qr80"], lw=1.8, label=LINES["qr80"])
        ax.plot(xs, fit_qr_line(g, tp["conditional"]["countries"][c]["used"])(grid), color=COLOR["hansen_qr"], lw=1.8,
                label=f"{LINES['hansen_qr']} (tau {tp['conditional']['countries'][c]['used']:.2f})")
        ax.axhline(g.cost.quantile(0.8), color=COLOR["raw_q80"], ls="-.", lw=1, label=LINES["raw_q80"])
        ax.axhline(g.cost.quantile(tp["unconditional"]["countries"][c]["used"]), color=COLOR["hansen"], ls="--", lw=1.2,
                   label=f"{LINES['hansen']} ({tp['unconditional']['countries'][c]['used']:.2f})")
        ax.set_xscale("log"); ax.set_xticks(xs, [str(v) for v in xs]); ax.minorticks_off()
        ax.set_ylim(0, float(g.cost.quantile(0.97)) * 1.15)
        ax.set_title(c, fontsize=10, loc="left"); ax.set_xlabel("floor area (m2, band midpoint)", fontsize=8)
    axes[0].set_ylabel("annual energy expenditure (EUR)", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("The four expenditure lines for a typical home (other need variables at the country's typical value); "
                 "a household above its line counts as high cost", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save_figure(fig, FIG / "expenditure_lines"); plt.close(fig)

    # Figure 4: SHAP of the Hansen + Quantile Regression line
    fig, ax = plt.subplots(figsize=(9, 4)); fig.patch.set_facecolor(SURFACE); style(ax, "x")
    order = st.groupby("feature").mean_abs_shap_eur.mean().sort_values().index
    for i, (c, col) in enumerate(zip(COUNTRIES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"])):
        v = st[st.country == c].set_index("feature").loc[order, "mean_abs_shap_eur"]
        ax.barh(np.arange(len(order)) + (i - 1.5) * 0.2, v, height=0.19, color=col, label=c)
    ax.set_yticks(range(len(order)), [NEED_NAMES[f] for f in order], fontsize=8.5)
    ax.set_xlabel("mean |SHAP| of the line (EUR per year)", fontsize=8.5)
    ax.set_title("What moves the Hansen + Quantile Regression line (SHAP)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); save_figure(fig, FIG / "expenditure_line_shap"); plt.close(fig)

    pd.set_option("display.width", 220)
    print(tpt.round(3).to_string(index=False))
    print(lt[["line", "country", "tau", "median_line_eur", "share_above", "No Risk", "Income Risk", "Expenditure Risk", "Double Risk",
              "four_class_ok", "hardship_high_cost_low_income", "hardship_low_cost_low_income"]].round(3).to_string(index=False))
    print(at.groupby("pair").kappa.agg(["min", "max"]).round(2))


if __name__ == "__main__":
    main()
