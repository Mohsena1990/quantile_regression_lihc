"""Step 3-5 of the flow: which predictors are effective, are they universal, and does a factors-only model
identify energy poverty as well in every country?

Van Hove et al. (2022) hypothesised universal predictors (valid across countries) and contextual ones (local).
Here that hypothesis is tested, per axis of the label (income axis, cost axis) and for the four-class target,
in two states: income EXCLUDED (proxy targeting) and INCLUDED (Van Hove's own design).

  1. importance per country on held-out CV folds: mean|SHAP| and permutation importance, bootstrap-style stability
  2. consensus: universal = stable under both methods in every gated country; top-2 per axis
  3. PDP for the selected factors; out-of-fold SHAP dependence
  4. leave-one-country-out universality test: select the factors WITHOUT country H, train a factors-only model on
     the other countries, score H, and compare with H's own full model
Only countries passing the four-class support gate enter (Spain fails: 21 Double-Risk households).
Development partitions only.

Outputs (outputs/tables): table_predictor_stability.csv, table_universal_predictors.csv, table_pdp.csv,
table_universality_test.csv; (outputs/figures): discovery_universal_map, discovery_country_pairs,
discovery_shap_dependence, discovery_pdp, discovery_universality_scatter. SHAP dependence data:
artifacts/identification/oof_shap_selected.parquet.
"""
from __future__ import annotations
import itertools, sys, time, warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import load_development_frame, tier_features
from lib.identification.discovery import (TARGETS, fold_importance, partial_dependence, select_universal, summarise_stability)
from lib.identification.factors import FEATURE_DOMAIN
from lib.identification.labels import add_labels, support_table, viability
from lib.identification.protocol import fit_predict_proba, loco_proba, oof_proba, summarise
from lib.reporting.figures import save_figure

STATES = ("excluded", "included")
UNIVERSAL_MODELS = ("lr4", "cb4")
COUNTRY_STYLE = {"Bulgaria": ("#2a78d6", "o"), "Hungary": ("#eb6834", "s"), "Italy": ("#1baf7a", "^"),
                 "Poland": ("#eda100", "D"), "Spain": ("#e87ba4", "v")}
DOMAIN_COLOR = {"Dwelling & efficiency": "#2a78d6", "Heating & appliances": "#eb6834", "Household": "#1baf7a",
                "Socio-economic": "#eda100", "Climate": "#e87ba4", "Income": "#0b0b0b", "Welfare screen": "#4a3aa7"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
TAB, FIG = ROOT / "outputs/tables", ROOT / "outputs/figures"


def style(ax, grid="both"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


def discover(lab, countries):
    stab_rows, shap_store = [], {}
    for state in STATES:
        feats = tier_features(include_income=(state == "included"))["T2"]
        for c in countries:
            g = lab[lab.country == c].reset_index(drop=True)
            for target in TARGETS:
                t0 = time.time()
                if target == "y":
                    imp = fold_importance(g, feats, target)
                else:
                    imp, oof = fold_importance(g, feats, target, collect_shap=True)
                    per_hh = oof.groupby("_row")[feats].mean()
                    shap_store[(state, c, target)] = (g.loc[per_hh.index, ["UID"] + feats].reset_index(drop=True), per_hh.reset_index(drop=True))
                st = summarise_stability(imp).assign(state=state, country=c, target=target)
                stab_rows.append(st)
                print(f"discovery {state} {c} {target}: {time.time() - t0:.0f}s", flush=True)
    stab = pd.concat(stab_rows, ignore_index=True)
    stab["domain"] = stab.feature.map(FEATURE_DOMAIN)
    return stab, shap_store


def universality(stab, countries):
    rows = []
    for (state, target), d in stab.groupby(["state", "target"]):
        u = select_universal(d, countries).assign(state=state, target=target, held_out="none")
        rows.append(u)
    out = pd.concat(rows, ignore_index=True)
    out["domain"] = out.feature.map(FEATURE_DOMAIN)
    return out


def loco_test(lab, stab, countries):
    """Factors chosen WITHOUT the held-out country, then scored on it."""
    rows = []
    for state in STATES:
        tiers = tier_features(include_income=(state == "included"))
        full = tiers["T2"]
        for h in countries:
            others = [c for c in countries if c != h]
            chosen = []
            for target in ("li", "hc"):
                d = stab[(stab.state == state) & (stab.target == target)]
                u = select_universal(d, others)
                chosen += u[u.selected].feature.tolist()
            factors = list(dict.fromkeys(chosen))
            if not factors:
                rows.append({"state": state, "held_out": h, "factors": "", "note": "no universal factor among the other countries"})
                continue
            g = lab[lab.country == h].reset_index(drop=True)
            for model in UNIVERSAL_MODELS:
                base = {"state": state, "held_out": h, "model": model, "factors": ";".join(factors), "n_factors": len(factors)}
                proba, tgt = loco_proba(lab, h, factors, model)
                rows.append({**base, "arm": "factors_only_loco", **summarise(tgt.y.values, proba)})
                rows.append({**base, "arm": "factors_only_local", **summarise(g.y.values, oof_proba(g, factors, model))})
                rows.append({**base, "arm": "full_local", **summarise(g.y.values, oof_proba(g, full, model))})
                proba, tgt = loco_proba(lab, h, full, model)
                rows.append({**base, "arm": "full_loco", **summarise(tgt.y.values, proba)})
            print(f"universality {state} held-out {h}: factors={factors}", flush=True)
    return pd.DataFrame(rows)


def pdp_table(lab, countries, universal):
    rows = []
    for state in STATES:
        feats = tier_features(include_income=(state == "included"))["T2"]
        sel = universal[(universal.state == state) & universal.selected & universal.target.isin(["li", "hc"])].feature.unique()
        for c in countries:
            g = lab[lab.country == c].reset_index(drop=True)
            for f in sel:
                rows.append(partial_dependence(g, feats, "y", f).assign(state=state, country=c))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ------------------------------------------------------------------ figures
def fig_universal_map(universal):
    fig, axes = plt.subplots(len(STATES), 2, figsize=(11.5, 8.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, state in enumerate(STATES):
        for j, target in enumerate(("li", "hc")):
            ax = axes[i, j]; style(ax)
            d = universal[(universal.state == state) & (universal.target == target)]
            for _, r in d.iterrows():
                col = DOMAIN_COLOR.get(r.domain, MUTED)
                ax.scatter(r.consensus_score * 100, min(r.min_shap_stability, r.min_perm_stability), s=95 if r.selected else 42,
                           color=col, edgecolor=INK if r.selected else SURFACE, linewidth=1.6 if r.selected else 0.8, zorder=3)
                if r.selected or r.consensus_score > 0.09:
                    tag = "" if r.universal_strict else f" ({int(r.n_countries_pass)}/{int(r.n_countries)} countries)"
                    ax.annotate(r.feature + tag, (r.consensus_score * 100, min(r.min_shap_stability, r.min_perm_stability)),
                                xytext=(6, 4), textcoords="offset points", fontsize=7.5, color=INK2)
            ax.axhline(0.5, color=MUTED, lw=0.9, ls="--"); ax.set_ylim(-0.04, 1.06)
            ax.set_title(f"{TARGETS[target]} -- income {state}", fontsize=10, color=INK, loc="left")
            ax.set_xlabel("Mean importance share across countries (%, SHAP and permutation averaged)", fontsize=8, color=INK2)
            if j == 0:
                ax.set_ylabel("Stability in the weakest country\n(share of CV folds in top-6)", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, ms=7, label=d) for d, c in DOMAIN_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Universal vs contextual predictors: importance (right = larger) against stability in the weakest country (up = stable everywhere)\n"
                 "Outlined, larger points = selected factors (top-2 per axis; '(3/4 countries)' = relaxed rule, weakest country fails). Van Hove's hypothesis needs points in the upper right.",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.92), w_pad=2.5); save_figure(fig, FIG / "discovery_universal_map"); plt.close(fig)


def fig_country_pairs(stab, countries):
    pairs = list(itertools.combinations(countries, 2))
    fig, axes = plt.subplots(2, len(pairs), figsize=(2.6 * len(pairs), 5.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for i, target in enumerate(("li", "hc")):
        d = stab[(stab.state == "excluded") & (stab.target == target)].pivot(index="feature", columns="country", values="shap_share") * 100
        for j, (a, b) in enumerate(pairs):
            ax = axes[i, j]; style(ax)
            ax.scatter(d[a], d[b], s=30, color=[DOMAIN_COLOR.get(FEATURE_DOMAIN.get(f), MUTED) for f in d.index], edgecolor=SURFACE, linewidth=0.6)
            top = d[[a, b]].sum(axis=1).nlargest(3).index
            for f in top:
                ax.annotate(f, (d.loc[f, a], d.loc[f, b]), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=INK2)
            m = max(d[a].max(), d[b].max()) * 1.05; ax.plot([0, m], [0, m], color=MUTED, lw=0.8, ls="--")
            rho = stats.spearmanr(d[a], d[b])[0]
            ax.set_title(f"{a[:3]} vs {b[:3]}  (rho={rho:.2f})", fontsize=8.5, color=INK, loc="left")
            if j == 0:
                ax.set_ylabel(f"{TARGETS[target]}\nSHAP share (%)", fontsize=8, color=INK2)
            ax.set_xlabel(f"{a} (%)", fontsize=7.5, color=INK2)
    fig.suptitle("Do countries rank the same predictors as important? SHAP share per feature, country against country (income excluded)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save_figure(fig, FIG / "discovery_country_pairs"); plt.close(fig)


def _xcodes(series):
    if series.dtype == object:
        cats = sorted(series.dropna().unique()); return series.map({c: i for i, c in enumerate(cats)}), cats
    return series, None


def fig_shap_dependence(shap_store, universal, countries):
    panels = []
    for target in ("li", "hc"):
        sel = universal[(universal.state == "excluded") & (universal.target == target) & universal.selected]
        panels += [(target, f) for f in sel.feature.tolist()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 4.2), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, (target, f) in zip(axes[0], panels):
        style(ax)
        for c in countries:
            vals, shp = shap_store[("excluded", c, target)]
            x, cats = _xcodes(vals[f].reset_index(drop=True))
            col, mk = COUNTRY_STYLE[c]
            xj = x.to_numpy(float) + np.random.default_rng(1).normal(0, 0.06, len(x))
            ax.scatter(xj, shp[f].to_numpy(), s=9, color=col, alpha=0.18, marker=mk, linewidths=0)
            bins = pd.qcut(x, 6, duplicates="drop") if x.nunique() > 8 else x
            m = pd.DataFrame({"x": x, "s": shp[f].to_numpy()}).groupby(bins, observed=True).agg(x=("x", "mean"), s=("s", "mean"))
            ax.plot(m.x, m.s, color=col, lw=2, label=c)
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_title(f"{f}\n({TARGETS[target]})", fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("feature value" + (" (categories: " + ", ".join(cats) + ")" if cats else ""), fontsize=7.5, color=INK2)
        ax.set_ylabel("SHAP value (log-odds of the axis)", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], marker=COUNTRY_STYLE[c][1], color=COUNTRY_STYLE[c][0], ls="-", ms=6, label=c) for c in countries]
    fig.legend(handles=handles, loc="lower center", ncol=len(countries), frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Same factor, same effect? Out-of-fold SHAP dependence of the universal factors (points = households, lines = binned means)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9)); save_figure(fig, FIG / "discovery_shap_dependence"); plt.close(fig)


def fig_pdp(pdp, countries):
    d = pdp[(pdp.state == "excluded") & (pdp.outcome == "Double Risk")]
    feats = d.feature.unique().tolist()
    if not feats:
        return
    fig, axes = plt.subplots(1, len(feats), figsize=(3.9 * len(feats), 4.0), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, f in zip(axes[0], feats):
        style(ax)
        for c in countries:
            s = d[(d.feature == f) & (d.country == c)]
            if s.empty:
                continue
            col, mk = COUNTRY_STYLE[c]
            xs = pd.to_numeric(s.value, errors="coerce")
            if xs.isna().any():
                ax.plot(range(len(s)), s.pd, color=col, marker=mk, lw=2, ms=5, label=c); ax.set_xticks(range(len(s)), s.value, rotation=45, fontsize=6.5)
            else:
                ax.plot(xs, s.pd, color=col, marker=mk, lw=2, ms=5, label=c)
        ax.set_title(f, fontsize=9.5, color=INK, loc="left"); ax.set_xlabel("feature value", fontsize=8, color=INK2)
        ax.set_ylabel("Partial dependence: P(Double Risk)", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], marker=COUNTRY_STYLE[c][1], color=COUNTRY_STYLE[c][0], ls="-", ms=6, label=c) for c in countries]
    fig.legend(handles=handles, loc="lower center", ncol=len(countries), frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle("Partial dependence of Double-Risk probability on the universal factors (income excluded)", fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.9)); save_figure(fig, FIG / "discovery_pdp"); plt.close(fig)


def fig_universality(test):
    t = test.dropna(subset=["arm"]) if "arm" in test else pd.DataFrame()
    if t.empty:
        return
    best = t.sort_values("f1_macro", ascending=False).groupby(["state", "held_out", "arm"], as_index=False).head(1)
    w = best.pivot_table(index=["state", "held_out"], columns="arm", values="f1_macro").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8)); fig.patch.set_facecolor(SURFACE)
    for ax, (xa, ya, xl, yl, title) in zip(axes, [
        ("factors_only_local", "full_local", "Factors-only model, own country (macro-F1)", "Full model, own country (macro-F1)", "Contextual gap: what the universal factors alone leave out"),
        ("factors_only_loco", "factors_only_local", "Factors-only, trained on OTHER countries (macro-F1)", "Factors-only, trained on own country (macro-F1)", "Transfer: do the universal factors travel?")]):
        style(ax)
        for _, r in w.iterrows():
            col, mk = COUNTRY_STYLE[r.held_out]
            ax.scatter(r[xa], r[ya], s=85, color=col, marker=mk, edgecolor=INK if r.state == "included" else SURFACE, linewidth=1.5, zorder=3)
            ax.annotate(f"{r.held_out[:3]} ({'+inc' if r.state == 'included' else 'no inc'})", (r[xa], r[ya]), xytext=(6, 4), textcoords="offset points", fontsize=7, color=INK2)
        lo = np.nanmin([w[xa].min(), w[ya].min()]) - 0.03; hi = np.nanmax([w[xa].max(), w[ya].max()]) + 0.03
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.9, ls="--"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(xl, fontsize=8, color=INK2); ax.set_ylabel(yl, fontsize=8, color=INK2); ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    fig.suptitle("Universality test: factors selected without the held-out country (outlined = income included; diagonal = no difference)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save_figure(fig, FIG / "discovery_universality_scatter"); plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True); (ROOT / "artifacts/identification").mkdir(parents=True, exist_ok=True)
    dev = load_development_frame()
    lab = add_labels(dev, cost_kind="need_adjusted")
    v = viability(support_table(lab))
    countries = v[v.four_class_ok].country.tolist()
    print("gated countries:", countries, "| excluded:", v[~v.four_class_ok].country.tolist(), flush=True)
    lab = lab[lab.country.isin(countries)]
    stab, shap_store = discover(lab, countries)
    stab.to_csv(TAB / "table_predictor_stability.csv", index=False)
    universal = universality(stab, countries)
    universal.to_csv(TAB / "table_universal_predictors.csv", index=False)
    sel = universal[universal.selected]
    print(sel[["state", "target", "feature", "domain", "consensus_score", "min_shap_stability", "min_perm_stability", "universal_strict"]].round(2).to_string(index=False), flush=True)
    keep = []
    for (state, c, target), (vals, shp) in shap_store.items():
        chosen = sel[(sel.state == state) & (sel.target == target)].feature.tolist()
        for f in chosen:
            keep.append(pd.DataFrame({"state": state, "country": c, "target": target, "feature": f, "UID": vals.UID, "value": vals[f].astype(str), "shap": shp[f]}))
    if keep:
        pd.concat(keep).to_parquet(ROOT / "artifacts/identification/oof_shap_selected.parquet", index=False)
    pdp = pdp_table(lab, countries, universal); pdp.to_csv(TAB / "table_pdp.csv", index=False)
    test = loco_test(lab, stab, countries); test.to_csv(TAB / "table_universality_test.csv", index=False)
    fig_universal_map(universal); fig_country_pairs(stab, countries)
    fig_shap_dependence(shap_store, universal, countries); fig_pdp(pdp, countries); fig_universality(test)
    print(f"Wrote discovery tables/figures for {countries}")


if __name__ == "__main__":
    main()
