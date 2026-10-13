"""Feature stages B0 (Van Hove) -> B1 (+ best of all variables) -> B2 (+ best climate indices), income always a predictor,
"attention + SHAP" feature selection, and three classifiers -- CatBoost, XGBoost and FT-Transformer (lib/identification/stages.py).

Four definitions of the energy-poverty classes (income axis = income bracket <= 3 in all four; expenditure lines:
lib/identification/expenditure_lines.py):
  vh         Van Hove: expenditure above the country's 80th percentile
  qr80       Quantile Regression: above the conditional 80th percentile given physical need
  hansen     Hansen Threshold Regression: above the country's expenditure tipping point
  hansen_qr  Hansen + Quantile Regression: above the conditional percentile at Hansen's tipping point
Development households only (train + validation); the test partition is spent. Repeated 5-fold CV (x3), household-bootstrap 95% CIs.
Countries whose four classes do not all reach 30 households under a definition are run but flagged (`four_class_ok`).

Outputs (outputs/tables):
  table_stages_metrics.csv          every metric with 95% CIs, label x country x stage x model (+ chance)
  table_stages_report.csv           the headline metrics only: per-class recall and precision, weighted F1, balanced accuracy,
                                    accuracy, ROC AUC (macro and per class)
  table_stages_deltas.csv           paired-bootstrap gains B1-B0, B2-B1, B2-B0, B2p-B1, B2-B2p per model
  table_stages_selection.csv        per candidate: share of the 15 outer folds selected, mean rank, attention, SHAP
  table_stages_selected_importance.csv  per model: held-out importance of each selected feature (SHAP or attention)
  table_stages_selection_curve.csv  inner-CV macro-F1 (mean, SE) for every k and the k chosen
(artifacts/identification) stages_oof_shap.parquet: held-out CatBoost SHAP per household.
(outputs/figures) stages_performance, stages_selection_frequency, stages_selection_curve, stages_selected_by_model,
stages_shap_beeswarm_<label>.
"""
from __future__ import annotations
import argparse, sys, time, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES, load_development_frame
from lib.identification.labels import add_labels, support_table, viability
from lib.identification.metrics import CLASS_KEYS
from lib.identification.models3 import MODELS3
from lib.identification.protocol import oof_proba, summarise
from lib.identification.stages import (ALL_FEATURES, B0_FEATURES, B2_CANDIDATES, FEATURE_NAMES, STAGES, paired_delta, run_stages)
from lib.reporting.figures import save_figure

LABELS = {"vh": "raw_q80", "qr80": "qr80", "hansen": "hansen", "hansen_qr": "hansen_qr"}
LABEL_NAMES = {"vh": "Van Hove (80th percentile)", "qr80": "Quantile Regression (80th)", "hansen": "Hansen Threshold Regression",
               "hansen_qr": "Hansen + Quantile Regression"}
MODEL_NAMES = {"catboost": "CatBoost", "xgboost": "XGBoost", "ft_transformer": "FT-Transformer"}
N_BOOT = 200
TAB, FIG, ART = ROOT / "outputs/tables", ROOT / "outputs/figures", ROOT / "artifacts/identification"
STAGE_COLOR = {"B0": "#898781", "B1": "#2a78d6", "B2": "#1baf7a", "B2p": "#e87ba4"}
MODEL_MARKER = {"catboost": "o", "xgboost": "s", "ft_transformer": "^"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
REPORT = (["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "auc_macro_ovr"]
          + [f"{m}_{k}" for m in ("recall", "precision", "auc") for k in CLASS_KEYS])


def one_cell(label, country, g, four_ok, n_repeats):
    t0 = time.time()
    out = run_stages(g, n_repeats=n_repeats)
    y = g.y.to_numpy()
    base = {"label": label, "country": country, "four_class_ok": four_ok}
    metrics = [{**base, "stage": "any", "model": "chance", **summarise(y, oof_proba(g, B0_FEATURES, "prior", n_repeats=n_repeats), n_boot=N_BOOT)}]
    deltas = []
    for (stage, model), p in out["proba"].items():
        metrics.append({**base, "stage": stage, "model": model, **summarise(y, p, n_boot=N_BOOT)})
    for model in MODELS3:
        for a, b in (("B0", "B1"), ("B1", "B2"), ("B0", "B2"), ("B1", "B2p"), ("B2p", "B2")):
            deltas.append({**base, "model": model, "comparison": f"{b} vs {a}", **paired_delta(y, out["proba"][(a, model)], out["proba"][(b, model)])})
    hh = []
    for stage, sv in out["shap"].items():
        signed = sv.mean(axis=0)
        for c, k in enumerate(CLASS_KEYS):
            d = pd.DataFrame(signed[:, c, :], columns=ALL_FEATURES)
            d.insert(0, "UID", g.UID.astype(str).to_numpy())
            hh.append(d.assign(label=label, country=country, stage=stage, shap_class=k))
    print(f"{label} {country}: {time.time() - t0:.0f}s", flush=True)
    return {"metrics": metrics, "deltas": deltas, "hh": pd.concat(hh), **{k: out[k].assign(**base) for k in ("importance", "selection", "curve")}}


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def fig_performance(m):
    fig, axes = plt.subplots(len(MODELS3), len(LABELS), figsize=(17, 11), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for r, model in enumerate(MODELS3):
        for ax, label in zip(axes[r], LABELS):
            style(ax)
            for ci, c in enumerate(COUNTRIES):
                for si, s in enumerate(STAGES):
                    row = m[(m.label == label) & (m.model == model) & (m.country == c) & (m.stage == s)].iloc[0]
                    x = ci + (si - 1.5) * 0.19
                    ax.errorbar(x, row.balanced_accuracy, yerr=[[row.balanced_accuracy - row.balanced_accuracy_lo],
                                                               [row.balanced_accuracy_hi - row.balanced_accuracy]],
                                fmt=MODEL_MARKER[model], color=STAGE_COLOR[s], ms=5, capsize=2, lw=1, label=s if ci == 0 else None)
                ch = m[(m.label == label) & (m.model == "chance") & (m.country == c)].iloc[0]
                ax.hlines(ch.balanced_accuracy, ci - 0.4, ci + 0.4, color=MUTED, ls=":", lw=1)
            ax.set_xticks(range(len(COUNTRIES)), COUNTRIES)
            ax.set_title(f"{MODEL_NAMES[model]} -- {LABEL_NAMES[label]}", fontsize=9, loc="left")
            ax.set_ylim(0.2, 0.9)
        axes[r, 0].set_ylabel("balanced accuracy (dotted = chance)", fontsize=8.5, color=INK2)
    axes[0, 0].legend(frameon=False, fontsize=8, title="stage", title_fontsize=8, loc="upper left")
    fig.suptitle("B0 (Van Hove) -> B1 (+ selected variables) -> B2 (+ selected climate); B2p = placebo climate; "
                 "repeated 5-fold CV, 95% household-bootstrap CI", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96)); save_figure(fig, FIG / "stages_performance"); plt.close(fig)


def fig_selection(selt):
    top = (selt[selt.stage == "B1"].groupby("feature").selection_rate.mean().sort_values(ascending=False).head(20).index.tolist())
    fig, axes = plt.subplots(2, len(LABELS), figsize=(17, 12), gridspec_kw={"height_ratios": [20, len(B2_CANDIDATES)]})
    for col, label in enumerate(LABELS):
        for row, (stage, feats) in enumerate((("B1", top), ("B2", B2_CANDIDATES))):
            d = selt[(selt.label == label) & (selt.stage == stage)].pivot(index="feature", columns="country", values="selection_rate")
            d = d.reindex(index=feats, columns=COUNTRIES)
            ax = axes[row, col]
            ax.imshow(d.to_numpy(float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax.set_yticks(range(len(feats)), [FEATURE_NAMES[f] for f in feats], fontsize=7)
            ax.set_xticks(range(len(COUNTRIES)), [c[:3] for c in COUNTRIES], fontsize=8)
            for (i, j), v in np.ndenumerate(d.to_numpy(float)):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6.5, color="white" if v > 0.6 else INK)
            ax.set_title(f"{LABEL_NAMES[label]}\n{stage}: share of 15 folds selected", fontsize=8.5, loc="left")
    fig.suptitle("Which candidates the attention + SHAP selector keeps (B1: the 20 most often selected of 82; B2: all 18 climate indices)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); save_figure(fig, FIG / "stages_selection_frequency"); plt.close(fig)


def fig_curve(curve):
    fig, axes = plt.subplots(2, len(LABELS), figsize=(17, 7.5), sharey="row")
    colors = {"Bulgaria": "#2a78d6", "Hungary": "#eb6834", "Italy": "#1baf7a", "Poland": "#eda100"}
    for col, label in enumerate(LABELS):
        for row, stage in enumerate(("B1", "B2")):
            ax = axes[row, col]; style(ax, "both")
            d = curve[(curve.label == label) & (curve.stage == stage)]
            for c in COUNTRIES:
                g = d[d.country == c]
                mm = g.groupby("k").inner_f1.mean()
                ax.plot(mm.index, mm.values, marker="o", ms=3, color=colors[c], lw=1.4, label=c)
                ax.text(0.99, 0.04 + 0.07 * COUNTRIES.index(c), f"{c[:3]}: median k = {g[g.chosen].k.median():.0f}",
                        transform=ax.transAxes, ha="right", fontsize=7, color=colors[c])
            ax.set_title(f"{LABEL_NAMES[label]} -- {stage}", fontsize=9, loc="left")
            ax.set_xlabel(f"number of {stage} candidates added (best-ranked first)", fontsize=8)
        axes[row, 0].set_ylabel("inner-CV macro-F1", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("How many candidates to add: the curve, and the number kept by the one-standard-error rule", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save_figure(fig, FIG / "stages_selection_curve"); plt.close(fig)


def fig_by_model(imp):
    """Top selected B1 features by each model's own held-out importance, B1 stage, averaged over labels and countries."""
    d = imp[(imp.stage == "B1") & ~imp.feature.isin(B0_FEATURES)]
    fig, axes = plt.subplots(1, len(MODELS3), figsize=(16, 6)); fig.patch.set_facecolor(SURFACE)
    for ax, model in zip(axes, MODELS3):
        style(ax, "x")
        s = d[d.model == model].groupby("feature").importance_share.mean().sort_values().tail(15)
        ax.barh([FEATURE_NAMES[f] for f in s.index], s.values, color="#2a78d6")
        ax.set_title(f"{MODEL_NAMES[model]} ({'attention' if model == 'ft_transformer' else 'mean |SHAP|'})", fontsize=10, loc="left")
        ax.set_xlabel("share of the model's importance (selected B1 features beyond B0)", fontsize=8)
    fig.suptitle("Selected features as seen by the three models (stage B1; all labels and countries)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save_figure(fig, FIG / "stages_selected_by_model"); plt.close(fig)


def fig_beeswarm(hh, dev, label):
    import shap
    fig = plt.figure(figsize=(20, 7.5))
    for i, c in enumerate(COUNTRIES):
        d = hh[(hh.label == label) & (hh.stage == "B2") & (hh.country == c) & (hh.shap_class == "double_risk")]
        feats = [f for f in ALL_FEATURES if d[f].abs().sum() > 0]
        x = dev.set_index(dev.UID.astype(str)).loc[d.UID, feats].copy()
        for f in feats:
            if not pd.api.types.is_numeric_dtype(x[f]):
                x[f] = x[f].astype("category").cat.codes.replace(-1, np.nan)
        plt.subplot(1, len(COUNTRIES), i + 1)
        shap.summary_plot(d[feats].to_numpy(), x.astype(float).to_numpy(), feature_names=[FEATURE_NAMES[f] for f in feats],
                          max_display=12, show=False, plot_size=None, color_bar=(i == len(COUNTRIES) - 1))
        plt.title(c, fontsize=10, loc="left"); plt.xlabel("SHAP value for Double Risk (CatBoost, log-odds)", fontsize=8)
    plt.suptitle(f"{LABEL_NAMES[label]}: what pushes a household towards Double Risk (stage B2, held-out folds)", fontsize=11, x=0.01, ha="left")
    plt.tight_layout(rect=(0, 0, 1, 0.94)); save_figure(plt.gcf(), FIG / f"stages_shap_beeswarm_{label}"); plt.close("all")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--jobs", type=int, default=4, help="parallel cells; each uses the GPU (more than 4 CUDA processes crashed under WSL)")
    a = p.parse_args()
    dev = load_development_frame()
    if [f for f in B2_CANDIDATES if f not in dev.columns]:
        raise SystemExit("B2 climate missing: run scripts/acquire_openmeteo_b2.py, acquire_era5_normals.py, fuse_openmeteo_b2.py")
    jobs, support = [], []
    for label, kind in LABELS.items():
        lab = add_labels(dev, cost_kind=kind)
        sup = viability(support_table(lab)).assign(label=label)
        support.append(sup)
        ok = dict(zip(sup.country, sup.four_class_ok))
        jobs += [(label, c, lab[lab.country == c].reset_index(drop=True), bool(ok[c])) for c in COUNTRIES]
    res = Parallel(n_jobs=a.jobs)(delayed(one_cell)(lab, c, g, ok, a.repeats) for lab, c, g, ok in jobs)
    metrics = pd.DataFrame([r for x in res for r in x["metrics"]])
    deltas = pd.DataFrame([r for x in res for r in x["deltas"]])
    imp, sel, curve = (pd.concat([x[k] for x in res], ignore_index=True) for k in ("importance", "selection", "curve"))
    metrics.to_csv(TAB / "table_stages_metrics.csv", index=False)
    rep = metrics[["label", "country", "four_class_ok", "stage", "model"] + REPORT + [f"{k}_lo" for k in ("balanced_accuracy", "f1_weighted")]
                  + [f"{k}_hi" for k in ("balanced_accuracy", "f1_weighted")]]
    rep.to_csv(TAB / "table_stages_report.csv", index=False)
    deltas.to_csv(TAB / "table_stages_deltas.csv", index=False)
    pd.concat(support).to_csv(TAB / "table_stages_support.csv", index=False)
    selt = (sel.groupby(["label", "country", "stage", "feature"])
            .agg(selection_rate=("selected", "mean"), mean_rank=("rank", "mean"), attention=("attention", "mean"),
                 shap_catboost=("shap_catboost", "mean"), shap_xgboost=("shap_xgboost", "mean"), mean_k=("k", "mean")).reset_index())
    selt["feature_name"] = selt.feature.map(FEATURE_NAMES)
    selt.to_csv(TAB / "table_stages_selection.csv", index=False)
    imp["importance_share"] = imp.importance / imp.groupby(["label", "country", "repeat", "fold", "stage", "model"]).importance.transform("sum")
    impt = imp.groupby(["label", "country", "stage", "model", "feature"]).agg(
        importance=("importance", "mean"), importance_share=("importance_share", "mean"), folds_selected=("fold", "size")).reset_index()
    impt["feature_name"] = impt.feature.map(FEATURE_NAMES)
    impt.to_csv(TAB / "table_stages_selected_importance.csv", index=False)
    curve.to_csv(TAB / "table_stages_selection_curve.csv", index=False)
    ART.mkdir(parents=True, exist_ok=True)
    hh = pd.concat([x["hh"] for x in res], ignore_index=True)
    hh.to_parquet(ART / "stages_oof_shap.parquet", index=False)
    fig_performance(metrics); fig_selection(selt); fig_curve(curve); fig_by_model(imp)
    for label in LABELS:
        fig_beeswarm(hh, dev, label)
    pd.set_option("display.width", 250)
    print(metrics[metrics.model != "chance"].pivot_table(index=["label", "country"], columns=["model", "stage"], values="balanced_accuracy").round(3).to_string())


if __name__ == "__main__":
    main()
