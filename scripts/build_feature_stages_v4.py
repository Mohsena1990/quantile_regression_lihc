"""Version-4 study: pooled four-country model, Models A / B / C, stages B0 -> B1 (two-level groups) -> B2 (top-3 climate) with a
placebo, per-country reading, and maps of each variable's effect (lib/identification/stages_v4.py).

Labels (lib/identification/lines_v4.py, thresholds from scripts/build_threshold_moo.py): Van Hove (bracket 3, 80th percentile --
also the optimised choice for the country-percentile family), Quantile Regression on need (optimised), the same line without
climate (fair test of climate predictors), and the energy burden line (optimised). Development households only; repeated 5-fold CV
(x3) stratified by country x class; 95% household-bootstrap CIs.

Outputs (outputs/tables):
  table_v4_metrics.csv          every metric with CIs: label x scope (pooled, each country) x model set (A/B/C) x stage x classifier,
                                plus per-country B0 models (model_set "country_model")
  table_v4_report.csv           headline metrics: per-class recall and precision, weighted F1, balanced accuracy, accuracy, ROC AUC
  table_v4_deltas.csv           paired gains per scope and classifier (C: B1-B0, B2-B1, B2-B2p, B0-A; pooled B0 vs country models)
  table_v4_selection_groups.csv, table_v4_selection_features.csv, table_v4_climate_top3.csv
  table_v4_importance.csv       held-out importance share per classifier (SHAP for the trees, attention for FT-Transformer)
  table_v4_country_effects.csv, table_v4_region_effects.csv   mean SHAP towards Double Risk (CatBoost, C model) per country / region
(outputs/figures) v4_models_abc, v4_pooled_vs_country, v4_effect_maps_<label>, v4_value_maps, v4_country_effects
"""
from __future__ import annotations
import argparse, pickle, sys, time, warnings
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
from lib.features.indices import NAMES as INDEX_NAMES
from lib.geography.climate_regions import region_geometries
from lib.identification.data import COUNTRIES, load_development_frame
from lib.identification.lines_v4 import LABEL_NAMES, study_labels
from lib.identification.metrics import CLASS_KEYS
from lib.identification.models3 import MODELS3
from lib.identification.protocol import summarise
from lib.identification.stages import FEATURE_NAMES as OLD_NAMES, paired_delta
from lib.identification.stages_v4 import B1_GROUPS, STAGES, UNIVERSE, outer_splits, run_fold
from lib.reporting.figures import save_figure

TAB, FIG, ART = ROOT / "outputs/tables", ROOT / "outputs/figures", ROOT / "artifacts/identification"
LABELS = ("vh", "needqr", "needqr_noclim", "burden")
N_BOOT = 200
N_REPEATS = 3
NAMES = {**OLD_NAMES, **INDEX_NAMES, "country": "country", "H11B": "dishwasher age"}
REPORT = (["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "auc_macro_ovr"]
          + [f"{m}_{k}" for m in ("recall", "precision", "auc") for k in CLASS_KEYS])
MODEL_NAMES = {"catboost": "CatBoost", "xgboost": "XGBoost", "ft_transformer": "FT-Transformer"}


def job(label, frame, fold_id, tr, te):
    t0 = time.time()
    out = run_fold(frame, tr, te, fold_id, fold_id // 5)
    print(f"{label} fold {fold_id}: {time.time() - t0:.0f}s", flush=True)
    return label, out


def assemble(frame: pd.DataFrame, results: list[dict]) -> dict:
    n = len(frame)
    keys = results[0]["proba"].keys()
    proba = {k: np.zeros((N_REPEATS, n, 4)) for k in keys}
    cmod = {}
    shap = {st: np.zeros((N_REPEATS, n, 4, len(UNIVERSE)), dtype=np.float32) for st in ("B1", "B2")}
    for r in results:
        for k, p in r["proba"].items():
            proba[k][r["repeat"], r["te"]] = p
        for (c, clf), (idx, p) in r["country_b0"].items():
            cmod.setdefault(clf, np.zeros((N_REPEATS, n, 4)))[r["repeat"], idx] = p
        for st, block in r["shap"].items():
            shap[st][r["repeat"], r["te"]] = block
    return {"proba": proba, "country_model": cmod, "shap": shap}


def metrics_for(label, frame, A) -> tuple[list, list]:
    y = frame.y.to_numpy()
    scopes = {"pooled": np.arange(len(frame)), **{c: np.where(frame.country == c)[0] for c in COUNTRIES}}
    rows, deltas = [], []
    for scope, idx in scopes.items():
        base = {"label": label, "scope": scope, "n": len(idx)}
        for (mset, st, clf), p in A["proba"].items():
            rows.append({**base, "model_set": mset, "stage": st, "model": clf, **summarise(y[idx], p[:, idx], n_boot=N_BOOT)})
        for clf, p in A["country_model"].items():
            rows.append({**base, "model_set": "country_model", "stage": "B0", "model": clf, **summarise(y[idx], p[:, idx], n_boot=N_BOOT)})
        for clf in MODELS3:
            P = A["proba"]
            for name, a, b in (("C B1 vs B0", ("C", "B0"), ("C", "B1")), ("C B2 vs B1", ("C", "B1"), ("C", "B2")),
                               ("C B2 vs B2p", ("C", "B2p"), ("C", "B2")), ("C B0 vs A", ("A", "all"), ("C", "B0")),
                               ("C B2 vs A", ("A", "all"), ("C", "B2"))):
                deltas.append({**base, "model": clf, "comparison": name,
                               **paired_delta(y[idx], P[(*a, clf)][:, idx], P[(*b, clf)][:, idx])})
            deltas.append({**base, "model": clf, "comparison": "pooled B0 vs country B0",
                           **paired_delta(y[idx], A["country_model"][clf][:, idx], P[("C", "B0", clf)][:, idx])})
    return rows, deltas


def effect_tables(label, frame, A, selected: list[str]):
    """Mean signed SHAP towards Double Risk (CatBoost, C model, B2) per country and per region, for the selected features."""
    sv = A["shap"]["B2"].mean(axis=0)[:, CLASS_KEYS.index("double_risk"), :]
    d = pd.DataFrame(sv, columns=UNIVERSE)[selected]
    d["country"], d["climate_region"] = frame.country.to_numpy(), frame.climate_region.to_numpy()
    c = d.groupby("country")[selected].mean().reset_index().melt("country", var_name="feature", value_name="mean_shap_double_risk")
    r = d.groupby(["country", "climate_region"])[selected].mean().reset_index().melt(["country", "climate_region"], var_name="feature",
                                                                                     value_name="mean_shap_double_risk")
    return c.assign(label=label), r.assign(label=label)


def map_effects(label, reg: pd.DataFrame, feats: list[str]):
    geo = region_geometries(reg.climate_region.unique())
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.4 * nrow))
    for ax, f in zip(np.ravel(axes), feats):
        g = geo.merge(reg[reg.feature == f], left_on="NUTS_ID", right_on="climate_region")
        lim = float(np.nanmax(np.abs(g.mean_shap_double_risk))) or 1e-6
        g.plot(column="mean_shap_double_risk", ax=ax, cmap="RdBu_r", vmin=-lim, vmax=lim, legend=True, edgecolor="white",
               linewidth=0.2, legend_kwds={"shrink": 0.6})
        ax.set_axis_off(); ax.set_title(NAMES.get(f, f), fontsize=9.5, loc="left")
    for ax in np.ravel(axes)[len(feats):]:
        ax.set_axis_off()
    fig.suptitle(f"{LABEL_NAMES[label]}: how each selected variable pushes households towards (red) or away from (blue) Double Risk, "
                 "regional mean SHAP (CatBoost, pooled model, stage B2)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save_figure(fig, FIG / f"v4_effect_maps_{label}"); plt.close(fig)


def map_values(frame: pd.DataFrame, feats: list[str]):
    reg = frame.groupby("climate_region")[[f for f in feats if pd.api.types.is_numeric_dtype(frame[f])]].mean().reset_index()
    feats = [f for f in feats if f in reg.columns]
    geo = region_geometries(reg.climate_region).merge(reg, left_on="NUTS_ID", right_on="climate_region")
    ncol = 3
    nrow = int(np.ceil(len(feats) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.4 * ncol, 4.4 * nrow))
    for ax, f in zip(np.ravel(axes), feats):
        geo.plot(column=f, ax=ax, cmap="viridis", legend=True, edgecolor="white", linewidth=0.2, legend_kwds={"shrink": 0.6})
        ax.set_axis_off(); ax.set_title(NAMES.get(f, f), fontsize=9.5, loc="left")
    for ax in np.ravel(axes)[len(feats):]:
        ax.set_axis_off()
    fig.suptitle("Regional average of each mapped variable (development households)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save_figure(fig, FIG / "v4_value_maps"); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=4, help="more than 4 CUDA processes crash under WSL")
    a = p.parse_args()
    dev = load_development_frame()
    labs = {k: v for k, v in study_labels(dev).items() if k in LABELS}
    frames = {k: v.reset_index(drop=True) for k, v in labs.items()}
    jobs = [(lab, f, i, tr, te) for lab, f in frames.items() for i, (tr, te) in enumerate(outer_splits(f, n_repeats=N_REPEATS))]
    res = Parallel(n_jobs=a.jobs)(delayed(job)(*j) for j in jobs)
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / "v4_results.pkl", "wb") as fh:
        pickle.dump(res, fh)
    rows, deltas, sel, curves, metas, imps, ceff, reff = [], [], [], [], [], [], [], []
    for lab, frame in frames.items():
        R = [r for l, r in res if l == lab]
        A = assemble(frame, R)
        m, dl = metrics_for(lab, frame, A)
        rows += m; deltas += dl
        for r in R:
            sel += [{**s, "label": lab, "fold_id": r["fold_id"]} for s in r["selection"]]
            curves += [{**c, "label": lab, "fold_id": r["fold_id"]} for c in r["curves"]]
            imps += [{**i, "label": lab, "fold_id": r["fold_id"]} for i in r["importance"]]
            metas.append({"label": lab, "fold_id": r["fold_id"], **r["meta"]})
        b1_freq = pd.Series([f for r in R for f in r["meta"]["b1"] if f not in UNIVERSE[:8]]).value_counts()
        top3 = pd.Series([f for r in R for f in r["meta"]["b2_top3"]]).value_counts()
        mapped = list(b1_freq.index[:6]) + [f for f in top3.index[:3] if f not in b1_freq.index[:6]]
        c, rg = effect_tables(lab, frame, A, mapped)
        ceff.append(c); reff.append(rg)
        map_effects(lab, rg, mapped)
    metrics, deltas = pd.DataFrame(rows), pd.DataFrame(deltas)
    metrics.to_csv(TAB / "table_v4_metrics.csv", index=False)
    metrics[["label", "scope", "model_set", "stage", "model", "n"] + REPORT].to_csv(TAB / "table_v4_report.csv", index=False)
    deltas.to_csv(TAB / "table_v4_deltas.csv", index=False)
    sel, curves, imps, metas = pd.DataFrame(sel), pd.DataFrame(curves), pd.DataFrame(imps), pd.DataFrame(metas)
    sel.groupby(["label", "stage", "feature"]).agg(selection_rate=("selected", "mean"), mean_rank=("rank", "mean"),
                                                    weight_attention=("weight_attention", "mean")).reset_index().assign(
        feature_name=lambda t: t.feature.map(NAMES)).to_csv(TAB / "table_v4_selection_features.csv", index=False)
    grp = metas.explode("groups").groupby(["label", "groups"]).size().div(metas.groupby("label").size(), level="label")
    grp.rename("selection_rate").reset_index().to_csv(TAB / "table_v4_selection_groups.csv", index=False)
    t3 = metas.explode("b2_top3").groupby(["label", "b2_top3"]).size().div(metas.groupby("label").size(), level="label")
    t3 = t3.rename("top3_rate").reset_index().rename(columns={"b2_top3": "feature"})
    t3["feature_name"] = t3.feature.map(NAMES)
    t3.merge(metas.groupby("label").b2_k_one_se.median().rename("median_k_one_se"), on="label").to_csv(TAB / "table_v4_climate_top3.csv", index=False)
    curves.to_csv(TAB / "table_v4_selection_curves.csv", index=False)
    imps["share"] = imps.importance / imps.groupby(["label", "fold_id", "model_set", "stage", "model"]).importance.transform("sum")
    imps.groupby(["label", "model_set", "stage", "model", "feature"]).share.mean().reset_index().assign(
        feature_name=lambda t: t.feature.map(NAMES)).to_csv(TAB / "table_v4_importance.csv", index=False)
    ceff, reff = pd.concat(ceff), pd.concat(reff)
    ceff.assign(feature_name=ceff.feature.map(NAMES)).to_csv(TAB / "table_v4_country_effects.csv", index=False)
    reff.to_csv(TAB / "table_v4_region_effects.csv", index=False)
    map_values(load_development_frame(), sorted(set(ceff.feature)))
    fig_abc(metrics); fig_pooled(metrics); fig_country_effects(ceff)
    pd.set_option("display.width", 250)
    x = metrics[(metrics.scope == "pooled") & (metrics.model_set != "country_model")]
    print(x.pivot_table(index=["label", "model"], columns=["model_set", "stage"], values="balanced_accuracy").round(3).to_string())


def fig_abc(m):
    x = m[(m.scope == "pooled") & m.model_set.isin(["A", "B", "C"])]
    fig, axes = plt.subplots(1, len(LABELS), figsize=(17, 4.6), sharey=True)
    order = [("A", "all"), ("B", "B0"), ("B", "B1"), ("B", "B2"), ("C", "B0"), ("C", "B1"), ("C", "B2"), ("C", "B2p")]
    for ax, lab in zip(axes, LABELS):
        for j, clf in enumerate(MODELS3):
            v = [x[(x.label == lab) & (x.model == clf) & (x.model_set == s) & (x.stage == st)].balanced_accuracy.iloc[0] for s, st in order]
            ax.plot(range(len(order)), v, marker="os^"[j], label=MODEL_NAMES[clf])
        ax.set_xticks(range(len(order)), [f"{s} {st}" if s != "A" else "A income" for s, st in order], rotation=45, fontsize=7.5)
        ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left"); ax.grid(color="#e1e0d9", lw=0.6)
    axes[0].set_ylabel("balanced accuracy (pooled)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Models A (income only), B (features without income), C (income + features) at each stage; pooled four-country model",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save_figure(fig, FIG / "v4_models_abc"); plt.close(fig)


def fig_pooled(m):
    x = m[(m.scope != "pooled") & (m.stage == "B0") & m.model_set.isin(["C", "country_model"])]
    fig, axes = plt.subplots(1, len(LABELS), figsize=(17, 4), sharey=True)
    for ax, lab in zip(axes, LABELS):
        for j, (ms, name) in enumerate((("C", "pooled model"), ("country_model", "per-country model"))):
            v = x[(x.label == lab) & (x.model_set == ms) & (x.model == "catboost")].set_index("scope").reindex(COUNTRIES).balanced_accuracy
            ax.bar(np.arange(4) + (j - 0.5) * 0.38, v, width=0.36, label=name)
        ax.set_xticks(range(4), COUNTRIES, fontsize=8); ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left")
    axes[0].set_ylabel("balanced accuracy, B0, CatBoost"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Is pooling right? One model for all four countries vs one model per country (scored on each country's households)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9)); save_figure(fig, FIG / "v4_pooled_vs_country"); plt.close(fig)


def fig_country_effects(c):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 5.5))
    for ax, lab in zip(axes, LABELS):
        d = c[c.label == lab].pivot(index="feature", columns="country", values="mean_shap_double_risk").reindex(columns=COUNTRIES)
        lim = float(np.nanmax(np.abs(d.to_numpy()))) or 1e-6
        ax.imshow(d.to_numpy(float), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_yticks(range(len(d)), [NAMES.get(f, f) for f in d.index], fontsize=7.5)
        ax.set_xticks(range(4), [k[:3] for k in COUNTRIES], fontsize=8)
        for (i, j), v in np.ndenumerate(d.to_numpy(float)):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left")
    fig.suptitle("Mean SHAP towards Double Risk per country (red = pushes towards energy poverty)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save_figure(fig, FIG / "v4_country_effects"); plt.close(fig)


if __name__ == "__main__":
    main()
