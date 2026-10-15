"""Version-5 study: pooled five-country model (Bulgaria, Hungary, Italy, Poland + Spain), stages B0 -> B1 -> B2 (+ placebo),
Models A / B / C, and the nine improvements (lib/identification/stages_v5.py):

  1 bill        predict the energy bill, then compare it with the household's line
  2 augmented   households with a missing energy cost added as imputed training households; households with a bill but no
                income used by the bill model
  3 region      the region (NUTS2) as one B1 variable, replacing the regional statistics
  4 robust      borderline and likely mislabelled households down-weighted
  5 ensemble    average of CatBoost, XGBoost and FT-Transformer ("ensemble"), and of the best four ("ensemble_plus": CatBoost,
                tuned XGBoost, FT-Transformer, bill model)
  6 tuned       Optuna-tuned XGBoost
  7 partial modules  environmental attitudes, energy-saving behaviour and energy-literacy items as a B1 group
  8 targeting curves  share of Double-Risk households reached when a scheme helps the top k% of households ranked by the model
  9 "not sure"  the least confident households flagged for a follow-up check; accuracy on the rest

Labels: lib/identification/lines_v5.py with thresholds from scripts/build_threshold_moo.py --v5. Development households only;
repeated 5-fold CV (x3) stratified by country x class; 95% household-bootstrap CIs; paired bootstrap for differences.

Outputs (outputs/tables): table_v5_metrics.csv, table_v5_report.csv, table_v5_deltas.csv, table_v5_sample.csv,
table_v5_variant_info.csv, table_v5_targeting.csv, table_v5_targeting_rule.csv, table_v5_abstention.csv,
table_v5_selection_groups.csv, table_v5_selection_features.csv, table_v5_climate_top3.csv, table_v5_selection_curves.csv,
table_v5_importance.csv, table_v5_country_effects.csv, table_v5_region_effects.csv
(outputs/figures) v5_models_abc, v5_variants, v5_targeting, v5_abstention, v5_effect_maps_<label>, v5_value_maps, v5_country_effects
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
from lib.features.partial_modules import PARTIAL_NAMES
from lib.geography.climate_regions import region_geometries
from lib.identification.data import COUNTRIES_V5 as COUNTRIES, load_development_frame
from lib.identification.lines_v4 import LABEL_NAMES
from lib.identification.lines_v5 import cost_only, study_labels_v5, unlabelled
from lib.identification.metrics import CLASS_KEYS
from lib.identification.models3 import MODELS3
from lib.identification.protocol import summarise
from lib.identification.stages import FEATURE_NAMES as OLD_NAMES, paired_delta
from lib.identification.stages_v5 import B1_GROUPS, STAGES, UNIVERSE, outer_splits, run_fold
from lib.reporting.figures import save_figure

TAB, FIG, ART = ROOT / "outputs/tables", ROOT / "outputs/figures", ROOT / "artifacts/identification"
LABELS = ("vh", "needqr", "needqr_noclim", "burden")
N_BOOT = 200
N_REPEATS = 3
DR = CLASS_KEYS.index("double_risk")
NAMES = {**OLD_NAMES, **INDEX_NAMES, **PARTIAL_NAMES, "country": "country", "H11B": "dishwasher age", "region": "region (NUTS2)"}
REPORT = (["accuracy", "balanced_accuracy", "f1_weighted", "f1_macro", "auc_macro_ovr"]
          + [f"{m}_{k}" for m in ("recall", "precision", "auc") for k in CLASS_KEYS])
MODEL_NAMES = {"catboost": "CatBoost", "xgboost": "XGBoost", "ft_transformer": "FT-Transformer", "ensemble": "ensemble (3 models)",
               "xgboost_regression": "bill model", "all": "ensemble_plus"}
# model keys followed in the policy analyses (targeting, "not sure")
POLICY = {("A", "all", "catboost"): "income only (A)", ("C", "B0", "catboost"): "Van Hove B0 (CatBoost)",
          ("C", "B1", "catboost"): "B1 CatBoost", ("C", "B1", "ensemble"): "B1 ensemble", ("V", "tuned", "xgboost"): "B1 tuned XGBoost",
          ("V", "bill", "xgboost_regression"): "B1 bill model", ("V", "ensemble_plus", "all"): "B1 ensemble_plus"}
TARGET_K = np.round(np.arange(0.05, 1.001, 0.05), 2)
COVERAGE = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
REACH = 0.70


def job(label, frame, unl, conly, spec, fold_id, tr, te):
    t0 = time.time()
    out = run_fold(frame, unl, conly, spec, tr, te, fold_id, fold_id // 5)
    print(f"{label} fold {fold_id}: {time.time() - t0:.0f}s", flush=True)
    return label, out


def assemble(frame: pd.DataFrame, results: list[dict]) -> dict:
    n = len(frame)
    proba = {k: np.zeros((N_REPEATS, n, 4)) for k in results[0]["proba"]}
    shap = {st: np.zeros((N_REPEATS, n, 4, len(UNIVERSE)), dtype=np.float32) for st in ("B1", "B2")}
    for r in results:
        for k, p in r["proba"].items():
            proba[k][r["repeat"], r["te"]] = p
        for st, block in r["shap"].items():
            shap[st][r["repeat"], r["te"]] = block
    # idea 5: averages of the three classifiers, and of the best four
    for mset, st in {(m, s) for m, s, _ in list(proba)}:
        if all((mset, st, c) in proba for c in MODELS3):
            proba[(mset, st, "ensemble")] = np.mean([proba[(mset, st, c)] for c in MODELS3], axis=0)
    proba[("V", "ensemble_plus", "all")] = np.mean([proba[("C", "B1", "catboost")], proba[("V", "tuned", "xgboost")],
                                                     proba[("C", "B1", "ft_transformer")], proba[("V", "bill", "xgboost_regression")]], axis=0)
    return {"proba": proba, "shap": shap}


COMPARISONS = ([("C B1 vs B0", ("C", "B0", c), ("C", "B1", c)) for c in MODELS3 + ("ensemble",)]
               + [("C B2 vs B1", ("C", "B1", c), ("C", "B2", c)) for c in MODELS3]
               + [("C B2 vs B2p", ("C", "B2p", c), ("C", "B2", c)) for c in MODELS3]
               + [("C B0 vs A", ("A", "all", c), ("C", "B0", c)) for c in MODELS3]
               + [(f"{v} vs C B1", ("C", "B1", c), ("V", v, c)) for v in ("robust", "augmented") for c in MODELS3 + ("ensemble",)]
               + [("bill vs C B1 XGBoost", ("C", "B1", "xgboost"), ("V", "bill", "xgboost_regression")),
                  ("tuned vs C B1 XGBoost", ("C", "B1", "xgboost"), ("V", "tuned", "xgboost"))]
               + [("ensemble vs C B1", ("C", "B1", c), ("C", "B1", "ensemble")) for c in MODELS3]
               + [("ensemble_plus vs C B1", ("C", "B1", c), ("V", "ensemble_plus", "all")) for c in MODELS3 + ("ensemble",)])


def scopes(frame: pd.DataFrame) -> dict:
    return {"pooled": np.arange(len(frame)), **{c: np.where(frame.country == c)[0] for c in COUNTRIES}}


def metrics_for(label, frame, A, only: str | None = None) -> tuple[list, list]:
    y = frame.y.to_numpy()
    rows, deltas = [], []
    for scope, idx in scopes(frame).items():
        if only is not None and scope != only:
            continue
        base = {"label": label, "scope": scope, "n": len(idx)}
        for (mset, st, clf), p in A["proba"].items():
            rows.append({**base, "model_set": mset, "stage": st, "model": clf, **summarise(y[idx], p[:, idx], n_boot=N_BOOT)})
        for name, a, b in COMPARISONS:
            deltas.append({**base, "comparison": name, "a": "|".join(a), "b": "|".join(b), "model": b[2],
                           **paired_delta(y[idx], A["proba"][a][:, idx], A["proba"][b][:, idx])})
    return rows, deltas


def targeting(label, frame, A) -> pd.DataFrame:
    """Idea 8: rank households by P(Double Risk); share of all Double-Risk households among the top k% (recall) and share of
    the top k% that are Double Risk (precision). Within each country = a national scheme. Averaged over the CV repeats."""
    y = frame.y.to_numpy()
    rows = []
    for key, name in POLICY.items():
        P = A["proba"][key]
        for scope, idx in scopes(frame).items():
            dr = y[idx] == DR
            for k in TARGET_K:
                n_top = max(1, int(round(k * len(idx))))
                rec, prec = [], []
                for r in range(N_REPEATS):
                    top = np.argsort(-P[r, idx, DR], kind="stable")[:n_top]
                    rec.append(dr[top].sum() / max(1, dr.sum())); prec.append(dr[top].mean())
                rows.append({"label": label, "scope": scope, "model": name, "top_share": float(k), "households_helped": n_top,
                             "double_risk_reached": float(np.mean(rec)), "precision": float(np.mean(prec)),
                             "double_risk_share": float(dr.mean())})
    return pd.DataFrame(rows)


def targeting_rule(t: pd.DataFrame) -> pd.DataFrame:
    """Smallest share of households to help so that at least REACH of Double-Risk households are reached."""
    out = []
    for (lab, scope, model), g in t.groupby(["label", "scope", "model"]):
        ok = g[g.double_risk_reached >= REACH].sort_values("top_share")
        r = ok.iloc[0] if len(ok) else g.sort_values("top_share").iloc[-1]
        out.append({"label": lab, "scope": scope, "model": model, "goal": f"reach {int(REACH * 100)}% of Double Risk",
                    "help_top_share": r.top_share, "households_helped": r.households_helped,
                    "double_risk_reached": r.double_risk_reached, "precision": r.precision,
                    "lift_over_random": r.precision / r.double_risk_share if r.double_risk_share else np.nan})
    return pd.DataFrame(out)


def abstention(label, frame, A) -> pd.DataFrame:
    """Idea 9: the least confident households (lowest top-class probability) are flagged "not sure" for a follow-up check; metrics
    on the households the model keeps. Averaged over the CV repeats."""
    from lib.identification.metrics import confusion
    y = frame.y.to_numpy()
    rows = []
    for key in [("C", "B1", "catboost"), ("C", "B1", "ensemble"), ("V", "tuned", "xgboost"), ("V", "ensemble_plus", "all")]:
        P = A["proba"][key]
        for scope, idx in scopes(frame).items():
            for cov in COVERAGE:
                vals = []
                for r in range(N_REPEATS):
                    p = P[r, idx]
                    conf = p.max(axis=1)
                    keep = np.argsort(-conf, kind="stable")[:max(1, int(round(cov * len(idx))))]
                    yk, pk = y[idx][keep], p[keep].argmax(1)
                    cm = confusion(yk, pk)
                    rec = np.divide(np.diag(cm), cm.sum(1), out=np.full(4, np.nan), where=cm.sum(1) > 0)
                    flagged = np.setdiff1d(np.arange(len(idx)), keep)
                    vals.append({"balanced_accuracy": float(np.nanmean(rec)), "accuracy": float((yk == pk).mean()),
                                 "recall_double_risk": float(rec[DR]),
                                 **{f"not_sure_share_{CLASS_KEYS[j]}": float((y[idx][flagged] == j).sum() / max(1, (y[idx] == j).sum()))
                                    for j in range(4)}})
                rows.append({"label": label, "scope": scope, "model": POLICY.get(key, "|".join(key)), "coverage": cov,
                             "not_sure_share": 1 - cov, **pd.DataFrame(vals).mean().to_dict()})
    return pd.DataFrame(rows)


def effect_tables(label, frame, A, selected: list[str]):
    sv = A["shap"]["B2"].mean(axis=0)[:, DR, :]
    d = pd.DataFrame(sv, columns=UNIVERSE)[selected]
    d["country"], d["climate_region"] = frame.country.to_numpy(), frame.climate_region.to_numpy()
    c = d.groupby("country")[selected].mean().reset_index().melt("country", var_name="feature", value_name="mean_shap_double_risk")
    r = d.groupby(["country", "climate_region"])[selected].mean().reset_index().melt(["country", "climate_region"], var_name="feature",
                                                                                     value_name="mean_shap_double_risk")
    return c.assign(label=label), r.assign(label=label)


def _grid(n: int, ncol: int = 3):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.8 * ncol, 4.2 * nrow))
    return fig, np.ravel(axes)


def map_effects(label, reg: pd.DataFrame, feats: list[str]):
    geo = region_geometries(reg.climate_region.unique())
    fig, axes = _grid(len(feats))
    for ax, f in zip(axes, feats):
        g = geo.merge(reg[reg.feature == f], left_on="NUTS_ID", right_on="climate_region")
        lim = float(np.nanmax(np.abs(g.mean_shap_double_risk))) or 1e-6
        g.plot(column="mean_shap_double_risk", ax=ax, cmap="RdBu_r", vmin=-lim, vmax=lim, legend=True, edgecolor="white",
               linewidth=0.2, legend_kwds={"shrink": 0.6})
        ax.set_xlim(-10, 30); ax.set_ylim(35, 55.5)                 # mainland view (the Canary Islands are left off the map)
        ax.set_axis_off(); ax.set_title(NAMES.get(f, f), fontsize=9.5, loc="left")
    for ax in axes[len(feats):]:
        ax.set_axis_off()
    fig.suptitle(f"{LABEL_NAMES[label]}: how each selected variable pushes households towards (red) or away from (blue) Double Risk, "
                 "regional mean SHAP (CatBoost, pooled five-country model, stage B2)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save_figure(fig, FIG / f"v5_effect_maps_{label}"); plt.close(fig)


def map_values(frame: pd.DataFrame, feats: list[str]):
    reg = frame.groupby("climate_region")[[f for f in feats if pd.api.types.is_numeric_dtype(frame[f])]].mean().reset_index()
    feats = [f for f in feats if f in reg.columns]
    geo = region_geometries(reg.climate_region).merge(reg, left_on="NUTS_ID", right_on="climate_region")
    fig, axes = _grid(len(feats))
    for ax, f in zip(axes, feats):
        geo.plot(column=f, ax=ax, cmap="viridis", legend=True, edgecolor="white", linewidth=0.2, legend_kwds={"shrink": 0.6},
                 missing_kwds={"color": "#dddddd"})
        ax.set_xlim(-10, 30); ax.set_ylim(35, 55.5)
        ax.set_axis_off(); ax.set_title(NAMES.get(f, f), fontsize=9.5, loc="left")
    for ax in axes[len(feats):]:
        ax.set_axis_off()
    fig.suptitle("Regional average of each mapped variable (development households; grey = not asked / no data)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); save_figure(fig, FIG / "v5_value_maps"); plt.close(fig)


def sample_table(dev: pd.DataFrame, labs: dict) -> pd.DataFrame:
    rows = []
    for lab, (frame, spec) in labs.items():
        u, co = unlabelled(dev, spec["family"], spec["cutoff"]), cost_only(dev)
        for c in COUNTRIES:
            f = frame[frame.country == c]
            rows.append({"label": lab, "country": c, "development_households": int((dev.country == c).sum()),
                         "labelled": len(f), **{f"class_{CLASS_KEYS[j]}": int((f.y == j).sum()) for j in range(4)},
                         "imputed_cost_missing": int((u.country == c).sum()), "bill_only_income_missing": int((co.country == c).sum())})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jobs", type=int, default=4, help="more than 4 CUDA processes crash under WSL")
    p.add_argument("--report-only", action="store_true", help="rebuild the tables and figures from artifacts/identification/v5_results.pkl")
    a = p.parse_args()
    dev = load_development_frame(extended=True)
    labs = {k: v for k, v in study_labels_v5(dev).items() if k in LABELS}
    sample_table(dev, labs).to_csv(TAB / "table_v5_sample.csv", index=False)
    if a.report_only:
        with open(ART / "v5_results.pkl", "rb") as fh:
            return report(dev, labs, pickle.load(fh))
    conly = cost_only(dev)
    jobs = []
    for lab, (frame, spec) in labs.items():
        unl = unlabelled(dev, spec["family"], spec["cutoff"])
        jobs += [(lab, frame, unl, conly, spec, i, tr, te) for i, (tr, te) in enumerate(outer_splits(frame, n_repeats=N_REPEATS))]
    print(f"{len(jobs)} fold jobs", flush=True)
    res = Parallel(n_jobs=a.jobs)(delayed(job)(*j) for j in jobs)
    ART.mkdir(parents=True, exist_ok=True)
    with open(ART / "v5_results.pkl", "wb") as fh:
        pickle.dump(res, fh)
    report(dev, labs, res)


def report(dev, labs, res):
    rows, deltas, sel, curves, metas, imps, ceff, reff, tgt, abst, vinfo = [], [], [], [], [], [], [], [], [], [], []
    assembled = {lab: assemble(frame, [r for l, r in res if l == lab]) for lab, (frame, _) in labs.items()}
    # the bootstrap tables are the slow part: one process per label and scope
    parts = Parallel(n_jobs=24)(delayed(metrics_for)(lab, frame, {"proba": assembled[lab]["proba"]}, scope)
                               for lab, (frame, _) in labs.items() for scope in scopes(frame))
    for m, dl in parts:
        rows += m; deltas += dl
    for lab, (frame, spec) in labs.items():
        R = [r for l, r in res if l == lab]
        A = assembled[lab]
        tgt.append(targeting(lab, frame, A)); abst.append(abstention(lab, frame, A))
        for r in R:
            sel += [{**s, "label": lab, "fold_id": r["fold_id"]} for s in r["selection"]]
            curves += [{**c, "label": lab, "fold_id": r["fold_id"]} for c in r["curves"]]
            imps += [{**i, "label": lab, "fold_id": r["fold_id"]} for i in r["importance"]]
            metas.append({"label": lab, "fold_id": r["fold_id"], **{k: v for k, v in r["meta"].items() if k != "variants"}})
            vinfo.append({"label": lab, "fold_id": r["fold_id"], **{f"{v}_{k}": x for v, d in r["meta"]["variants"].items() for k, x in d.items()}})
        b1_freq = pd.Series([f for r in R for f in r["meta"]["b1"] if f not in UNIVERSE[:8]]).value_counts()
        top3 = pd.Series([f for r in R for f in r["meta"]["b2_top3"]]).value_counts()
        mapped = list(b1_freq.index[:6]) + [f for f in top3.index[:3] if f not in b1_freq.index[:6]]
        c, rg = effect_tables(lab, frame, A, mapped)
        ceff.append(c); reff.append(rg)
        map_effects(lab, rg, mapped)
    metrics, deltas = pd.DataFrame(rows), pd.DataFrame(deltas)
    metrics.to_csv(TAB / "table_v5_metrics.csv", index=False)
    metrics[["label", "scope", "model_set", "stage", "model", "n"] + REPORT].to_csv(TAB / "table_v5_report.csv", index=False)
    deltas.to_csv(TAB / "table_v5_deltas.csv", index=False)
    tgt = pd.concat(tgt); tgt.to_csv(TAB / "table_v5_targeting.csv", index=False)
    targeting_rule(tgt).to_csv(TAB / "table_v5_targeting_rule.csv", index=False)
    abst = pd.concat(abst); abst.to_csv(TAB / "table_v5_abstention.csv", index=False)
    vinfo = pd.DataFrame(vinfo); vinfo.to_csv(TAB / "table_v5_variant_info.csv", index=False)
    sel, curves, imps, metas = pd.DataFrame(sel), pd.DataFrame(curves), pd.DataFrame(imps), pd.DataFrame(metas)
    sel.groupby(["label", "stage", "feature"]).agg(selection_rate=("selected", "mean"), mean_rank=("rank", "mean"),
                                                    weight_attention=("weight_attention", "mean")).reset_index().assign(
        feature_name=lambda t: t.feature.map(NAMES)).to_csv(TAB / "table_v5_selection_features.csv", index=False)
    grp = metas.explode("groups").groupby(["label", "groups"]).size().div(metas.groupby("label").size(), level="label")
    grp.rename("selection_rate").reset_index().to_csv(TAB / "table_v5_selection_groups.csv", index=False)
    t3 = metas.explode("b2_top3").groupby(["label", "b2_top3"]).size().div(metas.groupby("label").size(), level="label")
    t3 = t3.rename("top3_rate").reset_index().rename(columns={"b2_top3": "feature"})
    t3["feature_name"] = t3.feature.map(NAMES)
    t3.merge(metas.groupby("label").b2_k_one_se.median().rename("median_k_one_se"), on="label").to_csv(TAB / "table_v5_climate_top3.csv", index=False)
    curves.to_csv(TAB / "table_v5_selection_curves.csv", index=False)
    imps["share"] = imps.importance / imps.groupby(["label", "fold_id", "model_set", "stage", "model"]).importance.transform("sum")
    imps.groupby(["label", "model_set", "stage", "model", "feature"]).share.mean().reset_index().assign(
        feature_name=lambda t: t.feature.map(NAMES)).to_csv(TAB / "table_v5_importance.csv", index=False)
    ceff, reff = pd.concat(ceff), pd.concat(reff)
    ceff.assign(feature_name=ceff.feature.map(NAMES)).to_csv(TAB / "table_v5_country_effects.csv", index=False)
    reff.to_csv(TAB / "table_v5_region_effects.csv", index=False)
    map_values(dev, sorted(set(ceff.feature)))
    fig_abc(metrics); fig_variants(metrics); fig_targeting(tgt); fig_abstention(abst); fig_country_effects(ceff)
    pd.set_option("display.width", 250)
    x = metrics[metrics.scope == "pooled"]
    print(x.pivot_table(index=["model_set", "stage", "model"], columns="label", values="balanced_accuracy").round(3).to_string())


def _ba(m, lab, mset, st, clf, col="balanced_accuracy"):
    r = m[(m.scope == "pooled") & (m.label == lab) & (m.model_set == mset) & (m.stage == st) & (m.model == clf)]
    return float(r[col].iloc[0]) if len(r) else np.nan


def fig_abc(m):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 4.8), sharey=True)
    order = [("A", "all"), ("B", "B0"), ("B", "B1"), ("B", "B2"), ("C", "B0"), ("C", "B1"), ("C", "B2"), ("C", "B2p")]
    for ax, lab in zip(axes, LABELS):
        for j, clf in enumerate(MODELS3 + ("ensemble",)):
            ax.plot(range(len(order)), [_ba(m, lab, s, st, clf) for s, st in order], marker="os^D"[j], label=MODEL_NAMES[clf])
        ax.set_xticks(range(len(order)), [f"{s} {st}" if s != "A" else "A income" for s, st in order], rotation=45, fontsize=7.5)
        ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left"); ax.grid(color="#e1e0d9", lw=0.6)
    axes[0].set_ylabel("balanced accuracy (pooled)"); axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Models A (income only), B (without income), C (with income) at each stage; pooled five-country model",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save_figure(fig, FIG / "v5_models_abc"); plt.close(fig)


VARIANT_ROWS = [("C", "B1", "catboost", "CatBoost"), ("C", "B1", "xgboost", "XGBoost"), ("C", "B1", "ft_transformer", "FT-Transformer"),
                ("V", "robust", "catboost", "robust CatBoost"), ("V", "robust", "xgboost", "robust XGBoost"),
                ("V", "augmented", "catboost", "augmented CatBoost"), ("V", "augmented", "xgboost", "augmented XGBoost"),
                ("V", "tuned", "xgboost", "tuned XGBoost"), ("V", "bill", "xgboost_regression", "bill model"),
                ("C", "B1", "ensemble", "ensemble (3)"), ("V", "ensemble_plus", "all", "ensemble_plus")]


def fig_variants(m):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 5), sharey=True)
    for ax, lab in zip(axes, LABELS):
        v = [_ba(m, lab, *r[:3]) for r in VARIANT_ROWS]
        lo = [_ba(m, lab, *r[:3], col="balanced_accuracy_lo") for r in VARIANT_ROWS]
        hi = [_ba(m, lab, *r[:3], col="balanced_accuracy_hi") for r in VARIANT_ROWS]
        yy = np.arange(len(VARIANT_ROWS))
        ax.errorbar(v, yy, xerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="o", color="#2a6f97", capsize=2)
        ax.axvline(v[1], color="#999", lw=0.8, ls="--")
        ax.set_yticks(yy, [r[3] for r in VARIANT_ROWS], fontsize=8); ax.invert_yaxis()
        ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left"); ax.grid(color="#e1e0d9", lw=0.6, axis="x")
        ax.set_xlabel("balanced accuracy, pooled (95% CI)", fontsize=8)
    fig.suptitle("Version-5 improvements on the B1 model (dashed line: plain XGBoost)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93)); save_figure(fig, FIG / "v5_variants"); plt.close(fig)


def fig_targeting(t):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 4.6), sharey=True)
    for ax, lab in zip(axes, LABELS):
        g = t[(t.label == lab) & (t.scope == "pooled")]
        for name in POLICY.values():
            h = g[g.model == name]
            ax.plot(h.top_share * 100, h.double_risk_reached * 100, label=name, lw=1.3)
        ax.plot([0, 100], [0, 100], color="#bbb", lw=0.8, ls=":")
        ax.axhline(REACH * 100, color="#d1342f", lw=0.8, ls="--")
        ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left"); ax.grid(color="#e1e0d9", lw=0.6)
        ax.set_xlabel("households helped: top % ranked by the model", fontsize=8)
    axes[0].set_ylabel("% of Double-Risk households reached"); axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Targeting curves (dotted: random choice; dashed: goal of reaching {int(REACH * 100)}% of Double Risk)",
                 fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save_figure(fig, FIG / "v5_targeting"); plt.close(fig)


def fig_abstention(t):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(18, 4.4), sharey=True)
    for ax, lab in zip(axes, LABELS):
        g = t[(t.label == lab) & (t.scope == "pooled")]
        for name, h in g.groupby("model"):
            ax.plot(h.not_sure_share * 100, h.balanced_accuracy, marker="o", ms=3, label=name)
        ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left"); ax.grid(color="#e1e0d9", lw=0.6)
        ax.set_xlabel('% of households flagged "not sure"', fontsize=8)
    axes[0].set_ylabel("balanced accuracy on the rest"); axes[0].legend(frameon=False, fontsize=7.5)
    fig.suptitle('Letting the model say "not sure": accuracy on the households it keeps', fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92)); save_figure(fig, FIG / "v5_abstention"); plt.close(fig)


def fig_country_effects(c):
    fig, axes = plt.subplots(1, len(LABELS), figsize=(19, 5.8))
    for ax, lab in zip(axes, LABELS):
        d = c[c.label == lab].pivot(index="feature", columns="country", values="mean_shap_double_risk").reindex(columns=COUNTRIES)
        lim = float(np.nanmax(np.abs(d.to_numpy()))) or 1e-6
        ax.imshow(d.to_numpy(float), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
        ax.set_yticks(range(len(d)), [NAMES.get(f, f) for f in d.index], fontsize=7.5)
        ax.set_xticks(range(len(COUNTRIES)), [k[:3] for k in COUNTRIES], fontsize=8)
        for (i, j), v in np.ndenumerate(d.to_numpy(float)):
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6.5)
        ax.set_title(LABEL_NAMES[lab], fontsize=9, loc="left")
    fig.suptitle("Mean SHAP towards Double Risk per country (red = pushes towards energy poverty)", fontsize=10.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94)); save_figure(fig, FIG / "v5_country_effects"); plt.close(fig)


if __name__ == "__main__":
    main()
