"""Identification-accuracy benchmark: how reliably can each country's energy-poor households be
identified, under which definition, with which information, and is a country big enough to say?

Grid: cost-axis definition (need_adjusted | raw_q80 = Van Hove) x income state (excluded = proxy
targeting | included = Van Hove's design, income bracket as a predictor) x information tier (T1
structural, T2 + socio-economic, T3 + welfare screen; see lib/identification/data.py) x model (prior
chance floor, lr4, axis2, cb4) x scope (local per-country, pooled with country effects,
leave-one-country-out transfer). Income axis = Van Hove's fixed rule (bracket <= 3) throughout;
scripts/build_identification_sensitivity.py reruns a reduced grid under the other income rules.
Development partitions only (train+validation; the test partition is locked). Every score is
repeated 5-fold CV; every metric (accuracy, precision, recall, F1, specificity, ROC-AUC, PR-AUC, MCC,
kappa, log-loss, Brier, per class and per axis) carries a household-bootstrap 95% CI.

Outputs:
  outputs/tables/table_identification_support.csv       class support + viability gate per country/definition
  outputs/tables/table_identification_benchmark.csv     the full grid
  outputs/tables/table_identification_headline.csv      best model per country x definition x tier x scope
  outputs/tables/table_external_validation.csv          do the classes track real hardship (S8/S7)?
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import COUNTRIES, load_development_frame, tier_features
from lib.identification.labels import CLASS_NAMES, COST_KINDS, add_labels, support_table, viability
from lib.identification.protocol import MODELS, loco_proba, oof_proba, summarise
from lib.labels.income_harmonization import HEADLINE_INCOME_RULE

N_BOOT = 200
OUT = ROOT / "outputs/tables"


def run_grid(dev, *, cost_kinds, income_states, tiers_wanted, models, scopes, income_rule=HEADLINE_INCOME_RULE):
    rows, support_rows, validation_rows = [], [], []
    for kind in cost_kinds:
        lab = add_labels(dev, cost_kind=kind, income_rule=income_rule)
        sup = viability(support_table(lab)).assign(cost_kind=kind, income_rule=income_rule)
        support_rows.append(sup)
        ok = dict(zip(sup.country, sup.four_class_ok))
        v = lab.assign(class_name=lab.y.map(CLASS_NAMES))
        validation_rows.append(v.groupby(["country", "class_name"]).agg(
            n=("y", "size"), aid_received_rate=("S7", "mean"),
            hardship_rate=("S8", lambda s: float((s.dropna() >= 3).mean()) if s.notna().any() else np.nan),
            s8_answered=("S8", "count")).reset_index().assign(cost_kind=kind, income_rule=income_rule))
        for state in income_states:
            tiers = tier_features(include_income=(state == "included"))
            for tier in tiers_wanted:
                feats = tiers[tier]
                for model in models:
                    if model == "prior" and (tier != tiers_wanted[0] or state != income_states[0]):
                        continue  # the chance floor depends on neither the tier nor the income state
                    t0 = time.time()
                    pooled = oof_proba(lab, feats + ["country"], model) if ("pooled" in scopes and model != "prior") else None
                    for c in COUNTRIES:
                        g = lab[lab.country == c].reset_index(drop=True)
                        base = {"cost_kind": kind, "income_rule": income_rule,
                                "income_state": "any" if model == "prior" else state,
                                "tier": "any" if model == "prior" else tier, "model": model, "country": c, "four_class_ok": bool(ok[c])}
                        rows.append({**base, "scope": "local", **summarise(g.y.values, oof_proba(g, feats, model), n_boot=N_BOOT)})
                        if model == "prior":
                            continue
                        if "pooled" in scopes:
                            mask = (lab.country == c).to_numpy()
                            rows.append({**base, "scope": "pooled", **summarise(lab.y.values[mask], pooled[:, mask], n_boot=N_BOOT)})
                        if "loco" in scopes:
                            proba, tgt = loco_proba(lab, c, feats, model)
                            rows.append({**base, "scope": "loco", **summarise(tgt.y.values, proba, n_boot=N_BOOT)})
                    print(f"{income_rule} {kind} {state} {tier} {model}: {time.time() - t0:.0f}s", flush=True)
    bench = pd.DataFrame(rows)
    chance = bench[bench.model.eq("prior")].set_index(["cost_kind", "country"]).macro_f1.rename("chance_f1")
    bench = bench.join(chance, on=["cost_kind", "country"])
    bench["lift_over_chance"] = bench.macro_f1 - bench.chance_f1
    return bench, pd.concat(support_rows), pd.concat(validation_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dev = load_development_frame()
    bench, support, validation = run_grid(dev, cost_kinds=list(COST_KINDS), income_states=["excluded", "included"],
                                          tiers_wanted=["T1", "T2", "T3"], models=list(MODELS), scopes=("local", "pooled", "loco"))
    bench.to_csv(OUT / "table_identification_benchmark.csv", index=False)
    support.to_csv(OUT / "table_identification_support.csv", index=False)
    validation.to_csv(OUT / "table_external_validation.csv", index=False)
    real = bench[bench.model.ne("prior")]
    head = (real.sort_values("macro_f1", ascending=False).groupby(["cost_kind", "income_state", "tier", "scope", "country"], as_index=False)
            .head(1).sort_values(["cost_kind", "income_state", "tier", "scope", "country"]))
    head.to_csv(OUT / "table_identification_headline.csv", index=False)
    show = head[(head.cost_kind == "need_adjusted") & (head.scope == "local")]
    print(show[["income_state", "tier", "country", "model", "accuracy", "precision_macro", "recall_macro", "f1_macro",
                "auc_macro_ovr", "mcc", "chance_f1", "four_class_ok"]].round(3).to_string(index=False))
    print(f"Wrote {len(bench)} benchmark rows ({bench.shape[1]} columns) -> {OUT.relative_to(ROOT)}/table_identification_*.csv")


if __name__ == "__main__":
    main()
