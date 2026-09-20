"""Single evaluation on the locked test partition (or, with --dry-run, a rehearsal that touches no test data).

Real run: refuses unless config/final_selection.json records a human authorisation and no earlier access
(lib/identification/final_test.py). For every pre-registered selection the model is fit once on all eligible train +
validation households of its country and scored on that country's eligible test households, with every metric and
95% bootstrap CIs, next to the development cross-validation score. The managerial (missed / wrongly-included) analysis
is repeated on the test households. Not part of run_full_pipeline.sh: it is a one-time, human-authorised step.

--dry-run: dev = train partition only, "test" = validation partition. Same code path, no test data, nothing recorded,
tables written with a `_dryrun` suffix. Use it to prove the mechanics before spending the real test set.

Outputs: outputs/tables/table_final_test_{classification,managerial,generalization}.csv, artifacts/manifests/final_test_result.json
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import load_development_frame, tier_features
from lib.identification.final_test import load_test_frame, read_selection, record_evaluation
from lib.identification.labels import CLASS_NAMES, add_labels, apply_labels
from lib.identification.protocol import fit_predict_proba, summarise
from scripts.build_managerial_implications import DOUBLE, TIER_LABEL, TIERS, with_ci

OUT = ROOT / "outputs/tables"
MANAGERIAL_MODEL = "lr4"     # fixed in advance (config/final_selection.json protocol), not chosen on test


def evaluate(dev: pd.DataFrame, test: pd.DataFrame, sel: dict, suffix: str) -> dict:
    rows, gen, managerial = [], [], []
    labelled = {k: (add_labels(dev, cost_kind=k), apply_labels(dev, test, cost_kind=k)) for k in sorted({s["cost_kind"] for s in sel["selections"]})}
    for s in sel["selections"]:
        dev_l, test_l = labelled[s["cost_kind"]]
        g_dev, g_test = dev_l[dev_l.country == s["country"]], test_l[test_l.country == s["country"]]
        feats = tier_features(include_income=(s["income_state"] == "included"))[s["tier"]]
        proba = fit_predict_proba(s["model"], g_dev, g_test, feats)
        chance = fit_predict_proba("prior", g_dev, g_test, feats)
        m = summarise(g_test.y.values, proba, n_boot=500)
        c = summarise(g_test.y.values, chance, n_boot=50)
        counts = {f"n_{name.replace(' ', '_').lower()}": int((g_test.y == k).sum()) for k, name in CLASS_NAMES.items()}
        rows.append({**{k: s[k] for k in ("cost_kind", "income_state", "tier", "country", "model")}, "n_test": len(g_test), **counts,
                     "chance_f1": c["f1_macro"], **m})
        gen.append({**{k: s[k] for k in ("cost_kind", "income_state", "tier", "country", "model")}, "dev_cv_f1": s["dev_f1_macro"],
                    "test_f1": m["f1_macro"], "test_f1_lo": m["f1_macro_lo"], "test_f1_hi": m["f1_macro_hi"],
                    "test_minus_dev": m["f1_macro"] - s["dev_f1_macro"], "dev_inside_test_ci": bool(m["f1_macro_lo"] <= s["dev_f1_macro"] <= m["f1_macro_hi"])})
    dev_l, test_l = labelled["need_adjusted"]
    for c in sorted(test_l.country.unique()):
        g_dev, g_test = dev_l[dev_l.country == c], test_l[test_l.country == c]
        is_pos = g_test.y.to_numpy() == DOUBLE
        for tier, inc in TIERS:
            feats = tier_features(include_income=inc)[tier]
            proba = fit_predict_proba(MANAGERIAL_MODEL, g_dev, g_test, feats)
            managerial.append({"country": c, "tier": TIER_LABEL[(tier, inc)], "model": MANAGERIAL_MODEL,
                               **with_ci(is_pos, (proba.argmax(1) == DOUBLE)[None])})
    for name, df in (("classification", pd.DataFrame(rows)), ("generalization", pd.DataFrame(gen)), ("managerial", pd.DataFrame(managerial))):
        df.to_csv(OUT / f"table_final_test_{name}{suffix}.csv", index=False)
    return {"classification": pd.DataFrame(rows), "generalization": pd.DataFrame(gen), "managerial": pd.DataFrame(managerial)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="rehearse on train -> validation; touches no test data")
    a = ap.parse_args()
    sel = read_selection()
    dev_all = load_development_frame()
    if a.dry_run:
        dev, test, suffix = dev_all[dev_all.partition == "train"], dev_all[dev_all.partition == "validation"], "_dryrun"
        print("DRY RUN: train -> validation, no test data read")
    else:
        test, dev, suffix = load_test_frame(), dev_all, ""       # load_test_frame records the access and enforces the lock
        print(f"TEST PARTITION LOADED: {len(test)} households")
    res = evaluate(dev, test, sel, suffix)
    pd.set_option("display.width", 220)
    show = res["classification"]
    show = show[(show.cost_kind == "need_adjusted") & (show.tier == "T2")]
    print(show[["income_state", "country", "model", "n_test", "accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_macro_lo", "f1_macro_hi", "auc_macro_ovr", "chance_f1"]].round(3).to_string(index=False))
    print(res["generalization"].groupby("income_state")[["dev_cv_f1", "test_f1", "test_minus_dev"]].mean().round(3))
    if not a.dry_run:
        sha = hashlib.sha256((OUT / "table_final_test_classification.csv").read_bytes()).hexdigest()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
        record_evaluation(sha, commit)
        (ROOT / "artifacts/manifests").mkdir(parents=True, exist_ok=True)
        (ROOT / "artifacts/manifests/final_test_result.json").write_text(json.dumps({"result_sha256": sha, "preregistered_at_commit": commit,
                                                                                   "n_test_by_country": res["classification"].groupby("country").n_test.first().to_dict()}, indent=2) + "\n")
        print("test evaluation recorded; config/final_selection.json is now spent")


if __name__ == "__main__":
    main()
