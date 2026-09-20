"""Pre-register the model choices for the single test evaluation -- from DEVELOPMENT results only.

For every (cost definition, income state, information tier, country) cell of the local per-country grid, the model
(logistic / two-axis / CatBoost) with the best cross-validated macro-F1 on train + validation is recorded in
`config/final_selection.json`. Nothing else is chosen on the test partition: tiers and income states are comparison axes,
not selections. Commit the file BEFORE running scripts/run_final_test.py.

The authorisation fields (`test_unlock_authorized`, `authorized_by`, `authorization_note`) are never set here; they are
preserved if already present. If the test partition has already been accessed the selection is frozen for good and this
script refuses to rewrite it.
"""
from __future__ import annotations
import hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.final_test import SELECTION_PATH

TABLE = ROOT / "outputs/tables/table_identification_benchmark.csv"
MODEL_ORDER = {"lr4": 0, "axis2": 1, "cb4": 2}     # deterministic tie-break


def main() -> None:
    old = json.loads(SELECTION_PATH.read_text()) if SELECTION_PATH.exists() else {}
    if old.get("test_accessed_at"):
        raise SystemExit(f"FROZEN: the test partition was accessed at {old['test_accessed_at']}; the pre-registered selection must not change")
    b = pd.read_csv(TABLE)
    b = b[(b.scope == "local") & (b.model != "prior") & (b.income_rule == "bracket3")]
    b = b.assign(_o=b.model.map(MODEL_ORDER)).sort_values(["f1_macro", "_o"], ascending=[False, True])
    best = b.groupby(["cost_kind", "income_state", "tier", "country"], as_index=False).head(1)
    sel = [{"cost_kind": r.cost_kind, "income_state": r.income_state, "tier": r.tier, "country": r.country, "model": r.model,
            "dev_f1_macro": round(float(r.f1_macro), 4), "dev_f1_lo": round(float(r.f1_macro_lo), 4), "dev_f1_hi": round(float(r.f1_macro_hi), 4),
            "dev_accuracy": round(float(r.accuracy), 4)}
           for r in best.sort_values(["cost_kind", "income_state", "tier", "country"]).itertuples()]
    out = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "labels": "income axis = bracket <= 3; cost axis = need-based threshold (need_adjusted, q from lib/identification/labels.py) "
                      "or country Q80 (raw_q80, Van Hove); thresholds learned on train + validation only",
            "training": "each selected model is fit once on ALL eligible train + validation households of its country (out-of-fold labels)",
            "evaluation": "single evaluation of every selection on that country's eligible test households; 95% household bootstrap CIs",
            "managerial": "logistic model (lr4), rule 'assist predicted Double Risk', tiers T1, T2, T3 and T3 + income",
            "no_selection_on_test": "models were chosen from development cross-validation only; nothing is tuned after the test look",
        },
        "source_table_sha256": hashlib.sha256(TABLE.read_bytes()).hexdigest(),
        "selections": sel,
        "test_unlock_authorized": bool(old.get("test_unlock_authorized", False)),
        "authorized_by": old.get("authorized_by"),
        "authorization_note": old.get("authorization_note"),
        "test_accessed_at": None, "test_evaluated_at": None,
    }
    SELECTION_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"froze {len(sel)} selections -> {SELECTION_PATH.relative_to(ROOT)}; authorised={out['test_unlock_authorized']}")


if __name__ == "__main__":
    main()
