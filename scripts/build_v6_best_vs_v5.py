"""Version 6: paired comparison of the best version-6 recipe (ensemble_mixed: CatBoost + XGBoost + FT-Transformer + multilevel
bill model, trained with the extra countries) against the version-5 recipe (ensemble_plus with the XGBoost bill model, five study
countries), same households and folds (artifacts/identification/v6_pool_results.pkl). Output: table_v6_best_vs_v5best.csv."""
from __future__ import annotations
import pickle, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.identification.data import load_pool_frame
from lib.identification.lines_v5 import study_labels_v5
from lib.identification.stages import paired_delta


def main():
    res = pickle.load(open(ROOT / "artifacts/identification/v6_pool_results.pkl", "rb"))
    pool = load_pool_frame()
    labs = study_labels_v5(pool[pool.pool == "focus"].reset_index(drop=True))
    rows = []
    for k in ("vh", "needqr", "needqr_noclim", "burden"):
        f = labs[k][0]; y = f.y.to_numpy(); P = {}
        for lab, fold, te, out in res:
            if lab == k:
                for key, p in out.items():
                    P.setdefault(key, np.zeros((3, len(f), 4)))[fold // 5, te] = p
        a = np.mean([P[("five", "B1", c)] for c in ("catboost", "xgboost", "ft_transformer", "bill")], axis=0)
        b = np.mean([P[("pool", "B1", c)] for c in ("catboost", "xgboost", "ft_transformer", "bill_mixed")], axis=0)
        for scope, idx in {"pooled": np.arange(len(f)), **{c: np.where(f.country == c)[0] for c in sorted(f.country.unique())}}.items():
            rows.append({"label": k, "scope": scope, **paired_delta(y[idx], a[:, idx], b[:, idx], metrics=("balanced_accuracy",))})
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "outputs/tables/table_v6_best_vs_v5best.csv", index=False)
    print(t[["label", "scope", "delta_balanced_accuracy", "delta_balanced_accuracy_lo", "delta_balanced_accuracy_hi"]].round(3).to_string())


if __name__ == "__main__":
    main()
