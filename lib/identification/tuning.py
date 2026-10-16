"""Idea 6 -- tune XGBoost with Optuna (Akiba et al. 2019, KDD), nested inside each outer training fold.

Search (Tree-structured Parzen Estimator sampler, seeded; N_TRIALS trials): number of trees 100-800, depth 2-8, learning rate
0.01-0.3 (log), row subsampling 0.5-1, column subsampling 0.4-1, minimum child weight 1-20 (log), L2 penalty 0.1-20 (log),
minimum split gain 0-5. Objective: balanced accuracy over a 3-fold stratified split of the training fold. The outer fold's
held-out households are never seen; the chosen parameters are refitted on the whole training fold.
"""
from __future__ import annotations
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from .models import fit as fit_model

N_TRIALS = 25
optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_xgboost(train: pd.DataFrame, feats: list[str], seed: int, n_trials: int = N_TRIALS) -> tuple[dict, float]:
    train = train.reset_index(drop=True)
    y = train["y"].astype(int).to_numpy()
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(train, y))

    def objective(trial):
        params = {"n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
                  "max_depth": trial.suggest_int("max_depth", 2, 8),
                  "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                  "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                  "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                  "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
                  "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
                  "gamma": trial.suggest_float("gamma", 0.0, 5.0)}
        s = [balanced_accuracy_score(y[te], fit_model("xgboost", train.iloc[tr], feats, seed, params=params)
                                     .predict_proba(train.iloc[te]).argmax(1)) for tr, te in folds]
        return float(np.mean(s))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params, float(study.best_value)
