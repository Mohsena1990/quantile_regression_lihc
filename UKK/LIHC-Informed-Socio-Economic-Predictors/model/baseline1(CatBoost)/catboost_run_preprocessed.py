# ======================================================
# FINAL RUNNING MODULE
# 4 datasets × 4 model baselines × k = {2, 3, 4}
# SMOTE/SMOTENC-ready, leakage-aware, CPU/GPU-compatible
# ======================================================

from __future__ import annotations

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE, SMOTENC

from CatBoost import CatBoostML
from Stratify_train_test_split_by_country import (
    country_stratified_group_split,
    class_stratified_k_fold_split,
)
from fine_tuning import tune_catboost


# ======================================================
# 1 Paths and configuration
# ======================================================

BASE_DIR = str(Path(__file__).resolve().parents[4])
# Consolidated under outputs/catboost/ alongside the rest of the pipeline's
# generated artifacts (see README's Output Layout). A prior run's results
# may still exist at <repo_root>/final_saved_models_catboost/ from before
# this change -- move them into outputs/catboost/ once that run has
# finished; don't move a directory a running process is still writing to.
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "catboost", "final_saved_models_catboost")
os.makedirs(MODEL_DIR, exist_ok=True)

PREPROCESS_DIR = os.path.join(
    BASE_DIR,
    "UKK",
    "LIHC-Informed-Socio-Economic-Predictors",
    "preprocessing",
)
if PREPROCESS_DIR not in sys.path:
    sys.path.append(PREPROCESS_DIR)

from risk_category import assign_traditional_lihc, assign_hqrtm

BASE_DATA_PATH = os.path.join(BASE_DIR, "preprocessed_data_clean.csv")

DATASET_CONFIGS = {
    "traditional_lihc": {
        "label_type": "traditional",
        "income_rule": "country_median_60",
        "exp_quantile": 0.80,
    },
    "hqrtm_q60": {
        "label_type": "hqrtm",
        "income_rule": "country_median_60",
        "quantile": 0.60,
        "margin_scale": 0.10,
    },
    "hqrtm_q65": {
        "label_type": "hqrtm",
        "income_rule": "country_median_60",
        "quantile": 0.65,
        "margin_scale": 0.10,
    },
    "hqrtm_q70": {
        "label_type": "hqrtm",
        "income_rule": "country_median_60",
        "quantile": 0.70,
        "margin_scale": 0.10,
    },
}

TARGET = "risk_category"
K_VALUES = [2, 3, 4]
TEST_SIZE = 0.25
RANDOM_STATE = 42

# Fully separate training per country rather than pooled training with a
# per-country test-set breakdown: each country gets its own train/test
# split, its own K-fold CV, its own tuned CatBoost model. Real tradeoffs
# accepted by choosing this: ~7x the compute of pooled training, and some
# {country, dataset, k} combinations may have too few "Double risk"
# households for a given k to support stratified K-fold CV at all --
# those are skipped and logged rather than crashing the run (see
# MIN_ROWS_PER_CLASS below).
PER_COUNTRY_MODE = True
COUNTRY_NAMES = {1: "Bulgaria", 3: "Germany", 4: "Hungary", 5: "Italy", 8: "Serbia", 9: "Spain", 10: "Ukraine"}

TASK_TYPE = "CPU"
DEVICES = "0"
PRIMARY_METRICS = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]


# ======================================================
# 2 Feature blocks
# ======================================================
BASE_STRUCTURAL = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "heating_control",
    "household_size",
    "children_present",
    "elderly_present",
]

# C2 (air conditioner use) and C3 (heating strategy) were dropped from
# here: both are 0% present in Bulgaria, Italy, and Serbia (a module-level
# survey skip, not scattered non-response -- see README's Data
# Construction section), so including them gave those three countries a
# placeholder "0"/"Missing" category instead of real data in every B3/B4
# model. SettlementSize and S6 are 100% present in all 7 countries.
CONTEXT_FEATURES = [
    "Country",
    "SettlementSize",
    "S6",
]

SOCIOECONOMIC_AUX = [
    "equivalized_income",
]

MODEL_SPECS = {
    "B1_structural": BASE_STRUCTURAL,
    "B2_structural_socioeconomic": BASE_STRUCTURAL + SOCIOECONOMIC_AUX,
    "B3_structural_context": BASE_STRUCTURAL + CONTEXT_FEATURES,
    "B4_structural_context_socioeconomic": BASE_STRUCTURAL + CONTEXT_FEATURES + SOCIOECONOMIC_AUX,
}

CATEGORICAL_LIKE = {
    "Country",
    "dwelling_type",
    "main_heating_source",
    "heating_control",
    "SettlementSize",
    # S6 codes (big city / suburb / town / village / farm / don't know)
    # have no numeric/ordinal meaning -- was previously missing from this
    # set and so was fed to CatBoost as a raw continuous number.
    "S6",
}

QR_FEATURES = [
    "floor_area",
    "house_age",
    "dwelling_type",
    "insulation_count",
    "main_heating_source",
    "household_size",
]


# ======================================================
# 3 Helper functions
# ======================================================
def ensure_required_columns(df: pd.DataFrame, cols: list, dataset_name: str, model_name: str = "") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        prefix = f"[{dataset_name}"
        if model_name:
            prefix += f" - {model_name}"
        prefix += "]"
        raise KeyError(f"{prefix} Missing required columns: {missing}")


def create_labels_for_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if config["label_type"] == "traditional":
        train_labeled = assign_traditional_lihc(
            train_df,
            income_col="equivalized_income",
            exp_col="total_expenditure",
            country_col="Country",
            income_rule=config.get("income_rule", "country_median_60"),
            exp_quantile=config.get("exp_quantile", 0.80),
            fit_df=train_df,
        )
        test_labeled = assign_traditional_lihc(
            test_df,
            income_col="equivalized_income",
            exp_col="total_expenditure",
            country_col="Country",
            income_rule=config.get("income_rule", "country_median_60"),
            exp_quantile=config.get("exp_quantile", 0.80),
            fit_df=train_df,
        )
    elif config["label_type"] == "hqrtm":
        train_labeled = assign_hqrtm(
            train_df,
            qr_features=QR_FEATURES,
            income_col="equivalized_income",
            exp_col="total_expenditure",
            country_col="Country",
            income_rule=config.get("income_rule", "country_median_60"),
            quantile=config.get("quantile", 0.65),
            add_country_effects=True,
            margin_scale=config.get("margin_scale", 0.10),
            fit_df=train_df,
        )
        test_labeled = assign_hqrtm(
            test_df,
            qr_features=QR_FEATURES,
            income_col="equivalized_income",
            exp_col="total_expenditure",
            country_col="Country",
            income_rule=config.get("income_rule", "country_median_60"),
            quantile=config.get("quantile", 0.65),
            add_country_effects=True,
            margin_scale=config.get("margin_scale", 0.10),
            fit_df=train_df,
        )
    else:
        raise ValueError(f"Unknown label_type for dataset {dataset_name}: {config['label_type']}")

    train_labeled = train_labeled.dropna(subset=[TARGET]).copy()
    test_labeled = test_labeled.dropna(subset=[TARGET]).copy()
    return train_labeled, test_labeled


def fit_feature_preprocessor(
    train_df: pd.DataFrame,
    features: list,
    categorical_cols: list | None = None,
) -> dict:
    categorical_cols = set(categorical_cols or [])
    numeric_medians = {}

    for col in features:
        if col in categorical_cols:
            continue
        series = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = series.median()
        numeric_medians[col] = 0.0 if pd.isna(med) else float(med)

    return {
        "categorical_cols": categorical_cols,
        "numeric_medians": numeric_medians,
    }


def transform_features(
    df: pd.DataFrame,
    features: list,
    processor: dict,
) -> pd.DataFrame:
    out = df.copy()
    categorical_cols = processor["categorical_cols"]
    numeric_medians = processor["numeric_medians"]

    for col in features:
        if col not in out.columns:
            continue

        if col in categorical_cols:
            cat_series = out[col].copy()

            if col in {"C2", "C3"}:
                cat_series = pd.to_numeric(cat_series, errors="coerce")
                cat_series = cat_series.replace([0, 99], np.nan)

            out[col] = (
                cat_series
                .astype(str)
                .replace(["nan", "None", "NaN"], np.nan)
                .fillna("missing")
            )
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
            out[col] = out[col].fillna(numeric_medians.get(col, 0.0))

    return out


def assert_no_nan(df: pd.DataFrame, cols: list, dataset_name: str, model_name: str, where: str) -> None:
    nan_counts = df[cols].isna().sum()
    nan_counts = nan_counts[nan_counts > 0]
    if not nan_counts.empty:
        print(f"\nNaN counts in {where}:")
        print(nan_counts)
        raise ValueError(f"[{dataset_name} - {model_name}] NaN values remain in {where}.")


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(cm: pd.DataFrame, path: str) -> None:
    cm.to_csv(path, index=True)


def oversample_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list,
    random_state: int = 42,
    sampling_strategy: str = "not majority",
):
    X_res = X_train.copy()
    y_res = y_train.copy()

    if len(cat_features) > 0:
        cat_indices = [X_res.columns.get_loc(c) for c in cat_features if c in X_res.columns]
        sampler = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )
    else:
        sampler = SMOTE(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
        )

    X_res, y_res = sampler.fit_resample(X_res, y_res)
    return X_res, y_res


def build_metric_summary(metric_df: pd.DataFrame, source_cols: dict, prefix: str) -> dict:
    summary = {}
    for metric_name, column_name in source_cols.items():
        series = pd.to_numeric(metric_df[column_name], errors="coerce")
        summary[f"{prefix}_{metric_name}_mean"] = float(series.mean())
        summary[f"{prefix}_{metric_name}_std"] = float(series.std(ddof=1)) if len(series) > 1 else 0.0
    return summary


def evals_result_to_long_df(evals_result: dict, fold: int) -> pd.DataFrame:
    rows = []
    for split_name, metrics in (evals_result or {}).items():
        for metric_name, values in metrics.items():
            for iteration, value in enumerate(values, start=1):
                rows.append(
                    {
                        "fold": fold,
                        "split": split_name,
                        "metric": metric_name,
                        "iteration": iteration,
                        "value": float(value),
                    }
                )
    return pd.DataFrame(rows)


def save_json(data: dict, path: str) -> None:
    with open(path, "w") as handle:
        json.dump(data, handle, indent=4)


# ======================================================
# 4 Main experiment loop
# ======================================================
def run_grid(df: pd.DataFrame, output_root: str, country_label: str | None = None) -> tuple:
    """Run the full {dataset x model x k} grid on df, either pooled
    (country_label=None, country-stratified splitting/CV) or on a
    single country's data already filtered into df (country_label set,
    plain splitting + class-stratified CV, since country-grouping breaks
    down with only one country present)."""
    results_local = []
    country_results_local = []
    label_prefix = f"[{country_label}] " if country_label else ""

    for dataset_name, dataset_config in DATASET_CONFIGS.items():
        print(f"\n{'=' * 80}")
        print(f"{label_prefix}Dataset: {dataset_name}")
        print(f"Label config: {dataset_config}")
        print(f"{'=' * 80}")

        if country_label is None:
            train_df_raw, test_df_raw = train_test_split(
                df,
                test_size=TEST_SIZE,
                stratify=df["Country"],
                random_state=RANDOM_STATE,
            )
        else:
            train_df_raw, test_df_raw = train_test_split(
                df,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
            )

        train_df, test_df = create_labels_for_split(
            train_df=train_df_raw,
            test_df=test_df_raw,
            dataset_name=dataset_name,
            config=dataset_config,
        )

        print("Train size:", len(train_df), "| Test size:", len(test_df))
        print("Train class counts:\n", train_df[TARGET].value_counts())
        print("Test class counts:\n", test_df[TARGET].value_counts())

        for model_name, features in MODEL_SPECS.items():
            print(f"\n--- {label_prefix}Model: {model_name} ---")
            print("Features:", features)

            ensure_required_columns(train_df, features + [TARGET], dataset_name, model_name)
            ensure_required_columns(test_df, features + [TARGET], dataset_name, model_name)

            cat_features = [f for f in features if f in CATEGORICAL_LIKE]

            processor = fit_feature_preprocessor(train_df, features, categorical_cols=cat_features)
            train_df_model = transform_features(train_df, features, processor)
            test_df_model = transform_features(test_df, features, processor)

            assert_no_nan(train_df_model, features, dataset_name, model_name, "train_df_model")
            assert_no_nan(test_df_model, features, dataset_name, model_name, "test_df_model")

            train_class_counts = train_df_model[TARGET].value_counts()

            for k in K_VALUES:
                if len(train_class_counts) < 2 or train_class_counts.min() < k:
                    print(
                        f"{label_prefix}Skipping dataset={dataset_name} model={model_name} k={k}: "
                        f"smallest class ({train_class_counts.idxmin() if len(train_class_counts) else 'n/a'}) "
                        f"has {train_class_counts.min() if len(train_class_counts) else 0} training rows, "
                        f"fewer than k={k}."
                    )
                    continue

                try:
                    result_row, country_test_rows = run_one_combo(
                        dataset_name=dataset_name,
                        dataset_config=dataset_config,
                        model_name=model_name,
                        features=features,
                        cat_features=cat_features,
                        k=k,
                        train_df=train_df,
                        test_df=test_df,
                        train_df_model=train_df_model,
                        test_df_model=test_df_model,
                        output_root=output_root,
                        country_label=country_label,
                    )
                except Exception as exc:
                    print(
                        f"{label_prefix}FAILED dataset={dataset_name} model={model_name} k={k}: {exc}. "
                        f"Skipping this combination and continuing."
                    )
                    continue

                results_local.append(result_row)
                country_results_local.extend(country_test_rows)

    return results_local, country_results_local


def run_one_combo(
    dataset_name: str,
    dataset_config: dict,
    model_name: str,
    features: list,
    cat_features: list,
    k: int,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df_model: pd.DataFrame,
    test_df_model: pd.DataFrame,
    output_root: str,
    country_label: str | None,
) -> tuple:
    print(f"\n>>> Running: country={country_label or 'pooled'} | dataset={dataset_name} | model={model_name} | k={k}")

    run_dir_parts = [output_root]
    if country_label is not None:
        run_dir_parts.append(country_label)
    run_dir_parts += [dataset_name, model_name, f"k{k}"]
    run_dir = os.path.join(*run_dir_parts)
    os.makedirs(run_dir, exist_ok=True)

    split_df = train_df_model[features + [TARGET]].copy()
    if "Country" not in split_df.columns:
        split_df["Country"] = train_df_model["Country"].values

    if country_label is None:
        raw_folds = country_stratified_group_split(
            df=split_df,
            inputs=features,
            target=TARGET,
            n_splits=k,
            random_state=RANDOM_STATE,
            use_smote=False,
            categorical_cols=cat_features,
            sampling_strategy="not majority",
        )
        fit_folds = country_stratified_group_split(
            df=split_df,
            inputs=features,
            target=TARGET,
            n_splits=k,
            random_state=RANDOM_STATE,
            use_smote=True,
            categorical_cols=cat_features,
            sampling_strategy="not majority",
        )
    else:
        raw_folds = class_stratified_k_fold_split(
            df=split_df,
            inputs=features,
            target=TARGET,
            n_splits=k,
            random_state=RANDOM_STATE,
            use_smote=False,
            categorical_cols=cat_features,
            sampling_strategy="not majority",
        )
        fit_folds = class_stratified_k_fold_split(
            df=split_df,
            inputs=features,
            target=TARGET,
            n_splits=k,
            random_state=RANDOM_STATE,
            use_smote=True,
            categorical_cols=cat_features,
            sampling_strategy="not majority",
        )

    best_params = tune_catboost(
        folds=fit_folds,
        features=features,
        cat_features=cat_features,
        score_metric="macro_f1",
        task_type=TASK_TYPE,
        devices=DEVICES if TASK_TYPE.upper() == "GPU" else None,
        n_iter=8,
        random_state=RANDOM_STATE,
        output_file=os.path.join(run_dir, "tuning_results.csv"),
    )

    best_params = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in best_params.items()
    }

    save_json(best_params, os.path.join(run_dir, "best_params.json"))
    save_json(
        {
            "country": country_label,
            "dataset": dataset_name,
            "model_name": model_name,
            "features": features,
            "cat_features": cat_features,
            "k": k,
        },
        os.path.join(run_dir, "features.json"),
    )
    save_json(
        {
            "country": country_label,
            "dataset": dataset_name,
            "model_name": model_name,
            "k": k,
            "train_size": int(len(train_df_model)),
            "test_size": int(len(test_df_model)),
            "train_class_counts": train_df_model[TARGET].value_counts().to_dict(),
            "test_class_counts": test_df_model[TARGET].value_counts().to_dict(),
        },
        os.path.join(run_dir, "data_split_summary.json"),
    )

    cv_rows = []
    learning_curve_frames = []

    for fold_id, (raw_fold, fit_fold) in enumerate(zip(raw_folds, fit_folds), start=1):
        X_tr_raw, X_val_raw, y_tr_raw, y_val_raw = raw_fold
        X_tr_fit, _, y_tr_fit, _ = fit_fold

        model = CatBoostML(params=best_params)
        model.train(
            X_train=X_tr_fit[features],
            y_train=y_tr_fit,
            X_val=X_val_raw[features],
            y_val=y_val_raw,
            cat_features=cat_features,
            use_class_weights=False,
        )

        y_train_pred = model.predict(X_tr_raw[features], cat_features=cat_features)
        y_val_pred = model.predict(X_val_raw[features], cat_features=cat_features)

        train_metrics = compute_metrics(y_tr_raw, y_train_pred)
        validation_metrics = compute_metrics(y_val_raw, y_val_pred)

        fold_row = {
            "fold": fold_id,
            "train_size_raw": int(len(X_tr_raw)),
            "train_size_fit": int(len(X_tr_fit)),
            "validation_size": int(len(X_val_raw)),
        }
        fold_row.update({f"train_{key}": value for key, value in train_metrics.items()})
        fold_row.update(validation_metrics)
        fold_row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        cv_rows.append(fold_row)

        learning_curve_df = evals_result_to_long_df(model.get_evals_result(), fold=fold_id)
        if not learning_curve_df.empty:
            learning_curve_frames.append(learning_curve_df)
            learning_curve_df.to_csv(
                os.path.join(run_dir, f"learning_curve_fold_{fold_id}.csv"),
                index=False,
            )

        model.model.save_model(os.path.join(run_dir, f"catboost_fold_{fold_id}.cbm"))

    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(os.path.join(run_dir, "cv_results.csv"), index=False)

    if learning_curve_frames:
        learning_curves_long = pd.concat(learning_curve_frames, ignore_index=True)
        learning_curves_long.to_csv(os.path.join(run_dir, "learning_curves_long.csv"), index=False)
        learning_curve_summary = (
            learning_curves_long.groupby(["split", "metric", "iteration"], as_index=False)
            .agg(mean=("value", "mean"), std=("value", "std"))
        )
        learning_curve_summary.to_csv(
            os.path.join(run_dir, "learning_curve_summary.csv"),
            index=False,
        )

    train_summary = build_metric_summary(
        cv_df,
        {metric: f"train_{metric}" for metric in PRIMARY_METRICS},
        prefix="train",
    )
    validation_summary = build_metric_summary(
        cv_df,
        {metric: metric for metric in PRIMARY_METRICS},
        prefix="validation",
    )
    cv_summary = {
        "cv_accuracy_mean": validation_summary["validation_accuracy_mean"],
        "cv_accuracy_std": validation_summary["validation_accuracy_std"],
        "cv_macro_f1_mean": validation_summary["validation_macro_f1_mean"],
        "cv_macro_f1_std": validation_summary["validation_macro_f1_std"],
        "cv_weighted_f1_mean": validation_summary["validation_weighted_f1_mean"],
        "cv_weighted_f1_std": validation_summary["validation_weighted_f1_std"],
    }

    X_train_final_raw = train_df_model[features].copy()
    y_train_final_raw = train_df_model[TARGET].copy()
    X_train_final_fit, y_train_final_fit = oversample_training_data(
        X_train_final_raw,
        y_train_final_raw,
        cat_features=cat_features,
        random_state=RANDOM_STATE,
        sampling_strategy="not majority",
    )

    final_model = CatBoostML(params=best_params)
    final_model.train(
        X_train=X_train_final_fit,
        y_train=y_train_final_fit,
        X_val=None,
        y_val=None,
        cat_features=cat_features,
        use_class_weights=False,
    )
    final_model.model.save_model(os.path.join(run_dir, "catboost_final.cbm"))

    X_test = test_df_model[features].copy()
    y_test = test_df_model[TARGET].copy()

    y_train_final_pred = final_model.predict(X_train_final_raw, cat_features=cat_features)
    y_test_pred = final_model.predict(X_test, cat_features=cat_features)

    final_train_metrics = compute_metrics(y_train_final_raw, y_train_final_pred)
    test_metrics = compute_metrics(y_test, y_test_pred)

    split_metrics_df = pd.DataFrame(
        [
            {"split": "train_final", **final_train_metrics},
            {"split": "test", **test_metrics},
        ]
    )
    split_metrics_df.to_csv(os.path.join(run_dir, "final_split_metrics.csv"), index=False)

    test_cm = confusion_matrix(y_test, y_test_pred, labels=final_model.model.classes_)
    test_cm_df = pd.DataFrame(
        test_cm,
        index=[f"true_{cls}" for cls in final_model.model.classes_],
        columns=[f"pred_{cls}" for cls in final_model.model.classes_],
    )
    save_confusion_matrix(test_cm_df, os.path.join(run_dir, "confusion_matrix.csv"))

    train_cm = confusion_matrix(y_train_final_raw, y_train_final_pred, labels=final_model.model.classes_)
    train_cm_df = pd.DataFrame(
        train_cm,
        index=[f"true_{cls}" for cls in final_model.model.classes_],
        columns=[f"pred_{cls}" for cls in final_model.model.classes_],
    )
    save_confusion_matrix(train_cm_df, os.path.join(run_dir, "train_confusion_matrix.csv"))

    # Per-country test-set breakdown -- only meaningful in pooled mode
    # (country_label is None). In per-country mode, test_df_model already
    # contains only one country, so this would just reproduce test_metrics
    # under a different name.
    country_test_rows = []
    if country_label is None:
        test_country_values = test_df_model["Country"].values
        for country in sorted(pd.unique(test_country_values)):
            mask = test_country_values == country
            n_country = int(mask.sum())
            if n_country == 0:
                continue
            country_metrics = compute_metrics(y_test[mask], y_test_pred[mask])
            country_row = {
                "dataset": dataset_name,
                "model_name": model_name,
                "k": k,
                "Country": country,
                "n_test": n_country,
                **{f"test_{key}": val for key, val in country_metrics.items()},
            }
            country_test_rows.append(country_row)

        country_test_df = pd.DataFrame(country_test_rows)
        country_test_df.to_csv(os.path.join(run_dir, "test_metrics_by_country.csv"), index=False)

    result_row = {
        "country": country_label or "pooled",
        "dataset": dataset_name,
        "model_name": model_name,
        "k": k,
        **train_summary,
        **validation_summary,
        **cv_summary,
        "final_train_accuracy": final_train_metrics["accuracy"],
        "final_train_macro_f1": final_train_metrics["macro_f1"],
        "final_train_weighted_f1": final_train_metrics["weighted_f1"],
        "final_train_macro_precision": final_train_metrics["macro_precision"],
        "final_train_macro_recall": final_train_metrics["macro_recall"],
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_precision": test_metrics["macro_precision"],
        "test_macro_recall": test_metrics["macro_recall"],
        "train_test_accuracy_gap": final_train_metrics["accuracy"] - test_metrics["accuracy"],
        "train_test_macro_f1_gap": final_train_metrics["macro_f1"] - test_metrics["macro_f1"],
        "train_test_weighted_f1_gap": final_train_metrics["weighted_f1"] - test_metrics["weighted_f1"],
    }

    pd.DataFrame([result_row]).to_csv(os.path.join(run_dir, "summary_metrics.csv"), index=False)
    print(pd.DataFrame([result_row]))

    return result_row, country_test_rows


# ======================================================
# 5 Run the grid and save the master summary
# ======================================================
all_results = []
all_country_results = []

base_df = pd.read_csv(BASE_DATA_PATH, low_memory=False)
ensure_required_columns(base_df, ["Country", "equivalized_income", "total_expenditure"], "base_data")
base_df = base_df.dropna(subset=["Country", "equivalized_income", "total_expenditure"]).copy()

if PER_COUNTRY_MODE:
    countries_present = sorted(base_df["Country"].dropna().unique())
    for country_code in countries_present:
        country_label = COUNTRY_NAMES.get(country_code, str(country_code))
        country_df = base_df[base_df["Country"] == country_code].copy()
        print(f"\n{'#' * 80}\n# Country: {country_label} (n={len(country_df)})\n{'#' * 80}")
        results, country_results = run_grid(country_df, MODEL_DIR, country_label=country_label)
        all_results.extend(results)
        all_country_results.extend(country_results)
else:
    results, country_results = run_grid(base_df, MODEL_DIR, country_label=None)
    all_results.extend(results)
    all_country_results.extend(country_results)

all_results_df = pd.DataFrame(all_results)
sort_cols = (["country"] if PER_COUNTRY_MODE else []) + ["dataset", "model_name", "k", "test_macro_f1"]
sort_asc = ([True] if PER_COUNTRY_MODE else []) + [True, True, True, False]
all_results_df = all_results_df.sort_values(by=sort_cols, ascending=sort_asc)

master_path = os.path.join(MODEL_DIR, "all_results_summary.csv")
all_results_df.to_csv(master_path, index=False)

all_country_results_df = pd.DataFrame(all_country_results)
country_master_path = os.path.join(MODEL_DIR, "all_results_by_country.csv")
all_country_results_df.to_csv(country_master_path, index=False)

print(f"\nFinished. Master summary saved to: {master_path}")
print(f"Per-country breakdown saved to: {country_master_path}")
print(all_results_df)

if PER_COUNTRY_MODE:
    print("\nBest config per country, by test_macro_f1:")
    if not all_results_df.empty:
        best_per_country = all_results_df.sort_values("test_macro_f1", ascending=False).groupby("country").first()
        for country_label, best_row in best_per_country.iterrows():
            print(
                f"  {country_label}: {best_row['dataset']} ({best_row['model_name']}, k={int(best_row['k'])}) "
                f"test_macro_f1={best_row['test_macro_f1']:.4f}"
            )
else:
    print("\nBest config per dataset, macro-F1 range across countries (widest first):")
    if not all_country_results_df.empty:
        best_per_dataset = all_results_df.sort_values("test_macro_f1", ascending=False).groupby("dataset").first()
        for dataset_name, best_row in best_per_dataset.iterrows():
            subset = all_country_results_df[
                (all_country_results_df["dataset"] == dataset_name)
                & (all_country_results_df["model_name"] == best_row["model_name"])
                & (all_country_results_df["k"] == best_row["k"])
            ]
            if subset.empty:
                continue
            spread = subset["test_macro_f1"].max() - subset["test_macro_f1"].min()
            print(
                f"  {dataset_name} ({best_row['model_name']}, k={best_row['k']}): "
                f"overall test_macro_f1={best_row['test_macro_f1']:.4f}, "
                f"per-country range={subset['test_macro_f1'].min():.4f}-{subset['test_macro_f1'].max():.4f} "
                f"(spread={spread:.4f})"
            )
