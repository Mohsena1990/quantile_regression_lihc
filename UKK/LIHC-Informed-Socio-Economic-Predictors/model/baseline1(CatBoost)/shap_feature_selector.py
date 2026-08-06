import os

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from typing import List, Optional
from CatBoost import CatBoostML


class SHAPAnalyzer:
    """
    SHAP interpretation module for the revised strategy.

    Use cases
    ---------
    1. Global feature importance for one model
    2. Fold-level SHAP stability
    3. Country-level SHAP consistency
    """

    def __init__(self, sample_size: int = 1000, random_state: int = 42):
        self.sample_size = sample_size
        self.random_state = random_state
        self.model = None
        self.explainer = None
        self.global_importance_ = None
        self.fold_importance_ = None
        self.country_importance_ = None

    def _prepare_sample(self, X: pd.DataFrame) -> pd.DataFrame:
        if len(X) <= self.sample_size:
            return X.copy()
        return X.sample(self.sample_size, random_state=self.random_state).copy()

    def fit_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        cat_features: Optional[List[str]] = None,
        params: Optional[dict] = None,
        use_class_weights: bool = False
    ):
        """
        Train CatBoost model and create SHAP explainer.
        """
        model_wrapper = CatBoostML(params=params)
        model_wrapper.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            cat_features=cat_features,
            use_class_weights=use_class_weights
        )

        self.model = model_wrapper.model
        self.explainer = shap.TreeExplainer(self.model, feature_perturbation="tree_path_dependent")
        return self

    def compute_global_importance(
        self,
        X: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compute mean absolute SHAP importance on a sample of X.
        """
        if self.explainer is None:
            raise ValueError("Model must be fitted before computing SHAP.")

        X_sample = self._prepare_sample(X)
        shap_values = self.explainer.shap_values(X_sample)

        if isinstance(shap_values, list):  # multiclass
            stacked = np.stack(shap_values, axis=2)
            importance = np.abs(stacked).mean(axis=(0, 2))
        else:
            importance = np.abs(shap_values).mean(axis=0)

        self.global_importance_ = pd.DataFrame({
            "feature": X_sample.columns,
            "importance": importance
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        return self.global_importance_

    def compute_fold_stability(
        self,
        folds,
        features: List[str],
        cat_features: Optional[List[str]] = None,
        params: Optional[dict] = None,
        use_class_weights: bool = True
    ) -> pd.DataFrame:
        """
        Compute SHAP importance across folds and summarize stability.
        """
        fold_rows = []

        for fold_id, (X_tr, X_val, y_tr, y_val) in enumerate(folds, start=1):
            model_wrapper = CatBoostML(params=params)
            model_wrapper.train(
                X_train=X_tr[features],
                y_train=y_tr,
                X_val=X_val[features],
                y_val=y_val,
                cat_features=cat_features,
                use_class_weights=use_class_weights
            )

            explainer = shap.TreeExplainer(model_wrapper.model, feature_perturbation="tree_path_dependent")
            X_sample = self._prepare_sample(X_val[features])
            shap_values = explainer.shap_values(X_sample)

            if isinstance(shap_values, list):
                stacked = np.stack(shap_values, axis=2)
                importance = np.abs(stacked).mean(axis=(0, 2))
            else:
                importance = np.abs(shap_values).mean(axis=0)

            fold_df = pd.DataFrame({
                "fold": fold_id,
                "feature": X_sample.columns,
                "importance": importance
            })
            fold_rows.append(fold_df)

        fold_importance = pd.concat(fold_rows, ignore_index=True)

        summary = (
            fold_importance.groupby("feature")["importance"]
            .agg(["mean", "std"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        self.fold_importance_ = summary
        return summary

    def compute_country_importance(
        self,
        X: pd.DataFrame,
        country_col: str = "Country"
    ) -> pd.DataFrame:
        """
        Compute country-level SHAP importance from an already fitted model.
        """
        if self.explainer is None:
            raise ValueError("Model must be fitted before computing SHAP.")
        if country_col not in X.columns:
            raise KeyError(f"{country_col} not found in X")

        rows = []

        for country, group in X.groupby(country_col):
            X_country = self._prepare_sample(group)

            shap_values = self.explainer.shap_values(X_country)

            if isinstance(shap_values, list):
                stacked = np.stack(shap_values, axis=2)
                importance = np.abs(stacked).mean(axis=(0, 2))
            else:
                importance = np.abs(shap_values).mean(axis=0)

            temp = pd.DataFrame({
                "country": country,
                "feature": X_country.columns,
                "importance": importance
            })
            rows.append(temp)

        out = pd.concat(rows, ignore_index=True)
        self.country_importance_ = out
        return out

    def plot_global_importance(
        self,
        top_n: int = 10,
        output_path: Optional[str] = None,
        title: str = "Global SHAP Feature Importance"
    ):
        """
        Plot top-n global SHAP importance.
        """
        if self.global_importance_ is None:
            raise ValueError("Run compute_global_importance first.")

        plot_df = self.global_importance_.head(top_n).sort_values("importance", ascending=True)

        plt.figure(figsize=(10, 6))
        plt.barh(plot_df["feature"], plot_df["importance"])
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel("Mean |SHAP value|", fontsize=13, fontweight="bold")
        plt.ylabel("Feature", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if output_path:
            root, _ = os.path.splitext(output_path)
            plt.savefig(f"{root}.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{root}.pdf", bbox_inches="tight")
            plt.close()
        else:
            plt.show()

    def plot_country_heatmap(
        self,
        output_path: Optional[str] = None,
        title: str = "Country-level SHAP Importance"
    ):
        """
        Plot country-feature SHAP heatmap.
        """
        if self.country_importance_ is None:
            raise ValueError("Run compute_country_importance first.")

        pivot_df = self.country_importance_.pivot_table(
            index="country",
            columns="feature",
            values="importance",
            aggfunc="mean"
        ).fillna(0)

        plt.figure(figsize=(14, 8))
        plt.imshow(pivot_df, aspect="auto")
        plt.colorbar(label="Mean |SHAP value|")
        plt.xticks(range(len(pivot_df.columns)), pivot_df.columns, rotation=90)
        plt.yticks(range(len(pivot_df.index)), pivot_df.index)
        plt.title(title, fontsize=16, fontweight="bold")
        plt.tight_layout()

        if output_path:
            root, _ = os.path.splitext(output_path)
            plt.savefig(f"{root}.png", dpi=300, bbox_inches="tight")
            plt.savefig(f"{root}.pdf", bbox_inches="tight")
            plt.close()
        else:
            plt.show()