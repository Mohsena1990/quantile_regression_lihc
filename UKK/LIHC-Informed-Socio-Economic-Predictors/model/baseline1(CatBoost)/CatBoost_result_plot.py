import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score as sk_f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve


# ---------------------------
# Confusion Matrix
# ---------------------------
def plot_confusion(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# ---------------------------
# Cross-Validation F1 & Accuracy
# ---------------------------
def plot_cv_scores(folds, cat_features, model):
    fold_metrics = []
    for i, (X_train_fold, X_val_fold, y_train_fold, y_val_fold) in enumerate(folds, 1):
        print(f"\n--- Fold {i} ---")
        for col in cat_features:
            X_train_fold[col] = X_train_fold[col].astype(str)
            X_val_fold[col] = X_val_fold[col].astype(str)

        model.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold, cat_features=cat_features)
        acc, preds = model.evaluate(X_val_fold, y_val_fold, cat_features=cat_features)
        f1 = sk_f1_score(y_val_fold, preds, average="macro")
        fold_metrics.append((i, acc, f1))

    # Plot CV metrics
    metrics_df = pd.DataFrame(fold_metrics, columns=["Fold", "Accuracy", "Macro-F1"])
    metrics_df.set_index("Fold").plot(marker="o", figsize=(8, 5))
    plt.title("Cross-Validation Performance")
    plt.ylabel("Score")
    plt.show()

    return metrics_df


# ---------------------------
# Feature Importance
# ---------------------------
def plot_feature_importance(fi, top_n=20):
    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi.head(top_n), x="Importance", y="Feature", palette="viridis")
    plt.title(f"Top {top_n} Important Features")
    plt.show()


# ---------------------------
# Learning Curve
# ---------------------------
def plot_learning_curve(model):
    evals_result = model.model.get_evals_result()
    plt.figure(figsize=(8, 5))
    for dataset in evals_result:
        if "Accuracy" in evals_result[dataset]:
            plt.plot(evals_result[dataset]['Accuracy'], label=dataset)
    plt.legend()
    plt.title("Learning Curve")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.show()


# ---------------------------
# ROC Curve (Multi-class)
# ---------------------------
def plot_roc(y_true, y_pred_proba, classes):
    y_true_bin = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(8, 6))

    for i, class_name in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{class_name} (AUC={roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()


# ---------------------------
# Precision-Recall Curve (Multi-class)
# ---------------------------
def plot_pr(y_true, y_pred_proba, classes):
    y_true_bin = label_binarize(y_true, classes=classes)
    plt.figure(figsize=(8, 6))

    for i, class_name in enumerate(classes):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, lw=2, label=f"{class_name} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()


# ---------------------------
# Probability Distribution
# ---------------------------
def plot_class_probs(y_pred_proba, classes):
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(classes):
        sns.histplot(y_pred_proba[:, i], bins=20, kde=True, label=class_name, stat="density")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Density")
    plt.title("Predicted Probability Distribution")
    plt.legend()
    plt.show()


# ---------------------------
# Calibration Curve
# ---------------------------
def plot_calibration(y_true, y_pred_proba, class_idx, class_name):
    prob_true, prob_pred = calibration_curve((y_true == class_name).astype(int),
                                             y_pred_proba[:, class_idx], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', label=class_name)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Calibration Curve - {class_name}")
    plt.legend()
    plt.show()
