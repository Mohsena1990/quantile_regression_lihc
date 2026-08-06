import random

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from sklearn.model_selection import train_test_split, StratifiedKFold


def _build_smote_sampler(
    X_tr,
    y_tr,
    categorical_cols,
    sampling_strategy,
    random_state,
):
    """Build a fold-safe SMOTE/SMOTENC sampler."""
    train_counts = pd.Series(y_tr).value_counts()
    if train_counts.empty or len(train_counts) < 2:
        print("Skipping SMOTE: fewer than 2 classes are present in this training fold.")
        return None

    min_class_size = int(train_counts.min())
    if min_class_size < 2:
        print(
            "Skipping SMOTE: at least one training class has fewer than 2 samples, "
            "so synthetic neighbors cannot be constructed."
        )
        return None

    k_neighbors = min(5, min_class_size - 1)

    if categorical_cols:
        cat_indices = [X_tr.columns.get_loc(c) for c in categorical_cols]
        return SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
        )

    return SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=k_neighbors,
    )


def _group_class_table(groups: pd.Series, y: pd.Series) -> pd.DataFrame:
    table = pd.crosstab(groups, y)
    class_order = list(pd.Series(y).value_counts().index)
    table = table.reindex(columns=class_order, fill_value=0)
    table["__total__"] = table.sum(axis=1)
    return table


def _country_priority(row: pd.Series, total_class_counts: pd.Series) -> tuple:
    rare_burden = 0.0
    for cls, count in row.drop(labels="__total__").items():
        if total_class_counts[cls] > 0:
            rare_burden += (count / total_class_counts[cls]) ** 2
    return (rare_burden, row["__total__"])


def _fold_score(fold_counts: pd.Series, target_counts: pd.Series, rare_weights: pd.Series, target_size: float) -> float:
    score = 0.0
    for cls in target_counts.index:
        if target_counts[cls] > 0:
            diff = (fold_counts[cls] - target_counts[cls]) / target_counts[cls]
            score += rare_weights[cls] * (diff ** 2)
    if target_size > 0:
        size_diff = (fold_counts["__total__"] - target_size) / target_size
        score += size_diff ** 2
    return float(score)


def _build_balanced_group_folds(
    groups: pd.Series,
    y: pd.Series,
    n_splits: int,
    random_state: int = 42,
    n_trials: int = 256,
):
    """Greedy country-to-fold assignment that balances per-fold class counts."""
    group_table = _group_class_table(groups, y)
    unique_groups = group_table.index.tolist()

    if n_splits > len(unique_groups):
        raise ValueError(
            f"n_splits={n_splits} exceeds number of unique groups={len(unique_groups)}"
        )

    classes = [c for c in group_table.columns if c != "__total__"]
    total_class_counts = group_table[classes].sum(axis=0)
    target_counts = total_class_counts / n_splits
    target_size = float(group_table["__total__"].sum()) / n_splits
    rare_weights = (total_class_counts.max() / total_class_counts).replace([np.inf, -np.inf], 1.0).fillna(1.0)

    best_assignment = None
    best_score = None

    for trial in range(max(1, n_trials)):
        rng = random.Random(random_state + trial)
        ordered_groups = unique_groups[:]
        ordered_groups.sort(
            key=lambda g: _country_priority(group_table.loc[g], total_class_counts),
            reverse=True,
        )

        if trial > 0:
            head = ordered_groups[: min(6, len(ordered_groups))]
            tail = ordered_groups[min(6, len(ordered_groups)) :]
            rng.shuffle(head)
            rng.shuffle(tail)
            ordered_groups = head + tail

        fold_groups = [[] for _ in range(n_splits)]
        fold_counts = [pd.Series(0.0, index=classes + ["__total__"]) for _ in range(n_splits)]

        for group in ordered_groups:
            group_counts = group_table.loc[group, classes + ["__total__"]].astype(float)
            candidate_scores = []

            for fold_idx in range(n_splits):
                new_counts = fold_counts[fold_idx] + group_counts
                score = _fold_score(new_counts, target_counts, rare_weights, target_size)

                total_score = 0.0
                for idx in range(n_splits):
                    counts = new_counts if idx == fold_idx else fold_counts[idx]
                    total_score += _fold_score(counts, target_counts, rare_weights, target_size)

                total_score += 0.02 * len(fold_groups[fold_idx])
                candidate_scores.append((total_score + score, fold_idx))

            _, best_fold_idx = min(candidate_scores, key=lambda item: item[0])
            fold_groups[best_fold_idx].append(group)
            fold_counts[best_fold_idx] = fold_counts[best_fold_idx] + group_counts

        if any(len(fold) == 0 for fold in fold_groups):
            continue

        score = sum(_fold_score(counts, target_counts, rare_weights, target_size) for counts in fold_counts)
        min_rare = min(counts[classes[0]] for counts in fold_counts) if classes else 0.0
        score -= 0.01 * min_rare

        if best_score is None or score < best_score:
            best_score = score
            best_assignment = [fold[:] for fold in fold_groups]

    if best_assignment is None:
        raise RuntimeError("Failed to construct balanced grouped folds.")

    return best_assignment


def train_and_test_splitting(
    df,
    inputs,
    target,
    test_size=0.25,
    random_state=42
):
    """
    Standard stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X = df[inputs].copy()
    y = df[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print("X Train size:", X_train.shape)
    print("X Test size:", X_test.shape)
    print("Y Train size:", y_train.shape)
    print("Y Test size:", y_test.shape)

    return X_train, X_test, y_train, y_test


def country_stratified_group_split(
    df,
    inputs,
    target,
    n_splits=5,
    random_state=42,
    use_smote=False,
    categorical_cols=None,
    sampling_strategy="not majority",
    group_col="Country"
):
    """
    Country-aware grouped CV split with fold balancing at the country level.

    Returns
    -------
    list
        List of tuples: (X_tr, X_val, y_tr, y_val)
    """
    X = df[inputs].copy()
    y = df[target].copy()
    groups = df[group_col].copy()

    fold_groups = _build_balanced_group_folds(
        groups=groups,
        y=y,
        n_splits=n_splits,
        random_state=random_state,
    )

    folds = []
    all_classes = pd.Index(pd.Series(y).dropna().unique())

    for fold_id, val_groups in enumerate(fold_groups, start=1):
        val_mask = groups.isin(val_groups)
        train_mask = ~val_mask

        X_tr = X.loc[train_mask].copy()
        X_val = X.loc[val_mask].copy()
        y_tr = y.loc[train_mask].copy()
        y_val = y.loc[val_mask].copy()

        train_counts_before = y_tr.value_counts()
        missing_train_classes = [cls for cls in all_classes if cls not in train_counts_before.index]

        categorical_cols = categorical_cols or []
        categorical_cols = [c for c in categorical_cols if c in X_tr.columns]

        for col in X_tr.columns:
            if col in categorical_cols:
                X_tr[col] = X_tr[col].astype(str).fillna("missing")
                if col in X_val.columns:
                    X_val[col] = X_val[col].astype(str).fillna("missing")
            else:
                X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
                X_tr[col] = X_tr[col].replace([np.inf, -np.inf], np.nan)
                X_tr[col] = X_tr[col].fillna(X_tr[col].median())

                if col in X_val.columns:
                    X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
                    X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan)
                    X_val[col] = X_val[col].fillna(X_tr[col].median())

        if use_smote:
            sampler = _build_smote_sampler(
                X_tr=X_tr,
                y_tr=y_tr,
                categorical_cols=categorical_cols,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )
            if sampler is not None:
                X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

        print(f"\nFold {fold_id}")
        print("Validation countries:", sorted(pd.Series(val_groups).astype(str).tolist()))
        print("Training class counts before SMOTE:")
        print(train_counts_before)
        if missing_train_classes:
            print("Missing training classes in this fold:", missing_train_classes)
        if use_smote:
            print("Training class counts after SMOTE:")
            print(pd.Series(y_tr).value_counts())
        print("Validation class counts:")
        print(y_val.value_counts())

        folds.append((X_tr, X_val, y_tr, y_val))

    return folds


def class_stratified_k_fold_split(
    df,
    inputs,
    target,
    n_splits=5,
    random_state=42,
    use_smote=False,
    categorical_cols=None,
    sampling_strategy="not majority",
):
    """
    Plain class-stratified K-fold CV split (no country grouping).

    Use this instead of country_stratified_group_split when df has
    already been filtered to a single country -- that function requires
    n_splits <= number of unique countries in the group column, which is
    always 1 in that case and would raise immediately. Raises ValueError
    up front if any class has fewer than n_splits members, since
    StratifiedKFold would otherwise silently produce folds missing that
    class entirely rather than erroring.

    Returns
    -------
    list
        List of tuples: (X_tr, X_val, y_tr, y_val)
    """
    X = df[inputs].copy()
    y = df[target].copy()

    class_counts = y.value_counts()
    if len(class_counts) < 2:
        raise ValueError(f"Only {len(class_counts)} class(es) present; need at least 2.")
    if class_counts.min() < n_splits:
        raise ValueError(
            f"Smallest class ({class_counts.idxmin()}) has {class_counts.min()} samples, "
            f"fewer than n_splits={n_splits}."
        )

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    categorical_cols = categorical_cols or []
    categorical_cols = [c for c in categorical_cols if c in X.columns]

    folds = []
    for fold_id, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_tr = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        for col in X_tr.columns:
            if col in categorical_cols:
                X_tr[col] = X_tr[col].astype(str).fillna("missing")
                if col in X_val.columns:
                    X_val[col] = X_val[col].astype(str).fillna("missing")
            else:
                X_tr[col] = pd.to_numeric(X_tr[col], errors="coerce")
                X_tr[col] = X_tr[col].replace([np.inf, -np.inf], np.nan)
                X_tr[col] = X_tr[col].fillna(X_tr[col].median())

                if col in X_val.columns:
                    X_val[col] = pd.to_numeric(X_val[col], errors="coerce")
                    X_val[col] = X_val[col].replace([np.inf, -np.inf], np.nan)
                    X_val[col] = X_val[col].fillna(X_tr[col].median())

        train_counts_before = y_tr.value_counts()

        if use_smote:
            sampler = _build_smote_sampler(
                X_tr=X_tr,
                y_tr=y_tr,
                categorical_cols=categorical_cols,
                sampling_strategy=sampling_strategy,
                random_state=random_state,
            )
            if sampler is not None:
                X_tr, y_tr = sampler.fit_resample(X_tr, y_tr)

        print(f"\nFold {fold_id}")
        print("Training class counts before SMOTE:")
        print(train_counts_before)
        if use_smote:
            print("Training class counts after SMOTE:")
            print(pd.Series(y_tr).value_counts())
        print("Validation class counts:")
        print(y_val.value_counts())

        folds.append((X_tr, X_val, y_tr, y_val))

    return folds
