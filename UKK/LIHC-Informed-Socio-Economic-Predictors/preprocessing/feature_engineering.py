import pandas as pd


def get_categorical_features(df, max_unique_absolute=11):
    """
    Extract categorical features from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    max_unique_absolute : int, default=11
        If numeric, treat as categorical if number of unique values <= this threshold.
    
    Returns
    -------
    list
        List of column names considered categorical.
    """
    categorical_features = []

    for col in df.columns:
        series = df[col]

        # Object dtype is categorical
        if series.dtype == "object" or pd.api.types.is_categorical_dtype(series):
            categorical_features.append(col)
        else:
            # Numeric but with limited unique values
            n_unique = series.nunique(dropna=True)
            if n_unique <= max_unique_absolute:
                categorical_features.append(col)

    return categorical_features

