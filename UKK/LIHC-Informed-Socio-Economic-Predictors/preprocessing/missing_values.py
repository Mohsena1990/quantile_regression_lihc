import numpy as np
import pandas as pd


def find_unknown_tokens(df, max_unique=50):
    """
    Finds suspicious/unknown tokens in each column.
    Returns a dictionary: {column_name: [list of unknown tokens]}.
    """
    # Define common unknown markers

    standard_unknowns = ["", " ", "nan", "NaN", "NULL", "null", "None", "-", "?"]
    # standard_unknowns = {"", " ", "nan", "NaN", "NULL", "null", "None", "-", "?", "N/A", "n/a"}
    unknowns = {}

    for col in df.columns:
        col_values = df[col].dropna().unique()
        
        # Skip free-text / high cardinality columns
        if len(col_values) > max_unique:
            continue
        
        # Numeric column → check for non-numeric strings
        if np.issubdtype(df[col].dtype, np.number):   # 🔹 FIXED
            suspicious = [
                val for val in col_values 
                if isinstance(val, str) and not val.replace('.', '', 1).isdigit()
            ]
        else:
            # Categorical/text → check if in unknown tokens
            suspicious = [val for val in col_values if str(val).strip() in standard_unknowns]
        
        if suspicious:
            unknowns[col] = suspicious

    print("*****************",unknowns )
    
    return unknowns


def report_missing_and_unknowns(df):
    """
    Reports missing values and unknown tokens for each column.
    Internally calls find_unknown_tokens().
    Returns the dictionary of unknown tokens for reuse.
    """
    unknown_dict = find_unknown_tokens(df)  # 🔹 Call inside

    for col in df.columns:
        missing_count = df[col].isna().sum()
        unknown_count = 0
        if col in unknown_dict:
            unknown_count = df[col].astype(str).isin(map(str, unknown_dict[col])).sum()
        
        if missing_count > 0:
            print(f"⚠️ {col}: {missing_count} missing values")
        if unknown_count > 0:
            print(f"⚠️ {col}: {unknown_count} unknown token(s) -> {unknown_dict[col]}")

    return unknown_dict   



# def find_missing_and_unknown(df):
#     """
#     Prints a clear line-by-line report of missing and unknown values in each column.
#     """
#     unknown_tokens = ["", " ", "nan", "NaN", "NULL", "null", "None", "-", "?"]

#     for col in df.columns:
#         missing_count = df[col].isna().sum()
#         unknown_count = df[col].astype(str).isin(unknown_tokens).sum()

#         if missing_count > 0:
#             print(f"{col} contains {missing_count} missing values")
#         if unknown_count > 0:
#             print(f"{col} contains {unknown_count} unknown values")




# Cleaning strategies

def drop_high_missing_rows(df, threshold=0.6):
    """Drop rows with >threshold (e.g., 60%) missing values (#NULL! or NaN) to reduce missing bias"""
    # Count missing per row (treat #NULL! as NaN)
    missing_ratio = df.isnull().mean(axis=1)  # NaN count
    return df[missing_ratio <= threshold]


def drop_missing(df, feature):
    """Drop rows where `feature` has missing/unknown values"""
    unknown_tokens = ["", " ", "nan", "NaN", "NULL", "null", "None", "-", "?"]
    return df[~df[feature].astype(str).isin(unknown_tokens) & df[feature].notna()]

def fill_with_constant(df, feature, value):
    """Replace missing/unknown with a constant value"""
    unknown_tokens = ["", " ", "nan", "NaN", "NULL", "null", "None", "-", "?"]
    df[feature] = df[feature].replace(unknown_tokens, np.nan)
    df[feature] = df[feature].fillna(value)
    return df


def fill_with_mean(df, features):
    """
    Replace missing numeric values with mean.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    features : str or list-like
        Single column name or list of column names
    
    Returns
    -------
    pd.DataFrame
        DataFrame with NaNs replaced by mean for given features
    """
    if isinstance(features, (str, int)):  # single column
        features = [features]

    for feature in features:
        if pd.api.types.is_numeric_dtype(df[feature]):
            df[feature] = pd.to_numeric(df[feature], errors="coerce")
            df[feature] = df[feature].fillna(df[feature].mean())
    return df




# def fill_with_mean(df, feature):
#     """Replace missing numeric values with mean"""
#     df[feature] = pd.to_numeric(df[feature], errors="coerce")  # force numeric
#     df[feature] = df[feature].fillna(df[feature].mean())
#     return df

# def fill_with_median(df, feature):
#     """Replace missing numeric values with median"""
#     df[feature] = pd.to_numeric(df[feature], errors="coerce")
#     df[feature] = df[feature].fillna(df[feature].median())
#     return df


def fill_with_median(df):
    """Replace missing numeric values with median for all numeric columns"""
    for feature in df.select_dtypes(include=["number"]).columns:
        df[feature] = pd.to_numeric(df[feature], errors="coerce")
        df[feature] = df[feature].fillna(df[feature].median())
    return df


def fill_with_mode(df, feature):
    """Replace missing categorical values with mode"""
    df[feature] = df[feature].replace(["", " ", "nan", "NaN", "NULL", "null", "None", "-", "?"], np.nan)
    df[feature] = df[feature].fillna(df[feature].mode()[0])
    return df


def drop_feature(df, feature):
    """
    Drop irrelevant or redundant features from the DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame
    feature : str or list
        Column name (or list of column names) to drop
    
    Returns:
    --------
    pd.DataFrame
        DataFrame without the specified feature(s)
    """

    df = df.drop(columns=feature, errors="ignore")

    return df






def drop_nan(df, how="any", axis=0, subset=None):
    """
    Drop rows or columns with NaN or common null-like values.
    """
    return df.dropna(axis=axis, how=how, subset=subset)
