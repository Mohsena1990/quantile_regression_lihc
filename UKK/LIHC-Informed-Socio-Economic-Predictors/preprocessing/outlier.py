
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np









# def remove_outliers_iqr(df, feature, multiplier=1.5):
#     """Remove outliers for a numeric feature using IQR method"""
#     Q1 = df[feature].quantile(0.05)
#     Q3 = df[feature].quantile(0.95)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - multiplier * IQR
#     upper_bound = Q3 + multiplier * IQR
#     return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

def remove_outliers_iqr(df, features, multiplier=1.5, method='remove'):
    """
    Remove or cap outliers for numeric features using the IQR method.
    
    Args:
        df (pd.DataFrame): Input DataFrame.
        features (str or list): Single column name or list of column names to process.
        multiplier (float): IQR multiplier to define outlier bounds (default: 1.5).
        method (str): 'remove' to drop outliers, 'cap' to replace with bounds (default: 'remove').
    
    Returns:
        pd.DataFrame: DataFrame with outliers handled according to the method.
    """
    # Convert single feature to list for consistency
    if isinstance(features, str):
        features = [features]
    
    # Validate features
    for feature in features:
        if feature not in df.columns:
            raise KeyError(f"Feature '{feature}' not found in DataFrame.")
        if not pd.api.types.is_numeric_dtype(df[feature]):
            raise TypeError(f"Feature '{feature}' must be numeric.")
    
    df_clean = df.copy()
    
    for feature in features:
        # Handle NaN values temporarily to avoid issues in quantile calculation
        mask = df_clean[feature].notna()
        data = df_clean.loc[mask, feature]
        
        # Calculate IQR with standard 25th and 75th percentiles
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        if method == 'remove':
            # Remove outliers
            df_clean = df_clean[(df_clean[feature].notna()) & 
                              (df_clean[feature] >= lower_bound) & 
                              (df_clean[feature] <= upper_bound)]
        elif method == 'cap':
            # Cap outliers at bounds
            df_clean[feature] = np.where(df_clean[feature] < lower_bound, lower_bound,
                                       np.where(df_clean[feature] > upper_bound, upper_bound, df_clean[feature]))
        else:
            raise ValueError("Method must be 'remove' or 'cap'.")
    
    return df_clean

# def remove_outliers_iqr(df, feature, multiplier=1.5):
#     """Remove outliers for a numeric feature using IQR method"""
#     Q1 = df[feature].quantile(0.03)  # 3d percentile
#     Q3 = df[feature].quantile(0.99)  # 99th percentile
#     IQR = Q3 - Q1
#     lower_bound = Q1 - multiplier * IQR
#     upper_bound = Q3 + multiplier * IQR
#     return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

def get_numeric_dataset(df):
    """
    Return only numeric columns (int and float) from a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with only numeric columns
    """
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    return numeric_df


def report_outliers(df, method="IQR", factor=1.5):
    results = []

    for col in df.columns:
        # Try numeric conversion
        series = pd.to_numeric(
            df[col].astype(str).str.replace(",", "").str.replace("%", ""), 
            errors="coerce"
        ).dropna()

        if series.empty:
            continue  # skip if no numeric values

        if method == "IQR":
            Q1 = series.quantile(0.02)
            Q3 = series.quantile(0.99)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            outliers = series[(series < lower) | (series > upper)]

        n_outliers = outliers.shape[0]
        pct_outliers = n_outliers / len(series) * 100

        results.append({
            "Feature": col,
            "Num_Outliers": int(n_outliers),
            "Percent_Outliers": round(pct_outliers, 2),
            "Min": series.min(),
            "Max": series.max()
        })

    if not results:
        print("⚠️ No numeric data found or all columns skipped.")
        return pd.DataFrame(columns=["Feature", "Num_Outliers", "Percent_Outliers", "Min", "Max"])

    return pd.DataFrame(results).sort_values(by="Percent_Outliers", ascending=False)

# def report_outliers(df, method="IQR", factor=1.5):
#     """
#     Reports the number and percentage of outliers for numeric features.
#     Automatically converts numeric-like columns to float.
#     """
#     results = []

#     for col in df.columns:
#         # Convert column to numeric if possible, coerce errors to NaN
#         series = pd.to_numeric(df[col].astype(str).str.replace(",", "").str.replace("%", ""), errors="coerce").dropna()
        
#         if series.empty:
#             continue

#         if method == "IQR":
#             Q1 = series.quantile(0.25)
#             Q3 = series.quantile(0.75)
#             IQR = Q3 - Q1
#             lower = Q1 - factor * IQR
#             upper = Q3 + factor * IQR
#             outliers = series[(series < lower) | (series > upper)]

#         n_outliers = outliers.shape[0]
#         pct_outliers = n_outliers / len(series) * 100

#         results.append({
#             "Feature": col,
#             "Num_Outliers": n_outliers,
#             "Percent_Outliers": round(pct_outliers, 2)
#         })

#     if not results:
#         print("No numeric data found or no outliers detected.")
#         return pd.DataFrame(columns=["Feature", "Num_Outliers", "Percent_Outliers"])

#     return pd.DataFrame(results).sort_values(by="Percent_Outliers", ascending=False)





def plot_outliers_per_feature(df, pause=True):
    """
    Plots boxplots of numeric features one by one to inspect outliers.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    pause : bool, default True
        Whether to wait for user input before plotting the next feature.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

    if len(numeric_cols) == 0:
        print("No numeric features found.")
        return

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            print(f"{col} has no valid numeric data.")
            continue
        
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=series, color="skyblue")
        plt.title(f"Outliers in feature: {col}")
        plt.ylabel(col)
        plt.show()

        if pause:
            input("Press Enter to continue to the next feature...")




# def outlier_plot(df, max_cols=4):
#     """
#     Plots boxplots of numeric features to visually inspect outliers.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Input dataframe
#     max_cols : int
#         Number of subplots per row
#     """
#     # Select numeric-like columns (force conversion)
#     numeric_cols = []
#     for col in df.columns:
#         try:
#             pd.to_numeric(df[col], errors="raise")
#             numeric_cols.append(col)
#         except:
#             continue
    
#     if len(numeric_cols) == 0:
#         print("No numeric features found.")
#         return
    
#     n_features = len(numeric_cols)
#     n_rows = (n_features + max_cols - 1) // max_cols  # ceil division
    
#     plt.figure(figsize=(max_cols * 4, n_rows * 4))
    
#     for i, col in enumerate(numeric_cols, 1):
#         plt.subplot(n_rows, max_cols, i)
#         series = pd.to_numeric(df[col], errors="coerce").dropna()
#         if series.empty:
#             plt.text(0.5, 0.5, "No valid data", ha="center", va="center")
#         else:
#             sns.boxplot(y=series, color="skyblue")
#         plt.title(f"{col}", fontsize=10)
#         plt.xlabel("")
    
#     plt.tight_layout()
#     plt.show()



def remove_right_tail(df, features=None, keep_fraction=0.98, plot=True):
    """
    Remove the top (1 - keep_fraction) portion of values (right tail) 
    from the distribution of each selected feature.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (str or list, optional): Column(s) to process. 
                                          If None, all numeric columns are used.
        keep_fraction (float): Fraction of lower values to keep (default=0.8).
        plot (bool): If True, show before/after distribution plots.

    Returns:
        pd.DataFrame: Cleaned DataFrame with right-tail values removed.
    """
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()
    if isinstance(features, str):
        features = [features]

    df_clean = df.copy()

    for feature in features:
        if feature not in df_clean.columns:
            raise KeyError(f"Feature '{feature}' not found in DataFrame.")

        # Plot before trimming
        # if plot:
        #     plt.figure(figsize=(6, 4))
        #     sns.histplot(df_clean[feature], kde=True, bins=30)
        #     plt.title(f"Before trimming: {feature}")
        #     plt.show()

        # Compute cutoff: keep only lower X%
        cutoff = df_clean[feature].quantile(keep_fraction)
        df_clean = df_clean[df_clean[feature] <= cutoff]

        # Plot after trimming
        # if plot:
        #     plt.figure(figsize=(6, 4))
        #     sns.histplot(df_clean[feature], kde=True, bins=30, color="orange")
        #     plt.title(f"After trimming (kept {int(keep_fraction*100)}%): {feature}")
        #     plt.show()

    return df_clean



def remove_left_tail(df, features=None, keep_fraction=0.05, plot=True):
    """
    Remove the top (1 - keep_fraction) portion of values (right tail) 
    from the distribution of each selected feature.

    Args:
        df (pd.DataFrame): Input DataFrame.
        features (str or list, optional): Column(s) to process. 
                                          If None, all numeric columns are used.
        keep_fraction (float): Fraction of lower values to keep (default=0.8).
        plot (bool): If True, show before/after distribution plots.

    Returns:
        pd.DataFrame: Cleaned DataFrame with right-tail values removed.
    """
    if features is None:
        features = df.select_dtypes(include="number").columns.tolist()
    if isinstance(features, str):
        features = [features]

    df_clean = df.copy()

    for feature in features:
        if feature not in df_clean.columns:
            raise KeyError(f"Feature '{feature}' not found in DataFrame.")

        # Plot before trimming
        # if plot:
        #     plt.figure(figsize=(6, 4))
        #     sns.histplot(df_clean[feature], kde=True, bins=30)
        #     plt.title(f"Before trimming: {feature}")
        #     plt.show()

        # Compute cutoff: keep only lower X%
        cutoff = df_clean[feature].quantile(keep_fraction)
        df_clean = df_clean[df_clean[feature] <= cutoff]

        # Plot after trimming
        # if plot:
        #     plt.figure(figsize=(6, 4))
        #     sns.histplot(df_clean[feature], kde=True, bins=30, color="orange")
        #     plt.title(f"After trimming (kept {int(keep_fraction*100)}%): {feature}")
        #     plt.show()

    return df_clean





