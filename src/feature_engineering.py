import pandas as pd
import numpy as np

def create_time_features(df, date_col="Date"):
    """
    Extracts time-based features from datetime column.
    """
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col])
        
    df["Year"] = df[date_col].dt.year
    df["Month"] = df[date_col].dt.month
    df["Day"] = df[date_col].dt.day
    df["DayOfWeek"] = df[date_col].dt.dayofweek
    df["Quarter"] = df[date_col].dt.quarter
    df["DayOfYear"] = df[date_col].dt.dayofyear
    df["Is_Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df["Is_Month_Start"] = df[date_col].dt.is_month_start.astype(int)
    df["Is_Month_End"] = df[date_col].dt.is_month_end.astype(int)
    
    # Cyclical encodings for Month and DayOfWeek
    df["Month_Sin"] = np.sin(2 * np.pi * df["Month"] / 12.0)
    df["Month_Cos"] = np.cos(2 * np.pi * df["Month"] / 12.0)
    df["DayOfWeek_Sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7.0)
    df["DayOfWeek_Cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7.0)
    
    return df

def create_lag_and_rolling_features(df, target_col="Units_Sold", group_col=None):
    """
    Generates lag features (1, 7, 14, 30 days) and rolling window metrics (7, 30 days).
    """
    df = df.copy()
    
    # Define group apply helper if category level or single time-series
    def _add_lags(group):
        group = group.sort_values("Date")
        for lag in [1, 7, 14, 30]:
            group[f"{target_col}_Lag_{lag}"] = group[target_col].shift(lag)
            
        for window in [7, 14, 30]:
            # Shift by 1 day to prevent data leakage from the current target
            group[f"{target_col}_Rolling_Mean_{window}"] = group[target_col].shift(1).rolling(window=window).mean()
            group[f"{target_col}_Rolling_Std_{window}"] = group[target_col].shift(1).rolling(window=window).std()
            group[f"{target_col}_Rolling_Max_{window}"] = group[target_col].shift(1).rolling(window=window).max()
            group[f"{target_col}_Rolling_Min_{window}"] = group[target_col].shift(1).rolling(window=window).min()
            
        group[f"{target_col}_Expanding_Mean"] = group[target_col].shift(1).expanding().mean()
        return group

    if group_col and group_col in df.columns:
        df = df.groupby(group_col, group_keys=False).apply(_add_lags)
    else:
        df = _add_lags(df)
        
    return df

def build_feature_pipeline(df, target_col="Units_Sold", group_col=None):
    """
    Full feature engineering pipeline.
    """
    df_features = create_time_features(df)
    df_features = create_lag_and_rolling_features(df_features, target_col=target_col, group_col=group_col)
    
    # Drop initial NA rows created by lag/rolling windows
    initial_rows = len(df_features)
    df_features = df_features.dropna().reset_index(drop=True)
    print(f"Feature engineering complete. Retained {len(df_features)} rows after dropping {initial_rows - len(df_features)} lag NA rows.")
    
    return df_features

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_daily_aggregated.csv")
    df_feat = build_feature_pipeline(df)
    df_feat.to_csv("data/featured_daily_sales.csv", index=False)
