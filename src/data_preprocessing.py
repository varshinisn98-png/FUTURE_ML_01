import os
import pandas as pd
import numpy as np

def clean_data(input_path="data/raw_sales_data.csv", output_path="data/cleaned_sales_data.csv"):
    """
    Cleans raw sales dataset:
    - Parses dates and sorts chronologically
    - Imputes missing numerical values (e.g., Price using median by category)
    - Re-calculates Revenue for consistency
    - Aggregates daily total sales and per-category sales
    """
    df = pd.read_csv(input_path)
    print(f"Raw data shape: {df.shape}")
    
    # 1. Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["Date", "Category"]).reset_index(drop=True)
    
    # 2. Impute missing values (Price missing values filled with Category median)
    if df["Price"].isnull().sum() > 0:
        print(f"Imputing {df['Price'].isnull().sum()} missing values in 'Price'...")
        df["Price"] = df.groupby("Category")["Price"].transform(lambda x: x.fillna(x.median()))
        
    # 3. Recalculate Revenue
    df["Revenue"] = (df["Units_Sold"] * df["Price"]).round(2)
    
    # 4. Outlier capping (IQR method on Units_Sold per Category)
    for cat in df["Category"].unique():
        cat_mask = df["Category"] == cat
        q1 = df.loc[cat_mask, "Units_Sold"].quantile(0.25)
        q3 = df.loc[cat_mask, "Units_Sold"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 3 * iqr  # Using 3x IQR to only cap severe outliers
        df.loc[cat_mask & (df["Units_Sold"] > upper_bound), "Units_Sold"] = int(upper_bound)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to '{output_path}'. Shape: {df.shape}")
    
    # Aggregate to overall Daily Sales level for macro forecasting
    daily_aggregated = df.groupby("Date").agg({
        "Units_Sold": "sum",
        "Revenue": "sum",
        "Is_Promotion": "max",
        "Price": "mean",
        "Competitor_Price_Index": "mean"
    }).reset_index()
    daily_aggregated_path = os.path.join(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", "cleaned_daily_aggregated.csv")
    daily_aggregated.to_csv(daily_aggregated_path, index=False)
    print(f"Aggregated daily sales dataset saved to '{daily_aggregated_path}'")
    
    return df, daily_aggregated

if __name__ == "__main__":
    clean_data()
