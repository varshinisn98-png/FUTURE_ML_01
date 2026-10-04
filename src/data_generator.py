import os
import numpy as np
import pandas as pd

def generate_sales_data(start_date="2021-01-01", end_date="2023-12-31", output_path="data/raw_sales_data.csv", seed=42):
    """
    Generates a realistic multi-year daily retail sales dataset with multiple product categories,
    seasonal trends, holiday effects, promotional boosts, and price elasticity.
    """
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")
    categories = ["Electronics", "Clothing", "Groceries", "Home & Kitchen"]
    
    records = []
    
    # Base parameters per category
    cat_params = {
        "Electronics": {"base_sales": 150, "price_mean": 299.99, "volatility": 25, "promo_sensitivity": 1.4},
        "Clothing": {"base_sales": 220, "price_mean": 49.99, "volatility": 35, "promo_sensitivity": 1.3},
        "Groceries": {"base_sales": 500, "price_mean": 12.50, "volatility": 40, "promo_sensitivity": 1.15},
        "Home & Kitchen": {"base_sales": 180, "price_mean": 89.99, "volatility": 20, "promo_sensitivity": 1.25}
    }
    
    for dt in date_range:
        day_of_year = dt.dayofyear
        day_of_week = dt.dayofweek
        month = dt.month
        year = dt.year
        
        # Trend component (growing business over time)
        trend = (dt - pd.Timestamp(start_date)).days / 365.0 * 25.0
        
        # Annual Seasonality (higher sales during Q4 holidays and summer peak)
        annual_seasonality = 20 * np.sin(2 * np.pi * day_of_year / 365.25 - np.pi / 2)
        if month in [11, 12]:
            q4_boost = 35.0  # Holiday shopping surge
        else:
            q4_boost = 0.0
            
        # Weekly Seasonality (higher on weekends)
        weekend_boost = 20.0 if day_of_week in [5, 6] else 0.0
        
        # Black Friday boost (late Nov)
        is_black_friday = 1 if (month == 11 and dt.day >= 23 and dt.day <= 29 and day_of_week == 4) else 0
        bf_boost = 80.0 if is_black_friday else 0.0
        
        for cat in categories:
            params = cat_params[cat]
            
            # Random promotion assignment (approx 15% of days)
            is_promo = 1 if np.random.rand() < 0.15 else 0
            promo_mult = params["promo_sensitivity"] if is_promo else 1.0
            
            # Competitor pricing index
            comp_price_index = round(float(np.random.normal(100, 5)), 2)
            
            # Price fluctuation
            price = round(float(params["price_mean"] * (1 - 0.1 * is_promo + np.random.normal(0, 0.02))), 2)
            
            # Calculate total sales quantity
            base = params["base_sales"] + trend + annual_seasonality + q4_boost + weekend_boost + bf_boost
            noise = np.random.normal(0, params["volatility"])
            units_sold = int(max(10, round((base + noise) * promo_mult)))
            
            revenue = round(units_sold * price, 2)
            
            records.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "Category": cat,
                "Price": price,
                "Is_Promotion": is_promo,
                "Competitor_Price_Index": comp_price_index,
                "Units_Sold": units_sold,
                "Revenue": revenue
            })
            
    df = pd.DataFrame(records)
    
    # Introduce small realistic missing values / noise to demonstrate data cleaning
    missing_idx = df.sample(frac=0.01, random_state=42).index
    df.loc[missing_idx, "Price"] = np.nan
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated raw dataset with {len(df)} rows at '{output_path}'")
    return df

if __name__ == "__main__":
    generate_sales_data()
