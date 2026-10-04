import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set cohesive professional styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.sans-serif": "Arial",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#eeeeee",
    "figure.autolayout": True
})

def plot_sales_overview(df_raw, save_path="visualizations/sales_trends.png"):
    """
    Plots historical sales trends overall and by product category.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Daily aggregated total sales
    df_raw["Date"] = pd.to_datetime(df_raw["Date"])
    daily = df_raw.groupby("Date")["Units_Sold"].sum().reset_index()
    
    axes[0].plot(daily["Date"], daily["Units_Sold"], color="#1f77b4", linewidth=1.5, label="Total Daily Units Sold")
    axes[0].set_title("Historical Overall Daily Sales & Demand Trend", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_ylabel("Units Sold", fontsize=11)
    axes[0].legend(loc="upper left")
    
    # Sales by Category
    sns.lineplot(data=df_raw, x="Date", y="Units_Sold", hue="Category", ax=axes[1], palette="Set2", linewidth=1.2)
    axes[1].set_title("Daily Demand Breakdown by Product Category", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_ylabel("Units Sold", fontsize=11)
    axes[1].set_xlabel("Date", fontsize=11)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sales overview plot to '{save_path}'")

def plot_model_comparison(metrics_df, save_path="visualizations/model_comparison.png"):
    """
    Plots RMSE, MAE, and R2 comparison across models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # RMSE Plot
    sns.barplot(data=metrics_df, x="Model", y="RMSE", ax=axes[0], hue="Model", palette="Blues_r", legend=False)
    axes[0].set_title("Root Mean Squared Error (RMSE - Lower is better)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("RMSE", fontsize=10)
    axes[0].tick_params(axis="x", rotation=35)
    for p in axes[0].patches:
        if p.get_height() > 0:
            axes[0].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    # MAE Plot
    sns.barplot(data=metrics_df, x="Model", y="MAE", ax=axes[1], hue="Model", palette="Greens_r", legend=False)
    axes[1].set_title("Mean Absolute Error (MAE - Lower is better)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("MAE", fontsize=10)
    axes[1].tick_params(axis="x", rotation=35)
    for p in axes[1].patches:
        if p.get_height() > 0:
            axes[1].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    # R2 Score Plot
    sns.barplot(data=metrics_df, x="Model", y="R2 Score", ax=axes[2], hue="Model", palette="Oranges", legend=False)
    axes[2].set_title("R² Score (Variance Explained - Higher is better)", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("R² Score", fontsize=10)
    axes[2].tick_params(axis="x", rotation=35)
    for p in axes[2].patches:
        if p.get_height() is not None:
            axes[2].annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha="center", va="center", xytext=(0, 5), textcoords="offset points", fontsize=9)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved model comparison plot to '{save_path}'")

def plot_actual_vs_predicted(test_df, predictions_dict, best_models=["XGBoost", "LightGBM", "SARIMAX"], save_path="visualizations/actual_vs_predicted.png"):
    """
    Plots Actual vs Predicted daily sales overlay.
    """
    plt.figure(figsize=(14, 6))
    dates = pd.to_datetime(test_df["Date"])
    actuals = test_df["Units_Sold"]
    
    plt.plot(dates, actuals, label="Actual Sales", color="black", linewidth=2.0, linestyle="-")
    
    colors = ["#d62728", "#2ca02c", "#ff7f0e", "#9467bd"]
    for idx, model_name in enumerate(best_models):
        if model_name in predictions_dict:
            plt.plot(dates, predictions_dict[model_name], label=f"{model_name} Forecast", color=colors[idx % len(colors)], linewidth=1.8, linestyle="--")
            
    plt.title("90-Day Out-of-Sample Sales Forecast vs Ground Truth", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Units Sold", fontsize=11)
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved Actual vs Predicted plot to '{save_path}'")

def plot_feature_importance(importance_series, top_n=12, model_name="XGBoost", save_path="visualizations/feature_importance.png"):
    """
    Plots top N feature importances for tree-based or regression models.
    """
    plt.figure(figsize=(10, 6))
    top_features = importance_series.sort_values(ascending=False).head(top_n)
    
    sns.barplot(x=top_features.values, y=top_features.index, hue=top_features.index, palette="viridis", legend=False)
    plt.title(f"Top {top_n} Predictive Features ({model_name})", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Relative Importance Score", fontsize=11)
    plt.ylabel("Feature Name", fontsize=11)
    
    for i, v in enumerate(top_features.values):
        plt.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importance plot to '{save_path}'")

def plot_residual_analysis(y_true, y_pred, model_name="XGBoost", save_path="visualizations/residual_analysis.png"):
    """
    Plots error distribution and residual plot to inspect for systematic bias.
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Residual Histogram / KDE
    sns.histplot(residuals, kde=True, ax=axes[0], color="#2b5c8f", bins=25)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"Residuals Error Distribution ({model_name})", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Forecast Error (Actual - Predicted)", fontsize=10)
    
    # Residuals vs Fitted values
    axes[1].scatter(y_pred, residuals, alpha=0.7, color="#348abd", edgecolors="k", s=35)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[1].set_title("Residuals vs Predicted Sales", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Predicted Units Sold", fontsize=10)
    axes[1].set_ylabel("Residual Error", fontsize=10)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residual analysis plot to '{save_path}'")

def plot_future_30day_forecast(test_df, best_model, feature_cols, days_ahead=30, save_path="visualizations/future_forecast.png"):
    """
    Generates and plots a 30-day future demand forecast into the future.
    """
    last_date = pd.to_datetime(test_df["Date"].max())
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead, freq="D")
    
    future_preds = []
    curr_df = test_df.copy()
    
    for f_date in future_dates:
        row_feat = {
            "Year": f_date.year,
            "Month": f_date.month,
            "Day": f_date.day,
            "DayOfWeek": f_date.dayofweek,
            "Quarter": f_date.quarter,
            "DayOfYear": f_date.dayofyear,
            "Is_Weekend": 1 if f_date.dayofweek in [5, 6] else 0,
            "Is_Month_Start": 1 if f_date.is_month_start else 0,
            "Is_Month_End": 1 if f_date.is_month_end else 0,
            "Month_Sin": np.sin(2 * np.pi * f_date.month / 12.0),
            "Month_Cos": np.cos(2 * np.pi * f_date.month / 12.0),
            "DayOfWeek_Sin": np.sin(2 * np.pi * f_date.dayofweek / 7.0),
            "DayOfWeek_Cos": np.cos(2 * np.pi * f_date.dayofweek / 7.0),
            "Price": curr_df["Price"].mean(),
            "Is_Promotion": 1 if f_date.dayofweek in [4, 5] else 0,
            "Competitor_Price_Index": 100.0,
            "Units_Sold_Lag_1": curr_df["Units_Sold"].iloc[-1],
            "Units_Sold_Lag_7": curr_df["Units_Sold"].iloc[-7],
            "Units_Sold_Lag_14": curr_df["Units_Sold"].iloc[-14],
            "Units_Sold_Lag_30": curr_df["Units_Sold"].iloc[-30],
            "Units_Sold_Rolling_Mean_7": curr_df["Units_Sold"].iloc[-7:].mean(),
            "Units_Sold_Rolling_Std_7": curr_df["Units_Sold"].iloc[-7:].std(),
            "Units_Sold_Rolling_Max_7": curr_df["Units_Sold"].iloc[-7:].max(),
            "Units_Sold_Rolling_Min_7": curr_df["Units_Sold"].iloc[-7:].min(),
            "Units_Sold_Rolling_Mean_14": curr_df["Units_Sold"].iloc[-14:].mean(),
            "Units_Sold_Rolling_Std_14": curr_df["Units_Sold"].iloc[-14:].std(),
            "Units_Sold_Rolling_Max_14": curr_df["Units_Sold"].iloc[-14:].max(),
            "Units_Sold_Rolling_Min_14": curr_df["Units_Sold"].iloc[-14:].min(),
            "Units_Sold_Rolling_Mean_30": curr_df["Units_Sold"].iloc[-30:].mean(),
            "Units_Sold_Rolling_Std_30": curr_df["Units_Sold"].iloc[-30:].std(),
            "Units_Sold_Rolling_Max_30": curr_df["Units_Sold"].iloc[-30:].max(),
            "Units_Sold_Rolling_Min_30": curr_df["Units_Sold"].iloc[-30:].min(),
            "Units_Sold_Expanding_Mean": curr_df["Units_Sold"].mean()
        }
        
        # Ensure all feature_cols exist in row_feat
        input_dict = {col: row_feat.get(col, 0.0) for col in feature_cols}
        input_vector = pd.DataFrame([input_dict])[feature_cols]
        pred_val = float(best_model.predict(input_vector)[0])
        future_preds.append(pred_val)
        
        # Append prediction to curr_df for next iteration
        new_row = {"Date": f_date, "Units_Sold": pred_val, "Price": row_feat["Price"]}
        curr_df = pd.concat([curr_df, pd.DataFrame([new_row])], ignore_index=True)
        
    plt.figure(figsize=(14, 6))
    hist_dates = pd.to_datetime(test_df["Date"].iloc[-60:])
    hist_sales = test_df["Units_Sold"].iloc[-60:]
    
    plt.plot(hist_dates, hist_sales, label="Recent Historical Sales (Last 60 Days)", color="#1f77b4", linewidth=2.0)
    plt.plot(future_dates, future_preds, label="30-Day Future Demand Forecast", color="#e377c2", linewidth=2.2, linestyle="--", marker="o", markersize=4)
    
    # Confidence Interval simulation
    std_err = np.std(hist_sales - test_df["Units_Sold"].iloc[-60:]) * 0.5
    lower_bound = np.array(future_preds) - 1.96 * std_err
    upper_bound = np.array(future_preds) + 1.96 * std_err
    plt.fill_between(future_dates, lower_bound, upper_bound, color="#e377c2", alpha=0.2, label="95% Forecast Confidence Band")
    
    plt.title("Business Planning: 30-Day Future Sales & Demand Projection", fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Date", fontsize=11)
    plt.ylabel("Units Sold", fontsize=11)
    plt.legend(loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved future 30-day forecast plot to '{save_path}'")
