import os
import pandas as pd
from src.data_generator import generate_sales_data
from src.data_preprocessing import clean_data
from src.feature_engineering import build_feature_pipeline
from src.models import time_based_train_test_split, SalesForecaster
from src.evaluate import evaluate_predictions, print_evaluation_summary
from src.visualize import (
    plot_sales_overview,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residual_analysis,
    plot_future_30day_forecast
)

def run_sales_forecasting_pipeline():
    print("=" * 60)
    print("      FUTURE_ML_01: SALES & DEMAND FORECASTING PIPELINE      ")
    print("=" * 60)
    
    # Step 1: Data Generation / Loading
    raw_path = "data/raw_sales_data.csv"
    if not os.path.exists(raw_path):
        print("\n--> Step 1: Generating raw sales dataset...")
        df_raw = generate_sales_data(output_path=raw_path)
    else:
        print("\n--> Step 1: Loading raw sales dataset...")
        df_raw = pd.read_csv(raw_path)
        
    # Plot initial overall sales trends
    plot_sales_overview(df_raw, save_path="visualizations/sales_trends.png")
    
    # Step 2: Data Preprocessing & Aggregation
    print("\n--> Step 2: Preprocessing and aggregating sales data...")
    df_clean, df_daily = clean_data(input_path=raw_path)
    
    # Step 3: Feature Engineering
    print("\n--> Step 3: Engineering time, lag, and rolling features...")
    df_featured = build_feature_pipeline(df_daily, target_col="Units_Sold")
    df_featured.to_csv("data/featured_daily_sales.csv", index=False)
    
    # Step 4: Time-Based Train/Test Split
    print("\n--> Step 4: Splitting time series data chronologically (90-day test set)...")
    train_df, test_df, X_train, y_train, X_test, y_test, feature_cols = time_based_train_test_split(
        df_featured, target_col="Units_Sold", test_days=90
    )
    
    # Step 5: Model Training & Forecasting
    print("\n--> Step 5: Training forecasting models (Linear, Tree-based, GBDT, & SARIMAX)...")
    forecaster = SalesForecaster()
    predictions = forecaster.fit_and_predict(X_train, y_train, X_test, y_test)
    
    # Fit SARIMAX baseline
    forecaster.train_sarimax(y_train, y_test)
    
    # Step 6: Model Evaluation
    print("\n--> Step 6: Evaluating models and computing metrics...")
    metrics_df = evaluate_predictions(y_test, forecaster.predictions)
    print_evaluation_summary(metrics_df)
    
    # Save evaluation table
    os.makedirs("data", exist_ok=True)
    metrics_df.to_csv("data/model_evaluation_metrics.csv", index=False)
    print("Saved evaluation metrics to 'data/model_evaluation_metrics.csv'.")
    
    # Determine best model based on lowest RMSE
    best_model_name = metrics_df.iloc[0]["Model"]
    print(f"\n--> Best performing model: '{best_model_name}' (RMSE: {metrics_df.iloc[0]['RMSE']}, R2: {metrics_df.iloc[0]['R2 Score']})")
    
    # Save best model joblib artifact
    forecaster.save_best_model(best_model_name, model_dir="models")
    
    # Step 7: Generating Visual Artifacts
    print("\n--> Step 7: Creating business forecast visualizations...")
    plot_model_comparison(metrics_df, save_path="visualizations/model_comparison.png")
    plot_actual_vs_predicted(test_df, forecaster.predictions, best_models=[best_model_name, "XGBoost", "LightGBM", "SARIMAX"], save_path="visualizations/actual_vs_predicted.png")
    
    if best_model_name in forecaster.feature_importances:
        plot_feature_importance(forecaster.feature_importances[best_model_name], top_n=12, model_name=best_model_name, save_path="visualizations/feature_importance.png")
    elif "XGBoost" in forecaster.feature_importances:
        plot_feature_importance(forecaster.feature_importances["XGBoost"], top_n=12, model_name="XGBoost", save_path="visualizations/feature_importance.png")
        
    best_pred = forecaster.predictions[best_model_name]
    plot_residual_analysis(y_test.values, best_pred, model_name=best_model_name, save_path="visualizations/residual_analysis.png")
    
    best_model_obj = forecaster.fitted_models[best_model_name]
    if best_model_name == "SARIMAX":
        best_model_obj = forecaster.fitted_models["XGBoost"]
    plot_future_30day_forecast(test_df, best_model_obj, feature_cols, days_ahead=30, save_path="visualizations/future_forecast.png")
    
    print("\n" + "=" * 60)
    print("  FUTURE_ML_01 SALES FORECASTING PIPELINE COMPLETED SUCCESSFULLY! ")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    run_sales_forecasting_pipeline()
