import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from statsmodels.tsa.statespace.sarimax import SARIMAX

def time_based_train_test_split(df, target_col="Units_Sold", test_days=90):
    """
    Splits time series dataset chronologically into train and test sets.
    """
    df = df.sort_values("Date").reset_index(drop=True)
    split_index = len(df) - test_days
    
    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()
    
    ignore_cols = ["Date", "Category", "Revenue", target_col]
    feature_cols = [c for c in df.columns if c not in ignore_cols]
    
    X_train = train_df[feature_cols]
    y_train = train_df[target_col]
    X_test = test_df[feature_cols]
    y_test = test_df[target_col]
    
    print(f"Train set: {len(train_df)} rows ({train_df['Date'].min()} to {train_df['Date'].max()})")
    print(f"Test set: {len(test_df)} rows ({test_df['Date'].min()} to {test_df['Date'].max()})")
    
    return train_df, test_df, X_train, y_train, X_test, y_test, feature_cols

class SalesForecaster:
    """
    Unified forecasting container for multiple ML models and baseline methods.
    """
    def __init__(self):
        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42),
            "LightGBM": LGBMRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, random_state=42, verbose=-1)
        }
        self.fitted_models = {}
        self.predictions = {}
        self.feature_importances = {}
        
    def fit_and_predict(self, X_train, y_train, X_test, y_test):
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            self.fitted_models[name] = model
            self.predictions[name] = preds
            
            if hasattr(model, "feature_importances_"):
                self.feature_importances[name] = pd.Series(model.feature_importances_, index=X_train.columns)
            elif hasattr(model, "coef_"):
                self.feature_importances[name] = pd.Series(np.abs(model.coef_), index=X_train.columns)
                
        return self.predictions

    def train_sarimax(self, train_series, test_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
        """
        Fits a SARIMAX statistical baseline model.
        """
        print("Training SARIMAX baseline model...")
        try:
            model = SARIMAX(train_series, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            results = model.fit(disp=False)
            forecast = results.forecast(steps=len(test_series))
            self.fitted_models["SARIMAX"] = results
            self.predictions["SARIMAX"] = forecast.values
            print("SARIMAX training complete.")
        except Exception as e:
            print(f"SARIMAX failed: {e}")
            # Fallback to naive seasonal lag forecast
            self.predictions["SARIMAX"] = train_series.iloc[-len(test_series):].values

    def save_best_model(self, best_name, model_dir="models"):
        os.makedirs(model_dir, exist_ok=True)
        path = os.path.join(model_dir, "best_sales_forecaster.joblib")
        joblib.dump(self.fitted_models[best_name], path)
        print(f"Best model ('{best_name}') saved successfully to '{path}'.")
