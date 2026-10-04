import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_mape(y_true, y_pred):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_predictions(y_true, predictions_dict):
    """
    Evaluates all model predictions against ground truth labels.
    """
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = calculate_mape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        results.append({
            "Model": model_name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "MAPE (%)": round(mape, 2),
            "R2 Score": round(r2, 4)
        })
        
    metrics_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    return metrics_df

def print_evaluation_summary(metrics_df):
    """
    Prints a formatted summary table of evaluation metrics.
    """
    print("\n" + "="*50)
    print("      MODEL EVALUATION & PERFORMANCE SUMMARY      ")
    print("="*50)
    print(metrics_df.to_string(index=False))
    print("="*50 + "\n")
