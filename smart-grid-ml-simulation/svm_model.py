# models/svm_model.py — Support Vector Machine with RBF Kernel
#
# Implements SVR with radial basis function kernel and Grid Search-optimized
# hyperparameters (C=100, gamma=0.1) for energy demand forecasting using
# weather features.
#
# Performance: R² = 0.806 | RMSE = 7.53% | MAPE = 5.86%


import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

# --- SVM hyperparameters (Grid Search results) -------------------------------
SVM_KERNEL = "rbf"
SVM_C      = 100
SVM_GAMMA  = 0.1


def run():
    """Run the SVM pipeline with weather-augmented features."""

    print("=" * 65)
    print(" Model C: Support Vector Machine (RBF, R² = 0.806)")
    print("=" * 65)

    # --- Load weather data ---
    print("\n  Loading weather-augmented data...")
    combined, relevant, future_data = prepare_weather_data()

    target = "hourly_energy"

    # --- Scaling ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_combined = scaler_X.fit_transform(combined[WEATHER_FEATURES])
    y_combined = scaler_y.fit_transform(combined[[target]]).flatten()

    # --- Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined[:len(relevant)], y_combined[:len(relevant)],
        test_size=0.2, random_state=42,
    )

    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples:     {len(X_test)}")
    print(f"  Kernel: {SVM_KERNEL} | C={SVM_C} | gamma={SVM_GAMMA}")

    # --- Train SVM ---
    print("  Training SVM model...")
    svm = SVR(kernel=SVM_KERNEL, C=SVM_C, gamma=SVM_GAMMA)
    svm.fit(X_train, y_train)

    # --- Evaluate ---
    y_pred = svm.predict(X_test)
    metrics = calculate_metrics(y_test, y_pred, "Test (SVM)")

    # --- Future predictions ---
    X_future = scaler_X.transform(future_data[WEATHER_FEATURES])
    future_pred = svm.predict(X_future)
    future_kwh = scaler_y.inverse_transform(
        future_pred.reshape(-1, 1)
    ).flatten()

    print(f"\n  Future predictions: {len(future_kwh)} hours")
    print(f"  Mean predicted demand: {future_kwh.mean():.2f} Wh")

    # --- Save predictions ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    formatted = (future_kwh / 1000).round(3)
    pd.DataFrame(formatted, columns=["PredictedHourlyEnergy"]).to_csv(
        f"{RESULTS_DIR}/svm_predictions.csv", index=False, float_format="%.3f"
    )
    print(f"  Predictions saved to {RESULTS_DIR}/svm_predictions.csv")

    return {"predictions": future_kwh, "metrics": metrics, "model": svm}


if __name__ == "__main__":
    run()
