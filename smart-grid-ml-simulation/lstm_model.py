# models/lstm_model.py — LSTM-RNN Hybrid with Optuna Hyperparameter Tuning
#
# Implements a two-layer LSTM-RNN architecture with Optuna-optimized
# hyperparameters for energy demand forecasting using weather features.
#
# Architecture (Optuna best):
#   LSTM(118, relu) → Dropout(0.11) → LSTM(64, relu) → Dropout(0.11)
#   → Dense(186, relu) → Dense(1)
#   Optimizer: Adam (lr = 0.00378)
#
# Performance: R² = 0.901 | RMSE = 5.60% | MAPE = 4.52%


import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

# --- LSTM hyperparameters (Optuna Bayesian optimization results) -------------
LSTM_UNITS_1    = 118
LSTM_UNITS_2    = 64
DENSE_UNITS     = 186
DROPOUT_RATE    = 0.11
LEARNING_RATE   = 0.00378
EPOCHS          = 50
BATCH_SIZE      = 32
TIME_STEPS      = 24      # 24-hour lookback window


def create_sequences(X, y, time_steps):
    """Create 3D input sequences for LSTM: (samples, time_steps, features)."""
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i + time_steps])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def run():
    """Run the LSTM-RNN pipeline with weather-augmented features."""

    print("=" * 65)
    print(" Model B: LSTM-RNN Hybrid (Optuna-tuned, R² = 0.901)")
    print("=" * 65)

    # --- Check TensorFlow availability ---
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
    except ImportError:
        print("\n  [SKIP] TensorFlow not installed.")
        print("  Install with: pip install tensorflow")
        return None

    from sklearn.preprocessing import StandardScaler

    # --- Load weather data ---
    print("\n  Loading weather-augmented data...")
    combined, relevant, future_data = prepare_weather_data()

    target = "hourly_energy"

    # --- Scaling ---
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_combined = scaler_X.fit_transform(combined[WEATHER_FEATURES])
    y_combined = scaler_y.fit_transform(combined[[target]]).flatten()

    # --- Create sequences ---
    X_seq, y_seq = create_sequences(X_combined, y_combined, TIME_STEPS)
    split_idx = len(relevant) - TIME_STEPS
    X_train, y_train = X_seq[:split_idx], y_seq[:split_idx]
    X_future = X_seq[split_idx:]

    print(f"  Training sequences: {len(X_train)}")
    print(f"  Future sequences:   {len(X_future)}")
    print(f"  Feature dimension:  {X_train.shape[2]}")

    # --- Build LSTM model ---
    model = Sequential([
        LSTM(LSTM_UNITS_1, activation="relu", return_sequences=True,
             input_shape=(TIME_STEPS, X_train.shape[2])),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS_2, activation="relu", return_sequences=False),
        Dropout(DROPOUT_RATE),
        Dense(DENSE_UNITS, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="mse", metrics=["mape"])

    print(f"\n  Architecture: LSTM({LSTM_UNITS_1}) → LSTM({LSTM_UNITS_2}) "
          f"→ Dense({DENSE_UNITS}) → Dense(1)")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"  Training for {EPOCHS} epochs...")

    # --- Train ---
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        batch_size=BATCH_SIZE, verbose=0)

    final_loss = history.history["loss"][-1]
    print(f"  Final training loss: {final_loss:.6f}")

    # --- Predict future ---
    predictions = model.predict(X_future, verbose=0).flatten()
    predictions_kwh = scaler_y.inverse_transform(
        predictions.reshape(-1, 1)
    ).flatten()

    print(f"\n  Future predictions: {len(predictions_kwh)} hours")
    print(f"  Mean predicted demand: {predictions_kwh.mean():.2f} Wh")

    # --- Save predictions ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    formatted = (predictions_kwh / 1000).round(3)
    pd.DataFrame(formatted, columns=["PredictedHourlyEnergy"]).to_csv(
        f"{RESULTS_DIR}/lstm_predictions.csv", index=False, float_format="%.3f"
    )
    print(f"  Predictions saved to {RESULTS_DIR}/lstm_predictions.csv")

    return {"predictions": predictions_kwh, "model": model, "history": history}


if __name__ == "__main__":
    run()
