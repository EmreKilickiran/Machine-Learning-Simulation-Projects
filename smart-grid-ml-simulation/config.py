# =============================================================================
# config.py — Shared Configuration, Feature Engineering & Evaluation Utilities
# =============================================================================
#
# Provides all shared functionality used across the three model pipelines
# and the renewable integration simulation.
#
# Author : Yunus Emre Kılıçkıran
# =============================================================================

import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, r2_score)

# =============================================================================
# FILE PATHS (modify for your environment)
# =============================================================================

DATA_DIR            = "data"
RESULTS_DIR         = "results"
ENERGY_DATA_FILE    = f"{DATA_DIR}/FilteredEnergyData.xlsx"
WEATHER_TRAIN_FILE  = f"{DATA_DIR}/cleaned_energy_data.csv"
WEATHER_FUTURE_FILE = f"{DATA_DIR}/processed_future_weather_data.csv"

# =============================================================================
# PARAMETERS
# =============================================================================

WINDOW_SIZE    = 12       # Sliding window length (hours)
PRICE_PER_KWH = 2.0      # TL/kWh for cost analysis

# Dataset split (days)
TRAIN_DAYS = 22
VAL_DAYS   = 4
TEST_DAYS  = 4

# XGBoost hyperparameters (tuned via cross-validation)
XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=1200,
    max_depth=3,
    learning_rate=0.03,
    reg_alpha=0,
    reg_lambda=1,
    subsample=0.7,
    colsample_bytree=0.8,
    gamma=0,
)

# Weather-based model features (shared by LSTM and SVM)
WEATHER_FEATURES = [
    "AirTemperature", "ComfortTemperature", "EffectiveCloudCover",
    "RelativeHumidity", "WindSpeed", "WWCodeNumeric", "Weekend",
    "Hour", "Weekday", "Month",
    "TimeOfDay_Morning", "TimeOfDay_Afternoon",
    "TimeOfDay_Evening", "TimeOfDay_Night",
]

# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def add_fourier_features(df, column, periods):
    """Add sin/cos Fourier decomposition for cyclical pattern capture."""
    for period in periods:
        df[f"sin_{period}"] = np.sin(2 * np.pi * df[column] / period)
        df[f"cos_{period}"] = np.cos(2 * np.pi * df[column] / period)
    return df


def assign_daypart(hour):
    """Activity-based time-of-day segmentation."""
    if 0 <= hour <= 5:    return 0  # Night
    elif 6 <= hour <= 8:  return 1  # Morning
    elif 9 <= hour <= 11: return 2  # Late morning
    elif 12 <= hour <= 14: return 3  # Afternoon
    elif 15 <= hour <= 17: return 4  # Late afternoon
    elif 18 <= hour <= 20: return 5  # Evening
    else:                  return 0  # Night


def engineer_features(df, n_houses_pv):
    """
    Full feature engineering pipeline for the energy demand dataset:

    - Sinusoidal hour/day transformations for cyclical pattern capture
    - Fourier-based periodic decomposition (24h, 12h, 6h harmonics)
    - Rolling statistics: mean (3/6/12/24h), std, max, min, skewness
    - First-order differencing for trend extraction
    - Daypart categorical segmentation
    - Net energy demand: consumption − (n_houses × PV production)
    """
    df = df.copy()
    df["Hour"] = df["Hour"] - 1  # Adjust to 0–23 range

    # Weekend indicator
    weekend_days = [6, 7, 13, 14, 20, 21, 27, 28]
    df["is_weekend"] = df["Day"].apply(lambda x: 1 if x in weekend_days else 0)

    # Sinusoidal temporal encoding
    df["sin_hour"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["sin_day"]  = np.sin(2 * np.pi * df["Day"] / 30)
    df["cos_day"]  = np.cos(2 * np.pi * df["Day"] / 30)

    # Multi-scale moving averages
    for w in [3, 6, 12, 24]:
        df[f"MA_{w}"] = df["TotalEnergyKWh"].rolling(window=w).mean()

    # Fourier features (multi-period)
    df = add_fourier_features(df, "Hour", periods=[24, 12, 6])

    # Daypart
    df["daypart"] = df["Hour"].apply(assign_daypart)

    # Rolling statistics (6-hour window)
    df["diff_energy"] = df["TotalEnergyKWh"].diff()
    df["roll_std_6"]  = df["TotalEnergyKWh"].rolling(6).std()
    df["roll_max_6"]  = df["TotalEnergyKWh"].rolling(6).max()
    df["roll_min_6"]  = df["TotalEnergyKWh"].rolling(6).min()
    df["roll_skew_6"] = df["TotalEnergyKWh"].rolling(6).skew()

    # Net energy demand
    df["NetEnergyKWh"] = df["TotalEnergyKWh"] - n_houses_pv * df["ProductionKWh"]

    df.dropna(inplace=True)
    return df


def create_sliding_windows(data, window_size):
    """
    Create supervised learning samples from time-series data using
    a sliding window approach. Past `window_size` target values are
    concatenated with the current timestep's auxiliary features.
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        energy_window = data[i:i + window_size, 0]
        other_features = data[i + window_size, 1:]
        X.append(np.concatenate((energy_window, other_features)))
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)


def prepare_weather_data():
    """
    Load and prepare weather-augmented datasets for LSTM and SVM models.
    Returns combined dataframe with temporal features and one-hot encoding.
    """
    train_data  = pd.read_csv(WEATHER_TRAIN_FILE)
    future_data = pd.read_csv(WEATHER_FUTURE_FILE)

    for d in [train_data, future_data]:
        d["date"]    = pd.to_datetime(d["date"])
        d["Weekend"] = d["date"].dt.weekday >= 5
        d["Hour"]    = d["date"].dt.hour
        d["Weekday"] = d["date"].dt.weekday
        d["Month"]   = d["date"].dt.month
        d["TimeOfDay"] = d["Hour"].apply(
            lambda h: "Morning" if 6 <= h < 12 else
                      "Afternoon" if 12 <= h < 18 else
                      "Evening" if 18 <= h < 24 else "Night"
        )

    train_data  = pd.get_dummies(train_data, columns=["TimeOfDay"])
    future_data = pd.get_dummies(future_data, columns=["TimeOfDay"])

    # Include last training day for sequence continuity
    relevant = train_data[train_data["date"] <= "2024-04-21 23:59:59"]
    combined = pd.concat([relevant, future_data], ignore_index=True)

    return combined, relevant, future_data


# =============================================================================
# EVALUATION UTILITIES
# =============================================================================

def calculate_metrics(y_true, y_pred, set_name="Set"):
    """Compute and display regression performance metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2   = r2_score(y_true, y_pred)
    mean_val = np.mean(y_true)

    print(f"\n  {set_name}:")
    print(f"    RMSE: {rmse:.2f} ({rmse / mean_val * 100:.2f}%)")
    print(f"    MAE:  {mae:.2f} ({mae / mean_val * 100:.2f}%)")
    print(f"    MAPE: {mape:.2f}%")
    print(f"    R²:   {r2:.4f}")
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def total_energy_stats(y_true, y_pred, label="", price=PRICE_PER_KWH):
    """Compute total energy and cost comparison over the test period."""
    total_true = np.sum(y_true)
    total_pred = np.sum(y_pred)
    diff = total_true - total_pred

    print(f"\n  {label} — 4-Day Energy & Cost Summary:")
    print(f"    Actual:    {total_true:.2f} kWh ({total_true * price:.2f} TL)")
    print(f"    Predicted: {total_pred:.2f} kWh ({total_pred * price:.2f} TL)")
    print(f"    Error:     {diff:.2f} kWh ({diff / total_true * 100:.2f}%)")
