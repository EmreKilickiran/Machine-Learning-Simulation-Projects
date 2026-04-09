# Machine Learning-Based Computational Modeling for Energy Demand and Renewable Integration

A complete computational modeling and simulation framework for short-term regional energy demand forecasting, integrating stochastic data generation, feature engineering, gradient boosting-based predictive modeling, and multi-scenario simulation of photovoltaic and storage systems. Developed as a **TÜBİTAK 2209-A funded** bachelor thesis at Marmara University.

## Overview

The pipeline independently develops and compares three machine learning architectures end-to-end for short-term energy demand prediction, then combines the best-performing model with stochastic PV outputs for multi-scenario renewable integration simulation.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────┐
│              INPUT DATA                                  │
│  Power consumption + Weather data + Calendar features    │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │  config.py          │
              │  Shared Feature     │
              │  Engineering        │
              │                     │
              │  • Sinusoidal enc.  │
              │  • Fourier (24/12/6)│
              │  • Rolling stats    │
              │  • Daypart segments │
              └──────────┬──────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
   ┌──────▼──────┐ ┌────▼────┐ ┌──────▼──────┐
   │ xgboost     │ │  lstm   │ │    svm      │
   │ _model.py   │ │ _model  │ │  _model.py  │
   │             │ │  .py    │ │             │
   │ Sliding     │ │ 2-layer │ │ RBF kernel  │
   │ Window +    │ │ LSTM +  │ │ C=100       │
   │ XGBRegressor│ │ Optuna  │ │ gamma=0.1   │
   │             │ │         │ │             │
   │ R²=0.905   │ │ R²=0.901│ │ R²=0.806     │
   └──────┬──────┘ └────┬────┘ └──────┬──────┘
          │              │              │
          └──────────────┼──────────────┘
                         │
              ┌──────────▼──────────┐
              │ storage_simulation  │
              │ .py                 │
              │                     │
              │ • 3 PV penetration  │
              │   levels (5-100%)   │
              │ • 5 capacity mult.  │
              │   (C=1 to C=5)     │
              │ • Threshold-based   │
              │   charge/discharge  │
              └─────────────────────┘
```

## Model Performance Comparison

| Model | R² | RMSE (Norm.) | MAE (Norm.) | MAPE |
|-------|-----|-------------|-------------|------|
| **XGBoost** | **0.905** | 5.26% | 4.53% | 4.49% |
| LSTM-RNN Hybrid | 0.901 | 5.60% | 4.59% | 4.52% |
| Support Vector Machine | 0.806 | 7.53% | 5.73% | 5.86% |

## Feature Engineering

Engineered temporal and statistical features applied across all models (defined in `config.py`):

- **Sinusoidal temporal encoding**: sin/cos of hour and day for cyclical pattern capture
- **Fourier-based periodic decomposition**: 24h, 12h, 6h harmonic components
- **Rolling statistics**: Moving averages (3/6/12/24h), standard deviation, max, min, skewness over 6-hour windows
- **Differencing**: First-order differences for trend extraction
- **Daypart segmentation**: Activity-based time categorization (night/morning/afternoon/evening)
- **Feature importance-based selection**: Reduces dimensionality and prevents overfitting

## Requirements

```
numpy
pandas
scikit-learn
xgboost
matplotlib
tensorflow      # for LSTM model only (optional — skipped if not installed)
openpyxl
```

## Repository Structure

```
.
├── README.md
├── run_all.py                          # Pipeline entry point
├── config.py                           # Shared config, feature engineering, utilities
├── models/
│   ├── __init__.py
│   ├── xgboost_model.py               # Model A: XGBoost (R² = 0.905)
│   ├── lstm_model.py                   # Model B: LSTM-RNN Hybrid (R² = 0.901)
│   ├── svm_model.py                    # Model C: SVM (R² = 0.806)
│   └── storage_simulation.py           # Multi-scenario PV + storage simulation
├── data/                               # (not tracked)
│   ├── FilteredEnergyData.xlsx
│   ├── cleaned_energy_data.csv
│   └── processed_future_weather_data.csv
└── results/                            # (generated)
```

## Usage

```bash
# Run all models + storage simulation
python run_all.py

# Run individual models
python run_all.py xgboost
python run_all.py lstm
python run_all.py svm
python run_all.py storage

# Run models directly
python -m models.xgboost_model
python -m models.lstm_model
python -m models.svm_model
```
