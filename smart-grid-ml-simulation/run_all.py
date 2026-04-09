#!/usr/bin/env python3
# =============================================================================
# run_all.py — Execute the Full Energy Demand Forecasting Pipeline
# =============================================================================
#
# Usage:
#   python run_all.py              # Run all models + simulation
#   python run_all.py xgboost      # Run only XGBoost
#   python run_all.py lstm         # Run only LSTM-RNN
#   python run_all.py svm          # Run only SVM
#   python run_all.py storage      # Run only storage simulation
# =============================================================================

import sys
import os

os.makedirs("results", exist_ok=True)

modules = {
    "xgboost": ("models.xgboost_model", "Model A: XGBoost"),
    "lstm":    ("models.lstm_model",     "Model B: LSTM-RNN"),
    "svm":     ("models.svm_model",      "Model C: SVM"),
    "storage": ("models.storage_simulation", "Storage Simulation"),
}

args = sys.argv[1:] if len(sys.argv) > 1 else list(modules.keys())

print("=" * 65)
print(" Energy Demand Forecasting & Renewable Integration Pipeline")
print(" TÜBİTAK 2209-A / Bachelor Thesis — Marmara University")
print("=" * 65)

for key in args:
    if key in modules:
        mod_path, label = modules[key]
        print(f"\n>>> Running {label}...\n")
        mod = __import__(mod_path, fromlist=["run"])
        mod.run()
    else:
        print(f"Unknown module: {key} (options: {', '.join(modules.keys())})")

print("\n" + "=" * 65)
print(" Pipeline complete.")
print("=" * 65)
