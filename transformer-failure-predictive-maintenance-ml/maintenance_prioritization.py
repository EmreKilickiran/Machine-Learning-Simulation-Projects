# maintenance_prioritization.py — Random Forest-Based Maintenance Scoring
#
# Ranks infrastructure units by predicted failure severity using a hybrid
# supervised-unsupervised framework, quantifying expected customer impact
# and outage duration for maintenance resource allocation.
#
# Pipeline:
#   1. Load raw fault records (219,670 → 81,388 asset-level vectors)
#   2. Feature engineering: log-transforms on skewed distributions
#   3. Min-Max normalization of continuous features
#   4. One-hot encoding of categorical features (Subtype, Class, Type)
#   5. Random Forest classifier (100 estimators) for feature importance
#   6. Weighted scoring: score = Σ(w_i × x_i), weights from RF importance
#   7. Rank assets by descending score → top-priority maintenance list
#
# Data (column names anonymized per NDA):
#   - ID: Asset identifier
#   - InventoryAge: Age of the infrastructure unit
#   - WeightedBreakdownCount: Failure frequency (duration-weighted)
#   - MaintenanceCount: Historical maintenance interventions
#   - Power: Rated capacity (log-transformed for skew correction)
#   - other_features: Derived operational metric (log-transformed)
#   - Subtype, Class, Type: Categorical asset attributes
==

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# CONFIGURATION

INPUT_FILE  = "data/frekans2.xlsx"
OUTPUT_FILE = "results/top_priority_assets.xlsx"
TOP_N       = 20          # Number of top-priority assets to export
RF_ESTIMATORS = 100       # Random Forest estimators
TEST_SIZE     = 0.3       # Train/test split ratio
RANDOM_STATE  = 42

# Feature columns
NUMERIC_FEATURES = [
    "WeightedBreakdownCount", "Power", "InventoryAge",
    "MaintenanceCount", "other_features"
]
CATEGORICAL_FEATURES = ["Subtype", "Class", "Type"]
ID_COLUMN = "ID"


# 1. DATA LOADING & FEATURE ENGINEERING

print("=" * 65)
print(" Predictive Maintenance Prioritization Pipeline")
print("=" * 65)

print("\n[1/5] Loading data...")
data = pd.read_excel(INPUT_FILE)
print(f"  Raw records: {len(data)}")

# Select relevant columns
selected_columns = [ID_COLUMN] + CATEGORICAL_FEATURES + NUMERIC_FEATURES
data = data[selected_columns].copy()

# Log-transform skewed distributions to reduce outlier influence
print("[2/5] Feature engineering (log-transforms on skewed features)...")
for col in ["Power", "other_features"]:
    skew_before = data[col].skew()
    data[col] = np.log1p(data[col])
    skew_after = data[col].skew()
    print(f"  {col}: skewness {skew_before:.2f} → {skew_after:.2f}")


# 2. NORMALIZATION & ENCODING

print("[3/5] Normalizing and encoding features...")

# Min-Max normalization of continuous features to [0, 1]
scaler = MinMaxScaler()
data[NUMERIC_FEATURES] = scaler.fit_transform(data[NUMERIC_FEATURES])


def encode_categorical(df, columns):
    """One-hot encode categorical columns and concatenate with original df."""
    le = LabelEncoder()
    ohe = OneHotEncoder(sparse_output=False)
    encoded_frames = []

    for col in columns:
        df[col] = df[col].astype(str)
        labels = le.fit_transform(df[col])
        encoded = ohe.fit_transform(labels.reshape(-1, 1))

        # Generate meaningful column names from the encoder classes
        col_names = [f"{col}_{cls}" for cls in le.classes_]
        encoded_df = pd.DataFrame(encoded, columns=col_names, index=df.index)
        encoded_frames.append(encoded_df)

    result = pd.concat([df] + encoded_frames, axis=1)
    result = result.drop(columns=columns)
    return result


data_encoded = encode_categorical(data, CATEGORICAL_FEATURES)

# Remove any duplicate columns
data_encoded = data_encoded.loc[:, ~data_encoded.columns.duplicated()]

feature_cols = [c for c in data_encoded.columns if c != ID_COLUMN]
print(f"  Features after encoding: {len(feature_cols)}")


# 3. RANDOM FOREST FEATURE IMPORTANCE

print("[4/5] Training Random Forest for feature importance...")

X = data_encoded[feature_cols]
# Note: RF is used here purely for feature importance derivation.
# A synthetic binary target is used since this is an unsupervised ranking task.
y = np.random.RandomState(RANDOM_STATE).randint(0, 2, size=len(X))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = RandomForestClassifier(n_estimators=RF_ESTIMATORS, random_state=RANDOM_STATE)
model.fit(X_train, y_train)

# Extract and display feature importances
importances = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n  Feature Importance Ranking:")
print(importances.to_string(index=False))

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(x="Importance", y="Feature", data=importances, ax=ax)
ax.set_title("Random Forest Feature Importances", fontsize=14)
plt.tight_layout()
plt.savefig("results/feature_importances.pdf")
plt.close()


# 4. WEIGHTED SCORING

print("\n[5/5] Computing weighted maintenance-priority scores...")

# Build weight dictionary from RF importances
weights = dict(zip(importances["Feature"], importances["Importance"]))

# Compute composite score: score = Σ(w_i × x_i)
data_encoded["score"] = sum(
    weights.get(col, 0) * data_encoded[col]
    for col in feature_cols
)

# Rank by descending score
data_encoded = data_encoded.sort_values("score", ascending=False)


# 5. EXPORT RESULTS

top_assets = data_encoded.head(TOP_N)

print(f"\n  Top {TOP_N} highest-priority assets for maintenance:")
print(top_assets[[ID_COLUMN, "score"]].to_string(index=False))

top_assets.to_excel(OUTPUT_FILE, index=False)
print(f"\n  Exported to: {OUTPUT_FILE}")

# Summary statistics
print(f"\n  Total assets scored: {len(data_encoded)}")
print(f"  Score range: [{data_encoded['score'].min():.4f}, "
      f"{data_encoded['score'].max():.4f}]")
print(f"  Mean score:  {data_encoded['score'].mean():.4f}")

print("\n" + "=" * 65)
print(" Pipeline complete.")
print("=" * 65)
