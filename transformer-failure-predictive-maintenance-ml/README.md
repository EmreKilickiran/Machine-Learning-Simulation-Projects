# Predictive Maintenance Optimization for Power Distribution Networks

Two computational models for predictive maintenance prioritization across a power distribution network, developed during an ML internship at **Enerjisa JSC**. Both models were validated against real-world fault data and deployed into Enerjisa's operational maintenance workflow.

## Data Preprocessing

The raw dataset consisted of unstructured transformer maintenance records collected from multiple cities, containing missing values, inconsistent entries, and redundant features. The full pipeline consolidated **219,670 raw fault records** into **81,388 asset-level feature vectors** through data cleaning, feature engineering, one-hot encoding, Min-Max normalization, and log-transforms on skewed distributions.

## Project 2.1: Anomaly Detection (Isolation Forest)

**Objective:** Identify grid assets exhibiting statistically abnormal operational behavior using unsupervised learning, enabling early detection of units at highest risk of major service disruptions.

### Technical Approach

```
Raw Outage Data → Statistical Indicators (z-score, IQR)
       ↓
  Isolation Forest (contamination = 0.1)
       ↓
  SHAP Feature Importance Analysis
       ↓
  Anomaly Score Distribution → Flag High-Priority Assets
```

- Trained Isolation Forest on the preprocessed feature space to compute per-asset anomaly scores
- Computed statistical anomaly indicators (z-score, IQR) across all numerical features
- Analyzed SHAP values to quantify individual feature contributions - inventory age, outage duration, and customer impact emerged as dominant drivers
- Computed anomaly score distribution across ~60,000 assets; flagged lowest-scoring units as high-priority candidates

### Run

```bash
streamlit run anomaly_detection.py
```

## Project 2.2: Maintenance Prioritization (Random Forest Scoring)

**Objective:** Rank infrastructure units by predicted failure severity using a hybrid supervised-unsupervised framework, quantifying expected customer impact and outage duration for maintenance resource allocation.

### Technical Approach

```
Raw Fault Records → Log-Transform → Min-Max Normalization
       ↓
  One-Hot Encoding (Subtype, Class, Type)
       ↓
  Random Forest (100 estimators) → Feature Importance Weights
       ↓
  Weighted Score: score = Σ(w_i × x_i) for ~60,000 assets
       ↓
  Rank by Descending Score → Top-Priority Maintenance List
```

- Trained Random Forest (100 estimators) to derive feature importance weights via mean decrease in impurity
- Designed a custom weighted scoring function using RF-derived importance as coefficients
- Ranked all assets by descending score and exported top-priority units as actionable maintenance recommendations

### Run

```bash
python maintenance_prioritization.py
```

## Validation Results

The combined models flagged **40 critical infrastructure units** for preventive maintenance. Validated against actual 2024 fault records, **82% of flagged assets experienced confirmed major failures** within the first six months.

| Model | Avg Breakdown Duration (hrs) | Avg Affected People | Avg Breakdown Count |
|-------|-----|---------|-----|
| Random Forest + Scoring | 2.66 | 11,267.65 | 3.75 |
| Isolation Forest + Anomaly | 11.11 | 655.96 | 20.45 |

The two models provided complementary risk profiles: one optimizing for customer impact, the other for failure recurrence.

## Requirements

```
numpy
pandas
scikit-learn
shap
matplotlib
seaborn
streamlit       # for anomaly_detection.py dashboard
openpyxl        # for Excel I/O
```

## Repository Structure

```
.
├── README.md
├── anomaly_detection.py            # Project 2.1: Isolation Forest + SHAP (Streamlit)
├── maintenance_prioritization.py   # Project 2.2: Random Forest scoring pipeline
├── requirements.txt
├── data/                           # (not tracked — NDA)
│   ├── Data.csv                    # Feeder outage data (anonymized)
│   └── frekans2.xlsx               # Transformer maintenance records
└── results/                        # (generated)
    ├── feature_importances.pdf
    └── top_priority_assets.xlsx
```

## Note

This version uses modified data to comply with Enerjisa's non-disclosure agreement. All sensitive information (asset identifiers, geographic coordinates, customer data) has been anonymized. The analytical methodology and system architecture remain true to the original implementation.
