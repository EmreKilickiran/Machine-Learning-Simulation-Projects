
# anomaly_detection.py — Isolation Forest Anomaly Detection Dashboard
#
# Identifies grid assets exhibiting statistically abnormal operational behavior
# using unsupervised learning (Isolation Forest), with SHAP-based feature
# importance analysis for model interpretability.
#
# Pipeline:
#   1. Load feeder outage data (~60,000 assets)
#   2. Exploratory data analysis (age distribution, blackout frequency)
#   3. Train Isolation Forest (contamination = 0.1)
#   4. Compute SHAP values for feature importance
#   5. Visualize anomaly score distribution
#   6. Interactive feeder inspection and maintenance tracking
#
# Run:  streamlit run anomaly_detection.py


import shap
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest


# 1. DATA LOADING & PREPROCESSING

@st.cache_data
def load_data(path="data/Data.csv"):
    """Load and preprocess feeder outage data."""
    data = pd.read_csv(path, encoding="ISO-8859-1")
    # Bin age into 5-year intervals for distribution analysis
    bins = range(0, data["Age"].max() + 5, 5)
    data["AgeGroup"] = pd.cut(data["Age"], bins=bins)
    return data


@st.cache_resource
def train_isolation_forest(numeric_features, contamination=0.1):
    """Train Isolation Forest and compute anomaly scores."""
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(numeric_features)
    scores = model.decision_function(numeric_features)
    return model, scores


@st.cache_data
def compute_shap_values(_model, numeric_features, sample_size=1000):
    """Compute SHAP values for feature importance interpretation."""
    n = min(sample_size, len(numeric_features))
    X_sample = numeric_features.sample(n=n, random_state=28)
    explainer = shap.TreeExplainer(_model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, X_sample



# 2. STREAMLIT DASHBOARD

sns.set(style="whitegrid")
data = load_data()

st.title("Anomaly Detection — Enerjisa Feeder Outage Analysis")
st.caption("Isolation Forest with SHAP Feature Importance")

# --- 2.1 Age Distribution ---------------------------------------------------

st.header("Age Distribution Analysis")
st.write(
    "Distribution of feeder ages in 5-year intervals. Identifies potential "
    "maintenance needs and lifecycle patterns in the infrastructure."
)

age_counts = data["AgeGroup"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=age_counts.index.astype(str), y=age_counts.values,
            palette="coolwarm", ax=ax)
ax.set_xlabel("Age Group", fontsize=14)
ax.set_ylabel("Count", fontsize=14)
ax.set_title("Age Distribution in 5-Year Intervals", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)

# --- 2.2 Blackout Frequency Distribution ------------------------------------

st.header("Blackout Frequency Distribution")
st.write(
    "Blackout frequency across feeders. Counts above 50 are grouped "
    "for visualization clarity."
)

modified_blackouts = data["Blackout_count"].copy()
modified_blackouts[modified_blackouts > 50] = 51

fig, ax = plt.subplots(figsize=(12, 6))
bc = modified_blackouts.value_counts().sort_index()
sns.barplot(x=bc.index.astype(str), y=bc.values, palette="coolwarm", ax=ax)
ax.set_xlabel("Blackout Count", fontsize=14)
ax.set_ylabel("Frequency", fontsize=14)
ax.set_title("Distribution of Blackout Count (>50 grouped)", fontsize=16)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(fig)


# 3. ISOLATION FOREST MODEL

st.header("Isolation Forest Anomaly Detection")
st.write(
    "The Isolation Forest algorithm isolates anomalous data points by "
    "exploiting the principle that anomalies require fewer random partitions "
    "to isolate in a binary tree structure. Contamination = 0.1 (10% of "
    "data points assumed anomalous)."
)

numeric_features = data.select_dtypes(include=[np.number])
model, anomaly_scores = train_isolation_forest(numeric_features)
data["Anomaly_Score"] = anomaly_scores

# --- 3.1 SHAP Feature Importance --------------------------------------------

st.subheader("Feature Importance (SHAP Values)")

shap_values, X_sample = compute_shap_values(model, numeric_features)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_sample, plot_type="bar",
                  show=False, max_display=10)
plt.tight_layout()
st.pyplot(fig)

st.markdown("""
**Key findings:**
- **Age** is the most influential feature (highest SHAP value)
- **Customer count** ranks second, reflecting societal impact
- **Inventory blackout count** provides comparative context within asset groups
- Infrastructure age and comparative metrics drive anomaly predictions
""")

# --- 3.2 Anomaly Score Visualization ----------------------------------------

st.subheader("Anomaly Score Distribution")

fig, ax = plt.subplots(figsize=(12, 6))
sns.scatterplot(x=data.index, y="Anomaly_Score", data=data,
                hue="Anomaly_Score", palette="coolwarm", alpha=0.7, ax=ax)
ax.set_title("Anomaly Scores per Feeder", fontsize=16)
ax.legend(loc="lower right")
plt.tight_layout()
st.pyplot(fig)

st.write(
    "Negative scores indicate potential anomalies. Points closer to -1 "
    "represent the most anomalous feeders requiring immediate attention."
)


# 4. FEEDER INSPECTION

st.header("Feeder Anomaly Rankings")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Ranked by Anomaly Score")
    st.dataframe(
        data[["Feeder", "Anomaly_Score"]]
        .sort_values("Anomaly_Score", ascending=True)
        .reset_index(drop=True),
        height=400
    )

with col2:
    st.subheader("Feeder Detail View")
    selected = st.selectbox("Select a feeder:", data["Feeder"].unique())
    display_cols = [c for c in data.columns if c != "AgeGroup"]
    st.dataframe(data[data["Feeder"] == selected][display_cols])


# 5. MAINTENANCE TRACKING SYSTEM

st.header("Maintenance Management")


class Feeder:
    def __init__(self, name):
        self.name = name
        self.needs_maintenance = True


class MaintenanceSystem:
    def __init__(self):
        self.feeders = []
        self.maintenance_done = pd.DataFrame(columns=["Feeder"])

    def add_feeder(self, feeder):
        self.feeders.append(feeder)

    def list_maintenance_required(self):
        return [f for f in self.feeders if f.needs_maintenance]

    def perform_maintenance(self, feeder_name):
        feeder = next(
            (f for f in self.feeders
             if f.name == feeder_name and f.needs_maintenance),
            None
        )
        if feeder:
            feeder.needs_maintenance = False
            self.feeders.remove(feeder)
            new_row = pd.DataFrame([{"Feeder": feeder_name}])
            self.maintenance_done = pd.concat(
                [self.maintenance_done, new_row], ignore_index=True
            )


# Initialize session state
if "system" not in st.session_state:
    st.session_state.system = MaintenanceSystem()
    for name in data["Feeder"].unique():
        st.session_state.system.add_feeder(Feeder(name))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feeder Selection")
    feeder_list = [f.name for f in st.session_state.system.list_maintenance_required()]
    selected_feeders = st.multiselect("Select feeders for maintenance:", feeder_list)
    if st.button("Add to Maintenance List"):
        st.session_state.selected_feed = selected_feeders

with col2:
    st.subheader("Maintenance Queue")
    if "selected_feed" in st.session_state and st.session_state.selected_feed:
        maint_df = pd.DataFrame(st.session_state.selected_feed, columns=["Feeder"])
        maint_df.index = maint_df.index + 1
        st.table(maint_df)
    else:
        st.info("No feeders in maintenance queue.")
