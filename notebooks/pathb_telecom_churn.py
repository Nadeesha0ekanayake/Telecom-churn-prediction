# Databricks notebook source
# MAGIC %md
# MAGIC # Telecom Churn — Path β (Databricks)
# MAGIC
# MAGIC Port-and-compare of the local Path α project onto Databricks with **managed MLflow**.
# MAGIC This notebook is **self-contained** (inlines the `src/` logic) and covers
# MAGIC **Steps 1–3**: load → clean/feature-engineer → baseline models.
# MAGIC
# MAGIC **The whole point:** the data-science logic is identical to Path α. The only real
# MAGIC change is the MLflow tracking backend — local `sqlite:///mlflow.db` becomes the
# MAGIC Databricks-managed tracking server (one line).
# MAGIC
# MAGIC **Cluster:** attach to an **ML runtime** cluster (pandas / scikit-learn / mlflow
# MAGIC pre-installed). Steps 4–5 (XGBoost/LightGBM) and SHAP come in a later notebook.

# COMMAND ----------

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
import mlflow
import mlflow.sklearn

# --- config (mirrors src/config.py) ---
DATA_URL = ("https://raw.githubusercontent.com/IBM/"
            "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
TARGET, ID_COL, SEED = "Churn", "customerID", 42
ADDON = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
         "TechSupport", "StreamingTV", "StreamingMovies"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1 · Load & quick look
# MAGIC Loading straight from the public IBM URL — no file upload needed.

# COMMAND ----------

raw = pd.read_csv(DATA_URL)
print(f"shape: {raw.shape[0]:,} rows x {raw.shape[1]} cols")
print("\nchurn balance:")
print(raw[TARGET].value_counts(normalize=True).round(4).to_string())
print("\nNote: TotalCharges dtype =", raw["TotalCharges"].dtype,
      "(text, because of 11 blank tenure-0 rows — same quirk as Path α)")
display(raw.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2 · Clean + feature engineering
# MAGIC Identical logic to `src/preprocess.py` (`clean` + `add_features`).

# COMMAND ----------

def clean(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    return df.replace(["No internet service", "No phone service"], "No")

def add_features(df):
    df = df.copy()
    df["num_addon_services"] = (df[ADDON] == "Yes").sum(axis=1)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-0.1, 12, 24, 48, 72],
                                labels=["0-12", "12-24", "24-48", "48-72"]).astype(str)
    df["has_internet"] = (df["InternetService"] != "No").map({True: "Yes", False: "No"})
    return df

df = add_features(clean(raw))
drop = {ID_COL, TARGET}
numeric = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
categorical = [c for c in df.columns if c not in drop and c not in numeric]
print(f"numeric ({len(numeric)}): {numeric}")
print(f"categorical ({len(categorical)}): {categorical}")
display(df[["tenure", "tenure_group", "num_addon_services", "has_internet", TARGET]].head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3 · Baseline models + **managed MLflow**
# MAGIC
# MAGIC 👇 **This is the one line that differs from Path α.** Locally we used
# MAGIC `sqlite:///mlflow.db`; here we use the Databricks-managed tracking server.

# COMMAND ----------

# Path α:  mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("databricks")   # <-- the only real change
mlflow.set_experiment("/Users/nadeeshaekanayake@hipagesgroup.com.au/telecom-churn-pathb")

# COMMAND ----------

X = df.drop(columns=[ID_COL, TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

def make_pipeline(estimator):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])

models = {
    "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
    "logreg": LogisticRegression(max_iter=1000, random_state=SEED),
    "logreg_balanced": LogisticRegression(max_iter=1000, class_weight="balanced",
                                          random_state=SEED),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows = []
for name, est in models.items():
    pipe = make_pipeline(est)
    cv_auc = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=cv)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    metrics = {
        "cv_roc_auc": cv_auc.mean(),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    rows.append({"model": name, **{k: round(v, 3) for k, v in metrics.items()}})
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("path", "beta_databricks")
        mlflow.log_param("model", name)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(pipe, name="model")

results = pd.DataFrame(rows).set_index("model")
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare to Path α (local)
# MAGIC
# MAGIC | model | Path α CV AUC | expected here |
# MAGIC |---|---|---|
# MAGIC | dummy | 0.500 | 0.500 |
# MAGIC | logreg | 0.846 | ~0.846 |
# MAGIC | logreg_balanced | 0.846 | ~0.846 |
# MAGIC
# MAGIC Same data + same logic ⇒ **same numbers** (identical seed & split). What changed:
# MAGIC - **MLflow** runs now live in the **workspace** (open the Experiments UI) — no local
# MAGIC   `mlflow ui`, no SQLite, no setup.
# MAGIC - **Environment**: the cluster already had pandas/sklearn/mlflow — no venv, no
# MAGIC   `brew install libomp`.
# MAGIC
# MAGIC **Next (later notebook):** Steps 4–5 (XGBoost/LightGBM) + SHAP, then Databricks
# MAGIC Model Serving in place of the local `mlflow models serve`.
