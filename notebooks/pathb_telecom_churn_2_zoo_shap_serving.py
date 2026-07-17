# Databricks notebook source
# MAGIC %md
# MAGIC # Telecom Churn — Path β (Databricks) · Part 2
# MAGIC
# MAGIC Steps 4–5 (model zoo) + Step 6 (SHAP) + S2 (serving), continuing the Path β port.
# MAGIC Self-contained. Managed MLflow, same `src/` logic as Path α.
# MAGIC
# MAGIC **Cluster note:** the shared cluster is a *standard* runtime, so we `%pip install`
# MAGIC the tree/SHAP libraries below (an **ML runtime** would already have them).

# COMMAND ----------

# MAGIC %pip install -q xgboost lightgbm shap
# MAGIC # (%pip auto-restarts the Python process; all state below is fresh)

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

DATA_URL = ("https://raw.githubusercontent.com/IBM/"
            "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv")
TARGET, ID_COL, SEED = "Churn", "customerID", 42
ADDON = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
         "TechSupport", "StreamingTV", "StreamingMovies"]

# --- load + prepare (same logic as src/, re-run because %pip reset the kernel) ---
def prepare(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["tenure"] == 0, "TotalCharges"] = df.loc[df["tenure"] == 0, "TotalCharges"].fillna(0.0)
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df = df.replace(["No internet service", "No phone service"], "No")
    df["num_addon_services"] = (df[ADDON] == "Yes").sum(axis=1)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-0.1, 12, 24, 48, 72],
                                labels=["0-12", "12-24", "24-48", "48-72"]).astype(str)
    df["has_internet"] = (df["InternetService"] != "No").map({True: "Yes", False: "No"})
    return df

df = prepare(pd.read_csv(DATA_URL))
drop = {ID_COL, TARGET}
numeric = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
categorical = [c for c in df.columns if c not in drop and c not in numeric]
X = df.drop(columns=[ID_COL, TARGET]); y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/nadeeshaekanayake@hipagesgroup.com.au/telecom-churn-pathb")
print("ready:", X_train.shape, "train /", X_test.shape, "test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Steps 4–5 · Model zoo (same as Path α)
# MAGIC Preprocessing matched per model; imbalance handled per model. Best selected by CV AUC.

# COMMAND ----------

def make_pipeline(estimator, scaler="standard"):
    num_t = StandardScaler() if scaler == "standard" else "passthrough"
    pre = ColumnTransformer([
        ("num", num_t, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return Pipeline([("pre", pre), ("clf", estimator)])

specs = {
    "logreg_balanced": (LogisticRegression(max_iter=1000, class_weight="balanced",
                                           random_state=SEED), "standard"),
    "random_forest": (RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                             random_state=SEED, n_jobs=-1), "none"),
    "xgboost": (XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                              scale_pos_weight=2.77, random_state=SEED, n_jobs=-1), "none"),
    "lightgbm": (LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                class_weight="balanced", random_state=SEED,
                                n_jobs=-1, verbose=-1), "none"),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
rows, run_ids = [], {}
for name, (est, scaler) in specs.items():
    pipe = make_pipeline(est, scaler)
    cv_auc = cross_val_score(pipe, X_train, y_train, scoring="roc_auc", cv=cv).mean()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]; pred = pipe.predict(X_test)
    m = {"cv_roc_auc": cv_auc, "accuracy": accuracy_score(y_test, pred),
         "precision": precision_score(y_test, pred, zero_division=0),
         "recall": recall_score(y_test, pred, zero_division=0),
         "f1": f1_score(y_test, pred, zero_division=0),
         "roc_auc": roc_auc_score(y_test, proba)}
    rows.append({"model": name, **{k: round(v, 3) for k, v in m.items()}})
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tag("path", "beta_databricks"); mlflow.set_tag("stage", "model_zoo")
        mlflow.log_param("model", name)
        for k, v in m.items():
            mlflow.log_metric(k, v)
        mlflow.sklearn.log_model(pipe, name="model")
        run_ids[name] = run.info.run_id

results = pd.DataFrame(rows).set_index("model").sort_values("cv_roc_auc", ascending=False)
best = results.index[0]
print("best by CV AUC:", best)
display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6 · SHAP on the selected model

# COMMAND ----------

import shap
import matplotlib.pyplot as plt

best_pipe = make_pipeline(*specs[best]).fit(X_train, y_train)
pre, clf = best_pipe.named_steps["pre"], best_pipe.named_steps["clf"]
names = list(pre.get_feature_names_out())
X_test_t = pre.transform(X_test)

if hasattr(clf, "coef_"):            # linear winner → exact LinearExplainer
    expl = shap.LinearExplainer(clf, pre.transform(X_train))
    sv = expl.shap_values(X_test_t)
else:                                # tree winner → TreeExplainer
    expl = shap.TreeExplainer(clf)
    sv = expl.shap_values(X_test_t)
    if isinstance(sv, list):
        sv = sv[1]

shap.summary_plot(sv, X_test_t, feature_names=names, max_display=15, show=False)
display(plt.gcf())
plt.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## S2 · Serving — batch via Spark UDF (the scale-out Databricks pattern)
# MAGIC Load the best model from MLflow and score a Spark DataFrame in parallel.
# MAGIC (Returns class labels — the pyfunc contract, same as the local REST endpoint.)

# COMMAND ----------

from pyspark.sql import functions as F

model_uri = f"runs:/{run_ids[best]}/model"
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri, result_type="integer")

sdf = spark.createDataFrame(X_test.reset_index(drop=True))
scored = sdf.withColumn("churn_pred", predict_udf(F.struct(*[F.col(c) for c in X_test.columns])))
display(scored.select("tenure", "Contract", "InternetService",
                      "MonthlyCharges", "churn_pred").limit(12))

# COMMAND ----------

# MAGIC %md
# MAGIC ## S2 · Real-time endpoint (OPTIONAL — reference / opt-in)
# MAGIC
# MAGIC Modern Databricks Model Serving needs the model in **Unity Catalog** and a serving
# MAGIC entitlement, and a live endpoint **provisions paid compute**. First see which UC
# MAGIC locations your identity can write to:

# COMMAND ----------

# Discover Unity Catalog locations available to YOUR interactive identity
try:
    display(spark.sql("SHOW CATALOGS"))
except Exception as e:
    print("Could not list catalogs:", e)

# COMMAND ----------

# MAGIC %md
# MAGIC Then, to actually deploy (only if you have a writable UC schema + serving access):
# MAGIC flip `CREATE_ENDPOINT = True` and set `CATALOG` / `SCHEMA`. Left OFF so nothing
# MAGIC paid is created by accident.

# COMMAND ----------

CREATE_ENDPOINT = False          # <-- set True to actually deploy (paid infra!)
CATALOG, SCHEMA = "<your_catalog>", "<your_schema>"
ENDPOINT_NAME = "telecom-churn-pathb"

if CREATE_ENDPOINT:
    mlflow.set_registry_uri("databricks-uc")
    full_name = f"{CATALOG}.{SCHEMA}.telecom_churn"
    mv = mlflow.register_model(model_uri, full_name)          # register to UC
    from mlflow.deployments import get_deploy_client
    client = get_deploy_client("databricks")
    client.create_endpoint(
        name=ENDPOINT_NAME,
        config={"served_entities": [{
            "entity_name": full_name, "entity_version": mv.version,
            "workload_size": "Small", "scale_to_zero_enabled": True,
        }]},
    )
    print(f"Deploying endpoint '{ENDPOINT_NAME}' from {full_name} v{mv.version} ...")
else:
    print("CREATE_ENDPOINT is False — no endpoint created (safe default).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recap
# MAGIC - **Zoo + SHAP** reproduce Path α on Databricks (expect `logreg_balanced` best,
# MAGIC   CV AUC ≈ 0.846).
# MAGIC - **Batch serving** via `spark_udf` — the realistic Databricks inference pattern,
# MAGIC   no extra infra.
# MAGIC - **Real-time endpoint** is one opt-in cell away, gated on a UC location + serving
# MAGIC   entitlement + willingness to run paid compute.
