# Telecom Churn Prediction — Roadmap

A **learning project**. Goal: predict customer churn on the IBM Telco dataset, while
deliberately trying **multiple options at each stage** to understand how things work.

## Guiding approach

- **Separate data-science logic from the execution environment.** Real logic lives in
  `src/` (plain, environment-agnostic Python). Thin "runners" (scripts / notebooks) call
  into it. This lets the *same* logic run in VS Code and in Databricks.
- **Finish one path end-to-end first, then traverse the others.** A single working path
  becomes the reference implementation; later paths are "port + compare" exercises.

## Cross-cutting tracks (pick one per path)

| Track | Option 1 | Option 2 |
|-------|----------|----------|
| **E — Environment** | E1 VS Code local (venv + pip) | E2 Databricks notebooks |
| **T — Experiment tracking (MLflow)** | T1 Local MLflow (file/sqlite) | T2 Databricks-managed MLflow |
| **S — Serving / Inference** | S1 MLflow (registry + pyfunc batch + `mlflow models serve`) | S2 Databricks Model Serving + Spark UDF batch |

## The 7-step spine (options per step)

**Step 1 · Data Loading & Exploration**
- Load: pandas · Spark DataFrame · polars
- EDA: manual matplotlib/seaborn · ydata-profiling · Databricks `display()` profiler
- Checks: shape, dtypes, nulls, duplicates, target balance (~26.5% churn), `TotalCharges` blanks

**Step 2 · Cleaning & Feature Engineering**
- Missing (`TotalCharges` blanks): drop · impute median · impute 0
- Encoding: OneHot · Ordinal · target encoding
- Scaling: StandardScaler · MinMax · none (trees don't need it)
- Pipeline: sklearn `ColumnTransformer`+`Pipeline` (leakage-safe) · manual
- Imbalance: `class_weight` · SMOTE · threshold tuning
- Split: stratified train/test (+ CV folds)

**Step 3 · Baseline**
- `DummyClassifier` (sanity floor) · Logistic Regression (real baseline)
- Metrics: accuracy, AUC-ROC, precision, recall, F1 + coefficients

**Step 4–5 · Model zoo (train & compare)**
- Linear: Logistic Regression (L1/L2)
- Trees: RandomForest, XGBoost, LightGBM (+ CatBoost, GradientBoosting)
- Contrast (optional): SVM / KNN / NaiveBayes
- Tuning: GridSearchCV · RandomizedSearchCV · Optuna
- All runs logged to MLflow → comparison table + side-by-side confusion matrices / ROC

**Step 6 · Interpretation**
- SHAP: TreeExplainer (trees) / LinearExplainer (LogReg)
- Global: summary/beeswarm · Local: force/waterfall for one customer
- Cross-check: permutation importance, partial dependence

**Step 7 · Business Angle**
- Cost-based threshold (cost of churn vs. retention-offer cost)
- Segment high-value at-risk customers
- Link to Next Best Action / retention playbook

## Path plan

- **Path α (first, in progress):** `E1 + T1 + full model zoo + SHAP + S1`
  → VS Code local, local MLflow, train all models, SHAP, MLflow batch + local REST inference.
- **Path β (later):** `E2 + T2 + S2`
  → Same `src/` logic from Databricks notebooks, managed MLflow, Databricks Model Serving.

## Progress log

- [x] Project scaffold + dataset downloaded (`data/raw/Telco-Customer-Churn.csv`)
- [x] Git repo + first commit + pushed to personal GitHub
- [x] Local venv + dependencies installed + libomp (`docs/00-environment-setup.md`)
- [ ] Step 1 · Data Loading & Exploration
- [ ] Step 2 · Cleaning & Feature Engineering
- [ ] Step 3 · Baseline
- [ ] Step 4–5 · Model zoo
- [ ] Step 6 · SHAP interpretation
- [ ] Step 7 · Business angle
- [ ] Path β · Databricks traversal
