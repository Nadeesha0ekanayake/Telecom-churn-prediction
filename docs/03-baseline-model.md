# 03 · Baseline Model + MLflow (Step 3, Path α)

> Learning notes. Code: `src/tracking.py`, `src/models.py`, `src/evaluate.py`;
> runner: `scripts/03_baseline.py`. Run: `./venv/bin/python scripts/03_baseline.py`

## MLflow, set up locally — `src/tracking.py`

MLflow tracks **experiments** (each training run's params + metrics + artifacts) and,
later, a **model registry** (versioned, named models for deployment).

- We point it at a **SQLite** backend: `sqlite:///mlflow.db`. Why not the default file
  store? The **Model Registry requires a database backend** — SQLite gives us one with
  zero server setup. Artifacts (the pickled models, plots) go to `mlartifacts/`.
- Both `mlflow.db` and `mlartifacts/` are **gitignored** (regenerable, and can get large).
- **Path β contrast:** on Databricks this entire helper becomes
  `mlflow.set_tracking_uri("databricks")` — the workspace hosts the backend for you.

**Open the UI to click through runs:**
```bash
./venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
# then visit http://127.0.0.1:5000
```

## The leakage-safe pipeline — `src/models.py`

Every model is `make_pipeline(estimator, ...)` = **preprocessor + estimator** in one
sklearn `Pipeline`. This is the pay-off of Step 2's design: during cross-validation the
scaler/encoder are refit on each fold's *training* portion only, so no test information
leaks in. The saved model also preprocesses raw input itself at inference time.

## The three baselines & why

| Model | Purpose |
|---|---|
| `dummy_most_frequent` | Always predicts the majority class ("No"). The **floor** any real model must beat. |
| `logreg` | Plain logistic regression — interpretable, honest linear baseline. |
| `logreg_balanced` | `class_weight='balanced'` — re-weights the 26.5% minority so churners aren't ignored. |

## Results (5-fold CV on train + held-out test)

|                     | cv_roc_auc | accuracy | precision | recall | f1 | roc_auc |
|---------------------|-----------|----------|-----------|--------|------|---------|
| dummy_most_frequent | 0.500±0.000 | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| logreg              | 0.846±0.012 | **0.799** | **0.653** | 0.519 | 0.578 | 0.842 |
| logreg_balanced     | 0.846±0.011 | 0.732 | 0.497 | **0.791** | **0.611** | 0.842 |

### What this teaches

1. **Accuracy is a trap on imbalanced data.** The dummy scores **73.5% accuracy** while
   being useless — it catches **0 of 374** churners (see `06_baseline_confusion.png`).
   Never lead with accuracy here.

2. **Logistic regression is a strong baseline: ROC-AUC 0.842.** For a linear model on
   this dataset that's genuinely good — the bar for the tree models in Step 4–5 is high.

3. **`class_weight='balanced'` is a recall/precision *trade*, not a free lunch:**
   - plain logreg catches **194/374** churners (recall 0.52) with 103 false alarms;
   - balanced catches **296/374** (recall 0.79) but with 299 false alarms (precision 0.50).
   - F1 improves (0.58 → 0.61); accuracy drops (0.80 → 0.73).
   - **Which is "better" depends on business cost** — is missing a churner worse than a
     wasted retention offer? We make that explicit in Step 7 (cost-based threshold).

4. **Both logregs have the *same* ROC-AUC (0.842).** Subtle but important: `class_weight`
   mostly shifts the **decision threshold** (where probability → Yes/No), it doesn't
   change how well the model *ranks* customers by risk. ROC-AUC measures ranking across
   *all* thresholds, so it's ~unchanged. This is why AUC is our **model-selection**
   metric, and threshold/precision/recall are **operating-point** decisions layered on top.

## Figures
- `05_baseline_roc.png` — ROC curves; both logregs sit well above the 0.5 chance line.
- `06_baseline_confusion.png` — the three confusion matrices side by side (the story above).

## MLflow logged (3 runs in experiment `telecom-churn`)
Each run stores params (model, scaler, encoder, class_weight), CV + test metrics, the
fitted pipeline (with signature + input example), ready to compare against Step 4–5 in
the same UI.

## Carried into Step 4–5
- Beat **ROC-AUC 0.842** with tree models (RandomForest, XGBoost, LightGBM).
- Log them to the same experiment → one comparison table / plot across all models.
- Revisit imbalance handling (`class_weight` / SMOTE) per model.

---

## Appendix · What is MLflow, and who owns it? (background)

A common misconception is that MLflow is an Azure/Microsoft product. It isn't.

**What it is:** an **open-source** (Apache-2.0) platform for the ML lifecycle —
experiment tracking, model packaging, and deployment. Vendor-neutral; runs anywhere.

**Origin/ownership:**
- **Created by Databricks** (the Apache Spark company), first released **2018**.
- **Donated to the Linux Foundation in 2020** → community-governed, not owned by any
  single cloud. You can use it with zero connection to Databricks (as we do: `pip install
  mlflow` + a local SQLite file).

**Why the Azure confusion:** several clouds embed the *same* open-source MLflow as a
managed feature, so people meet it there first:

| Where you meet MLflow | What's actually happening |
|---|---|
| **Azure Machine Learning** | Uses MLflow as its native tracking API — Microsoft just *hosts* the OSS tool |
| **Azure Databricks** | Microsoft-managed Databricks, ships with "Managed MLflow" |
| **AWS SageMaker** | Also offers managed MLflow |
| **Databricks (any cloud)** | Managed MLflow is a core feature |

Each vendor runs the server for you; the library and API are identical. Only *who hosts
the backend* changes.

**Three ways to run it:**
1. **Local** (this project) — file or SQLite backend, `mlflow ui`, no server/cloud/account.
2. **Self-hosted server** — `mlflow server` + a DB + artifact store (e.g. S3) for a team.
3. **Managed** — a cloud (Databricks / Azure ML / SageMaker) hosts it; point at its URI.

**Four components:** Tracking (params/metrics/artifacts) · Models (standard packaging) ·
Model Registry (versioned named models + stages — needs a DB backend, hence our SQLite) ·
Projects (reproducible run format).

**Why this matters for our Path α vs β design:** the *only* line that changes between
local and Databricks is the tracking URI —

```python
# Path α (now):   local, self-hosted
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# Path β (later): company-managed Databricks
mlflow.set_tracking_uri("databricks")
```

Same MLflow, same API, same logging code in `src/tracking.py`. That portability is
exactly *because* MLflow is an open standard, not an Azure-locked tool.
