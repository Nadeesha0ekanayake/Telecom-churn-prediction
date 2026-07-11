# 00 · Environment Setup (Path α — VS Code local)

> Learning notes: **what we did and why**. This is the E1 environment from the roadmap
> (VS Code local, `venv` + pip). Path β (Databricks) will *not* need most of this — the
> cluster provides Python and most libraries — which is itself a useful contrast.

## 1. Why a virtual environment (`venv`)?

```bash
python3 -m venv venv          # create an isolated Python in ./venv
./venv/bin/pip install ...    # installs land here, not in system Python
```

- **Isolation** — the project's packages don't collide with your system Python or other
  projects. Delete `venv/` and the project's footprint is gone.
- **Reproducibility** — everyone (and future-you) gets the same versions.
- `venv/` is **gitignored** — it's large, OS-specific, and fully rebuildable from
  `requirements.txt`. You commit the *recipe*, not the *cooked meal*.

Python used: **3.9.6** (system, via pyenv shims).

## 2. The macOS-specific gotcha: `libomp`

**XGBoost and LightGBM need the OpenMP runtime** (`libomp`) to run their multi-threaded
tree building. On Linux this ships with the compiler; on **macOS it is missing by
default**, so importing the libraries fails with a `Library not loaded: libomp.dylib`
error. We install it with Homebrew (a *system* dependency, not a pip package):

```bash
brew install libomp        # installed: libomp 22.1.8
```

Why this matters for learning: pip installs the Python wheels, but those wheels
dynamically link to a native `libomp.dylib` that pip does **not** provide on macOS.
This is a classic "Python package depends on a system library" situation. On Databricks
(Path β) the cluster already has OpenMP, so this step disappears — a good example of what
a managed environment hides for you.

## 3. Installing the Python dependencies

```bash
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt
```

### Two files, on purpose (a reproducibility pattern)

| File | Role | Analogy |
|------|------|---------|
| `requirements.txt` | Curated, human-readable, **what we asked for** (unpinned, grouped, commented) | shopping list |
| `requirements-lock.txt` | `pip freeze` output — **exact resolved versions** of every transitive dependency (166 packages) | itemised receipt |

To rebuild the *exact* environment later: `pip install -r requirements-lock.txt`.
To upgrade deliberately: edit `requirements.txt`, reinstall, re-freeze the lock.

### What each top-level dependency is for

| Package | Version | Why it's here |
|---------|---------|---------------|
| pandas | 2.3.3 | load/clean the tabular data |
| numpy | 2.0.2 | numeric arrays underneath everything |
| scipy | 1.13.1 | stats helpers (tests, distributions) |
| scikit-learn | 1.6.1 | pipelines, preprocessing, LogisticRegression, CV, metrics |
| xgboost | 2.1.4 | gradient-boosted trees — strong tabular model |
| lightgbm | 4.6.0 | gradient-boosted trees — fast, another to compare |
| imbalanced-learn | 0.12.4 | SMOTE / resampling for the ~26.5% churn imbalance |
| mlflow | 3.1.4 | experiment tracking + model registry + local serving (Track T1/S1) |
| shap | 0.49.1 | model explainability (Step 6) |
| matplotlib | 3.9.4 | base plotting |
| seaborn | 0.13.2 | statistical plots on top of matplotlib |
| jupyter + ipykernel | — | interactive notebooks in VS Code |
| optuna | 4.9.0 | hyperparameter search (Step 4–5 tuning) |

## 4. Verifying the install

```bash
./venv/bin/python -c "import pandas, numpy, sklearn, xgboost, lightgbm, imblearn, mlflow, shap, matplotlib, seaborn, optuna; print('all imports OK')"
```

All 12 imported cleanly.

**Benign warning you may see:**
```
NotOpenSSLWarning: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl'
module is compiled with 'LibreSSL 2.8.3'.
```
This is just macOS shipping LibreSSL instead of OpenSSL. It's a warning, not an error,
and doesn't affect our modelling. Safe to ignore.

## 5. Reproduce this environment from scratch

```bash
brew install libomp                              # macOS system dep for xgboost/lightgbm
cd telecom-churn-prediction
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt       # or requirements-lock.txt for exact pins
```

Optional (to use this venv as a named Jupyter kernel in VS Code):
```bash
./venv/bin/python -m ipykernel install --user --name telecom-churn --display-name "Telecom Churn (venv)"
```
