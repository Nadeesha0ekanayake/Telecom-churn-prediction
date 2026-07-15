"""Step 3 runner: baseline models (Dummy + Logistic Regression) with MLflow.

    ./venv/bin/python scripts/03_baseline.py

For each model: cross-validate ROC-AUC on train, fit on train, evaluate on test,
and log params/metrics/model/figures to local MLflow. Prints a comparison table.
Inspect runs with:  ./venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Spurious NumPy-2.0 / Apple-Accelerate BLAS warnings on macOS. Results are
# correct and stable; a real overflow would produce NaNs. Silence the noise.
for _msg in ("divide by zero encountered in matmul",
             "overflow encountered in matmul",
             "invalid value encountered in matmul"):
    warnings.filterwarnings("ignore", message=_msg, category=RuntimeWarning)
# Benign for us: our features have no missing values at inference time.
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer",
                        category=UserWarning)

import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402

from src import data_loader, evaluate, models, preprocess, tracking  # noqa: E402


def main() -> None:
    # Data (rebuilt from raw so we don't depend on the gitignored processed file)
    df = preprocess.prepare(data_loader.load_raw())
    numeric, categorical = preprocess.feature_columns(df)
    X_train, X_test, y_train, y_test = preprocess.split(df)

    tracking.setup_mlflow()
    results: dict = {}

    for name, estimator in models.get_baseline_models().items():
        pipe = models.make_pipeline(estimator, numeric, categorical,
                                    scaler="standard", encoder="onehot")
        cv_mean, cv_std = evaluate.cross_val_auc(pipe, X_train, y_train)

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        test_metrics = evaluate.classification_metrics(y_test, y_pred, y_proba)

        results[name] = {
            "cv_auc_mean": cv_mean, "cv_auc_std": cv_std,
            "test": test_metrics, "y_pred": y_pred, "y_proba": y_proba,
        }

        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "model": name,
                "scaler": "standard",
                "encoder": "onehot",
                "class_weight": getattr(estimator, "class_weight", None),
                "n_features_in": len(numeric) + len(categorical),
            })
            mlflow.log_metric("cv_roc_auc_mean", cv_mean)
            mlflow.log_metric("cv_roc_auc_std", cv_std)
            for k, v in test_metrics.items():
                mlflow.log_metric(f"test_{k}", v)
            mlflow.sklearn.log_model(
                pipe, name="model",
                signature=infer_signature(X_train, y_pred),
                input_example=X_train.head(3),
            )

    # --- comparison table ---
    print("\n" + "=" * 70 + "\nBASELINE COMPARISON (test set)\n" + "=" * 70)
    table = pd.DataFrame({
        name: {
            "cv_roc_auc": f"{r['cv_auc_mean']:.3f}±{r['cv_auc_std']:.3f}",
            **{k: round(v, 3) for k, v in r["test"].items()},
        }
        for name, r in results.items()
    }).T
    print(table.to_string())

    # --- figures ---
    print("\nFigures:")
    print(" ", evaluate.plot_roc(results, y_test))
    print(" ", evaluate.plot_confusions(results, y_test))
    print(f"\nMLflow runs logged to {tracking.config.MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
