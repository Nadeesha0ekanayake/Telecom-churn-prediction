"""MLflow setup for local tracking (Path α / Track T1).

Why SQLite (not the default file store)? The MLflow **Model Registry** — which we'll
use in the inference step — requires a database-backed tracking store. SQLite gives us
that with zero server setup. Artifacts (models, plots) go to a local folder.

Path β note: on Databricks this whole file collapses to
`mlflow.set_tracking_uri("databricks")` — the managed workspace provides the backend.
"""
from __future__ import annotations

import mlflow

from src import config


def setup_mlflow(experiment: str = config.MLFLOW_EXPERIMENT) -> None:
    """Point MLflow at the local SQLite store and select the experiment."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment) is None:
        config.MLARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(
            experiment, artifact_location=config.MLARTIFACTS_DIR.as_uri()
        )
    mlflow.set_experiment(experiment)
