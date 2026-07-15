"""Model definitions and the preprocessor+estimator pipeline factory.

Keeping every model behind a single `make_pipeline()` means preprocessing is always
bundled with the estimator into one object — so cross-validation fits the scaler/encoder
on each training fold only (leakage-safe), and the saved model preprocesses raw input
itself at inference time.
"""
from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config, preprocess


def make_pipeline(
    estimator,
    numeric: list[str],
    categorical: list[str],
    scaler: str = "standard",
    encoder: str = "onehot",
) -> Pipeline:
    """Compose an unfitted preprocessor with an estimator into one Pipeline."""
    pre = preprocess.build_preprocessor(numeric, categorical, scaler, encoder)
    return Pipeline([("pre", pre), ("clf", estimator)])


def get_baseline_models() -> dict:
    """Step 3 baselines.

    - dummy_most_frequent: predicts the majority class — the score floor any real
      model must beat (its ROC-AUC is 0.5 by construction).
    - logreg: plain logistic regression — an interpretable, honest linear baseline.
    - logreg_balanced: same but class_weight='balanced' — re-weights the 26.5%
      minority so the model stops ignoring churners. Lets us *see* the trade-off.
    """
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logreg": LogisticRegression(max_iter=1000, random_state=config.RANDOM_SEED),
        "logreg_balanced": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=config.RANDOM_SEED
        ),
    }
