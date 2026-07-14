"""Central config: paths, column names, and the random seed.

Paths are derived from this file's location, so imports work regardless of the
current working directory (VS Code, a script, or a notebook).
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "Telco-Customer-Churn.csv"
PROCESSED_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

# Dataset-specific column names
TARGET = "Churn"          # Yes / No
ID_COL = "customerID"     # unique id, dropped before modelling

# One seed everywhere → reproducible splits and models
RANDOM_SEED = 42
