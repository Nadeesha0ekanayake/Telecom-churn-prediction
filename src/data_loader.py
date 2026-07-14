"""Load the raw Telco dataset.

Step 1 deliberately loads the data *exactly as stored* (no cleaning), so the
exploration can surface the quirks (e.g. blank `TotalCharges`) honestly.
Cleaning happens in Step 2.
"""
from __future__ import annotations

import pandas as pd

from src import config


def load_raw(path=config.RAW_CSV) -> pd.DataFrame:
    """Read the Telco CSV as-is, with no type coercion or cleaning."""
    return pd.read_csv(path)
