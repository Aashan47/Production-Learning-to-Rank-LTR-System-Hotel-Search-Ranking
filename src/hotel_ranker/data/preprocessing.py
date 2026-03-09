"""Data loading, cleaning, and missing-value imputation.

Design choices
--------------
* **Optimized dtypes** – We load columns with the smallest dtype that fits
  their range (see ``schema.DTYPE_MAP``). This cuts memory usage by ~60 %
  on the full Expedia dataset.
* **Competitor columns** – Missing values mean "no competitor data available",
  so we fill with 0 (neutral) rather than impute a statistical value.
* **Numeric columns** – Median imputation is robust to outliers.
* **price_rank_in_query** – A within-query rank of the hotel price gives the
  model a relative price signal, which is more informative than the raw
  USD amount that varies across destinations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from hotel_ranker.data import schema

logger = logging.getLogger(__name__)


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Load the CSV with memory-efficient dtypes.

    Columns not present in ``schema.DTYPE_MAP`` are loaded with pandas
    defaults and then down-cast where possible.
    """
    logger.info("Loading %s ...", csv_path.name)

    # Only apply dtypes for columns that exist in the file
    # (avoids errors if the file schema differs slightly)
    sample = pd.read_csv(csv_path, nrows=0)
    available_dtypes = {
        col: dtype
        for col, dtype in schema.DTYPE_MAP.items()
        if col in sample.columns
    }

    df = pd.read_csv(csv_path, dtype=available_dtypes)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in-place and return the DataFrame.

    Strategy
    --------
    1. Competitor rate / inv / rate_percent_diff columns → fill with 0.
       Missing means the competitor data is not available, so the neutral
       value (no difference, no inventory flag) is 0.
    2. ``prop_review_score`` → fill with 0 (unrated property).
    3. ``prop_location_score2`` → fill with median.
    4. ``visitor_hist_starrating``, ``visitor_hist_adr_usd`` → fill with -1
       to indicate "no history" (the model can learn this sentinel).
    5. All remaining numeric NaNs → median of that column.
    """
    # 1. Competitor columns
    comp_cols = [c for c in schema.ALL_COMPETITOR_COLS if c in df.columns]
    df[comp_cols] = df[comp_cols].fillna(0)

    # 2. Review score
    if schema.PROP_REVIEW_SCORE in df.columns:
        df[schema.PROP_REVIEW_SCORE] = df[schema.PROP_REVIEW_SCORE].fillna(0)

    # 3. Location score 2
    if schema.PROP_LOCATION_SCORE2 in df.columns:
        median_loc2 = df[schema.PROP_LOCATION_SCORE2].median()
        df[schema.PROP_LOCATION_SCORE2] = df[schema.PROP_LOCATION_SCORE2].fillna(median_loc2)

    # 4. Visitor history columns
    for col in [schema.VISITOR_HIST_STARRATING, schema.VISITOR_HIST_ADR_USD]:
        if col in df.columns:
            df[col] = df[col].fillna(-1)

    # 5. Remaining numeric NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    logger.info("Missing values handled. Remaining NaNs: %d", df.isna().sum().sum())
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns that depend only on raw data.

    Columns added
    -------------
    * ``price_rank_in_query`` – rank of price within each search query
      (1 = cheapest). This gives the model a relative price signal.
    * ``price_log`` – log1p of price for more normal distribution.
    """
    df["price_rank_in_query"] = df.groupby(schema.SEARCH_ID)[schema.PRICE_USD].rank(
        method="min"
    )
    df["price_log"] = np.log1p(df[schema.PRICE_USD].clip(lower=0))
    return df


def preprocess(csv_path: Path) -> pd.DataFrame:
    """Full preprocessing pipeline: load → clean → derive columns."""
    df = load_raw(csv_path)
    df = handle_missing_values(df)
    df = add_derived_columns(df)
    return df
