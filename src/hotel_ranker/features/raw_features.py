"""Select and transform raw (direct) features from the dataset.

These are columns that can be used as-is or with minimal transformation.
No cross-row aggregation is performed here.
"""

import numpy as np
import pandas as pd

from hotel_ranker.data.schema import (
    PRICE_USD,
    PROMOTION_FLAG,
    PROP_BRAND_BOOL,
    PROP_LOCATION_SCORE1,
    PROP_LOCATION_SCORE2,
    PROP_LOG_HISTORICAL_PRICE,
    PROP_REVIEW_SCORE,
    PROP_STARRATING,
    SRCH_ADULTS_COUNT,
    SRCH_BOOKING_WINDOW,
    SRCH_CHILDREN_COUNT,
    SRCH_LENGTH_OF_STAY,
    SRCH_ROOM_COUNT,
    SRCH_SATURDAY_NIGHT_BOOL,
    VISITOR_HIST_ADR_USD,
    VISITOR_HIST_STARRATING,
)

# Columns to pass through directly
PASSTHROUGH_COLS = [
    PROP_STARRATING,
    PROP_REVIEW_SCORE,
    PROP_BRAND_BOOL,
    PROP_LOCATION_SCORE1,
    PROP_LOCATION_SCORE2,
    PROP_LOG_HISTORICAL_PRICE,
    PRICE_USD,
    PROMOTION_FLAG,
    SRCH_LENGTH_OF_STAY,
    SRCH_BOOKING_WINDOW,
    SRCH_ADULTS_COUNT,
    SRCH_CHILDREN_COUNT,
    SRCH_ROOM_COUNT,
    SRCH_SATURDAY_NIGHT_BOOL,
    VISITOR_HIST_STARRATING,
    VISITOR_HIST_ADR_USD,
]


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of raw features (passthrough + simple transforms).

    New columns
    -----------
    * ``price_log`` – log1p(price_usd), already added in preprocessing but
      included here for completeness.
    * ``price_per_night`` – price divided by length of stay.
    * ``total_guests`` – adults + children.
    * ``star_review_ratio`` – star rating / review score (quality consistency).
    """
    available = [c for c in PASSTHROUGH_COLS if c in df.columns]
    out = df[available].copy()

    # Price per night (avoids division by zero)
    los = df[SRCH_LENGTH_OF_STAY].clip(lower=1)
    out["price_per_night"] = df[PRICE_USD] / los

    # Total guest count
    out["total_guests"] = df[SRCH_ADULTS_COUNT] + df[SRCH_CHILDREN_COUNT]

    # Star / review consistency (0 review → NaN → fill 0)
    review = df[PROP_REVIEW_SCORE].replace(0, np.nan)
    out["star_review_ratio"] = (df[PROP_STARRATING] / review).fillna(0)

    # Price relative to visitor's historical ADR
    if VISITOR_HIST_ADR_USD in df.columns:
        hist_adr = df[VISITOR_HIST_ADR_USD].replace(-1, np.nan)
        out["price_vs_visitor_hist"] = (df[PRICE_USD] / hist_adr.clip(lower=1)).fillna(0)

    # Already-derived columns from preprocessing
    if "price_log" in df.columns:
        out["price_log"] = df["price_log"]
    if "price_rank_in_query" in df.columns:
        out["price_rank_in_query"] = df["price_rank_in_query"]

    return out
