"""Historical aggregate features (CTR, booking rate, etc.).

These features capture the *past performance* of a property or destination.
They are the strongest predictive features in most ranking systems because
they encode real user preferences.

Bayesian smoothing
------------------
Naively computing ``click_rate = clicks / impressions`` is unreliable for
properties with few impressions (e.g., a hotel shown 3 times and clicked 2
times has a 67% CTR, which is noise, not signal).

We use **Bayesian averaging** (also called "shrinkage" or "smoothed
estimates"):

    smoothed_rate = (n * raw_rate + C * prior_rate) / (n + C)

where ``C`` (``PRIOR_COUNT``) is a pseudo-count and ``prior_rate`` is the
global average. Properties with many impressions converge to their true rate;
properties with few impressions shrink toward the global mean. This is
equivalent to a Beta-Binomial posterior mean.
"""

import logging

import numpy as np
import pandas as pd

from hotel_ranker.config import PRIOR_COUNT, PRIOR_RATE
from hotel_ranker.data.schema import (
    BOOKING_BOOL,
    CLICK_BOOL,
    POSITION,
    PRICE_USD,
    PROPERTY_ID,
    PROP_STARRATING,
    SEARCH_ID,
    SRCH_DESTINATION_ID,
)

logger = logging.getLogger(__name__)


def _bayesian_smooth(
    count: pd.Series,
    raw_rate: pd.Series,
    prior_count: int = PRIOR_COUNT,
    prior_rate: float = PRIOR_RATE,
) -> pd.Series:
    """Apply Bayesian smoothing to a rate estimate."""
    return (count * raw_rate + prior_count * prior_rate) / (count + prior_count)


def build_property_history(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-property historical aggregates from training data only.

    Features
    --------
    * ``prop_click_rate`` – Smoothed click-through rate.
    * ``prop_booking_rate`` – Smoothed booking rate.
    * ``prop_avg_position`` – Average position the property was shown at.
    * ``prop_impression_count`` – Total impressions (log-scaled).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split only (to prevent leakage).

    Returns
    -------
    pd.DataFrame
        Indexed by ``prop_id`` with one row per property.
    """
    agg = train_df.groupby(PROPERTY_ID).agg(
        impressions=(CLICK_BOOL, "count"),
        total_clicks=(CLICK_BOOL, "sum"),
        total_bookings=(BOOKING_BOOL, "sum"),
        avg_position=(POSITION, "mean"),
    )

    agg["prop_click_rate"] = _bayesian_smooth(
        agg["impressions"], agg["total_clicks"] / agg["impressions"].clip(lower=1)
    )
    agg["prop_booking_rate"] = _bayesian_smooth(
        agg["impressions"], agg["total_bookings"] / agg["impressions"].clip(lower=1)
    )
    agg["prop_avg_position"] = agg["avg_position"]
    agg["prop_impression_count"] = np.log1p(agg["impressions"])

    result = agg[
        ["prop_click_rate", "prop_booking_rate", "prop_avg_position", "prop_impression_count"]
    ]
    logger.info("Built property history for %d properties", len(result))
    return result


def build_destination_history(train_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-destination aggregates from training data.

    Features
    --------
    * ``dest_avg_price`` – Average price at this destination.
    * ``dest_avg_star`` – Average star rating at this destination.
    * ``dest_booking_rate`` – Smoothed booking rate for the destination.
    """
    agg = train_df.groupby(SRCH_DESTINATION_ID).agg(
        dest_avg_price=(PRICE_USD, "mean"),
        dest_avg_star=(PROP_STARRATING, "mean"),
        count=(BOOKING_BOOL, "count"),
        total_bookings=(BOOKING_BOOL, "sum"),
    )
    agg["dest_booking_rate"] = _bayesian_smooth(
        agg["count"], agg["total_bookings"] / agg["count"].clip(lower=1)
    )
    result = agg[["dest_avg_price", "dest_avg_star", "dest_booking_rate"]]
    logger.info("Built destination history for %d destinations", len(result))
    return result


def merge_historical_features(
    df: pd.DataFrame,
    prop_history: pd.DataFrame,
    dest_history: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join historical features onto the main DataFrame.

    Properties / destinations not found in the history tables get filled
    with the global prior (cold-start fallback).
    """
    out = pd.DataFrame(index=df.index)

    # Property history
    prop_feats = df[[PROPERTY_ID]].merge(
        prop_history, left_on=PROPERTY_ID, right_index=True, how="left"
    )
    for col in prop_history.columns:
        default = PRIOR_RATE if "rate" in col else 0
        out[col] = prop_feats[col].fillna(default).values

    # Destination history
    dest_feats = df[[SRCH_DESTINATION_ID]].merge(
        dest_history, left_on=SRCH_DESTINATION_ID, right_index=True, how="left"
    )
    for col in dest_history.columns:
        out[col] = dest_feats[col].fillna(dest_feats[col].median()).values

    # Derived: price relative to destination average
    out["price_vs_dest_avg"] = df[PRICE_USD] / out["dest_avg_price"].clip(lower=1)

    return out
