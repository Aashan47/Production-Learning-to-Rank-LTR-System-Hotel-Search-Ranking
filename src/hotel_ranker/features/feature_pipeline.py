"""Orchestrates all feature generators into a single feature matrix.

This module is the main entry point for feature engineering. It chains
raw features, match features, and historical features, then concatenates
them into a single DataFrame ready for model training.

Feature leakage prevention
--------------------------
Historical features (CTR, booking rates) are computed *only from the
training set* and then joined onto train/val/test by property/destination ID.
This ensures the model never sees future information during evaluation.
"""

import logging

import pandas as pd

from hotel_ranker.features.raw_features import build_raw_features
from hotel_ranker.features.match_features import build_match_features
from hotel_ranker.features.historical_features import (
    build_destination_history,
    build_property_history,
    merge_historical_features,
)

logger = logging.getLogger(__name__)


def build_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the full feature matrix for a given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The split (train, val, or test) to generate features for.
    train_df : pd.DataFrame
        The training split, used to compute historical aggregates.
        If ``df`` IS the training split, pass the same DataFrame.

    Returns
    -------
    pd.DataFrame
        Combined feature matrix with all feature groups concatenated.
    """
    logger.info("Building features for %d rows ...", len(df))

    # 1. Raw features (per-row transforms)
    raw = build_raw_features(df)
    logger.info("  Raw features: %d columns", raw.shape[1])

    # 2. Match features (within-query interactions)
    match = build_match_features(df)
    logger.info("  Match features: %d columns", match.shape[1])

    # 3. Historical features (from training data only)
    prop_hist = build_property_history(train_df)
    dest_hist = build_destination_history(train_df)
    hist = merge_historical_features(df, prop_hist, dest_hist)
    logger.info("  Historical features: %d columns", hist.shape[1])

    features = pd.concat([raw, match, hist], axis=1)

    # Final sanity check: no NaN should remain
    nan_count = features.isna().sum().sum()
    if nan_count > 0:
        logger.warning("Feature matrix has %d NaN values — filling with 0", nan_count)
        features = features.fillna(0)

    logger.info("Total features: %d columns", features.shape[1])
    return features
