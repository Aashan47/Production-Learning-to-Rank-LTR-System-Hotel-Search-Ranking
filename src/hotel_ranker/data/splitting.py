"""Query-level train / validation / test splitting.

Why query-level?
----------------
In Learning-to-Rank, the model sees a *list* of items per query and learns
relative preferences within that list. If we split row-by-row, items from the
same query could appear in both train and test sets, causing data leakage—the
model would "remember" the query context and give artificially high metrics.

By splitting on unique ``srch_id`` values, every item belonging to a query
stays together in exactly one partition.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

from hotel_ranker.config import RANDOM_SEED, TEST_RATIO, TRAIN_RATIO, VAL_RATIO
from hotel_ranker.data.schema import SEARCH_ID

logger = logging.getLogger(__name__)


def _compute_group_sizes(df: pd.DataFrame) -> np.ndarray:
    """Return an array of group sizes (number of items per query), ordered
    to match the DataFrame row order.

    LGBMRanker expects a ``group`` array where ``group[i]`` is the number
    of items in the i-th query.
    """
    return df.groupby(SEARCH_ID).size().loc[df[SEARCH_ID].unique()].values


def query_level_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame by unique search IDs into train / val / test.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset, must contain the ``SEARCH_ID`` column.
    train_ratio, val_ratio, test_ratio : float
        Proportions that must sum to 1.0 (±tolerance).

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        Disjoint subsets of ``df``.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    unique_queries = df[SEARCH_ID].unique()
    rng = np.random.RandomState(RANDOM_SEED)
    rng.shuffle(unique_queries)

    n = len(unique_queries)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_ids = set(unique_queries[:train_end])
    val_ids = set(unique_queries[train_end:val_end])
    test_ids = set(unique_queries[val_end:])

    train_df = df[df[SEARCH_ID].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[SEARCH_ID].isin(val_ids)].reset_index(drop=True)
    test_df = df[df[SEARCH_ID].isin(test_ids)].reset_index(drop=True)

    logger.info(
        "Split: train=%d queries (%d rows), val=%d queries (%d rows), "
        "test=%d queries (%d rows)",
        len(train_ids), len(train_df),
        len(val_ids), len(val_df),
        len(test_ids), len(test_df),
    )
    return train_df, val_df, test_df


def get_groups(df: pd.DataFrame) -> np.ndarray:
    """Compute the ``group`` array required by LGBMRanker.

    The DataFrame must be sorted by ``SEARCH_ID`` so that rows belonging
    to the same query are contiguous.
    """
    df_sorted = df.sort_values(SEARCH_ID).reset_index(drop=True)
    groups = df_sorted.groupby(SEARCH_ID).size().values
    return groups
