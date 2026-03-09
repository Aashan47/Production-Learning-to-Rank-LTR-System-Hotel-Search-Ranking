"""Multi-objective label engineering.

Standard ranking datasets have binary labels (clicked / not clicked). This is
limiting because:
1. A click on a $50 hotel and a booking on a $500 hotel are not equally
   valuable to the business.
2. Bookings are a much stronger relevance signal than clicks.

Multi-objective composite label
-------------------------------
We create a composite relevance score using within-query price percentile
to differentiate booking value:

    booking_value = w2 * (0.5 + 0.5 * price_pct_within_query)
    y = w1 * click_bool + booking_value * booking_bool

* ``w1`` (W_CLICK=3.0) – base reward for a click.
* ``w2`` (W_BOOK=5.0) – booking multiplier.
* ``price_pct`` – within-query price percentile (0→1), scales booking value
  from w2*0.5 (cheapest) to w2*1.0 (most expensive in query).

Discretization into relevance grades
-------------------------------------
LambdaRank / NDCG work with ordinal relevance grades (e.g., 0–4). We
normalise the composite score within each query to [0, 1] and then map
to discrete grades via uniform binning. With default weights this produces:

- Grade 0 = no interaction (score = 0).
- Grade 1 = click only (score = 3.0, normalised ≈ 0.375).
- Grade 2 = booking, cheap hotel  (score ≈ 5.5, normalised ≈ 0.69).
- Grade 3 = booking, mid-price hotel (score ≈ 6.5–7.0, normalised ≈ 0.81–0.88).
- Grade 4 = booking, most expensive hotel in query (score = 8.0, normalised = 1.0).
"""

import logging

import numpy as np
import pandas as pd

from hotel_ranker.config import MAX_RELEVANCE_GRADE, W_BOOK, W_CLICK
from hotel_ranker.data.schema import BOOKING_BOOL, CLICK_BOOL, PRICE_USD, SEARCH_ID

logger = logging.getLogger(__name__)


def compute_composite_label(
    df: pd.DataFrame,
    w_click: float = W_CLICK,
    w_book: float = W_BOOK,
) -> pd.Series:
    """Compute the raw composite relevance score.

    Returns
    -------
    pd.Series
        Non-negative float scores, same index as ``df``.
    """
    click = df[CLICK_BOOL].astype(float)
    book = df[BOOKING_BOOL].astype(float)

    # Within-query price percentile (0→1) differentiates booking value.
    # Falls back to global rank when srch_id is absent (e.g. unit tests).
    if SEARCH_ID in df.columns:
        price_pct = df.groupby(SEARCH_ID)[PRICE_USD].rank(pct=True)
    else:
        price_pct = df[PRICE_USD].rank(pct=True)

    # Booking score scales from w_book*0.5 (cheapest) to w_book*1.0 (priciest).
    # After within-query normalisation this maps click-only → grade 1 and
    # bookings → grades 2, 3, or 4 depending on relative price, giving
    # LambdaMART richer pairwise gradient signal than binary 0/4 labels.
    booking_value = w_book * (0.5 + 0.5 * price_pct)
    score = w_click * click + booking_value * book
    return score


def discretize_labels(
    scores: pd.Series,
    query_ids: pd.Series,
    max_grade: int = MAX_RELEVANCE_GRADE,
) -> np.ndarray:
    """Normalise composite scores within each query and discretize to grades.

    Steps
    -----
    1. Within each query, min-max normalise scores to [0, 1].
    2. Map to integer grades in [0, max_grade] via floor(normalised * max_grade).
    3. Items with score == 0 always get grade 0.

    Parameters
    ----------
    scores : pd.Series
        Raw composite scores from ``compute_composite_label``.
    query_ids : pd.Series
        Corresponding search IDs for grouping.
    max_grade : int
        Maximum relevance grade (inclusive).

    Returns
    -------
    np.ndarray of int
        Discrete relevance grades.
    """
    temp = pd.DataFrame({"score": scores.values, "qid": query_ids.values})

    # Per-query min-max normalisation
    q_min = temp.groupby("qid")["score"].transform("min")
    q_max = temp.groupby("qid")["score"].transform("max")
    q_range = (q_max - q_min).replace(0, 1)  # avoid div-by-zero for constant queries

    normalised = (temp["score"] - q_min) / q_range

    # Discretize
    grades = np.floor(normalised * max_grade).astype(int)
    grades = np.clip(grades, 0, max_grade)

    # Ensure zero-score items stay at grade 0
    grades[temp["score"] == 0] = 0

    logger.info(
        "Label distribution: %s",
        pd.Series(grades).value_counts().sort_index().to_dict(),
    )
    return np.asarray(grades, dtype=int)
