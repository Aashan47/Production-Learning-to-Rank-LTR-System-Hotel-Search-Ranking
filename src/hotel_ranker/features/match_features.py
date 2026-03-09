"""Match features that capture query-item interactions.

These features measure how well a particular hotel "matches" the user's
search context. They require within-query aggregations (e.g. how does this
hotel's price compare to the average price in the same search result list?).

Why match features matter
-------------------------
Raw features like ``price_usd = 150`` are absolute. But relevance is
*relative*: $150 is cheap for New York but expensive for rural Thailand.
Match features make the model context-aware by computing per-query
statistics and measuring each item's deviation.
"""

import numpy as np
import pandas as pd

from hotel_ranker.data.schema import (
    COMPETITOR_RATE_COLS,
    COMPETITOR_INV_COLS,
    PRICE_USD,
    PROP_LOCATION_SCORE1,
    PROP_LOCATION_SCORE2,
    PROP_STARRATING,
    SEARCH_ID,
    VISITOR_HIST_STARRATING,
)


def build_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build query-item match features.

    Features
    --------
    * ``price_diff_from_query_mean`` – How much this hotel's price deviates
      from the mean price in its search result list. Negative = cheaper.
    * ``price_ratio_to_query_median`` – Ratio to query median price.
    * ``star_diff_from_query_mean`` – Star rating deviation within query.
    * ``star_match_visitor_pref`` – Absolute difference between property
      star rating and visitor's historical preference.
    * ``location_score_composite`` – Weighted combination of location scores.
    * ``competitor_rate_advantage`` – How many competitors have a higher rate
      (positive = this hotel is cheaper than competitors).
    * ``competitor_inv_advantage`` – How many competitors are sold out.
    """
    out = pd.DataFrame(index=df.index)

    # --- Price match features ---
    query_price_mean = df.groupby(SEARCH_ID)[PRICE_USD].transform("mean")
    query_price_median = df.groupby(SEARCH_ID)[PRICE_USD].transform("median")
    query_price_std = df.groupby(SEARCH_ID)[PRICE_USD].transform("std").fillna(1)

    out["price_diff_from_query_mean"] = df[PRICE_USD] - query_price_mean
    out["price_ratio_to_query_median"] = df[PRICE_USD] / query_price_median.clip(lower=1)
    out["price_zscore_in_query"] = (df[PRICE_USD] - query_price_mean) / query_price_std

    # --- Star rating match ---
    query_star_mean = df.groupby(SEARCH_ID)[PROP_STARRATING].transform("mean")
    out["star_diff_from_query_mean"] = df[PROP_STARRATING] - query_star_mean

    if VISITOR_HIST_STARRATING in df.columns:
        visitor_pref = df[VISITOR_HIST_STARRATING].replace(-1, np.nan)
        out["star_match_visitor_pref"] = (
            (df[PROP_STARRATING] - visitor_pref).abs().fillna(0)
        )

    # --- Location score composite ---
    loc1 = df.get(PROP_LOCATION_SCORE1, pd.Series(0, index=df.index))
    loc2 = df.get(PROP_LOCATION_SCORE2, pd.Series(0, index=df.index))
    out["location_score_composite"] = 0.6 * loc1 + 0.4 * loc2

    # --- Competitor advantage features ---
    rate_cols = [c for c in COMPETITOR_RATE_COLS if c in df.columns]
    inv_cols = [c for c in COMPETITOR_INV_COLS if c in df.columns]

    if rate_cols:
        # comp_rate > 0 means competitor is more expensive → advantage for us
        rate_matrix = df[rate_cols]
        out["competitor_rate_advantage"] = (rate_matrix > 0).sum(axis=1)
        out["competitor_rate_disadvantage"] = (rate_matrix < 0).sum(axis=1)
        out["competitor_rate_mean"] = rate_matrix.mean(axis=1)

    if inv_cols:
        # comp_inv = 1 means competitor is sold out → advantage
        inv_matrix = df[inv_cols]
        out["competitor_inv_advantage"] = (inv_matrix == 1).sum(axis=1)

    return out
