"""Position bias estimation and Inverse Propensity Scoring (IPS).

The position bias problem
-------------------------
In search/recommendation systems, users are more likely to click items shown
at higher positions simply because they *see* them first—not because those
items are more relevant. If we naively train on click data, the model learns
to mimic the old ranker's position ordering instead of true relevance.

Formally, the observed click probability decomposes as:

    P(click | item, position) = P(relevant | item) * P(examine | position)

The examination probability P(examine | position) is the **propensity**—it
depends only on where the item was displayed, not on what it is.

Inverse Propensity Scoring (IPS)
--------------------------------
IPS re-weights training samples to undo position bias:

    weight_i = 1 / P(examine | position_i)    for clicked items
    weight_i = 1                                for unclicked items

An item clicked at position 10 is "worth more" than one clicked at position 1
because the user had to scroll further to find it, suggesting stronger
relevance intent.

Propensity estimation via randomisation
---------------------------------------
The Expedia dataset includes a ``random_bool`` column: when True, the item
was placed at a random position. For these items, position is independent
of relevance, so the observed click rate at each position is a clean
estimate of the examination propensity:

    P(examine | pos) ≈ CTR(pos | random_bool=True)

We fit a **power-law** model: P(pos) = alpha * pos^(-beta), which is the
standard parametric form for examination probabilities (Joachims et al., 2017).
"""

import logging

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from hotel_ranker.config import IPS_CLIP_PERCENTILE
from hotel_ranker.data.schema import CLICK_BOOL, POSITION, RANDOM_BOOL

logger = logging.getLogger(__name__)


def _power_law(position: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Power-law examination model: P(examine | pos) = alpha * pos^(-beta)."""
    return alpha * np.power(position.astype(float), -beta)


def estimate_propensity(df: pd.DataFrame) -> dict:
    """Estimate position propensity from randomly-placed items.

    Returns
    -------
    dict with keys:
        - "empirical" : pd.Series – raw CTR by position (from random items)
        - "alpha", "beta" : float – fitted power-law parameters
        - "propensity_func" : callable – position → propensity
    """
    # Filter to randomly placed items
    random_mask = df[RANDOM_BOOL] == 1
    random_df = df[random_mask]

    if len(random_df) == 0:
        logger.warning("No random items found; falling back to uniform propensity")
        return {
            "empirical": pd.Series(dtype=float),
            "alpha": 1.0,
            "beta": 0.0,
            "propensity_func": lambda pos: np.ones_like(pos, dtype=float),
        }

    # Empirical CTR by position
    empirical = random_df.groupby(POSITION)[CLICK_BOOL].mean()
    logger.info(
        "Empirical propensity: pos=1 CTR=%.3f, pos=10 CTR=%.3f",
        empirical.get(1, float("nan")),
        empirical.get(10, float("nan")),
    )

    # Fit power-law
    positions = empirical.index.values.astype(float)
    ctrs = empirical.values.astype(float)

    try:
        (alpha, beta), _ = curve_fit(
            _power_law, positions, ctrs, p0=[0.5, 1.0], maxfev=5000
        )
    except RuntimeError:
        logger.warning("Power-law fit failed; using empirical values directly")
        alpha, beta = ctrs[0] if len(ctrs) > 0 else 0.5, 1.0

    logger.info("Fitted propensity: alpha=%.4f, beta=%.4f", alpha, beta)

    def propensity_func(pos):
        return _power_law(np.asarray(pos, dtype=float).clip(min=1), alpha, beta)

    return {
        "empirical": empirical,
        "alpha": alpha,
        "beta": beta,
        "propensity_func": propensity_func,
    }


def compute_ips_weights(
    df: pd.DataFrame,
    propensity_result: dict,
    clip_percentile: int = IPS_CLIP_PERCENTILE,
) -> np.ndarray:
    """Compute IPS sample weights for each row.

    Clicked items get weight = 1/propensity(position).
    Unclicked items get weight = 1 (they are not affected by examination bias
    in the same way—they might have been examined but deemed irrelevant).

    Weight clipping
    ---------------
    IPS weights can have high variance when propensity is very small (e.g.,
    position 40). We clip at the ``clip_percentile``-th percentile to
    stabilise training. This is a standard variance-reduction technique
    (clipped IPS / capped IPS).
    """
    propensity_func = propensity_result["propensity_func"]
    positions = df[POSITION].values
    clicks = df[CLICK_BOOL].values

    propensities = propensity_func(positions)
    propensities = np.clip(propensities, 1e-6, None)  # avoid division by zero

    weights = np.where(clicks == 1, 1.0 / propensities, 1.0)

    # Clip extreme weights — compute cap only on clicked items' weights,
    # otherwise the ~95% of unclicked items (all weight=1.0) drag the
    # percentile down to 1.0 and cancel out the IPS correction entirely.
    clicked_weights = weights[clicks == 1]
    cap = np.percentile(clicked_weights, clip_percentile) if len(clicked_weights) > 0 else weights.max()
    weights = np.clip(weights, None, cap)

    logger.info(
        "IPS weights: min=%.2f, median=%.2f, max=%.2f (capped at %.2f)",
        weights.min(), np.median(weights), weights.max(), cap,
    )
    return weights.astype(np.float32)
