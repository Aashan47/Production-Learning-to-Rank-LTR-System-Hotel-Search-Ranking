"""Ranking evaluation metrics: NDCG@k and MRR.

NDCG (Normalized Discounted Cumulative Gain)
--------------------------------------------
NDCG measures ranking quality by comparing the model's ordering to the ideal
ordering. It accounts for:
- **Relevance grade** – higher-graded items contribute more.
- **Position discount** – items ranked lower contribute less (log2 discount).

    DCG@k  = sum_{i=1}^{k}  (2^{rel_i} - 1) / log2(i + 1)
    NDCG@k = DCG@k / IDCG@k

where IDCG@k is the DCG of the ideal (perfectly sorted) ranking.

NDCG ranges from 0 to 1, where 1 means perfect ranking.

MRR (Mean Reciprocal Rank)
--------------------------
MRR measures how quickly the *first* relevant item appears:

    RR   = 1 / rank_of_first_relevant_item
    MRR  = mean(RR) over all queries

MRR is useful when only the top result matters (e.g., "I'm Feeling Lucky").
"""

import logging
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from hotel_ranker.config import NDCG_CUTOFFS
from hotel_ranker.data.schema import SEARCH_ID

logger = logging.getLogger(__name__)


def ndcg_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    k: int,
) -> float:
    """Compute mean NDCG@k across all queries.

    Parameters
    ----------
    y_true : np.ndarray
        True relevance grades (integers), concatenated across queries.
    y_pred : np.ndarray
        Model predicted scores, same shape as y_true.
    groups : np.ndarray
        Group sizes (items per query).
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Mean NDCG@k over all queries.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    offset = 0
    for size in groups:
        if size < 2:
            offset += size
            continue
        true_slice = y_true[offset : offset + size]
        pred_slice = y_pred[offset : offset + size]

        # sklearn expects 2D arrays: (1, n_items)
        score = ndcg_score(
            true_slice.reshape(1, -1),
            pred_slice.reshape(1, -1),
            k=k,
        )
        scores.append(score)
        offset += size

    return float(np.mean(scores)) if scores else 0.0


def mean_reciprocal_rank(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    relevance_threshold: int = 1,
) -> float:
    """Compute Mean Reciprocal Rank (MRR).

    Parameters
    ----------
    y_true : np.ndarray
        True relevance grades.
    y_pred : np.ndarray
        Predicted scores.
    groups : np.ndarray
        Group sizes.
    relevance_threshold : int
        Minimum grade to be considered "relevant".

    Returns
    -------
    float
        MRR score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    reciprocal_ranks = []
    offset = 0
    for size in groups:
        true_slice = y_true[offset : offset + size]
        pred_slice = y_pred[offset : offset + size]

        # Sort by predicted score (descending)
        order = np.argsort(-pred_slice)
        sorted_true = true_slice[order]

        # Find first relevant item
        relevant_positions = np.where(sorted_true >= relevance_threshold)[0]
        if len(relevant_positions) > 0:
            rr = 1.0 / (relevant_positions[0] + 1)
        else:
            rr = 0.0
        reciprocal_ranks.append(rr)
        offset += size

    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0


def per_query_ndcg(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    """Return NDCG@k for each individual query (for distribution analysis)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    scores = []
    offset = 0
    for size in groups:
        if size < 2:
            scores.append(float("nan"))
            offset += size
            continue
        true_slice = y_true[offset : offset + size]
        pred_slice = y_pred[offset : offset + size]
        score = ndcg_score(
            true_slice.reshape(1, -1),
            pred_slice.reshape(1, -1),
            k=k,
        )
        scores.append(score)
        offset += size
    return np.array(scores)


def evaluate_all(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    cutoffs: List[int] = NDCG_CUTOFFS,
) -> Dict[str, float]:
    """Run all evaluation metrics and return a summary dict."""
    results = {}
    for k in cutoffs:
        results[f"NDCG@{k}"] = ndcg_at_k(y_true, y_pred, groups, k)
    results["MRR"] = mean_reciprocal_rank(y_true, y_pred, groups)

    logger.info("Evaluation results:")
    for metric, value in results.items():
        logger.info("  %s = %.4f", metric, value)

    return results
