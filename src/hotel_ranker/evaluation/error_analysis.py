"""Error analysis: before/after comparison and diagnostic utilities.

This module provides tools to understand *why* the model ranks items the way
it does and to identify where it succeeds or fails compared to the original
ranking (by position in the dataset).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

from hotel_ranker.data.schema import (
    BOOKING_BOOL,
    CLICK_BOOL,
    POSITION,
    PRICE_USD,
    PROPERTY_ID,
    PROP_STARRATING,
    SEARCH_ID,
)
from hotel_ranker.evaluation.metrics import per_query_ndcg

logger = logging.getLogger(__name__)


def compare_rankings(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_queries: int = 5,
) -> list[pd.DataFrame]:
    """Show side-by-side "Before vs After" for sample queries.

    "Before" = original position order (the old ranker).
    "After"  = model-predicted order.

    Returns a list of DataFrames, one per query, with columns:
    [prop_id, position_original, model_rank, relevance, price, stars, clicked, booked]
    """
    query_ids = df[SEARCH_ID].unique()
    rng = np.random.RandomState(42)
    sample_qids = rng.choice(query_ids, size=min(n_queries, len(query_ids)), replace=False)

    results = []
    for qid in sample_qids:
        mask = df[SEARCH_ID] == qid
        q_df = df[mask].copy()
        q_pred = y_pred[mask.values] if isinstance(mask, pd.Series) else y_pred[mask]
        q_true = y_true[mask.values] if isinstance(mask, pd.Series) else y_true[mask]

        q_df = q_df.assign(
            model_score=q_pred,
            relevance=q_true,
            model_rank=pd.Series(q_pred, index=q_df.index)
            .rank(ascending=False, method="min")
            .astype(int),
        )

        display_cols = [
            PROPERTY_ID,
            POSITION,
            "model_rank",
            "relevance",
            "model_score",
        ]
        for col in [PRICE_USD, PROP_STARRATING, CLICK_BOOL, BOOKING_BOOL]:
            if col in q_df.columns:
                display_cols.append(col)

        comparison = q_df[display_cols].sort_values("model_rank")
        results.append(comparison)

    return results


def find_biggest_improvements(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
    top_n: int = 10,
) -> pd.DataFrame:
    """Find queries where the model improves most over the original ranking.

    Compares NDCG@k of model ranking vs. original position ranking.
    """
    query_ids = df[SEARCH_ID].unique()

    # Model NDCG
    model_ndcg = per_query_ndcg(y_true, y_pred, groups, k=k)

    # Original (position-based) NDCG
    # Lower position = higher original rank, so we invert
    original_pred = -df[POSITION].values.astype(float)
    original_ndcg = per_query_ndcg(y_true, original_pred, groups, k=k)

    improvements = pd.DataFrame({
        "query_id": query_ids[: len(model_ndcg)],
        "model_ndcg": model_ndcg,
        "original_ndcg": original_ndcg,
        "improvement": model_ndcg - original_ndcg,
    }).dropna()

    return improvements.nlargest(top_n, "improvement")


def plot_feature_importance(
    feature_importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None,
) -> None:
    """Plot horizontal bar chart of top feature importances."""
    top = feature_importance_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=top, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance (split count)")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Feature importance plot saved to %s", save_path)
    plt.close(fig)


def plot_ndcg_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    k: int = 5,
    save_path: Optional[str] = None,
) -> None:
    """Plot the distribution of per-query NDCG@k scores."""
    query_scores = per_query_ndcg(y_true, y_pred, groups, k=k)
    query_scores = query_scores[~np.isnan(query_scores)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(query_scores, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax.axvline(np.mean(query_scores), color="red", linestyle="--", label=f"Mean: {np.mean(query_scores):.3f}")
    ax.set_title(f"Distribution of Per-Query NDCG@{k}")
    ax.set_xlabel(f"NDCG@{k}")
    ax.set_ylabel("Number of Queries")
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("NDCG distribution plot saved to %s", save_path)
    plt.close(fig)
