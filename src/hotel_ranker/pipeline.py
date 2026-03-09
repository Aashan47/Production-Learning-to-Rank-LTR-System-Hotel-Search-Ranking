"""End-to-end LTR pipeline: data -> features -> train -> evaluate.

This module ties every component together into a single runnable pipeline.
It can be invoked from the command line or imported and called from notebooks.

Usage
-----
    python -m hotel_ranker.pipeline                  # full pipeline
    python -m hotel_ranker.pipeline --sample 0.1     # use 10% of queries
"""

import argparse
import logging
import time

import numpy as np

from hotel_ranker.config import MODEL_DIR, NDCG_CUTOFFS
from hotel_ranker.data.acquisition import find_csv
from hotel_ranker.data.preprocessing import preprocess
from hotel_ranker.data.schema import SEARCH_ID, POSITION
from hotel_ranker.data.splitting import get_groups, query_level_split
from hotel_ranker.bias.propensity import compute_ips_weights, estimate_propensity
from hotel_ranker.features.feature_pipeline import build_features
from hotel_ranker.training.label_engineering import compute_composite_label, discretize_labels
from hotel_ranker.training.trainer import (
    get_feature_importance,
    save_model,
    train_ranker,
)
from hotel_ranker.evaluation.metrics import evaluate_all
from hotel_ranker.evaluation.error_analysis import (
    compare_rankings,
    find_biggest_improvements,
    plot_feature_importance,
    plot_ndcg_distribution,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_pipeline(
    sample_fraction: float = 1.0,
) -> dict:
    """Execute the full LTR pipeline.

    Parameters
    ----------
    sample_fraction : float
        Fraction of queries to use (for quick smoke tests).

    Returns
    -------
    dict
        Evaluation metrics on the test set.
    """
    t0 = time.time()

    # -- Step 1: Data acquisition & preprocessing ----------------------
    logger.info("=" * 60)
    logger.info("STEP 1: Data Acquisition & Preprocessing")
    logger.info("=" * 60)
    csv_path = find_csv()
    df = preprocess(csv_path)

    if sample_fraction < 1.0:
        unique_q = df[SEARCH_ID].unique()
        n_sample = max(1, int(len(unique_q) * sample_fraction))
        rng = np.random.RandomState(42)
        sampled_q = rng.choice(unique_q, size=n_sample, replace=False)
        df = df[df[SEARCH_ID].isin(sampled_q)].reset_index(drop=True)
        logger.info("Subsampled to %d queries (%d rows)", n_sample, len(df))

    # -- Step 2: Query-level splitting ---------------------------------
    logger.info("=" * 60)
    logger.info("STEP 2: Query-Level Splitting")
    logger.info("=" * 60)
    train_df, val_df, test_df = query_level_split(df)

    train_df = train_df.sort_values(SEARCH_ID).reset_index(drop=True)
    val_df = val_df.sort_values(SEARCH_ID).reset_index(drop=True)
    test_df = test_df.sort_values(SEARCH_ID).reset_index(drop=True)

    groups_train = get_groups(train_df)
    groups_val = get_groups(val_df)
    groups_test = get_groups(test_df)

    # -- Step 3: Feature engineering -----------------------------------
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 60)
    X_train = build_features(train_df, train_df)
    X_val = build_features(val_df, train_df)
    X_test = build_features(test_df, train_df)

    common_cols = sorted(set(X_train.columns) & set(X_val.columns) & set(X_test.columns))
    X_train = X_train[common_cols]
    X_val = X_val[common_cols]
    X_test = X_test[common_cols]

    # -- Step 4: Position bias & IPS weights ---------------------------
    logger.info("=" * 60)
    logger.info("STEP 4: Position Bias Estimation & IPS")
    logger.info("=" * 60)
    propensity_result = estimate_propensity(train_df)
    ips_weights = compute_ips_weights(train_df, propensity_result)

    # -- Step 5: Label engineering -------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 5: Multi-Objective Label Engineering")
    logger.info("=" * 60)
    y_train = discretize_labels(compute_composite_label(train_df), train_df[SEARCH_ID])
    y_val = discretize_labels(compute_composite_label(val_df), val_df[SEARCH_ID])
    y_test = discretize_labels(compute_composite_label(test_df), test_df[SEARCH_ID])

    # -- Step 6: Model training ----------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 6: LGBMRanker Training")
    logger.info("=" * 60)
    model = train_ranker(
        X_train, y_train, groups_train,
        X_val, y_val, groups_val,
        sample_weight=ips_weights,
    )
    save_model(model)

    # -- Step 7: Evaluation --------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 7: Evaluation on Test Set")
    logger.info("=" * 60)
    y_pred_test = model.predict(X_test)
    results = evaluate_all(y_test, y_pred_test, groups_test)

    fi = get_feature_importance(model, list(X_test.columns))
    logger.info("\nTop 10 features:\n%s", fi.head(10).to_string(index=False))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(fi, save_path=str(MODEL_DIR / "feature_importance.png"))
    plot_ndcg_distribution(
        y_test, y_pred_test, groups_test,
        k=5, save_path=str(MODEL_DIR / "ndcg_distribution.png"),
    )

    comparisons = compare_rankings(test_df, y_test, y_pred_test, groups_test)
    logger.info("\n" + "=" * 60)
    logger.info("Before vs After (sample queries):")
    logger.info("=" * 60)
    for i, comp in enumerate(comparisons):
        logger.info("\n--- Query %d ---\n%s", i + 1, comp.to_string(index=False))

    improvements = find_biggest_improvements(test_df, y_test, y_pred_test, groups_test)
    logger.info("\nQueries with biggest ranking improvement:\n%s", improvements.to_string(index=False))

    logger.info("\nPipeline completed in %.1f seconds", time.time() - t0)
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Hotel Ranker LTR Pipeline")
    parser.add_argument(
        "--sample",
        type=float,
        default=1.0,
        help="Fraction of queries to use (e.g., 0.1 for 10%%)",
    )
    args = parser.parse_args()

    results = run_pipeline(sample_fraction=args.sample)
    print("\nFinal Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
