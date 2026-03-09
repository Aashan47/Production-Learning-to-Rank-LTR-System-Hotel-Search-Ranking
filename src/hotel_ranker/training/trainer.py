"""LightGBM LGBMRanker training with IPS-weighted samples.

Why LambdaMART / LGBMRanker?
-----------------------------
LambdaMART is the industry-standard algorithm for Learning-to-Rank. It
combines:
1. **Lambda gradients** – derived from NDCG, so each gradient step directly
   optimises the ranking metric (not a surrogate like MSE).
2. **MART (Multiple Additive Regression Trees)** – gradient-boosted decision
   trees that handle non-linear feature interactions naturally.

LightGBM's ``lambdarank`` objective implements LambdaMART with histogram-
based splitting for fast training on large datasets.

Group parameter
---------------
LGBMRanker.fit() requires a ``group`` array where ``group[i]`` = number of
items in query *i*. The model uses this to compute pairwise lambda gradients
*within* each query (items from different queries never form pairs).

IPS sample weights
------------------
We pass position-debiased IPS weights via the ``sample_weight`` parameter.
LightGBM multiplies each sample's gradient contribution by its weight,
effectively up-weighting items clicked at low positions (strong relevance
signal) and down-weighting items clicked at high positions (possibly due
to position bias rather than true relevance).
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from hotel_ranker.config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS, MODEL_DIR
from hotel_ranker.data.schema import SEARCH_ID

logger = logging.getLogger(__name__)


def train_ranker(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    groups_val: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
    params: Optional[dict] = None,
) -> lgb.LGBMRanker:
    """Train an LGBMRanker with early stopping on validation NDCG.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices for train and validation splits.
    y_train, y_val : np.ndarray
        Discrete relevance grades (integers).
    groups_train, groups_val : np.ndarray
        Group sizes (number of items per query) for each split.
    sample_weight : np.ndarray, optional
        IPS weights for the training set. If None, uniform weights.
    params : dict, optional
        Override default LGBMRanker hyperparameters.

    Returns
    -------
    lgb.LGBMRanker
        Trained model.
    """
    model_params = {**LGBM_PARAMS, **(params or {})}
    logger.info("Training LGBMRanker with params: %s", model_params)

    model = lgb.LGBMRanker(**model_params)

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(period=50),
    ]

    model.fit(
        X_train,
        y_train,
        group=groups_train,
        eval_set=[(X_val, y_val)],
        eval_group=[groups_val],
        eval_metric="ndcg",
        eval_at=[5, 10],
        sample_weight=sample_weight,
        callbacks=callbacks,
    )

    logger.info("Training complete. Best iteration: %d", model.best_iteration_)
    return model


def get_feature_importance(model: lgb.LGBMRanker, feature_names: list) -> pd.DataFrame:
    """Extract feature importance as a sorted DataFrame."""
    importance = model.feature_importances_
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importance,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return fi


def save_model(model: lgb.LGBMRanker, name: str = "lgbm_ranker") -> Path:
    """Save the trained model to disk."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)
    return path


def load_model(name: str = "lgbm_ranker") -> lgb.LGBMRanker:
    """Load a saved model from disk."""
    path = MODEL_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model
