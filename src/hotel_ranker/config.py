"""Central configuration for the Hotel Ranker LTR system.

All hyperparameters, file paths, and constants are defined here to ensure
reproducibility and easy experimentation.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Random seed (for reproducibility across all modules)
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Data splitting ratios (query-level split)
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ---------------------------------------------------------------------------
# Multi-objective label weights
#   y = W_CLICK * click_bool + W_BOOK * booking_bool * log1p(price_usd)
#
# W_BOOK is larger because bookings are a stronger signal of relevance and
# directly tied to revenue. log1p(price) scales the booking signal so that
# higher-revenue bookings receive proportionally higher relevance, without
# letting raw price dominate.
# ---------------------------------------------------------------------------
W_CLICK = 3.0   # click-only → grade 1; click+book → grade 4
W_BOOK = 5.0

# Relevance grades after discretization (for LambdaRank / NDCG)
MAX_RELEVANCE_GRADE = 4

# ---------------------------------------------------------------------------
# LightGBM LGBMRanker hyperparameters
# ---------------------------------------------------------------------------
LGBM_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [5, 10],
    "n_estimators": 1000,      # more trees (early stopping will find optimum)
    "num_leaves": 127,         # deeper trees capture more interactions (was 63)
    "learning_rate": 0.02,     # slower learning finds better optima (was 0.05)
    "min_child_samples": 20,   # allow splits on smaller groups (was 50)
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

EARLY_STOPPING_ROUNDS = 150   # more patience for slower learning rate (was 100)

# ---------------------------------------------------------------------------
# IPS (Inverse Propensity Scoring) configuration
# ---------------------------------------------------------------------------
IPS_CLIP_PERCENTILE = 75  # clip extreme IPS weights at this percentile

# ---------------------------------------------------------------------------
# Historical feature smoothing (Bayesian averaging)
#   smoothed_rate = (count * raw_rate + PRIOR_COUNT * PRIOR_RATE)
#                   / (count + PRIOR_COUNT)
# ---------------------------------------------------------------------------
PRIOR_COUNT = 30
PRIOR_RATE = 0.05  # global baseline CTR assumption

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
NDCG_CUTOFFS = [5, 10]
