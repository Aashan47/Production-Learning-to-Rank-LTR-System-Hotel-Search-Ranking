# Model Training

## LambdaMART Theory

### From Pairwise to Listwise

LambdaMART belongs to the family of **listwise** learning-to-rank algorithms, though its gradient computation is rooted in pairwise comparisons. The evolution of LTR approaches:

1. **Pointwise**: Treat each item independently, predict relevance as regression/classification. Problem: ignores the relative structure of ranking.
2. **Pairwise (RankNet)**: For each pair of items (i, j) where i is more relevant than j, learn to score i higher. The RankNet loss for a pair is:

```
L_ij = log(1 + exp(-(s_i - s_j)))
```

where `s_i` and `s_j` are the model scores.

3. **LambdaRank**: Modifies the RankNet gradient by multiplying by `|delta_NDCG_ij|`, the change in NDCG from swapping items i and j. This focuses learning on swaps that matter for the ranking metric:

```
lambda_ij = (1 / (1 + exp(s_i - s_j))) * |delta_NDCG_ij|
```

4. **LambdaMART**: Combines LambdaRank gradients with gradient-boosted decision trees (MART = Multiple Additive Regression Trees).

### Lambda Gradients

The key insight of LambdaMART is that we never need to define an explicit loss function. Instead, we directly define the gradients (lambdas) that each training example contributes:

For item i in a query, the total lambda is:

```
lambda_i = sum_{j: y_j > y_i} lambda_ij - sum_{j: y_j < y_i} lambda_ij
```

Where `lambda_ij` is the pairwise lambda for the pair (i, j):

```
lambda_ij = sigma / (1 + exp(sigma * (s_i - s_j))) * |delta_NDCG_ij|
```

And `delta_NDCG_ij` is the change in NDCG from swapping items i and j in the current ranking:

```
delta_NDCG_ij = |DCG_gain(y_i, pos_j) + DCG_gain(y_j, pos_i) - DCG_gain(y_i, pos_i) - DCG_gain(y_j, pos_j)|
```

Where:

```
DCG_gain(y, pos) = (2^y - 1) / log2(pos + 1)
```

### Why This Works

The `|delta_NDCG_ij|` weighting ensures that:
- Swaps between highly relevant and irrelevant items near the top of the list receive large gradients.
- Swaps between items deep in the list receive small gradients.
- Swaps between items of similar relevance receive small gradients.

This implicitly optimizes NDCG without needing NDCG to be differentiable (it is not, due to the sorting operation).

---

## LightGBM's Implementation

### Why LightGBM?

LightGBM implements LambdaMART with several engineering optimizations that make it practical for large datasets:

- **Histogram-based splitting**: Continuous features are bucketed into 255 bins, reducing the number of split candidates from O(n) to O(255). This makes training time linear in the number of data points.
- **Leaf-wise growth**: Unlike XGBoost's level-wise growth, LightGBM grows the leaf with the highest gain, producing deeper, more accurate trees with fewer leaves.
- **Exclusive Feature Bundling (EFB)**: Mutually exclusive sparse features are bundled together, reducing the effective number of features.
- **Native ranking support**: LightGBM's `LGBMRanker` class handles the group parameter, lambda gradient computation, and NDCG evaluation internally.

### The Group Parameter

LightGBM requires a `group` array that specifies the boundaries between queries:

```python
# If the data has 3 queries with 5, 3, and 7 items respectively:
group = [5, 3, 7]

# This tells LightGBM:
#   Query 0: rows 0-4
#   Query 1: rows 5-7
#   Query 2: rows 8-14
```

Lambda gradients are computed within each query group. Items from different queries are never compared.

---

## Hyperparameter Configuration

All hyperparameters are centralized in `config.py`:

```python
LGBM_PARAMS = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'n_estimators': 500,
    'num_leaves': 63,
    'learning_rate': 0.05,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'importance_type': 'gain',
    'n_jobs': -1,
    'random_state': 42,
    'verbosity': -1,
}
```

### Rationale for Each Parameter

| Parameter | Value | Rationale |
|---|---|---|
| `objective` | `lambdarank` | Enables LambdaMART's NDCG-optimizing gradients |
| `metric` | `ndcg` | Evaluation metric for early stopping and logging |
| `n_estimators` | 500 | Upper bound on boosting rounds; actual count determined by early stopping |
| `num_leaves` | 63 | Controls tree complexity. 63 = 2^6 - 1 allows up to depth-6 trees. Higher values capture more interactions but risk overfitting |
| `learning_rate` | 0.05 | Lower rates require more trees but produce more stable ensembles. 0.05 is a standard starting point |
| `min_child_samples` | 50 | Minimum examples per leaf. Prevents overfitting to small groups. Set relatively high given the large dataset |
| `subsample` | 0.8 | Row sampling per tree. Introduces randomness for regularization |
| `colsample_bytree` | 0.8 | Feature sampling per tree. Reduces correlation between trees and improves generalization |
| `reg_alpha` | 0.1 | L1 regularization on leaf weights. Encourages sparse leaf values |
| `reg_lambda` | 1.0 | L2 regularization on leaf weights. Shrinks leaf values toward zero |
| `importance_type` | `gain` | Feature importance measured by total gain from splits on each feature |
| `n_jobs` | -1 | Use all available CPU cores |
| `random_state` | 42 | Reproducibility |
| `verbosity` | -1 | Suppress LightGBM logging noise |

---

## IPS Integration via Sample Weights

### Mechanism

LGBMRanker's `fit` method accepts a `sample_weight` parameter that scales the gradient contribution of each training example:

```python
model = lgb.LGBMRanker(**LGBM_PARAMS)

model.fit(
    X_train, y_train,
    group=train_groups,
    sample_weight=ips_weights,
    eval_set=[(X_val, y_val)],
    eval_group=[val_groups],
    callbacks=[
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(50),
    ],
)
```

### How Weights Affect Lambda Gradients

When sample weights are provided, the lambda gradient for each pair (i, j) is modified:

```
lambda_ij_weighted = lambda_ij * w_i
```

Where `w_i = 1 / propensity(position_i)` is the IPS weight for item i. This means:

- Clicks from high-propensity positions (top of page) contribute normally.
- Clicks from low-propensity positions (bottom of page) are amplified.
- The model learns from the counterfactual question: "What would the click rate be if all positions had equal examination probability?"

### Weight Normalization

IPS weights are normalized to have a mean of 1.0 before being passed to LightGBM:

```python
ips_weights = ips_weights / ips_weights.mean()
```

This ensures the effective learning rate is not changed by the weighting scheme. Without normalization, very large weights would require a proportionally smaller learning rate.

---

## Early Stopping

### Configuration

```python
EARLY_STOPPING_ROUNDS = 50
```

### Behavior

1. After each boosting round, LightGBM evaluates NDCG on the validation set.
2. If the validation NDCG does not improve for 50 consecutive rounds, training stops.
3. The model is rolled back to the best iteration (the one with the highest validation NDCG).

### Why 50 Rounds?

- Too few (e.g., 10): Premature stopping. LambdaMART's validation NDCG can plateau for 20-30 rounds before improving again, especially early in training.
- Too many (e.g., 200): Wastes compute time training overfit trees that will be discarded.
- 50 rounds is a standard choice that balances patience with efficiency.

### Typical Training Curve

```
Round   Train NDCG@5    Val NDCG@5
  50      0.42            0.38
 100      0.48            0.43
 200      0.53            0.46
 300      0.56            0.47     <-- best
 350      0.58            0.47     <-- no improvement for 50 rounds, stop
```

The final model uses the checkpoint from round 300 (best validation NDCG), not round 350.

---

## Training Pipeline Integration

The trainer module (`training/trainer.py`) exposes a simple interface:

```python
from hotel_ranker.training.trainer import train_model

model = train_model(
    X_train, y_train, train_groups,
    X_val, y_val, val_groups,
    ips_weights=ips_weights,
)
```

This returns a fitted `LGBMRanker` instance that can be used for prediction:

```python
scores = model.predict(X_test)
```

The scores are real-valued. Higher scores indicate higher predicted relevance. To produce a ranking, items within each query are sorted by score in descending order.
