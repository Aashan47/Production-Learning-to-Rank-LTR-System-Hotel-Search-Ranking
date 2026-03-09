# Evaluation

## Overview

The evaluation module measures ranking quality using standard information retrieval metrics and provides diagnostic tools for error analysis. All metrics are computed on the held-out test set to estimate real-world performance.

---

## NDCG (Normalized Discounted Cumulative Gain)

### Theory

NDCG is the primary metric for evaluating ranking quality. It measures how well the predicted ranking places relevant items near the top of the list.

#### Discounted Cumulative Gain (DCG)

DCG accumulates the gain from each item, discounted by its position:

```
DCG@k = sum_{i=1}^{k} (2^{rel_i} - 1) / log2(i + 1)
```

Where:
- `k` is the cutoff (e.g., 5 or 10)
- `rel_i` is the relevance grade of the item at position i in the predicted ranking
- The numerator `2^{rel_i} - 1` is the **gain**: exponential scaling rewards higher relevance grades disproportionately
- The denominator `log2(i + 1)` is the **discount**: logarithmic decay penalizes items placed lower in the ranking

#### Ideal DCG (IDCG)

IDCG is the maximum possible DCG, obtained by sorting items by their true relevance in descending order:

```
IDCG@k = sum_{i=1}^{k} (2^{rel*_i} - 1) / log2(i + 1)
```

Where `rel*_i` is the relevance grade of the i-th most relevant item.

#### NDCG

NDCG normalizes DCG by IDCG to produce a value in [0, 1]:

```
NDCG@k = DCG@k / IDCG@k
```

- NDCG@k = 1.0 means the predicted ranking is perfect (matches the ideal ranking up to position k).
- NDCG@k = 0.0 means no relevant items appear in the top k.

#### Example

Consider a query with 5 items with true relevance grades [3, 0, 2, 1, 4]:

Predicted ranking order: [item5, item1, item3, item4, item2]
Relevance at each position: [4, 3, 2, 1, 0]

```
DCG@5  = (2^4-1)/log2(2) + (2^3-1)/log2(3) + (2^2-1)/log2(4) + (2^1-1)/log2(5) + (2^0-1)/log2(6)
       = 15/1 + 7/1.585 + 3/2 + 1/2.322 + 0/2.585
       = 15 + 4.416 + 1.5 + 0.431 + 0
       = 21.347
```

Ideal ranking: [4, 3, 2, 1, 0] (already sorted)

```
IDCG@5 = 21.347  (same in this case, since the predicted ranking happens to be perfect)
NDCG@5 = 21.347 / 21.347 = 1.0
```

### Cutoff Values

The system evaluates NDCG at two cutoffs, configured in `config.py`:

```python
NDCG_CUTOFFS = [5, 10]
```

| Metric | What It Measures |
|---|---|
| NDCG@5 | Quality of the top 5 results (above the fold). Most important for user experience. |
| NDCG@10 | Quality of the top 10 results (first page). Captures broader ranking quality. |

### Implementation

NDCG is computed using `sklearn.metrics.ndcg_score`:

```python
from sklearn.metrics import ndcg_score

def compute_ndcg(y_true, y_pred, k):
    """
    Compute NDCG@k for a single query.

    y_true: array of true relevance grades
    y_pred: array of predicted scores
    k: cutoff position
    """
    return ndcg_score([y_true], [y_pred], k=k)
```

For multi-query evaluation, NDCG is computed per query and then averaged (macro-average). This gives equal weight to each query regardless of the number of items:

```python
def mean_ndcg(df, score_col, label_col, k):
    ndcgs = []
    for srch_id, group in df.groupby('srch_id'):
        ndcg = ndcg_score(
            [group[label_col].values],
            [group[score_col].values],
            k=k
        )
        ndcgs.append(ndcg)
    return np.mean(ndcgs)
```

---

## MRR (Mean Reciprocal Rank)

### Definition

MRR measures how quickly the model surfaces the first highly relevant result:

```
RR(query) = 1 / rank_of_first_relevant_item
```

```
MRR = mean(RR(query) for all queries)
```

Where "relevant" is defined as having a relevance grade >= threshold (typically grade >= 3, i.e., a booking).

### Interpretation

| MRR Value | Meaning |
|---|---|
| 1.0 | The best item is always ranked first |
| 0.5 | On average, the best item is ranked second |
| 0.33 | On average, the best item is ranked third |
| 0.1 | On average, the best item is ranked tenth |

### Implementation

```python
def compute_mrr(df, score_col, label_col, relevance_threshold=3):
    """
    Compute MRR across all queries.

    Items with label >= relevance_threshold are considered "relevant."
    """
    reciprocal_ranks = []
    for srch_id, group in df.groupby('srch_id'):
        sorted_group = group.sort_values(score_col, ascending=False)
        relevant_mask = sorted_group[label_col] >= relevance_threshold
        if relevant_mask.any():
            first_relevant_rank = relevant_mask.values.argmax() + 1  # 1-indexed
            reciprocal_ranks.append(1.0 / first_relevant_rank)
        else:
            reciprocal_ranks.append(0.0)  # No relevant item in query
    return np.mean(reciprocal_ranks)
```

### MRR vs. NDCG

| Aspect | NDCG | MRR |
|---|---|---|
| Focus | Overall ranking quality | First relevant result |
| Relevance levels | Graded (0-4) | Binary (above/below threshold) |
| Multiple relevant items | Considers all | Only first |
| Use case | General ranking quality | Navigational queries |

Both metrics are reported because they capture complementary aspects of ranking quality.

---

## Per-Query NDCG Analysis

### Purpose

Aggregate metrics (mean NDCG) can mask important patterns. Per-query NDCG analysis reveals:

- **Distribution shape**: Is performance uniformly good, or are there many perfect queries dragging up a long tail of poor ones?
- **Query difficulty**: Which queries are hard to rank? What do they have in common?
- **Failure modes**: Are failures concentrated in specific destination types, price ranges, or user segments?

### Implementation

`evaluation/metrics.py` provides `per_query_ndcg`, which returns a DataFrame with one row per query:

```python
per_query = per_query_ndcg(df, score_col='pred_score', label_col='label', k=5)
# Returns: DataFrame with columns [srch_id, ndcg_at_5, n_items, n_relevant]
```

This enables downstream analysis such as:

```python
# Queries where the model performs poorly
hard_queries = per_query[per_query['ndcg_at_5'] < 0.5]

# Correlation between query size and performance
per_query[['n_items', 'ndcg_at_5']].corr()
```

---

## Before/After Comparison

### Methodology

The error analysis module (`evaluation/error_analysis.py`) compares the model's ranking against the original ranking (by position) to quantify improvement.

#### Baseline

The baseline ranking is the original Expedia ranking, represented by the `position` column. Lower position values mean higher original rank.

#### Comparison Metrics

For each query, the module computes:

```
delta_ndcg = ndcg(model_ranking) - ndcg(original_ranking)
```

Queries are then categorized:

| Category | Condition | Interpretation |
|---|---|---|
| Improved | delta_ndcg > 0.01 | Model ranking is better |
| Degraded | delta_ndcg < -0.01 | Model ranking is worse |
| Unchanged | abs(delta_ndcg) <= 0.01 | No meaningful difference |

A tolerance of 0.01 is used to avoid counting noise as change.

### Biggest Improvements and Degradations

The module identifies the top-k queries with the largest positive and negative delta_ndcg. For each, it reports:

- The query ID and context features (destination, dates, party size)
- The original and predicted rankings side by side
- Which items moved up or down

This diagnostic helps identify systematic patterns, such as "the model improves ranking for queries with large result sets but degrades it for queries with very few options."

---

## Feature Importance

### Gain-Based Importance

LightGBM's `feature_importances_` attribute (with `importance_type='gain'`) reports the total reduction in the training objective contributed by splits on each feature:

```
importance(feature_j) = sum over all trees, all nodes split on feature_j, of gain from that split
```

Higher gain means the feature contributes more to separating relevant from irrelevant items.

### Importance Plots

`error_analysis.py` generates two visualizations:

1. **Bar chart of top-20 features by gain.** Shows which features drive the model's decisions.
2. **NDCG distribution histogram.** Shows the per-query NDCG distribution to assess consistency.

These plots are saved to the output directory and displayed when running in interactive mode.

---

## Metric Summary Table

| Metric | Formula | Range | Higher Is Better | Primary Use |
|---|---|---|---|---|
| NDCG@5 | DCG@5 / IDCG@5 | [0, 1] | Yes | Top-of-page quality |
| NDCG@10 | DCG@10 / IDCG@10 | [0, 1] | Yes | Full-page quality |
| MRR | mean(1/rank_first_relevant) | [0, 1] | Yes | First-result quality |
| delta_NDCG | NDCG(model) - NDCG(baseline) | [-1, 1] | Yes | Improvement over baseline |
| % Improved | queries with delta > 0.01 | [0, 100] | Yes | Breadth of improvement |
