# Position Bias Correction

## The Problem

Users interact with search results sequentially from top to bottom. Items at position 1 receive far more attention than items at position 20, regardless of their true relevance. This creates a fundamental confound in click data: observed clicks reflect both relevance and position, but a ranking model should learn only relevance.

If we train on raw click data, the model learns to replicate the existing ranking policy (which placed items at those positions) rather than learning true relevance. This creates a rich-get-richer feedback loop where items ranked highly in the past continue to be ranked highly, even if better alternatives exist.

---

## The Examination Hypothesis

The standard theoretical framework for position bias is the **examination hypothesis** (Richardson et al., 2007):

```
P(click | item, position) = P(relevant | item) * P(examine | position)
```

This decomposes the click probability into two independent factors:

- **P(relevant | item)**: The probability that the item is relevant to the user's need. This is what the ranking model should learn.
- **P(examine | position)**: The probability that the user examines the item at the given position. This is a nuisance factor that depends only on position, not on the item.

The key assumption is **conditional independence**: given that the user examines an item, the probability of clicking depends only on relevance, not on position. This is a simplification (in reality, examination probability may depend on the surrounding items), but it is empirically well-supported for vertical search result lists.

---

## Propensity Estimation

### Randomized Data

The Expedia dataset includes a `random_bool` column that indicates whether a search result page was randomly shuffled. When `random_bool = 1`, item positions are independent of their relevance, meaning:

```
P(click | item, position, random=1) = P(relevant | item) * P(examine | position)
```

Since the position assignment is random, we can estimate the examination probability (propensity) by computing the click rate at each position across all randomized queries:

```
P(examine | position=k) proportional to (clicks at position k) / (impressions at position k)
    where only random_bool=1 rows are used
```

This is normalized so that `P(examine | position=1) = 1.0` (the top position is the reference).

### Power-Law Model

Empirically, position propensities follow a power-law decay:

```
P(examine | position=k) = k^(-eta)
```

Where `eta` is fit via least-squares regression on the log-transformed empirical propensities:

```
log(propensity_k) = -eta * log(k) + intercept
```

The power-law model has two advantages over using raw empirical propensities:

1. **Smoothing.** Empirical propensities at high positions are noisy due to few observations. The parametric model smooths these estimates.
2. **Extrapolation.** If the training data has a maximum position of 38, the power-law model can estimate propensities for positions 39+.

Typical values of `eta` range from 0.5 to 1.5. Higher values indicate more aggressive position bias (users rarely look past the top few results).

---

## Inverse Propensity Scoring (IPS)

### Concept

IPS is a technique from causal inference (Horvitz & Thompson, 1952) adapted for unbiased LTR by Joachims et al. (2017). The idea is to reweight each observation by the inverse of its probability of being observed (examined):

```
IPS_weight(item at position k) = 1 / P(examine | position=k)
```

**Intuition**: A click at position 20 is more informative than a click at position 1 because the user was much less likely to even see the item at position 20. By upweighting clicks at low-examination positions, we correct for the fact that some items are underrepresented in the click data simply because they were placed in low-visibility positions.

### Application to Training

IPS weights are passed to LGBMRanker via the `sample_weight` parameter:

```python
model.fit(
    X_train, y_train,
    group=train_groups,
    sample_weight=ips_weights,  # <-- position bias correction
    ...
)
```

Each training example is weighted by `1 / propensity(position)`. This modifies the lambda gradients computed by LambdaMART so that the model pays more attention to informative (low-propensity) observations.

### Formal Guarantee

Under the examination hypothesis, the IPS-weighted loss is an unbiased estimator of the true loss computed under uniform examination:

```
E[1/P(examine|k) * loss(item_k)] = sum_k P(examine|k) * 1/P(examine|k) * loss(item_k)
                                   = sum_k loss(item_k)
```

This holds regardless of the specific loss function used, making IPS compatible with LambdaMART's NDCG-based optimization.

---

## Variance Reduction via Weight Clipping

### The Variance Problem

While IPS is unbiased in expectation, it can have high variance. Items at very low positions have very small propensities, producing very large IPS weights. A single item at position 40 might receive a weight of 40^eta, which could be 100x or more. This creates unstable gradient estimates and noisy training.

### Clipping Strategy

Weight clipping caps the maximum IPS weight at a percentile-based threshold:

```python
clip_value = np.percentile(raw_weights, IPS_CLIP_PERCENTILE)  # 95th percentile
clipped_weights = np.minimum(raw_weights, clip_value)
```

With `IPS_CLIP_PERCENTILE = 95`, the top 5% of weights are capped at the 95th percentile value. This introduces a small bias (items at the lowest positions are slightly underweighted) in exchange for a large reduction in variance.

### Bias-Variance Tradeoff

| Clipping Percentile | Bias | Variance | Typical Use |
|---|---|---|---|
| 100 (no clipping) | None | High | Theoretical analysis |
| 99 | Very low | Moderate | Large datasets |
| 95 | Low | Low | **Default (production)** |
| 90 | Moderate | Very low | Small datasets |

The choice of 95th percentile is a pragmatic default that works well across a range of dataset sizes and position distributions.

---

## Implementation Details

### Module: `bias/propensity.py`

The propensity estimation pipeline:

1. **Filter** to rows where `random_bool == 1`.
2. **Compute** empirical click rate at each position.
3. **Normalize** so position 1 has propensity 1.0.
4. **Fit** power-law model via log-log linear regression.
5. **Predict** propensity for all positions using the fitted model.
6. **Compute** IPS weights as `1 / propensity(position)`.
7. **Clip** weights at the 95th percentile.

### Edge Cases

- **Position 0 or missing**: Assigned propensity of 1.0 (no correction).
- **Non-clicked items**: IPS weights are still computed for all items. The weight affects the gradient contribution of both positive (clicked) and negative (not clicked) examples.
- **Insufficient random data**: If fewer than 100 randomized queries exist, the system logs a warning and falls back to uniform propensities (no correction). This prevents fitting a power-law model on insufficient data.

---

## References

- Joachims, T., Swaminathan, A., & Schnabel, T. (2017). Unbiased Learning-to-Rank with Biased Feedback. *WSDM*.
- Wang, X., Golbandi, N., Bendersky, M., et al. (2018). Position Bias Estimation for Unbiased Learning to Rank in Personal Search. *WSDM*.
- Richardson, M., Dominowska, E., & Ragno, R. (2007). Predicting Clicks: Estimating the Click-Through Rate for New Ads. *WWW*.
- Horvitz, D. G., & Thompson, D. J. (1952). A Generalization of Sampling Without Replacement from a Finite Universe. *JASA*.
