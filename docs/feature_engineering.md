# Feature Engineering

The feature pipeline produces four categories of features, each capturing a different aspect of hotel relevance. All feature generators implement a consistent interface and are orchestrated by `feature_pipeline.py`.

---

## 1. Raw Features

**Module**: `features/raw_features.py`

Raw features are either direct passthroughs from the dataset or simple derivations that do not require cross-row computation.

### Passthrough Features

These columns are used as-is from the preprocessed dataset:

- `prop_starrating` - Hotel star rating (0-5)
- `prop_review_score` - Guest review score (0-5, 0 = no reviews)
- `prop_brand_bool` - Whether the property belongs to a brand chain
- `prop_location_score1` - Desirability score of the hotel location (higher = better)
- `prop_location_score2` - Second location score (different methodology)
- `prop_log_historical_price` - Log of the hotel's historical average price
- `promotion_flag` - Whether the hotel is running a promotion
- `srch_length_of_stay` - Number of nights
- `srch_booking_window` - Days between search date and check-in
- `srch_adults_count` - Number of adults
- `srch_children_count` - Number of children
- `srch_room_count` - Number of rooms requested
- `price_usd` - Current price in USD
- `srch_saturday_night_bool` - Whether the stay includes a Saturday night

### Derived Features

#### Price Per Night

```
price_per_night = price_usd / max(srch_length_of_stay, 1)
```

Normalizes the total price by stay duration, enabling fair comparison between a 1-night stay at $200 and a 5-night stay at $1000.

#### Total Guests

```
total_guests = srch_adults_count + srch_children_count
```

Captures the total party size, which affects hotel suitability (e.g., suite vs. standard room).

#### Star-Review Ratio

```
star_review_ratio = prop_starrating / max(prop_review_score, 0.1)
```

A ratio > 1 indicates the hotel's official star rating exceeds its guest review score (potentially overrated). A ratio < 1 indicates guest satisfaction exceeds the official rating (potentially underrated). The denominator floor of 0.1 prevents division by zero for unreviewed properties.

#### Price vs. Visitor History

```
price_vs_visitor_hist = price_usd / max(visitor_hist_adr, 1.0)
```

Compares the current hotel price to the user's historical average daily rate. Values > 1 indicate a hotel above the user's typical spend; values < 1 indicate a hotel below it. This captures price sensitivity at the user level.

---

## 2. Match Features

**Module**: `features/match_features.py`

Match features measure how well each hotel fits the specific query context. These require within-query aggregation.

### Price Positioning

#### Price Difference from Query Mean

```
price_diff = price_usd - mean(price_usd | srch_id)
```

Positive values indicate above-average pricing within the query.

#### Price Ratio to Query Mean

```
price_ratio = price_usd / mean(price_usd | srch_id)
```

A scale-invariant version. A ratio of 1.5 means 50% above the query average.

#### Price Z-Score

```
price_zscore = (price_usd - mean(price_usd | srch_id)) / std(price_usd | srch_id)
```

Standardized price positioning. A z-score of +2 means the hotel is two standard deviations above the query mean price. This is the most informative price-positioning feature because it accounts for both the level and the spread of prices in the query.

For queries with zero variance (all hotels same price), the z-score is set to 0.

### Star Matching

#### Star Difference from User History

```
star_diff = prop_starrating - visitor_hist_starrating
```

When the user has booking history, this measures the gap between the hotel's star rating and the user's historical preference. Positive values indicate an upgrade; negative values indicate a downgrade. For new users (visitor_hist_starrating = -1), this feature is set to 0.

### Location Composite Score

```
location_composite = prop_location_score1 + prop_location_score2
```

Combines both location scores into a single signal. While the two scores are computed differently, they are positively correlated and their sum provides a more robust location quality estimate than either alone.

### Competitor Advantage Features

For each competitor `i` (1 through 8):

#### Rate Advantage

```
comp_rate_advantage_i = -comp{i}_rate
```

`comp{i}_rate` is +1 if the competitor is cheaper, -1 if more expensive. Negating it so that +1 means "we are cheaper" makes the feature direction consistent.

#### Availability Advantage

```
comp_inv_advantage_i = comp{i}_inv
```

`comp{i}_inv` is 1 if the competitor has the property available. This captures competitive pressure: if many competitors list the same hotel, the user has alternatives and price sensitivity increases.

#### Rate Difference Magnitude

```
comp_rate_pct_diff_i = comp{i}_rate_percent_diff * (-comp{i}_rate)
```

The magnitude of the price difference, signed so positive means we are cheaper by that percentage. This combines direction and magnitude into a single feature.

---

## 3. Historical Features

**Module**: `features/historical_features.py`

Historical features aggregate past interaction data at the property and destination level. These are the strongest features in most LTR systems but require careful handling to prevent leakage.

### Bayesian Smoothed Rates

Raw rates (e.g., clicks / impressions) are noisy for properties with few impressions. Bayesian smoothing (empirical Bayes) regularizes toward a global prior:

```
smoothed_rate = (observed_events + prior_count * prior_rate) / (impressions + prior_count)
```

Where:
- `observed_events` = number of clicks (or bookings) for the property
- `impressions` = number of times the property was shown
- `prior_count` = 30 (configurable in config.py; controls regularization strength)
- `prior_rate` = 0.05 (configurable; the global average rate)

**Interpretation**: A property with 0 clicks in 2 impressions gets a smoothed CTR of `(0 + 30*0.05) / (2 + 30) = 1.5 / 32 = 0.047`, close to the prior. A property with 50 clicks in 200 impressions gets `(50 + 1.5) / (200 + 30) = 51.5 / 230 = 0.224`, close to the observed rate. The prior dominates when data is sparse; observed data dominates when data is abundant.

### Per-Property Features

| Feature | Formula | Description |
|---|---|---|
| `prop_ctr` | `smoothed(clicks, impressions)` | Click-through rate for this property |
| `prop_booking_rate` | `smoothed(bookings, impressions)` | Booking rate for this property |
| `prop_avg_position` | `mean(position | prop_id)` | Average position this property is shown at |

### Per-Destination Features

| Feature | Formula | Description |
|---|---|---|
| `dest_avg_price` | `mean(price_usd | srch_destination_id)` | Average price in this destination |
| `dest_avg_starrating` | `mean(prop_starrating | srch_destination_id)` | Average star rating in this destination |
| `dest_avg_review` | `mean(prop_review_score | srch_destination_id)` | Average review score in this destination |

### Leakage Prevention

Historical features are computed **only on the training set** and then joined to the validation and test sets by property ID or destination ID. This prevents future information from leaking into training:

```python
# Computed on train only
prop_stats = compute_property_stats(train_df)
dest_stats = compute_destination_stats(train_df)

# Applied to all splits
train_df = train_df.merge(prop_stats, on='prop_id', how='left')
val_df   = val_df.merge(prop_stats, on='prop_id', how='left')
test_df  = test_df.merge(prop_stats, on='prop_id', how='left')
```

Properties or destinations that appear in val/test but not in train receive the prior rate (via `fillna(prior_rate)`), which is the correct Bayesian treatment for unseen entities.

---

## Feature Count Summary

| Category | Approximate Count | Computation Cost |
|---|---|---|
| Raw features | ~18 | O(n) |
| Match features | ~15 | O(n) per query |
| Historical features | ~8 | O(n) aggregate + join |
| **Total** | **~40** | |

The exact count depends on the number of active competitor columns present in the dataset.
