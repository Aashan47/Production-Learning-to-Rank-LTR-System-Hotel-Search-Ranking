# Data Pipeline

## Data Acquisition

### Source

The system uses the **Expedia Hotel Search** dataset, originally published as a Kaggle competition. The dataset contains anonymized hotel search logs including:

- **Search context**: destination, check-in/check-out dates, number of adults/children/rooms, user country, search date.
- **Hotel properties**: star rating, review scores, location scores, historical price data, competitor pricing.
- **User history**: visitor historical star rating and average daily rate.
- **Outcomes**: click and booking boolean flags, plus position in the original ranking.

### Download Mechanism

Data acquisition uses `kagglehub`, which handles authentication and caching automatically:

```python
# acquisition.py
import kagglehub

path = kagglehub.dataset_download("c/expedia-personalized-sort")
csv_path = find_csv(path)  # Locates the training CSV within the download directory
```

The `find_csv` helper searches the download directory recursively for the main CSV file, handling version-dependent directory structures from kagglehub.

### Data Volume

The full dataset contains approximately 10 million rows across ~400,000 unique search queries. Each row represents one hotel displayed in a search result. The `--sample` CLI flag can reduce this for development:

```bash
python -m hotel_ranker.pipeline --sample 0.1  # ~1M rows, ~40K queries
```

---

## Preprocessing Strategy

Preprocessing is handled in `preprocessing.py` and consists of three stages: dtype optimization, missing value handling, and derived column creation.

### Dtype Optimization

The raw CSV loads with default float64/int64 dtypes, consuming excessive memory. The schema module defines optimal dtypes:

| Column Type | Raw Dtype | Optimized Dtype | Rationale |
|---|---|---|---|
| Boolean flags | float64 | Int8 (nullable) | 0/1 values with NaN |
| Star ratings | float64 | float32 | 0.5 increments, 0-5 range |
| Prices | float64 | float32 | Sufficient precision |
| Counts | float64 | Int16 (nullable) | Small integers with NaN |
| IDs | int64 | int32 | Sufficient range |

This typically reduces memory usage by 50-60%.

### Missing Value Handling

Different column types have different missingness mechanisms, and each requires a distinct imputation strategy:

#### Competitor Columns (fill with 0)

```
comp1_rate, comp1_inv, comp1_rate_percent_diff, ..., comp8_rate, comp8_inv, comp8_rate_percent_diff
```

**Rationale**: A missing competitor value means that competitor does not operate in the given market. Filling with 0 encodes "no competitive pressure from this competitor," which is the semantically correct interpretation. This is not imputation in the statistical sense; it is encoding a known business fact.

#### Review Scores (fill with 0)

```
prop_review_score
```

**Rationale**: A missing review score indicates a property with no reviews. This is distinct from a property with a low review score. Filling with 0 creates a natural ordering: no-review properties score below reviewed properties. This is preferable to median imputation, which would incorrectly assign average quality to unreviewed properties.

#### Visitor History (fill with -1)

```
visitor_hist_starrating, visitor_hist_adr
```

**Rationale**: These columns represent the user's historical average star rating and average daily rate from prior bookings. Missing values indicate a new user with no booking history. Using -1 as a sentinel value (outside the natural range of both features) allows the tree-based model to learn a distinct split for "no history" versus "has history." This is superior to both zero-fill (which conflates "no history" with "low-end preference") and median-fill (which obscures the new-user signal).

#### Remaining Numeric Columns (median imputation)

For all other numeric columns with missing values, the system applies per-column median imputation. Median is preferred over mean because:

- It is robust to outliers (price distributions are heavily right-skewed).
- It preserves the central tendency without being pulled by extreme values.
- For tree-based models, the exact imputed value matters less than preserving the relative ordering of non-missing values.

### Derived Columns

Two columns are added during preprocessing because they are useful across multiple feature generators:

- **`price_rank_in_query`**: The rank of each hotel's price within its search query (1 = cheapest). Computed via `groupby('srch_id')['price_usd'].rank()`.
- **`price_log`**: `log1p(price_usd)`. Log-transformed price compresses the heavy right tail and makes price differences more perceptually meaningful (the difference between $100 and $200 matters more than between $1000 and $1100).

---

## Query-Level Splitting

### Strategy

The dataset is split into train, validation, and test sets at the **query level** (by `srch_id`), not at the row level. The split ratio is **70% train / 15% validation / 15% test**, controlled by `RANDOM_SEED = 42` for reproducibility.

```python
# splitting.py
unique_queries = df['srch_id'].unique()
np.random.seed(RANDOM_SEED)
np.random.shuffle(unique_queries)

n = len(unique_queries)
train_queries = unique_queries[:int(0.70 * n)]
val_queries   = unique_queries[int(0.70 * n):int(0.85 * n)]
test_queries  = unique_queries[int(0.85 * n):]
```

### Why Not Row-Level Splitting?

Row-level splitting (randomly assigning individual hotel rows to train/val/test) introduces **data leakage** in ranking problems:

1. **Query context leakage.** If some hotels from query Q are in train and others in test, the model has partial knowledge of the competitive context at test time. In production, the model sees no hotels from the query during training.

2. **Group structure violation.** LGBMRanker requires the `group` parameter, which specifies how many consecutive rows belong to each query. Splitting rows across sets would break group boundaries, making it impossible to compute NDCG correctly.

3. **Temporal correlation.** Hotels within the same query share temporal features (search date, check-in date). Row-level splitting leaks temporal information.

### Group Parameter

LGBMRanker takes a `group` array where `group[i]` is the number of items in the i-th query. This is computed by `get_groups`:

```python
def get_groups(df):
    return df.groupby('srch_id').size().values
```

The `group` array tells LightGBM which rows to compare when computing lambda gradients. It is essential that the data is sorted by `srch_id` and that the group array matches exactly.

### Leakage Prevention Summary

| Leakage Vector | Mitigation |
|---|---|
| Same query in train and test | Query-level split |
| Future data in training | No temporal features that reference future |
| Historical features computed on test data | Historical features computed on training set only, then applied to val/test |
| Position bias in labels | IPS weighting (see position_bias.md) |
