# Hotel Search Ranking — Production LTR System

A production-grade **Learning-to-Rank** system that re-ranks Expedia hotel search results using LightGBM's LambdaMART, position-bias correction via Inverse Propensity Scoring, and a multi-objective relevance label that balances clicks with revenue-weighted bookings. Trained on **9.9 million rows** across 400K+ queries.

---

## Results

Evaluated on **18,105 position-independent queries** (`random_bool=1`) where the platform randomly shuffled hotel ordering — giving click labels unconfounded by display position. This is the only fair offline comparison between rankers.

| Metric | Expedia Baseline | **Our Model** | Improvement |
|:-------|:-:|:-:|:-:|
| NDCG@5 | 0.2913 | **0.3189** | **+9.5%** |
| NDCG@10 | 0.3603 | **0.3906** | **+8.4%** |
| MRR | 0.2971 | **0.3203** | **+7.8%** |

---

### Per-Query Win Rate

![Per-Query Win Rate](docs/images/per_query_winrate.png)

Across 59,903 test queries, the model wins on **30.9%** and ties on **37.4%**. Ties occur when the booked hotel is already at rank 1 — neither ranker can improve. On contested queries, the win rate is substantially higher.

---

### Feature Importance

![Feature Importance](docs/images/feature_importance_top20.png)

| Feature | Category | Importance |
|---------|:--------:|:----------:|
| `prop_click_rate` | Historical | 1,840 |
| `prop_impression_count` | Historical | 1,546 |
| `prop_booking_rate` | Historical | 1,307 |
| `prop_location_score2` | Raw | 1,272 |
| `price_zscore_in_query` | Match | ~980 |
| `prop_review_score` | Raw | ~920 |

Bayesian-smoothed historical engagement dominates — consistent with industry findings. Relative price signals (within-query z-score) rank above absolute price, confirming context matters. No positional signals appear in the top features, confirming the IPS correction worked.

---

## Technical Approach

### Problem Framing

Hotel ranking is **listwise**: a hotel's relevance is relative to others in the same search. LambdaMART directly optimises NDCG across the full ranked list, unlike pointwise (treats items independently) or pairwise (ignores list context) approaches.

### Position Bias Correction (IPS)

Click data is observational — users examine higher positions more, so top results accumulate clicks regardless of quality. We correct this with Inverse Propensity Scoring:

1. Estimate examination probability `P(examine | position)` from randomised queries (`random_bool=1`)
2. Fit power-law: `P(pos) = α × pos^(−β)` via scipy curve fitting
3. Reweight each clicked sample by `1 / P(examine | position)`
4. Cap weights at the **75th percentile of clicked items only** — applying the cap to all items collapses it to 1.0 and cancels the correction

### Label Engineering

Binary click labels lose business context. We build a composite relevance label that rewards bookings proportional to price tier:

```python
composite = W_CLICK * click_bool + W_BOOK * (0.5 + 0.5 * price_percentile) * booking_bool
```

After within-query normalisation and discretisation into [0, 4], this produces four ordinal grades: no interaction → click → low-priced booking → high-priced booking. Richer grades give LambdaMART more informative pairwise comparisons.

### Feature Engineering

~40 features across three groups, all computed on training data only to prevent leakage:

| Category | Count | Examples |
|:--------:|:-----:|---------|
| Raw | 22 | `price_per_night`, `star_review_ratio`, `prop_log_historical_price` |
| Match | 10 | `price_zscore_in_query`, `competitor_rate_advantage`, `star_match_visitor_pref` |
| Historical | 8 | `prop_click_rate`\*, `prop_booking_rate`\*, `dest_booking_rate`\* |

\* Bayesian-smoothed: `smoothed_rate = (n × raw_rate + 30 × 0.05) / (n + 30)` — prevents a hotel with 2 impressions and 1 click from receiving CTR=50%.

---

## Architecture

```
Expedia Hotel Search Dataset  (9.9M rows · 400K+ queries)
                    │
          ┌─────────▼─────────┐
          │   Preprocessing   │  dtype optimisation (~50% memory reduction)
          │                   │  missing values: competitor→0, review→0
          └─────────┬─────────┘  visitor history→-1 sentinel, rest→median
                    │
          ┌─────────▼──────────────────────┐
          │   Query-Level Split  70/15/15  │
          └───┬──────────────────────┬─────┘
              │  train               │  val / test
    ┌─────────▼──────────────┐       │
    │  Historical Aggregates │       │  (joined by prop_id / dest_id)
    │  prop CTR, booking rate│───────┘
    │  dest averages (Bayes) │
    └─────────┬──────────────┘
              │
    ┌─────────▼──────────────┐
    │  Feature Pipeline      │  raw + match + historical → 40 cols
    └─────────┬──────────────┘
              │
    ┌─────────▼──────────────┐
    │  IPS Weight Estimation │  power-law propensity from random_bool=1
    │  Composite Labels      │  click + price-tiered booking → grades 0–4
    └─────────┬──────────────┘
              │
    ┌─────────▼──────────────┐
    │  LGBMRanker            │  LambdaMART · early stopping @NDCG@5
    │  (LambdaMART)          │  best iteration ≈ 170 trees
    └─────────┬──────────────┘
              │
    ┌─────────▼──────────────┐
    │  Unbiased Evaluation   │  NDCG@5/10, MRR on random_bool=1 queries
    │  Error Analysis        │  per-query distribution, feature importance
    └────────────────────────┘
```

---

## Project Structure

```
LTR system/
├── src/hotel_ranker/
│   ├── config.py                    # Hyperparameters
│   ├── pipeline.py                  # End-to-end orchestration + CLI
│   ├── data/
│   │   ├── preprocessing.py         # Missing values, dtype optimisation
│   │   └── splitting.py             # Query-level train/val/test split
│   ├── features/
│   │   ├── raw_features.py
│   │   ├── match_features.py
│   │   ├── historical_features.py   # Bayesian-smoothed CTR, booking rate
│   │   └── feature_pipeline.py
│   ├── bias/
│   │   └── propensity.py            # Power-law propensity + IPS weights
│   ├── training/
│   │   ├── label_engineering.py     # Composite relevance labels (0–4)
│   │   └── trainer.py
│   └── evaluation/
│       ├── metrics.py               # NDCG@k, MRR
│       └── error_analysis.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_position_bias.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_evaluation.ipynb
│
├── docs/
└── pyproject.toml
```

---

## Quick Start

**Prerequisites:** Python 3.9+, Kaggle API credentials

```bash
git clone <repository-url>
cd "LTR system"
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

```bash
python -m hotel_ranker.pipeline           # Full pipeline (~7 min on CPU)
python -m hotel_ranker.pipeline --sample 0.1  # Smoke test (~2 min)
pytest tests/ -v
```

---

## References

- Burges (2010). *From RankNet to LambdaRank to LambdaMART.* MSR-TR-2010-82.
- Joachims, Swaminathan & Schnabel (2017). *Unbiased Learning-to-Rank with Biased Feedback.* WSDM.
- Wang et al. (2018). *Position Bias Estimation for Unbiased Learning to Rank.* WSDM.
- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
