# Architecture Overview

## System Design

This project implements a production Learning-to-Rank (LTR) system for hotel search ranking, built on the Expedia Hotel Search dataset. The system re-ranks hotel search results to maximize both user satisfaction (click-through) and business value (bookings and revenue), using a LambdaMART model via LightGBM's `LGBMRanker`.

### Design Philosophy

The system is designed around three guiding principles:

1. **Listwise ranking over pointwise classification.** Hotel relevance is inherently relative within a search context. A hotel that is excellent for one query may be irrelevant for another. LambdaMART directly optimizes NDCG, a listwise metric, rather than reducing ranking to independent binary classification.

2. **Causal debiasing over naive supervision.** Historical click data is confounded by position bias: users click higher-ranked results more often regardless of relevance. The system applies Inverse Propensity Scoring (IPS) to correct for this, producing a model that learns true relevance rather than memorizing prior ranking policy.

3. **Graceful degradation for new properties.** Properties with no interaction history lack the behavioral signals (CTR, booking rate) that drive most ranking models. Bayesian smoothing on historical features shrinks new properties toward a global prior, ensuring they receive a reasonable ranking rather than being permanently buried.

---

## Data Flow Diagram

```
+---------------------+
|   Kaggle Dataset    |
|  (Expedia Hotel     |
|   Search Data)      |
+----------+----------+
           |
           v
+----------+----------+
|  Data Acquisition   |  acquisition.py
|  (kagglehub DL)     |
+----------+----------+
           |
           v
+----------+----------+
|   Preprocessing     |  preprocessing.py
|  - Missing values   |
|  - Dtype optimize   |
|  - Derived columns  |
+----------+----------+
           |
           v
+----------+----------+
|   Query-Level       |  splitting.py
|   Train/Val/Test    |
|   Split (70/15/15)  |
+----------+----------+
           |
           v
+----------+-------------------+-------------------+
|                              |                   |
v                              v                   v
+--------------+  +----------------+  +----------------+
| Raw Features |  | Match Features |  | Historical     |
| raw_features |  | match_features |  | Features       |
|   .py        |  |   .py          |  | historical_    |
|              |  |                |  | features.py    |
+--------------+  +----------------+  +----------------+
        |                  |                   |
        +------------------+-------------------+
                           |
                           v
                +----------+----------+
                |  Feature Pipeline   |  feature_pipeline.py
                |  (Orchestration)    |
                +----------+----------+
                           |
              +------------+------------+
              |                         |
              v                         v
  +-----------+-----------+  +----------+----------+
  | Position Bias         |  | Label Engineering   |
  | Correction            |  | (Composite label:   |
  | (IPS weights)         |  |  click + revenue)   |
  | propensity.py         |  | label_engineering.py|
  +-----------+-----------+  +----------+----------+
              |                         |
              +------------+------------+
                           |
                           v
                +----------+----------+
                |   LGBMRanker        |  trainer.py
                |   Training          |
                |   (LambdaMART)      |
                +----------+----------+
                           |
                           v
                +----------+----------+
                |   Evaluation        |
                |  - NDCG@5, NDCG@10 |
                |  - MRR              |
                |  - Error Analysis   |
                +---------------------+
```

---

## Key Design Decisions

### Why LambdaMART?

LambdaMART (Lambda Gradient-boosted Multiple Additive Regression Trees) is the industry standard for production ranking systems and has been the backbone of major search engines and recommendation systems. The key advantages:

- **Direct NDCG optimization.** LambdaMART defines implicit lambda gradients that directly optimize ranking quality metrics like NDCG. This avoids the metric mismatch inherent in pointwise (regression/classification) or pairwise (e.g., RankNet) approaches.
- **Handling of heterogeneous features.** Gradient-boosted trees naturally handle mixed feature types (continuous, categorical, ordinal) without normalization or encoding. This is critical for the Expedia dataset, which mixes price floats, star ratings, boolean flags, and count features.
- **Production efficiency.** LightGBM's histogram-based splitting and leaf-wise growth make inference fast enough for online serving. A trained model can score thousands of candidates in milliseconds.

### Why Inverse Propensity Scoring (IPS)?

Click data is observational, not experimental. Users tend to examine top positions more frequently, creating a feedback loop where already highly-ranked items accumulate more clicks regardless of true relevance. IPS corrects this by:

- Estimating position-dependent examination probabilities (propensities) from items shown in randomized positions (`random_bool = 1`).
- Weighting each training example inversely to its propensity, so clicks from low-examination positions count more than clicks from high-examination positions.
- Clipping extreme weights at the 95th percentile to control variance.

This approach follows the unbiased LTR framework of Joachims et al. (2017) and Wang et al. (2018).

---

## Module Dependency Graph

```
hotel_ranker/
    |
    +-- config.py  <------- (imported by all modules)
    |
    +-- data/
    |     +-- schema.py  <-- (column constants, used by preprocessing & features)
    |     +-- acquisition.py
    |     +-- preprocessing.py  --> schema.py
    |     +-- splitting.py
    |
    +-- features/
    |     +-- raw_features.py       --> schema.py
    |     +-- match_features.py     --> schema.py
    |     +-- historical_features.py --> config.py (priors)
    |     +-- feature_pipeline.py   --> all feature modules
    |
    +-- bias/
    |     +-- propensity.py  --> config.py (IPS clip percentile)
    |
    +-- training/
    |     +-- label_engineering.py  --> config.py (weights, max grade)
    |     +-- trainer.py           --> config.py (LGBM params, early stopping)
    |
    +-- evaluation/
    |     +-- metrics.py           --> config.py (NDCG cutoffs)
    |     +-- error_analysis.py    --> metrics.py
    |
    +-- pipeline.py  --> all modules (end-to-end orchestration)
```

### Dependency Rules

- **`config.py`** is a leaf dependency: it imports nothing from the project and is imported by nearly everything.
- **`data/schema.py`** defines column name constants and dtype maps. It is imported by preprocessing and feature modules to avoid hardcoded strings.
- **Feature modules** depend on schema and config but not on each other. They can be developed, tested, and toggled independently.
- **`pipeline.py`** is the composition root. It wires all modules together and is the only module that depends on everything.
- **No circular dependencies.** The dependency graph is a DAG.
