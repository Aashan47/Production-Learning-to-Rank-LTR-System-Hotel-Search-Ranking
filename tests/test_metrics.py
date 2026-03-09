"""Tests for hotel_ranker.evaluation.metrics module.

Validates NDCG@k, MRR, and per-query NDCG using hand-crafted synthetic data.
"""

import numpy as np
import pytest

from hotel_ranker.evaluation.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    per_query_ndcg,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def perfect_ranking():
    """Predictions that perfectly mirror the true relevance ordering.

    Two queries, each with 4 items. Predicted scores produce the ideal
    ranking (highest relevance gets highest predicted score).
    """
    y_true = np.array([3, 2, 1, 0, 4, 3, 1, 0])
    y_pred = np.array([3.0, 2.0, 1.0, 0.0, 4.0, 3.0, 1.0, 0.0])
    groups = np.array([4, 4])
    return y_true, y_pred, groups


@pytest.fixture
def reversed_ranking():
    """Predictions that are the exact reverse of the ideal ordering.

    The model ranks the least relevant item highest.
    """
    y_true = np.array([3, 2, 1, 0, 4, 3, 1, 0])
    y_pred = np.array([0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 3.0, 4.0])
    groups = np.array([4, 4])
    return y_true, y_pred, groups


@pytest.fixture
def mrr_first_relevant():
    """Single query where the first predicted item is relevant."""
    y_true = np.array([2, 0, 0, 0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])  # item 0 ranked first
    groups = np.array([4])
    return y_true, y_pred, groups


@pytest.fixture
def mrr_second_relevant():
    """Single query where the relevant item is predicted second."""
    y_true = np.array([0, 2, 0, 0])
    y_pred = np.array([3.0, 4.0, 2.0, 1.0])  # item 1 ranked first, item 0 second
    # After sorting by pred desc: order = [1,0,2,3] -> true = [2,0,0,0]
    # First relevant at position 1 (0-indexed) -> RR = 1/1 = 1.0
    # Wait: item 1 has pred=4.0 (highest), so it's ranked first.
    # true[1]=2 >= 1 -> first relevant at rank 1 -> RR = 1.0
    # Need to adjust so the relevant item is second.
    groups = np.array([4])
    return y_true, y_pred, groups


@pytest.fixture
def mrr_second_relevant_corrected():
    """Single query where the first relevant item appears at rank 2."""
    y_true = np.array([0, 2, 0, 0])
    # pred: item 0 has highest score -> ranked first (true=0, not relevant)
    #        item 1 has second highest -> ranked second (true=2, relevant)
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])
    groups = np.array([4])
    return y_true, y_pred, groups


@pytest.fixture
def multi_group_data():
    """Three queries of varying sizes for group-based testing."""
    y_true = np.array([3, 1, 0, 2, 0, 4, 2])
    y_pred = np.array([3.0, 1.0, 0.0, 2.0, 0.0, 4.0, 2.0])
    groups = np.array([3, 2, 2])
    return y_true, y_pred, groups


# ---------------------------------------------------------------------------
# Tests for ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:
    """Tests for the ndcg_at_k function."""

    def test_perfect_ranking_gives_ndcg_one(self, perfect_ranking):
        """A perfect ranking should produce NDCG@k = 1.0."""
        y_true, y_pred, groups = perfect_ranking
        score = ndcg_at_k(y_true, y_pred, groups, k=4)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_reversed_ranking_below_one(self, reversed_ranking):
        """A reversed (worst) ranking should produce NDCG < 1.0."""
        y_true, y_pred, groups = reversed_ranking
        score = ndcg_at_k(y_true, y_pred, groups, k=4)
        assert score < 1.0

    def test_reversed_ranking_non_negative(self, reversed_ranking):
        """NDCG should never be negative."""
        y_true, y_pred, groups = reversed_ranking
        score = ndcg_at_k(y_true, y_pred, groups, k=4)
        assert score >= 0.0

    def test_ndcg_at_1_perfect(self):
        """NDCG@1 should be 1.0 if the top-ranked item has the highest relevance."""
        y_true = np.array([3, 2, 1, 0])
        y_pred = np.array([10.0, 5.0, 2.0, 0.0])
        groups = np.array([4])
        score = ndcg_at_k(y_true, y_pred, groups, k=1)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_different_k_values(self, perfect_ranking):
        """NDCG should be valid for different k cutoffs."""
        y_true, y_pred, groups = perfect_ranking
        for k in [1, 2, 3, 4]:
            score = ndcg_at_k(y_true, y_pred, groups, k=k)
            assert 0.0 <= score <= 1.0, f"NDCG@{k} = {score} out of [0, 1] range"

    def test_single_item_group_skipped(self):
        """Groups with size < 2 should be skipped (no meaningful ranking)."""
        y_true = np.array([3, 2, 1, 5])
        y_pred = np.array([3.0, 2.0, 1.0, 5.0])
        groups = np.array([3, 1])  # second group has only 1 item
        # Only the first group contributes; it's a perfect ranking -> 1.0
        score = ndcg_at_k(y_true, y_pred, groups, k=3)
        assert score == pytest.approx(1.0, abs=1e-6)

    def test_all_single_item_groups_returns_zero(self):
        """If all groups have size 1, return 0.0 (no valid queries)."""
        y_true = np.array([3, 2])
        y_pred = np.array([3.0, 2.0])
        groups = np.array([1, 1])
        score = ndcg_at_k(y_true, y_pred, groups, k=1)
        assert score == pytest.approx(0.0)

    def test_multiple_queries_averaged(self, multi_group_data):
        """NDCG should be averaged across all valid queries."""
        y_true, y_pred, groups = multi_group_data
        score = ndcg_at_k(y_true, y_pred, groups, k=3)
        # All queries have perfect ranking, so average should be 1.0
        assert score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests for mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestMeanReciprocalRank:
    """Tests for the mean_reciprocal_rank function."""

    def test_first_item_relevant_gives_mrr_one(self, mrr_first_relevant):
        """When the highest-predicted item is relevant, MRR = 1.0."""
        y_true, y_pred, groups = mrr_first_relevant
        mrr = mean_reciprocal_rank(y_true, y_pred, groups, relevance_threshold=1)
        assert mrr == pytest.approx(1.0)

    def test_second_item_relevant_gives_mrr_half(self, mrr_second_relevant_corrected):
        """When the first relevant item is at predicted rank 2, MRR = 0.5."""
        y_true, y_pred, groups = mrr_second_relevant_corrected
        mrr = mean_reciprocal_rank(y_true, y_pred, groups, relevance_threshold=1)
        assert mrr == pytest.approx(0.5)

    def test_no_relevant_items_gives_zero(self):
        """When no items meet the relevance threshold, RR = 0 for that query."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([3.0, 2.0, 1.0])
        groups = np.array([3])
        mrr = mean_reciprocal_rank(y_true, y_pred, groups, relevance_threshold=1)
        assert mrr == pytest.approx(0.0)

    def test_mrr_averaged_across_queries(self):
        """MRR should be the mean of per-query reciprocal ranks."""
        # Query 1: relevant item at rank 1 -> RR = 1.0
        # Query 2: relevant item at rank 2 -> RR = 0.5
        y_true = np.array([2, 0, 0, 0, 2, 0])
        y_pred = np.array([3.0, 2.0, 1.0, 2.0, 3.0, 1.0])
        # Query 1: sorted by pred desc -> [item0(3.0), item1(2.0), item2(1.0)]
        #          true = [2, 0, 0], first relevant at rank 1 -> RR = 1.0
        # Query 2: sorted by pred desc -> [item1(3.0), item0(2.0), item2(1.0)]
        #          true = [0, 2, 0] -> sorted true = [0, 2, 0]
        #          Wait: items are [idx3, idx4, idx5] with pred [2.0, 3.0, 1.0]
        #          sorted order: idx4(3.0), idx3(2.0), idx5(1.0)
        #          true sorted: [2, 0, 0], first relevant at rank 1 -> RR = 1.0

        # Let me fix to actually produce RR=0.5 for query 2
        y_true2 = np.array([2, 0, 0, 0, 2, 0])
        y_pred2 = np.array([3.0, 2.0, 1.0, 3.0, 2.0, 1.0])
        # Query 2: items [idx3,idx4,idx5] pred [3.0, 2.0, 1.0]
        # sorted: idx3(3.0), idx4(2.0), idx5(1.0) -> true [0, 2, 0]
        # first relevant at rank 2 -> RR = 0.5
        groups2 = np.array([3, 3])

        mrr = mean_reciprocal_rank(y_true2, y_pred2, groups2, relevance_threshold=1)
        expected = (1.0 + 0.5) / 2.0  # 0.75
        assert mrr == pytest.approx(expected)

    def test_mrr_with_custom_threshold(self):
        """Only items with true >= threshold are considered relevant."""
        y_true = np.array([1, 3, 0])
        y_pred = np.array([3.0, 2.0, 1.0])
        groups = np.array([3])

        # threshold=2: only item with true=3 is relevant, ranked second
        mrr = mean_reciprocal_rank(y_true, y_pred, groups, relevance_threshold=2)
        assert mrr == pytest.approx(0.5)

        # threshold=1: item with true=1 is also relevant, ranked first
        mrr = mean_reciprocal_rank(y_true, y_pred, groups, relevance_threshold=1)
        assert mrr == pytest.approx(1.0)

    def test_single_item_group(self):
        """A single-item group should still be handled without error."""
        y_true = np.array([2])
        y_pred = np.array([1.0])
        groups = np.array([1])
        mrr = mean_reciprocal_rank(y_true, y_pred, groups, relevance_threshold=1)
        assert mrr == pytest.approx(1.0)

    def test_single_item_group_irrelevant(self):
        """A single irrelevant item should produce RR = 0."""
        y_true = np.array([0])
        y_pred = np.array([1.0])
        groups = np.array([1])
        mrr = mean_reciprocal_rank(y_true, y_pred, groups, relevance_threshold=1)
        assert mrr == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests for per_query_ndcg
# ---------------------------------------------------------------------------

class TestPerQueryNdcg:
    """Tests for the per_query_ndcg function."""

    def test_returns_correct_number_of_scores(self, multi_group_data):
        """Should return one NDCG score per query (group)."""
        y_true, y_pred, groups = multi_group_data
        scores = per_query_ndcg(y_true, y_pred, groups, k=5)
        assert len(scores) == len(groups)

    def test_perfect_ranking_per_query(self, perfect_ranking):
        """Each query with perfect ranking should have NDCG = 1.0."""
        y_true, y_pred, groups = perfect_ranking
        scores = per_query_ndcg(y_true, y_pred, groups, k=4)
        for i, s in enumerate(scores):
            assert s == pytest.approx(1.0, abs=1e-6), (
                f"Query {i} should have NDCG=1.0, got {s}"
            )

    def test_single_item_group_returns_nan(self):
        """Groups with size < 2 should return NaN in per-query scores."""
        y_true = np.array([3, 2, 1, 5])
        y_pred = np.array([3.0, 2.0, 1.0, 5.0])
        groups = np.array([3, 1])
        scores = per_query_ndcg(y_true, y_pred, groups, k=3)

        assert len(scores) == 2
        assert scores[0] == pytest.approx(1.0, abs=1e-6)
        assert np.isnan(scores[1])  # single-item group

    def test_mixed_quality_queries(self):
        """Different queries should produce different NDCG scores."""
        # Query 1: perfect ranking
        # Query 2: reversed ranking
        y_true = np.array([3, 2, 1, 0, 0, 1, 2, 3])
        y_pred = np.array([3.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0])
        groups = np.array([4, 4])
        scores = per_query_ndcg(y_true, y_pred, groups, k=4)

        assert scores[0] == pytest.approx(1.0, abs=1e-6)  # perfect
        assert scores[1] < 1.0  # reversed

    def test_per_query_scores_bounded(self, multi_group_data):
        """All per-query NDCG scores should be in [0, 1] (excluding NaN)."""
        y_true, y_pred, groups = multi_group_data
        scores = per_query_ndcg(y_true, y_pred, groups, k=5)

        valid_scores = scores[~np.isnan(scores)]
        assert (valid_scores >= 0.0).all()
        assert (valid_scores <= 1.0 + 1e-9).all()

    def test_default_k_is_five(self):
        """When k is not specified, it should default to 5."""
        y_true = np.array([5, 4, 3, 2, 1, 0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])
        groups = np.array([6])

        # Call with default k
        scores_default = per_query_ndcg(y_true, y_pred, groups)
        # Call with explicit k=5
        scores_k5 = per_query_ndcg(y_true, y_pred, groups, k=5)

        np.testing.assert_array_almost_equal(scores_default, scores_k5)
