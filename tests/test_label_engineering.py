"""Tests for hotel_ranker.training.label_engineering module.

Validates composite label computation and discretization into relevance grades.
"""

import numpy as np
import pandas as pd
import pytest

from hotel_ranker.training.label_engineering import (
    compute_composite_label,
    discretize_labels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def interaction_df():
    """DataFrame with various click/booking combinations."""
    return pd.DataFrame({
        "srch_id": [1, 1, 1, 1, 2, 2, 2],
        "click_bool": [0, 1, 1, 0, 1, 0, 1],
        "booking_bool": [0, 0, 1, 0, 1, 0, 0],
        "price_usd": [100.0, 150.0, 200.0, 250.0, 300.0, 50.0, 120.0],
    })


@pytest.fixture
def multi_query_df():
    """DataFrame with multiple queries for discretization testing."""
    return pd.DataFrame({
        "srch_id": [1, 1, 1, 1, 2, 2, 2, 2],
        "click_bool": [0, 0, 1, 1, 0, 1, 0, 1],
        "booking_bool": [0, 0, 0, 1, 0, 0, 0, 1],
        "price_usd": [100.0, 120.0, 150.0, 200.0, 80.0, 90.0, 110.0, 250.0],
    })


# ---------------------------------------------------------------------------
# Tests for compute_composite_label
# ---------------------------------------------------------------------------

class TestComputeCompositeLabel:
    """Tests for the compute_composite_label function."""

    def test_no_interaction_gets_zero(self, interaction_df):
        """Row with click=0 and booking=0 should have score 0."""
        scores = compute_composite_label(interaction_df)

        # Row 0: click=0, booking=0
        assert scores.iloc[0] == pytest.approx(0.0)

        # Row 3: click=0, booking=0
        assert scores.iloc[3] == pytest.approx(0.0)

    def test_click_only_gets_w_click(self, interaction_df):
        """Row with click=1 and booking=0 should get score = w_click * 1."""
        w_click = 1.0
        scores = compute_composite_label(interaction_df, w_click=w_click)

        # Row 1: click=1, booking=0
        assert scores.iloc[1] == pytest.approx(w_click)

    def test_click_only_custom_weight(self, interaction_df):
        """Click-only score with custom w_click."""
        w_click = 2.5
        scores = compute_composite_label(interaction_df, w_click=w_click, w_book=5.0)

        # Row 1: click=1, booking=0 -> 2.5 * 1 + 5.0 * 0 * ... = 2.5
        assert scores.iloc[1] == pytest.approx(w_click)

    def test_booking_gets_click_plus_book_component(self, interaction_df):
        """Booking row: score = w_click + w_book * (0.5 + 0.5 * price_pct)."""
        w_click = 1.0
        w_book = 5.0
        scores = compute_composite_label(interaction_df, w_click=w_click, w_book=w_book)

        # Row 2: click=1, booking=1, srch_id=1, price=200.0
        # srch_id=1 prices: [100, 150, 200, 250] -> price_pct of 200 = 3/4 = 0.75
        price_pct = 0.75
        expected = w_click + w_book * (0.5 + 0.5 * price_pct)
        assert scores.iloc[2] == pytest.approx(expected)

    def test_booking_without_click_flag(self):
        """Even if click=0 but booking=1, booking component still applies."""
        df = pd.DataFrame({
            "click_bool": [0],
            "booking_bool": [1],
            "price_usd": [100.0],
        })
        scores = compute_composite_label(df, w_click=1.0, w_book=5.0)
        # Single item: price_pct=1.0 -> booking_value = 5.0 * (0.5 + 0.5*1.0) = 5.0
        expected = 5.0
        assert scores.iloc[0] == pytest.approx(expected)

    def test_scores_are_non_negative(self, interaction_df):
        """All composite scores should be >= 0."""
        scores = compute_composite_label(interaction_df)
        assert (scores >= 0).all()

    def test_booking_score_higher_than_click_only(self, interaction_df):
        """A booked item should have a higher score than a clicked-only item."""
        scores = compute_composite_label(interaction_df)

        click_only_score = scores.iloc[1]  # click=1, book=0
        booked_score = scores.iloc[2]  # click=1, book=1
        assert booked_score > click_only_score

    def test_output_length_matches_input(self, interaction_df):
        """Output Series should have same length as input DataFrame."""
        scores = compute_composite_label(interaction_df)
        assert len(scores) == len(interaction_df)

    def test_default_weights_match_config(self, interaction_df):
        """Default weights should match W_CLICK and W_BOOK from config."""
        from hotel_ranker.config import W_CLICK, W_BOOK
        scores_default = compute_composite_label(interaction_df)
        scores_explicit = compute_composite_label(interaction_df, w_click=W_CLICK, w_book=W_BOOK)
        np.testing.assert_array_almost_equal(scores_default, scores_explicit)


# ---------------------------------------------------------------------------
# Tests for discretize_labels
# ---------------------------------------------------------------------------

class TestDiscretizeLabels:
    """Tests for the discretize_labels function."""

    def test_output_range(self, multi_query_df):
        """Discretized grades should be in [0, max_grade]."""
        scores = compute_composite_label(multi_query_df)
        query_ids = multi_query_df["srch_id"]

        max_grade = 4
        grades = discretize_labels(scores, query_ids, max_grade=max_grade)

        assert grades.min() >= 0
        assert grades.max() <= max_grade

    def test_zero_score_items_get_grade_zero(self, multi_query_df):
        """Items with composite score 0 should always receive grade 0."""
        scores = compute_composite_label(multi_query_df)
        query_ids = multi_query_df["srch_id"]

        grades = discretize_labels(scores, query_ids)

        zero_score_mask = scores == 0
        assert (grades[zero_score_mask.values] == 0).all(), (
            "Some zero-score items did not receive grade 0"
        )

    def test_within_query_normalization(self):
        """The highest-scoring item in a query should get grade == max_grade."""
        df = pd.DataFrame({
            "srch_id": [1, 1, 1],
            "click_bool": [0, 1, 1],
            "booking_bool": [0, 0, 1],
            "price_usd": [100.0, 100.0, 100.0],
        })
        scores = compute_composite_label(df)
        query_ids = df["srch_id"]

        max_grade = 4
        grades = discretize_labels(scores, query_ids, max_grade=max_grade)

        # The booked item should get the highest grade
        assert grades[2] == max_grade

    def test_output_dtype_is_integer(self, multi_query_df):
        """Grades should be integer values."""
        scores = compute_composite_label(multi_query_df)
        query_ids = multi_query_df["srch_id"]

        grades = discretize_labels(scores, query_ids)
        assert np.issubdtype(grades.dtype, np.integer)

    def test_output_length_matches_input(self, multi_query_df):
        """Output array length should match input scores length."""
        scores = compute_composite_label(multi_query_df)
        query_ids = multi_query_df["srch_id"]

        grades = discretize_labels(scores, query_ids)
        assert len(grades) == len(scores)

    def test_custom_max_grade(self, multi_query_df):
        """Different max_grade values should be respected."""
        scores = compute_composite_label(multi_query_df)
        query_ids = multi_query_df["srch_id"]

        for max_g in [2, 3, 5, 10]:
            grades = discretize_labels(scores, query_ids, max_grade=max_g)
            assert grades.max() <= max_g
            assert grades.min() >= 0

    def test_constant_query_all_same_score(self):
        """Query where all items have the same score -> all get same grade."""
        scores = pd.Series([0.0, 0.0, 0.0])
        query_ids = pd.Series([1, 1, 1])

        grades = discretize_labels(scores, query_ids, max_grade=4)
        # All zero scores -> all grade 0
        assert (grades == 0).all()

    def test_single_item_query(self):
        """A query with a single item should not cause errors."""
        scores = pd.Series([5.0])
        query_ids = pd.Series([1])

        grades = discretize_labels(scores, query_ids, max_grade=4)
        assert len(grades) == 1
        assert 0 <= grades[0] <= 4
