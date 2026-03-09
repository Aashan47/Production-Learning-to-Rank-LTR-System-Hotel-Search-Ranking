"""Tests for hotel_ranker.features (raw_features and match_features).

Validates feature engineering logic using synthetic DataFrames.
"""

import numpy as np
import pandas as pd
import pytest

from hotel_ranker.features.raw_features import build_raw_features
from hotel_ranker.features.match_features import build_match_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_df():
    """Minimal DataFrame with all columns needed by build_raw_features."""
    return pd.DataFrame({
        "srch_id": [1, 1, 1, 2, 2],
        "prop_id": [10, 20, 30, 40, 50],
        "price_usd": [100.0, 200.0, 150.0, 300.0, 250.0],
        "prop_starrating": [3, 4, 5, 3, 4],
        "prop_review_score": [4.0, 3.5, 5.0, 0.0, 4.5],
        "prop_brand_bool": [1, 0, 1, 0, 1],
        "prop_location_score1": [2.0, 3.0, 4.0, 5.0, 1.0],
        "prop_location_score2": [3.0, 4.0, 5.0, 2.0, 3.5],
        "prop_log_historical_price": [4.5, 5.0, 4.8, 5.5, 5.2],
        "promotion_flag": [0, 1, 0, 1, 0],
        "srch_length_of_stay": [2, 3, 1, 5, 2],
        "srch_booking_window": [10, 20, 5, 30, 15],
        "srch_adults_count": [2, 1, 2, 3, 2],
        "srch_children_count": [1, 0, 2, 1, 0],
        "srch_room_count": [1, 1, 2, 1, 1],
        "srch_saturday_night_bool": [1, 0, 1, 0, 1],
        "visitor_hist_starrating": [3.0, -1.0, 4.0, -1.0, 2.0],
        "visitor_hist_adr_usd": [120.0, -1.0, 80.0, -1.0, 200.0],
        # Derived columns that preprocessing would have created
        "price_log": np.log1p([100.0, 200.0, 150.0, 300.0, 250.0]),
        "price_rank_in_query": [1.0, 3.0, 2.0, 2.0, 1.0],
        # Competitor columns
        "comp1_rate": [1.0, 0.0, -1.0, 1.0, 0.0],
        "comp2_rate": [0.0, -1.0, 1.0, 0.0, 1.0],
        "comp3_rate": [1.0, 1.0, 0.0, -1.0, 0.0],
        "comp1_inv": [1.0, 0.0, 0.0, 1.0, 1.0],
        "comp2_inv": [0.0, 0.0, 1.0, 0.0, 0.0],
        "comp3_inv": [0.0, 1.0, 0.0, 0.0, 1.0],
    })


@pytest.fixture
def single_query_df():
    """Single-query DataFrame for simpler match-feature assertions."""
    return pd.DataFrame({
        "srch_id": [1, 1, 1, 1],
        "price_usd": [100.0, 200.0, 300.0, 400.0],
        "prop_starrating": [3, 4, 5, 2],
        "prop_review_score": [4.0, 3.5, 5.0, 2.0],
        "prop_location_score1": [2.0, 3.0, 4.0, 1.0],
        "prop_location_score2": [3.0, 4.0, 5.0, 2.0],
        "visitor_hist_starrating": [3.0, 3.0, 3.0, 3.0],
        "visitor_hist_adr_usd": [150.0, 150.0, 150.0, 150.0],
        "srch_length_of_stay": [2, 2, 2, 2],
        "srch_adults_count": [2, 2, 2, 2],
        "srch_children_count": [1, 1, 1, 1],
        "comp1_rate": [1.0, 0.0, -1.0, 1.0],
        "comp2_rate": [1.0, 1.0, 0.0, -1.0],
        "comp1_inv": [1.0, 0.0, 0.0, 1.0],
        "comp2_inv": [0.0, 1.0, 0.0, 0.0],
    })


# ---------------------------------------------------------------------------
# Tests for build_raw_features
# ---------------------------------------------------------------------------

class TestBuildRawFeatures:
    """Tests for the build_raw_features function."""

    def test_output_rows_match_input(self, base_df):
        """Output should have the same number of rows as input."""
        result = build_raw_features(base_df)
        assert len(result) == len(base_df)

    def test_no_nan_in_output(self, base_df):
        """Output should not contain NaN values (all fills applied)."""
        result = build_raw_features(base_df)
        total_nan = result.isna().sum().sum()
        assert total_nan == 0, f"Found {total_nan} NaN values in raw features output"

    def test_price_per_night_calculation(self, base_df):
        """price_per_night = price_usd / srch_length_of_stay."""
        result = build_raw_features(base_df)
        assert "price_per_night" in result.columns

        expected = base_df["price_usd"] / base_df["srch_length_of_stay"].clip(lower=1)
        np.testing.assert_array_almost_equal(result["price_per_night"], expected)

    def test_price_per_night_with_zero_stay(self):
        """Length of stay clipped to 1 should prevent division by zero."""
        df = pd.DataFrame({
            "price_usd": [100.0],
            "prop_starrating": [3],
            "prop_review_score": [4.0],
            "prop_brand_bool": [1],
            "prop_location_score1": [2.0],
            "prop_location_score2": [3.0],
            "prop_log_historical_price": [4.5],
            "promotion_flag": [0],
            "srch_length_of_stay": [0],  # edge case
            "srch_booking_window": [10],
            "srch_adults_count": [2],
            "srch_children_count": [1],
            "srch_room_count": [1],
            "srch_saturday_night_bool": [1],
            "visitor_hist_starrating": [3.0],
            "visitor_hist_adr_usd": [120.0],
        })
        result = build_raw_features(df)
        # With clip(lower=1), price_per_night should equal price_usd
        assert result.iloc[0]["price_per_night"] == pytest.approx(100.0)

    def test_total_guests_calculation(self, base_df):
        """total_guests = srch_adults_count + srch_children_count."""
        result = build_raw_features(base_df)
        assert "total_guests" in result.columns

        expected = base_df["srch_adults_count"] + base_df["srch_children_count"]
        np.testing.assert_array_equal(result["total_guests"], expected)

    def test_passthrough_columns_present(self, base_df):
        """Key passthrough columns should appear in output."""
        result = build_raw_features(base_df)

        expected_cols = [
            "prop_starrating", "prop_review_score", "price_usd",
            "promotion_flag", "srch_length_of_stay",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Passthrough column '{col}' missing"

    def test_price_log_and_rank_carried_through(self, base_df):
        """Derived columns from preprocessing should be included if present."""
        result = build_raw_features(base_df)

        assert "price_log" in result.columns
        assert "price_rank_in_query" in result.columns
        np.testing.assert_array_almost_equal(result["price_log"], base_df["price_log"])

    def test_star_review_ratio(self, base_df):
        """star_review_ratio = starrating / review_score, with 0-review -> 0."""
        result = build_raw_features(base_df)
        assert "star_review_ratio" in result.columns

        # Row 3 has review_score=0, so ratio should be 0
        assert result.iloc[3]["star_review_ratio"] == pytest.approx(0.0)

        # Row 0: star=3, review=4.0 -> 0.75
        assert result.iloc[0]["star_review_ratio"] == pytest.approx(3.0 / 4.0)


# ---------------------------------------------------------------------------
# Tests for build_match_features
# ---------------------------------------------------------------------------

class TestBuildMatchFeatures:
    """Tests for the build_match_features function."""

    def test_output_rows_match_input(self, base_df):
        """Output should have the same number of rows as input."""
        result = build_match_features(base_df)
        assert len(result) == len(base_df)

    def test_no_nan_in_output(self, base_df):
        """Match features should not contain NaN values."""
        result = build_match_features(base_df)
        total_nan = result.isna().sum().sum()
        assert total_nan == 0, f"Found {total_nan} NaN values in match features output"

    def test_price_diff_from_query_mean_sums_to_zero(self, single_query_df):
        """Within a single query, price_diff_from_query_mean should sum to ~0."""
        result = build_match_features(single_query_df)

        diff_sum = result["price_diff_from_query_mean"].sum()
        assert abs(diff_sum) < 1e-6, (
            f"price_diff_from_query_mean should sum to ~0 within query, got {diff_sum}"
        )

    def test_price_diff_sums_near_zero_per_query(self, base_df):
        """price_diff_from_query_mean should sum to ~0 within each query group."""
        result = build_match_features(base_df)
        result["srch_id"] = base_df["srch_id"]

        for _, group in result.groupby("srch_id"):
            group_sum = group["price_diff_from_query_mean"].sum()
            assert abs(group_sum) < 1e-6, (
                f"price_diff_from_query_mean does not sum to ~0 for a query: {group_sum}"
            )

    def test_competitor_rate_advantage_counts(self, single_query_df):
        """competitor_rate_advantage should count how many comp rates are > 0."""
        result = build_match_features(single_query_df)
        assert "competitor_rate_advantage" in result.columns

        # Row 0: comp1_rate=1 (>0), comp2_rate=1 (>0) -> advantage = 2
        assert result.iloc[0]["competitor_rate_advantage"] == 2

        # Row 2: comp1_rate=-1, comp2_rate=0 -> advantage = 0
        assert result.iloc[2]["competitor_rate_advantage"] == 0

    def test_competitor_rate_disadvantage_counts(self, single_query_df):
        """competitor_rate_disadvantage should count how many comp rates are < 0."""
        result = build_match_features(single_query_df)
        assert "competitor_rate_disadvantage" in result.columns

        # Row 2: comp1_rate=-1 (<0) -> disadvantage = 1
        assert result.iloc[2]["competitor_rate_disadvantage"] == 1

        # Row 0: no comp rate < 0 -> disadvantage = 0
        assert result.iloc[0]["competitor_rate_disadvantage"] == 0

    def test_competitor_inv_advantage_counts(self, single_query_df):
        """competitor_inv_advantage should count how many comp_inv == 1."""
        result = build_match_features(single_query_df)
        assert "competitor_inv_advantage" in result.columns

        # Row 0: comp1_inv=1, comp2_inv=0 -> inv advantage = 1
        assert result.iloc[0]["competitor_inv_advantage"] == 1

        # Row 1: comp1_inv=0, comp2_inv=1 -> inv advantage = 1
        assert result.iloc[1]["competitor_inv_advantage"] == 1

    def test_location_score_composite(self, single_query_df):
        """location_score_composite = 0.6 * loc1 + 0.4 * loc2."""
        result = build_match_features(single_query_df)
        assert "location_score_composite" in result.columns

        expected = (
            0.6 * single_query_df["prop_location_score1"]
            + 0.4 * single_query_df["prop_location_score2"]
        )
        np.testing.assert_array_almost_equal(
            result["location_score_composite"], expected
        )

    def test_expected_match_feature_columns(self, base_df):
        """All expected match feature columns should be present."""
        result = build_match_features(base_df)
        expected_cols = [
            "price_diff_from_query_mean",
            "price_ratio_to_query_median",
            "price_zscore_in_query",
            "star_diff_from_query_mean",
            "star_match_visitor_pref",
            "location_score_composite",
            "competitor_rate_advantage",
            "competitor_rate_disadvantage",
            "competitor_rate_mean",
            "competitor_inv_advantage",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Expected column '{col}' not found"
