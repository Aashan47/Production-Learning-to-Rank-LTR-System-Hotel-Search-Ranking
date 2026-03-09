"""Tests for hotel_ranker.data.preprocessing module.

Validates missing-value imputation strategies and derived column creation
using synthetic DataFrames (no external data dependencies).
"""

import numpy as np
import pandas as pd
import pytest

from hotel_ranker.data.preprocessing import add_derived_columns, handle_missing_values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_with_missing():
    """DataFrame with NaN values in columns that handle_missing_values targets."""
    return pd.DataFrame({
        "srch_id": [1, 1, 1, 2, 2],
        "prop_id": [10, 20, 30, 40, 50],
        "price_usd": [100.0, 200.0, 150.0, 300.0, 250.0],
        "prop_review_score": [4.0, np.nan, 3.5, np.nan, 5.0],
        "prop_location_score2": [3.0, np.nan, 5.0, 4.0, np.nan],
        "visitor_hist_starrating": [3.0, np.nan, 4.0, np.nan, 2.0],
        "visitor_hist_adr_usd": [120.0, np.nan, 80.0, np.nan, 200.0],
        "comp1_rate": [1.0, np.nan, -1.0, np.nan, 0.0],
        "comp2_rate": [np.nan, np.nan, 1.0, 0.0, np.nan],
        "comp1_inv": [1.0, np.nan, 0.0, np.nan, 1.0],
        "comp2_inv": [np.nan, 0.0, np.nan, 1.0, np.nan],
        "comp1_rate_percent_diff": [5.0, np.nan, -3.0, np.nan, 2.0],
        "prop_starrating": [3, 4, 5, 3, 4],
        "prop_location_score1": [2.0, 3.0, np.nan, 4.0, 1.0],
    })


@pytest.fixture
def df_for_derived():
    """DataFrame suitable for testing add_derived_columns."""
    return pd.DataFrame({
        "srch_id": [1, 1, 1, 2, 2],
        "price_usd": [100.0, 300.0, 200.0, 50.0, 150.0],
    })


# ---------------------------------------------------------------------------
# Tests for handle_missing_values
# ---------------------------------------------------------------------------

class TestHandleMissingValues:
    """Tests for the handle_missing_values function."""

    def test_competitor_columns_filled_with_zero(self, df_with_missing):
        """Competitor rate/inv/percent_diff NaNs should be filled with 0."""
        result = handle_missing_values(df_with_missing)

        comp_cols = [
            "comp1_rate", "comp2_rate",
            "comp1_inv", "comp2_inv",
            "comp1_rate_percent_diff",
        ]
        for col in comp_cols:
            assert result[col].isna().sum() == 0, f"{col} still has NaN"
            # Check that originally-NaN positions are now 0
            original_nan_mask = df_with_missing[col].isna()
            assert (result.loc[original_nan_mask, col] == 0).all(), (
                f"{col}: NaN positions were not filled with 0"
            )

    def test_review_score_filled_with_zero(self, df_with_missing):
        """prop_review_score NaN should be filled with 0 (unrated property)."""
        result = handle_missing_values(df_with_missing)

        assert result["prop_review_score"].isna().sum() == 0
        # Rows that were NaN should now be 0
        nan_mask = df_with_missing["prop_review_score"].isna()
        assert (result.loc[nan_mask, "prop_review_score"] == 0).all()

    def test_visitor_hist_filled_with_negative_one(self, df_with_missing):
        """visitor_hist_starrating and visitor_hist_adr_usd NaN -> -1."""
        result = handle_missing_values(df_with_missing)

        for col in ["visitor_hist_starrating", "visitor_hist_adr_usd"]:
            assert result[col].isna().sum() == 0, f"{col} still has NaN"
            nan_mask = df_with_missing[col].isna()
            assert (result.loc[nan_mask, col] == -1).all(), (
                f"{col}: NaN positions were not filled with -1"
            )

    def test_location_score2_filled_with_median(self, df_with_missing):
        """prop_location_score2 NaN should be filled with the column median."""
        original_median = df_with_missing["prop_location_score2"].median()
        result = handle_missing_values(df_with_missing)

        assert result["prop_location_score2"].isna().sum() == 0
        nan_mask = df_with_missing["prop_location_score2"].isna()
        assert np.allclose(result.loc[nan_mask, "prop_location_score2"], original_median)

    def test_no_nan_remains_after_imputation(self, df_with_missing):
        """After handle_missing_values, no NaN should remain in numeric cols."""
        result = handle_missing_values(df_with_missing)

        numeric_cols = result.select_dtypes(include=[np.number]).columns
        total_nan = result[numeric_cols].isna().sum().sum()
        assert total_nan == 0, (
            f"Found {total_nan} remaining NaN values in numeric columns"
        )

    def test_non_nan_values_preserved(self, df_with_missing):
        """Values that were not NaN should remain unchanged."""
        result = handle_missing_values(df_with_missing)

        # comp1_rate row 0 was 1.0, should still be 1.0
        assert result.loc[0, "comp1_rate"] == 1.0
        # prop_review_score row 0 was 4.0
        assert result.loc[0, "prop_review_score"] == 4.0
        # visitor_hist_starrating row 0 was 3.0
        assert result.loc[0, "visitor_hist_starrating"] == 3.0


# ---------------------------------------------------------------------------
# Tests for add_derived_columns
# ---------------------------------------------------------------------------

class TestAddDerivedColumns:
    """Tests for the add_derived_columns function."""

    def test_price_rank_in_query_correct(self, df_for_derived):
        """price_rank_in_query should reflect rank of price within each srch_id."""
        result = add_derived_columns(df_for_derived)

        assert "price_rank_in_query" in result.columns

        # Query 1: prices [100, 300, 200] -> ranks [1, 3, 2]
        q1 = result[result["srch_id"] == 1]
        assert q1.iloc[0]["price_rank_in_query"] == 1.0  # 100 is cheapest
        assert q1.iloc[1]["price_rank_in_query"] == 3.0  # 300 is most expensive
        assert q1.iloc[2]["price_rank_in_query"] == 2.0  # 200 is middle

        # Query 2: prices [50, 150] -> ranks [1, 2]
        q2 = result[result["srch_id"] == 2]
        assert q2.iloc[0]["price_rank_in_query"] == 1.0
        assert q2.iloc[1]["price_rank_in_query"] == 2.0

    def test_price_rank_with_ties(self):
        """Tied prices should receive the same (minimum) rank."""
        df = pd.DataFrame({
            "srch_id": [1, 1, 1],
            "price_usd": [100.0, 100.0, 200.0],
        })
        result = add_derived_columns(df)

        # Both 100.0 get rank 1 (method="min"), 200.0 gets rank 3
        assert result.iloc[0]["price_rank_in_query"] == 1.0
        assert result.iloc[1]["price_rank_in_query"] == 1.0
        assert result.iloc[2]["price_rank_in_query"] == 3.0

    def test_price_log_is_log1p_of_price(self, df_for_derived):
        """price_log should equal np.log1p(price_usd)."""
        result = add_derived_columns(df_for_derived)

        assert "price_log" in result.columns
        expected = np.log1p(df_for_derived["price_usd"])
        np.testing.assert_array_almost_equal(result["price_log"], expected)

    def test_price_log_with_zero_price(self):
        """price_log for zero price should be log1p(0) = 0."""
        df = pd.DataFrame({
            "srch_id": [1],
            "price_usd": [0.0],
        })
        result = add_derived_columns(df)
        assert result.iloc[0]["price_log"] == pytest.approx(0.0)

    def test_derived_columns_added_not_removed(self, df_for_derived):
        """Original columns should still be present after adding derived ones."""
        original_cols = set(df_for_derived.columns)
        result = add_derived_columns(df_for_derived)

        for col in original_cols:
            assert col in result.columns, f"Original column '{col}' was removed"
        assert len(result.columns) == len(original_cols) + 2  # +price_rank, +price_log
