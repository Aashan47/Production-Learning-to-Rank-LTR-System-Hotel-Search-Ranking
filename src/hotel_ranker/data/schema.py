"""Column name constants and dtype mappings for the Expedia Hotel Search dataset.

Centralising column names avoids typo-related bugs and provides a single
source of truth that every module imports.
"""

# ---------------------------------------------------------------------------
# Identifiers & grouping
# ---------------------------------------------------------------------------
SEARCH_ID = "srch_id"
PROPERTY_ID = "prop_id"
POSITION = "position"
DATE_TIME = "date_time"

# ---------------------------------------------------------------------------
# Target columns
# ---------------------------------------------------------------------------
CLICK_BOOL = "click_bool"
BOOKING_BOOL = "booking_bool"
GROSS_BOOKINGS_USD = "gross_bookings_usd"

# ---------------------------------------------------------------------------
# Randomisation flag (used for unbiased propensity estimation)
# ---------------------------------------------------------------------------
RANDOM_BOOL = "random_bool"

# ---------------------------------------------------------------------------
# Property (hotel) features
# ---------------------------------------------------------------------------
PROP_STARRATING = "prop_starrating"
PROP_REVIEW_SCORE = "prop_review_score"
PROP_BRAND_BOOL = "prop_brand_bool"
PROP_LOCATION_SCORE1 = "prop_location_score1"
PROP_LOCATION_SCORE2 = "prop_location_score2"
PROP_LOG_HISTORICAL_PRICE = "prop_log_historical_price"
PRICE_USD = "price_usd"
PROMOTION_FLAG = "promotion_flag"

# ---------------------------------------------------------------------------
# Search context features
# ---------------------------------------------------------------------------
SITE_ID = "site_id"
VISITOR_LOCATION_COUNTRY_ID = "visitor_location_country_id"
VISITOR_HIST_STARRATING = "visitor_hist_starrating"
VISITOR_HIST_ADR_USD = "visitor_hist_adr_usd"
SRCH_DESTINATION_ID = "srch_destination_id"
SRCH_LENGTH_OF_STAY = "srch_length_of_stay"
SRCH_BOOKING_WINDOW = "srch_booking_window"
SRCH_ADULTS_COUNT = "srch_adults_count"
SRCH_CHILDREN_COUNT = "srch_children_count"
SRCH_ROOM_COUNT = "srch_room_count"
SRCH_SATURDAY_NIGHT_BOOL = "srch_saturday_night_bool"

# ---------------------------------------------------------------------------
# Competitor columns (comp1–comp8, rate & inv)
# ---------------------------------------------------------------------------
COMPETITOR_RATE_COLS = [f"comp{i}_rate" for i in range(1, 9)]
COMPETITOR_INV_COLS = [f"comp{i}_inv" for i in range(1, 9)]
COMPETITOR_RATE_PERCENT_DIFF_COLS = [f"comp{i}_rate_percent_diff" for i in range(1, 9)]
ALL_COMPETITOR_COLS = COMPETITOR_RATE_COLS + COMPETITOR_INV_COLS + COMPETITOR_RATE_PERCENT_DIFF_COLS

# ---------------------------------------------------------------------------
# Optimized dtypes for memory-efficient loading
# ---------------------------------------------------------------------------
DTYPE_MAP = {
    SEARCH_ID: "int32",
    SITE_ID: "int8",
    VISITOR_LOCATION_COUNTRY_ID: "int16",
    PROPERTY_ID: "int32",
    PROP_STARRATING: "int8",
    PROP_REVIEW_SCORE: "float32",
    PROP_BRAND_BOOL: "int8",
    PROP_LOCATION_SCORE1: "float32",
    PROP_LOCATION_SCORE2: "float32",
    PROP_LOG_HISTORICAL_PRICE: "float32",
    PRICE_USD: "float32",
    PROMOTION_FLAG: "int8",
    SRCH_DESTINATION_ID: "int32",
    SRCH_LENGTH_OF_STAY: "int8",
    SRCH_BOOKING_WINDOW: "int16",
    SRCH_ADULTS_COUNT: "int8",
    SRCH_CHILDREN_COUNT: "int8",
    SRCH_ROOM_COUNT: "int8",
    SRCH_SATURDAY_NIGHT_BOOL: "int8",
    RANDOM_BOOL: "int8",
    CLICK_BOOL: "int8",
    BOOKING_BOOL: "int8",
    POSITION: "int8",
    GROSS_BOOKINGS_USD: "float32",
}

# Add competitor columns (all nullable float)
for _col in ALL_COMPETITOR_COLS:
    DTYPE_MAP[_col] = "float32"

# ---------------------------------------------------------------------------
# Feature groups (for convenient selection in feature pipeline)
# ---------------------------------------------------------------------------
RAW_PROPERTY_FEATURES = [
    PROP_STARRATING,
    PROP_REVIEW_SCORE,
    PROP_BRAND_BOOL,
    PROP_LOCATION_SCORE1,
    PROP_LOCATION_SCORE2,
    PROP_LOG_HISTORICAL_PRICE,
    PRICE_USD,
    PROMOTION_FLAG,
]

RAW_SEARCH_FEATURES = [
    SITE_ID,
    VISITOR_LOCATION_COUNTRY_ID,
    VISITOR_HIST_STARRATING,
    VISITOR_HIST_ADR_USD,
    SRCH_DESTINATION_ID,
    SRCH_LENGTH_OF_STAY,
    SRCH_BOOKING_WINDOW,
    SRCH_ADULTS_COUNT,
    SRCH_CHILDREN_COUNT,
    SRCH_ROOM_COUNT,
    SRCH_SATURDAY_NIGHT_BOOL,
]
