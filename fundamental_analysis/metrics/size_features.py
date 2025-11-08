"""Company size and scale feature calculations."""

import polars as pl

from fundamental_analysis.metrics.temporal_utils import temporal_change

# Raw size features from SF1 that this module uses (preserved separately by orchestrator)
SIZE_FEATURE_RAW_COLUMNS = ["marketcap", "revenue", "assets"]


def _marketcap_category_expr() -> pl.Expr:
    """
    Categorize market cap into standard size buckets.

    Categories (industry standard):
    - Mega cap: > $200B
    - Large cap: $10B - $200B
    - Mid cap: $2B - $10B
    - Small cap: $300M - $2B
    - Micro cap: < $300M
    """
    return (
        pl.when(pl.col("marketcap") > 200_000_000_000)
        .then(pl.lit("mega"))
        .when(pl.col("marketcap") > 10_000_000_000)
        .then(pl.lit("large"))
        .when(pl.col("marketcap") > 2_000_000_000)
        .then(pl.lit("mid"))
        .when(pl.col("marketcap") > 300_000_000)
        .then(pl.lit("small"))
        .when(pl.col("marketcap") > 0)
        .then(pl.lit("micro"))
        .otherwise(None)
    )


def get_size_feature_expressions() -> list[pl.Expr]:
    """
    Return list of size feature expressions for composing with other metrics.

    Use this for efficient batch calculation with other metric types.

    Includes:
    - Market cap category (categorical: mega/large/mid/small/micro)
    - Temporal growth features (QoQ and YoY)

    Note: Raw features (marketcap, revenue, assets) are preserved separately by orchestrator.
    Note: Normalization (e.g., sector-based t-score) should be applied in preprocessing.
    """
    return [
        # Categorical feature
        _marketcap_category_expr().alias("marketcap_category"),
        # Temporal features (growth - percentage change)
        temporal_change(pl.col("revenue"), 1).alias("revenue_growth_qoq"),
        temporal_change(pl.col("revenue"), 4).alias("revenue_growth_yoy"),
        temporal_change(pl.col("marketcap"), 1).alias("marketcap_growth_qoq"),
        temporal_change(pl.col("marketcap"), 4).alias("marketcap_growth_yoy"),
        temporal_change(pl.col("assets"), 1).alias("assets_growth_qoq"),
        temporal_change(pl.col("assets"), 4).alias("assets_growth_yoy"),
    ]
