"""Company size and scale feature calculations."""

import polars as pl

from fundamental_analysis.metrics.temporal_utils import temporal_change

# Raw size/fundamental features from SF1 (preserved separately by orchestrator)
SIZE_FEATURE_RAW_COLUMNS = ["marketcap", "revenue", "netinc", "equity", "assets"]


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


def get_size_snapshot_expressions() -> list[pl.Expr]:
    """
    Return snapshot size feature expressions (non-temporal).

    Includes:
    - Market cap category (categorical: mega/large/mid/small/micro)

    Note: Raw features (marketcap, revenue, assets) are preserved separately by orchestrator.
    Note: Normalization (e.g., sector-based t-score) should be applied in preprocessing.
    """
    return [
        _marketcap_category_expr().alias("marketcap_category"),
    ]


def get_size_growth_expressions() -> list[pl.Expr]:
    """
    Return temporal growth feature expressions for size metrics.

    Includes growth (QoQ and YoY) for revenue, netinc, equity, marketcap, and assets.
    """
    return [
        temporal_change(pl.col("revenue"), 1).alias("revenue_growth_qoq"),
        temporal_change(pl.col("revenue"), 4).alias("revenue_growth_yoy"),
        temporal_change(pl.col("netinc"), 1).alias("netinc_growth_qoq"),
        temporal_change(pl.col("netinc"), 4).alias("netinc_growth_yoy"),
        temporal_change(pl.col("equity"), 1).alias("equity_growth_qoq"),
        temporal_change(pl.col("equity"), 4).alias("equity_growth_yoy"),
        temporal_change(pl.col("marketcap"), 1).alias("marketcap_growth_qoq"),
        temporal_change(pl.col("marketcap"), 4).alias("marketcap_growth_yoy"),
        temporal_change(pl.col("assets"), 1).alias("assets_growth_qoq"),
        temporal_change(pl.col("assets"), 4).alias("assets_growth_yoy"),
    ]


def get_size_feature_expressions() -> list[pl.Expr]:
    """
    Return all size feature expressions (snapshot + growth).

    Use this for efficient batch calculation with other metric types.
    """
    return get_size_snapshot_expressions() + get_size_growth_expressions()
