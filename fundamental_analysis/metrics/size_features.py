"""Company size and scale feature calculations."""

import polars as pl

# Raw size features from SF1 that this module uses (preserved separately by orchestrator)
SIZE_FEATURE_RAW_COLUMNS = ["marketcap", "revenue", "assets"]


def _revenue_growth_expr(shift: int) -> pl.Expr:
    """
    Revenue growth rate (percentage change). Use shift=1 for QoQ, shift=4 for YoY.

    Formula: (current_revenue - previous_revenue) / previous_revenue

    Returns null when previous revenue is 0 or when no previous period exists.
    """
    current_revenue = pl.col("revenue")
    previous_revenue = pl.col("revenue").shift(shift).over(
        "ticker",
        order_by="reportperiod"
    )

    return pl.when(previous_revenue != 0).then(
        (current_revenue - previous_revenue) / previous_revenue
    ).otherwise(None)


def _marketcap_growth_expr(shift: int) -> pl.Expr:
    """
    Market capitalization growth rate (percentage change). Use shift=1 for QoQ, shift=4 for YoY.

    Formula: (current_marketcap - previous_marketcap) / previous_marketcap

    Returns null when previous marketcap is 0 or when no previous period exists.
    Captures company valuation momentum.
    """
    current_marketcap = pl.col("marketcap")
    previous_marketcap = pl.col("marketcap").shift(shift).over(
        "ticker",
        order_by="reportperiod"
    )

    return pl.when(previous_marketcap != 0).then(
        (current_marketcap - previous_marketcap) / previous_marketcap
    ).otherwise(None)


def _assets_growth_expr(shift: int) -> pl.Expr:
    """
    Total assets growth rate (percentage change). Use shift=1 for QoQ, shift=4 for YoY.

    Formula: (current_assets - previous_assets) / previous_assets

    Returns null when previous assets is 0 or when no previous period exists.
    Captures company expansion and scale growth.
    """
    current_assets = pl.col("assets")
    previous_assets = pl.col("assets").shift(shift).over(
        "ticker",
        order_by="reportperiod"
    )

    return pl.when(previous_assets != 0).then(
        (current_assets - previous_assets) / previous_assets
    ).otherwise(None)


def get_size_feature_expressions() -> list[pl.Expr]:
    """
    Return list of size feature expressions for composing with other metrics.

    Use this for efficient batch calculation with other metric types.

    Includes temporal growth features (QoQ and YoY).
    Note: Raw features (marketcap, revenue, assets) are preserved separately by orchestrator.
    Note: Normalization (e.g., sector-based t-score) should be applied in preprocessing.
    """
    return [
        # Temporal features (growth - percentage change)
        _revenue_growth_expr(1).alias("revenue_growth_qoq"),
        _revenue_growth_expr(4).alias("revenue_growth_yoy"),
        _marketcap_growth_expr(1).alias("marketcap_growth_qoq"),
        _marketcap_growth_expr(4).alias("marketcap_growth_yoy"),
        _assets_growth_expr(1).alias("assets_growth_qoq"),
        _assets_growth_expr(4).alias("assets_growth_yoy"),
    ]
