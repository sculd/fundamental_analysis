"""Growth metric calculations (time-series based)."""

import polars as pl


def _eps_growth_qoq_expr() -> pl.Expr:
    """
    EPS Growth Rate (Quarter-over-Quarter): change in diluted EPS vs previous quarter.

    Formula: (current_eps - previous_eps) / abs(previous_eps)

    Uses absolute value for denominator to handle negative earnings.
    Returns null when previous EPS is 0 or when no previous quarter exists.

    Higher values indicate better earnings growth.
    """
    current_eps = pl.col("epsdil")
    previous_eps = pl.col("epsdil").shift(1).over(
        "ticker",
        order_by="reportperiod"
    )

    return pl.when(previous_eps != 0).then(
        (current_eps - previous_eps) / previous_eps.abs()
    ).otherwise(None)


def _eps_growth_yoy_expr() -> pl.Expr:
    """
    EPS Growth Rate (Year-over-Year): change in diluted EPS vs same quarter last year.

    Formula: (current_eps - eps_4q_ago) / abs(eps_4q_ago)

    Compares to 4 quarters ago for seasonal adjustment.
    Returns null when comparison quarter EPS is 0 or doesn't exist.

    Higher values indicate better earnings growth.
    """
    current_eps = pl.col("epsdil")
    eps_4q_ago = pl.col("epsdil").shift(4).over(
        "ticker",
        order_by="reportperiod"
    )

    return pl.when(eps_4q_ago != 0).then(
        (current_eps - eps_4q_ago) / eps_4q_ago.abs()
    ).otherwise(None)


def _revenue_growth_qoq_expr() -> pl.Expr:
    """
    Revenue Growth Rate (Quarter-over-Quarter): change in revenue vs previous quarter.

    Formula: (current_revenue - previous_revenue) / previous_revenue

    Returns null when previous revenue is 0 or when no previous quarter exists.

    Higher values indicate better revenue growth.
    """
    current_revenue = pl.col("revenue")
    previous_revenue = pl.col("revenue").shift(1).over(
        "ticker",
        order_by="reportperiod"
    )

    return pl.when(previous_revenue != 0).then(
        (current_revenue - previous_revenue) / previous_revenue
    ).otherwise(None)


def _revenue_growth_yoy_expr() -> pl.Expr:
    """
    Revenue Growth Rate (Year-over-Year): change in revenue vs same quarter last year.

    Formula: (current_revenue - revenue_4q_ago) / revenue_4q_ago

    Compares to 4 quarters ago for seasonal adjustment.
    Returns null when comparison quarter revenue is 0 or doesn't exist.

    Higher values indicate better revenue growth.
    """
    current_revenue = pl.col("revenue")
    revenue_4q_ago = pl.col("revenue").shift(4).over(
        "ticker",
        order_by="reportperiod"
    )

    return pl.when(revenue_4q_ago != 0).then(
        (current_revenue - revenue_4q_ago) / revenue_4q_ago
    ).otherwise(None)


def get_growth_expressions() -> list[pl.Expr]:
    """
    Return list of growth metric expressions for composing with other metrics.

    Use this for efficient batch calculation with other metric types.

    Note: Growth metrics require time-series data. The DataFrame should contain
    multiple periods per ticker for meaningful results.
    """
    return [
        _eps_growth_qoq_expr().alias("eps_growth_qoq"),
        _eps_growth_yoy_expr().alias("eps_growth_yoy"),
        _revenue_growth_qoq_expr().alias("revenue_growth_qoq"),
        _revenue_growth_yoy_expr().alias("revenue_growth_yoy"),
    ]
