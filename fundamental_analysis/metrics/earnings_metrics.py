"""Earnings metric calculations."""

import polars as pl


def _eps_growth_expr(shift: int) -> pl.Expr:
    """
    EPS growth rate (percentage change). Use shift=1 for QoQ, shift=4 for YoY.

    Formula: (current_eps - previous_eps) / abs(previous_eps)

    Uses absolute value for denominator to handle negative earnings.
    Returns null when previous EPS is 0 or when no previous period exists.
    """
    current_eps = pl.col("epsdil")
    previous_eps = pl.col("epsdil").shift(shift).over(
        "ticker",
        order_by="reportperiod"
    )

    return pl.when(previous_eps != 0).then(
        (current_eps - previous_eps) / previous_eps.abs()
    ).otherwise(None)


def get_earnings_expressions() -> list[pl.Expr]:
    """
    Return list of earnings metric expressions for composing with other metrics.

    Use this for efficient batch calculation with other metric types.

    Includes temporal features (growth QoQ/YoY).
    Note: EPS growth requires time-series data. The DataFrame should contain
    multiple periods per ticker for meaningful results.
    """
    return [
        _eps_growth_expr(1).alias("eps_growth_qoq"),
        _eps_growth_expr(4).alias("eps_growth_yoy"),
    ]
