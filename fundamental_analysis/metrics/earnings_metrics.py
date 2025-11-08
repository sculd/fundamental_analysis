"""Earnings metric calculations."""

import polars as pl

from fundamental_analysis.metrics.temporal_utils import temporal_change


def get_earnings_expressions() -> list[pl.Expr]:
    """
    Return list of earnings metric expressions for composing with other metrics.

    Use this for efficient batch calculation with other metric types.

    Includes temporal features (growth QoQ/YoY).
    Note: EPS growth requires time-series data. The DataFrame should contain
    multiple periods per ticker for meaningful results.
    """
    return [
        temporal_change(pl.col("epsdil"), 1).alias("eps_growth_qoq"),
        temporal_change(pl.col("epsdil"), 4).alias("eps_growth_yoy"),
    ]
