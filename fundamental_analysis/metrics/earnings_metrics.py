"""Earnings metric calculations."""

import polars as pl

from fundamental_analysis.metrics.temporal_utils import temporal_change


def get_earnings_snapshot_expressions() -> list[pl.Expr]:
    """
    Return snapshot earnings metric expressions (non-temporal).

    Note: Earnings module currently has no snapshot metrics (only growth metrics).
    Returns empty list for consistency with other metric modules.
    """
    return []


def get_earnings_growth_expressions() -> list[pl.Expr]:
    """
    Return temporal growth expressions for earnings metrics.

    Includes EPS growth (QoQ and YoY).
    Note: EPS growth requires time-series data. The DataFrame should contain
    multiple periods per ticker for meaningful results.
    """
    return [
        temporal_change(pl.col("epsdil"), 1).alias("eps_growth_qoq"),
        temporal_change(pl.col("epsdil"), 4).alias("eps_growth_yoy"),
    ]


def get_earnings_expressions() -> list[pl.Expr]:
    """
    Return all earnings metric expressions (snapshot + growth).

    Use this for efficient batch calculation with other metric types.
    """
    return get_earnings_snapshot_expressions() + get_earnings_growth_expressions()
