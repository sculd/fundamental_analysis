"""Metrics calculation for fundamental analysis."""

import polars as pl

from fundamental_analysis.metrics.financial_health import \
    get_financial_health_expressions
from fundamental_analysis.metrics.fundamental_ratios import \
    get_fundamental_ratio_expressions
from fundamental_analysis.metrics.growth_metrics import get_growth_expressions
from fundamental_analysis.metrics.profitability import \
    get_profitability_expressions


def calculate_all_metrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate all available metrics in single pass.

    Metrics included:
    - Fundamental ratios (P/E, P/B, P/S, P/C, EV/EBITDA)
    - Financial health (debt-to-equity, current ratio, debt-to-assets, interest coverage)
    - Profitability (ROE, ROIC)
    - Growth (EPS growth QoQ/YoY, Revenue growth QoQ/YoY)

    All calculations happen in a single pass for maximum efficiency.

    Note: Growth metrics require time-series data with multiple periods per ticker.
    """
    return df.with_columns(
        get_fundamental_ratio_expressions() +
        get_financial_health_expressions() +
        get_profitability_expressions() +
        get_growth_expressions()
    )
