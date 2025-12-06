"""Metrics calculation for fundamental analysis."""

import polars as pl

from fundamental_analysis.metrics.earnings_metrics import (
    get_earnings_growth_expressions, get_earnings_snapshot_expressions)
from fundamental_analysis.metrics.financial_health import (
    get_financial_health_growth_expressions,
    get_financial_health_snapshot_expressions)
from fundamental_analysis.metrics.fundamental_ratios import (
    get_fundamental_ratio_growth_expressions,
    get_fundamental_ratio_snapshot_expressions)
from fundamental_analysis.metrics.profitability import (
    get_profitability_growth_expressions,
    get_profitability_snapshot_expressions)
from fundamental_analysis.metrics.size_features import (
    SIZE_FEATURE_RAW_COLUMNS, get_size_growth_expressions,
    get_size_snapshot_expressions)

# Identifiers to keep from SF1 data
IDENTIFIER_COLUMNS = ["ticker", "reportperiod", "datekey", "calendardate"]


def calculate_all_metrics(
    df: pl.DataFrame,
    include_snapshot_metrics: bool = True,
    include_growth_metrics: bool = True,
) -> pl.DataFrame:
    """
    Calculate all available metrics from SF1 data in single pass.

    Args:
        df: Input DataFrame with SF1 fundamental data
        include_snapshot_metrics: If True, include snapshot (non-temporal) metrics
        include_growth_metrics: If True, include growth (temporal) metrics

    Metrics included:
    - Size features (marketcap, revenue, assets + growth QoQ/YoY)
    - Fundamental ratios (P/E, P/B, P/S, P/C, EV/EBITDA + growth QoQ/YoY)
    - Financial health (debt-to-equity, current ratio, debt-to-assets, interest coverage + growth QoQ/YoY)
    - Profitability (ROE, ROIC + growth QoQ/YoY)
    - Earnings (EPS growth QoQ/YoY)

    All calculations happen in a single pass for maximum efficiency.

    Performance optimization: Input is sorted by (ticker, reportperiod) once upfront
    to avoid redundant sorting in temporal feature calculations (30+ operations).

    Returns DataFrame with ONLY identifiers and calculated metrics.
    All original SF1 columns are dropped after calculation.

    Note: Growth metrics require time-series data with multiple periods per ticker.
    Note: Normalization (e.g., sector-based t-score) should be applied in preprocessing.
    Note: Price metrics from SEP data should be calculated separately using
          calculate_price_metrics() to ensure correct as-of-date alignment.
    """
    # Sort once for efficient temporal calculations
    df = df.sort("ticker", "reportperiod")

    # Build expression list based on parameters
    expressions = []

    if include_snapshot_metrics:
        expressions.extend(get_size_snapshot_expressions())
        expressions.extend(get_fundamental_ratio_snapshot_expressions())
        expressions.extend(get_financial_health_snapshot_expressions())
        expressions.extend(get_profitability_snapshot_expressions())
        expressions.extend(get_earnings_snapshot_expressions())

    if include_growth_metrics:
        expressions.extend(get_size_growth_expressions())
        expressions.extend(get_fundamental_ratio_growth_expressions())
        expressions.extend(get_financial_health_growth_expressions())
        expressions.extend(get_profitability_growth_expressions())
        expressions.extend(get_earnings_growth_expressions())

    # Calculate all metrics
    df_with_metrics = df.with_columns(expressions)

    # Get list of all calculated metric columns (new columns not in original)
    metric_columns = [
        col for col in df_with_metrics.columns
        if col not in df.columns
    ]

    # Select: identifiers + size features (raw) + calculated metrics
    # Note: SIZE_FEATURE_RAW_COLUMNS must be explicitly included since they exist in input df
    return df_with_metrics.select(
        IDENTIFIER_COLUMNS +
        SIZE_FEATURE_RAW_COLUMNS +
        metric_columns
    )
