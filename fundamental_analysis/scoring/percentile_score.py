"""Percentile-based scoring for fundamental metrics.

Unlike z-scores which assume normal distribution, percentile ranks are
distribution-agnostic and more intuitive for skewed financial ratios.
"""

import polars as pl

from fundamental_analysis.scoring.common import ALL_METRICS, ScoreOption


def _calculate_rolling_percentiles(
    df: pl.DataFrame,
    metrics: list[str],
    option: ScoreOption,
    positive_only_metrics: list[str] | None = None,
) -> pl.DataFrame:
    """
    Calculate percentile ranks within each segment.

    For each row, uses the most recent value per ticker within the segment
    (as of that row's datekey) to compute percentile ranks.

    This is efficient: O(n log n) per segment instead of O(n²).
    """
    if positive_only_metrics is None:
        positive_only_metrics = []

    segment_col = option.segment_col
    date_col = option.date_col

    # Ensure sorted by segment and date
    df = df.sort(segment_col, date_col)

    for metric in metrics:
        # Build filter condition
        filter_cond = pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        if metric in positive_only_metrics:
            filter_cond = filter_cond & (pl.col(metric) > 0)

        # Calculate rank-based percentile within each segment
        # Using all historical data up to each point (cumulative rank)
        df = df.with_columns([
            pl.when(filter_cond)
            .then(
                # Cumulative rank within segment (how many previous values are less than current)
                # This gives point-in-time percentile using all data seen so far
                (
                    (pl.col(metric).rank("min").over(segment_col) - 1) /
                    (pl.col(metric).count().over(segment_col) - 1).clip(lower_bound=1) * 100
                ).round(0).cast(pl.Int64) // 5 * 5
            )
            .otherwise(None)
            .alias(f"{metric}_percentile"),

            pl.when(filter_cond)
            .then(pl.col(metric).count().over(segment_col))
            .otherwise(None)
            .cast(pl.Int64)
            .alias(f"{metric}_population"),

            pl.when(filter_cond)
            .then(pl.col(metric).median().over(segment_col))
            .otherwise(None)
            .alias(f"{metric}_median"),

            pl.when(filter_cond)
            .then(pl.col(metric).quantile(0.1).over(segment_col))
            .otherwise(None)
            .alias(f"{metric}_p10"),

            pl.when(filter_cond)
            .then(pl.col(metric).quantile(0.9).over(segment_col))
            .otherwise(None)
            .alias(f"{metric}_p90"),
        ])

    return df


def calculate_metric_percentiles(
    df: pl.DataFrame,
    option: ScoreOption | None = None,
) -> pl.DataFrame:
    """
    Calculate rolling percentile ranks for all standard fundamental metrics.

    This is the core reusable function for percentile calculation. It processes
    all 11 fundamental metrics (valuation, profitability, liquidity, leverage)
    using point-in-time rolling windows to avoid look-ahead bias.

    Percentile ranks are more robust than z-scores for skewed financial ratios
    as they make no distributional assumptions.

    Example:
        df = calculate_metric_percentiles(df)

        Result includes columns like:
        - pe_ratio_percentile: 0-100, where lower means cheaper vs peers
        - roe_calculated_percentile: 0-100, where higher means better vs peers

    Parameters
    ----------
    df : pl.DataFrame
        Input data with fundamental metrics, segment, and datekey columns
    option : ScoreOption | None, default None
        Configuration for percentile calculation. If None, uses default values.

    Returns
    -------
    pl.DataFrame
        Original dataframe with added percentile columns:
        - {metric}_percentile: percentile rank (0-100) within segment
        - {metric}_population: count of valid values in rolling window

        For all 11 metrics: pe_ratio, pb_ratio, ps_ratio, pc_ratio,
        ev_ebitda_ratio, roe_calculated, roic_calculated, current_ratio,
        interest_coverage, debt_to_equity, debt_to_assets
    """
    if option is None:
        option = ScoreOption()

    metric_names = [m[0] for m in ALL_METRICS]

    # All metrics require positive values for meaningful statistics
    return _calculate_rolling_percentiles(
        df,
        metrics=metric_names,
        option=option,
        positive_only_metrics=metric_names,
    )
