"""Percentile-based scoring for fundamental metrics.

Unlike z-scores which assume normal distribution, percentile ranks are
distribution-agnostic and more intuitive for skewed financial ratios.
"""

from dataclasses import dataclass
from datetime import timedelta

import polars as pl


@dataclass
class PercentileOption:
    """Configuration for percentile calculation.

    Parameters
    ----------
    window_days : int, default 180
        Rolling window size in days for calculating segment statistics
    segment_col : str, default "segment"
        Column name for segmentation
    date_col : str, default "datekey"
        Column name for date
    """

    window_days: int = 180
    segment_col: str = "segment"
    date_col: str = "datekey"


# Define all metrics with their favorable direction
# Format: (metric_name, direction)
# direction: "lower" = low values are good, "higher" = high values are good
ALL_METRICS = [
    # Valuation metrics (lower is better - cheaper)
    ("pe_ratio", "lower"),
    ("pb_ratio", "lower"),
    ("ps_ratio", "lower"),
    ("pc_ratio", "lower"),
    ("ev_ebitda_ratio", "lower"),
    # Profitability metrics (higher is better)
    ("roe_calculated", "higher"),
    ("roic_calculated", "higher"),
    # Liquidity metrics (higher is better)
    ("current_ratio", "higher"),
    ("interest_coverage", "higher"),
    # Leverage metrics (lower is better)
    ("debt_to_equity", "lower"),
    ("debt_to_assets", "lower"),
]


def _calculate_rolling_percentiles(
    df: pl.DataFrame,
    metrics: list[str],
    option: PercentileOption,
    positive_only_metrics: list[str] | None = None,
) -> pl.DataFrame:
    """
    Calculate percentile ranks using a rolling window of historical data.

    For each row at datekey D, calculate percentile rank using only data from
    [D - window_days, D] within the same segment. This ensures point-in-time
    correctness (no look-ahead bias).

    Parameters
    ----------
    df : pl.DataFrame
        Input data with metrics, segment, and datekey columns
    metrics : list[str]
        List of metric column names to calculate percentiles for
    option : PercentileOption
        Configuration for percentile calculation
    positive_only_metrics : list[str] | None, default None
        Metrics that should only use positive values for statistics.
        Percentiles will only be calculated for positive values of these metrics.

    Returns
    -------
    pl.DataFrame
        Original dataframe with added columns:
        - {metric}_percentile: percentile rank (0-100) within segment
        - {metric}_population: count of valid values in rolling window
    """
    if positive_only_metrics is None:
        positive_only_metrics = []

    segment_col = option.segment_col
    date_col = option.date_col
    window_days = option.window_days

    # Ensure sorted by segment and date
    df = df.sort(segment_col, date_col)

    # Create a dataframe with all unique (datekey, segment) combinations
    unique_dates_segments = df.select([date_col, segment_col]).unique()

    # For each metric, calculate rolling percentiles
    for metric in metrics:
        df_valid = df.filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        ).select([date_col, segment_col, "ticker", metric])

        # Filter to positive values if required
        if metric in positive_only_metrics:
            df_valid = df_valid.filter(pl.col(metric) > 0)

        # For each unique (datekey, segment), calculate percentiles
        percentile_list = []

        for row in unique_dates_segments.iter_rows(named=True):
            current_date = row[date_col]
            current_segment = row[segment_col]

            # Calculate window bounds
            window_start = current_date - timedelta(days=window_days)

            # Filter to window
            window_data = df_valid.filter(
                (pl.col(segment_col) == current_segment) &
                (pl.col(date_col) >= window_start) &
                (pl.col(date_col) <= current_date)
            )

            population_count = len(window_data)

            if population_count > 0:
                # Get all tickers in this window with their values
                tickers_in_window = window_data.select(["ticker", metric]).to_dicts()

                for ticker_row in tickers_in_window:
                    ticker = ticker_row["ticker"]
                    value = ticker_row[metric]

                    # Calculate percentile: proportion of values less than this value
                    values_below = window_data.filter(pl.col(metric) < value).height
                    values_equal = window_data.filter(pl.col(metric) == value).height

                    # Use average rank for ties: (values_below + (values_equal - 1) / 2) / total
                    # This gives the midpoint percentile for tied values
                    percentile = ((values_below + (values_equal - 1) / 2 + 0.5) / population_count) * 100

                    percentile_list.append({
                        date_col: current_date,
                        segment_col: current_segment,
                        "ticker": ticker,
                        f"{metric}_percentile": percentile,
                        f"{metric}_population": population_count,
                    })

        # Convert to dataframe and join back to original df
        if percentile_list:
            df_percentiles = pl.DataFrame(percentile_list)
            df = df.join(
                df_percentiles,
                on=[date_col, segment_col, "ticker"],
                how="left"
            )
        else:
            # No valid data, add null columns
            df = df.with_columns([
                pl.lit(None).alias(f"{metric}_percentile"),
                pl.lit(None).alias(f"{metric}_population"),
            ])

    return df


def calculate_metric_percentiles(
    df: pl.DataFrame,
    option: PercentileOption | None = None,
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
    option : PercentileOption | None, default None
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
        option = PercentileOption()

    metric_names = [m[0] for m in ALL_METRICS]

    # All metrics require positive values for meaningful statistics
    return _calculate_rolling_percentiles(
        df,
        metrics=metric_names,
        option=option,
        positive_only_metrics=metric_names,
    )
