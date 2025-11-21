"""Count-based scoring: count metrics beyond sigma threshold."""

from datetime import timedelta

import polars as pl


def _calculate_rolling_z_scores(
    df: pl.DataFrame,
    metrics: list[str],
    window_days: int,
    segment_col: str = "segment",
    date_col: str = "datekey",
    positive_only_metrics: list[str] | None = None,
) -> pl.DataFrame:
    """
    Calculate z-scores using a rolling window of historical data.

    For each row at datekey D, calculate mean/std using only data from
    [D - window_days, D] within the same segment. This ensures point-in-time
    correctness (no look-ahead bias).

    Parameters
    ----------
    df : pl.DataFrame
        Input data with metrics, segment, and datekey columns
    metrics : list[str]
        List of metric column names to calculate z-scores for
    window_days : int
        Size of rolling window in days (e.g., 180 for 6 months)
    segment_col : str, default "segment"
        Column name for segmentation (e.g., sector)
    date_col : str, default "datekey"
        Column name for date
    positive_only_metrics : list[str] | None, default None
        Metrics that should only use positive values for statistics.
        Z-scores will only be calculated for positive values of these metrics.

    Returns
    -------
    pl.DataFrame
        Original dataframe with added columns:
        - {metric}_mean: rolling mean within segment
        - {metric}_std: rolling std within segment
        - {metric}_zscore: (value - mean) / std
    """
    if positive_only_metrics is None:
        positive_only_metrics = []

    # Ensure sorted by segment and date
    df = df.sort(segment_col, date_col)

    # For each unique (datekey, segment), calculate rolling window stats
    # Strategy: For each row, we'll calculate stats using join_asof with a time window

    # Create a dataframe with all unique (datekey, segment) combinations
    unique_dates_segments = df.select([date_col, segment_col]).unique()

    # For each metric, calculate rolling stats
    for metric in metrics:
        df_valid = df.filter(
            pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        ).select([date_col, segment_col, metric])

        # Filter to valid data for this metric
        # If metric requires positive values only, add that filter
        if metric in positive_only_metrics:
            df_valid = df_valid.filter(
                (pl.col(metric) > 0)
            )

        # For each unique (datekey, segment), calculate stats from rolling window
        stats_list = []

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

            # Calculate stats
            if len(window_data) > 1:
                stats = window_data.select([
                    pl.col(metric).mean().alias("mean"),
                    pl.col(metric).std().alias("std"),
                ]).to_dicts()[0]

                stats_list.append({
                    date_col: current_date,
                    segment_col: current_segment,
                    f"{metric}_mean": stats["mean"],
                    f"{metric}_std": stats["std"],
                })
            else:
                # Not enough data in window
                stats_list.append({
                    date_col: current_date,
                    segment_col: current_segment,
                    f"{metric}_mean": None,
                    f"{metric}_std": None,
                })

        # Convert to dataframe and join back to original df
        df_stats = pl.DataFrame(stats_list)
        df = df.join(
            df_stats,
            on=[date_col, segment_col],
            how="left"
        )

        # Calculate z-scores
        # If this metric requires positive values, only calculate z-score for positive values
        if metric in positive_only_metrics:
            df = df.with_columns([
                pl.when(
                    pl.col(metric).is_not_null() &
                    pl.col(metric).is_finite() &
                    (pl.col(metric) > 0)
                )
                .then(
                    (pl.col(metric) - pl.col(f"{metric}_mean")) /
                    pl.col(f"{metric}_std")
                )
                .otherwise(None)
                .alias(f"{metric}_zscore")
            ])
        else:
            df = df.with_columns([
                (
                    (pl.col(metric) - pl.col(f"{metric}_mean")) /
                    pl.col(f"{metric}_std")
                ).alias(f"{metric}_zscore")
            ])

    return df


def calculate_signal_counts(
    df: pl.DataFrame,
    window_days: int = 180,
    sigma_threshold: float = 2.0,
    segment_col: str = "segment",
    date_col: str = "datekey",
) -> pl.DataFrame:
    """
    Calculate bidirectional outlier counts for all fundamental metrics.

    Treats all metrics equally, counting outliers in both favorable and
    unfavorable directions. Each metric has a defined "good" direction:
    - Valuation ratios: lower is better (cheap)
    - Profitability: higher is better
    - Liquidity: higher is better
    - Leverage: lower is better

    Parameters
    ----------
    df : pl.DataFrame
        Input data with fundamental metrics
    window_days : int, default 180
        Rolling window size in days
    sigma_threshold : float, default 2.0
        Z-score threshold for outlier detection
    segment_col : str, default "segment"
        Segmentation column name
    date_col : str, default "datekey"
        Date column name

    Returns
    -------
    pl.DataFrame
        Original dataframe with added columns:
        - favorable_count: number of metrics beyond threshold in "good" direction
        - unfavorable_count: number of metrics beyond threshold in "bad" direction
        - total_signal_count: favorable + unfavorable (magnitude of extremeness)
        - net_signal: favorable - unfavorable (overall direction)
        - metrics_available: number of metrics with valid z-scores
        - favorable_ratio: favorable / available
        - unfavorable_ratio: unfavorable / available
        - total_ratio: total_signal / available
    """
    # Define all metrics with their favorable direction
    # Format: (metric_name, direction)
    # direction: "lower" = low values are good, "higher" = high values are good
    all_metrics = [
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

    metric_names = [m[0] for m in all_metrics]

    # Calculate rolling z-scores for all metrics
    # All metrics require positive values for meaningful statistics
    df = _calculate_rolling_z_scores(
        df,
        metrics=metric_names,
        window_days=window_days,
        segment_col=segment_col,
        date_col=date_col,
        positive_only_metrics=metric_names,  # All require positive values
    )

    # Build conditions for favorable and unfavorable outliers
    favorable_conditions = []
    unfavorable_conditions = []
    available_conditions = []

    for metric_name, direction in all_metrics:
        zscore_col = f"{metric_name}_zscore"

        if direction == "lower":
            # Lower is better (valuation, leverage)
            # Favorable: z < -threshold (below average = good)
            favorable_conditions.append(
                pl.when(
                    pl.col(zscore_col).is_not_null() &
                    pl.col(zscore_col).is_finite() &
                    (pl.col(zscore_col) < -sigma_threshold)
                )
                .then(1)
                .otherwise(0)
            )
            # Unfavorable: z > +threshold (above average = bad)
            unfavorable_conditions.append(
                pl.when(
                    pl.col(zscore_col).is_not_null() &
                    pl.col(zscore_col).is_finite() &
                    (pl.col(zscore_col) > sigma_threshold)
                )
                .then(1)
                .otherwise(0)
            )
        else:  # direction == "higher"
            # Higher is better (profitability, liquidity)
            # Favorable: z > +threshold (above average = good)
            favorable_conditions.append(
                pl.when(
                    pl.col(zscore_col).is_not_null() &
                    pl.col(zscore_col).is_finite() &
                    (pl.col(zscore_col) > sigma_threshold)
                )
                .then(1)
                .otherwise(0)
            )
            # Unfavorable: z < -threshold (below average = bad)
            unfavorable_conditions.append(
                pl.when(
                    pl.col(zscore_col).is_not_null() &
                    pl.col(zscore_col).is_finite() &
                    (pl.col(zscore_col) < -sigma_threshold)
                )
                .then(1)
                .otherwise(0)
            )

        # Count available metrics
        available_conditions.append(
            pl.when(pl.col(zscore_col).is_not_null() & pl.col(zscore_col).is_finite())
            .then(1)
            .otherwise(0)
        )

    # Add count columns
    df = df.with_columns([
        pl.sum_horizontal(favorable_conditions).alias("favorable_count"),
        pl.sum_horizontal(unfavorable_conditions).alias("unfavorable_count"),
        pl.sum_horizontal(available_conditions).alias("metrics_available"),
    ])

    # Add derived columns (total and net signals)
    df = df.with_columns([
        # Total signal magnitude (how extreme/interesting is this stock?)
        (pl.col("favorable_count") + pl.col("unfavorable_count")).alias("total_signal_count"),

        # Net signal (overall good vs bad)
        (pl.col("favorable_count") - pl.col("unfavorable_count")).alias("net_signal"),
    ])

    # Add ratio columns
    df = df.with_columns([
        # Ratios (normalize for missing data)
        pl.when(pl.col("metrics_available") > 0)
        .then(pl.col("favorable_count") / pl.col("metrics_available"))
        .otherwise(None)
        .alias("favorable_ratio"),

        pl.when(pl.col("metrics_available") > 0)
        .then(pl.col("unfavorable_count") / pl.col("metrics_available"))
        .otherwise(None)
        .alias("unfavorable_ratio"),

        pl.when(pl.col("metrics_available") > 0)
        .then(pl.col("total_signal_count") / pl.col("metrics_available"))
        .otherwise(None)
        .alias("total_ratio"),
    ])

    return df
