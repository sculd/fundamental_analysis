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
        # Filter to valid data for this metric
        # If metric requires positive values only, add that filter
        if metric in positive_only_metrics:
            df_valid = df.filter(
                pl.col(metric).is_not_null() &
                pl.col(metric).is_finite() &
                (pl.col(metric) > 0)
            ).select([date_col, segment_col, metric])
        else:
            df_valid = df.filter(
                pl.col(metric).is_not_null() & pl.col(metric).is_finite()
            ).select([date_col, segment_col, metric])

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

        # Convert to dataframe and join back
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


def calculate_undervaluation_count(
    df: pl.DataFrame,
    window_days: int = 180,
    sigma_threshold: float = 2.0,
    segment_col: str = "segment",
    date_col: str = "datekey",
) -> pl.DataFrame:
    """
    Calculate count-based undervaluation score.

    Counts how many valuation metrics are outliers (beyond threshold).
    For undervaluation: counts metrics with z-score < -sigma_threshold
    (i.e., trading below sector average).

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
        - undervaluation_count: number of metrics below -sigma_threshold
        - undervaluation_available: number of metrics with valid z-scores
        - undervaluation_ratio: count / available (0-1)
    """
    undervaluation_metrics = [
        "pe_ratio",
        "pb_ratio",
        "ps_ratio",
        "pc_ratio",
        "ev_ebitda_ratio",
    ]

    # Calculate rolling z-scores for valuation metrics
    # Only use positive values for statistics (negative = unprofitable)
    df = _calculate_rolling_z_scores(
        df,
        metrics=undervaluation_metrics,
        window_days=window_days,
        segment_col=segment_col,
        date_col=date_col,
        positive_only_metrics=undervaluation_metrics,  # All require positive values
    )

    # Count outliers (z-score < -sigma_threshold = undervalued)
    zscore_cols = [f"{m}_zscore" for m in undervaluation_metrics]

    df = df.with_columns([
        # Count outliers (below -threshold)
        pl.sum_horizontal([
            pl.when(
                pl.col(col).is_not_null() &
                pl.col(col).is_finite() &
                (pl.col(col) < -sigma_threshold)
            )
            .then(1)
            .otherwise(0)
            for col in zscore_cols
        ]).alias("undervaluation_count"),

        # Count available metrics
        pl.sum_horizontal([
            pl.when(pl.col(col).is_not_null() & pl.col(col).is_finite())
            .then(1)
            .otherwise(0)
            for col in zscore_cols
        ]).alias("undervaluation_available"),
    ])

    # Calculate ratio
    df = df.with_columns([
        pl.when(pl.col("undervaluation_available") > 0)
        .then(pl.col("undervaluation_count") / pl.col("undervaluation_available"))
        .otherwise(None)
        .alias("undervaluation_ratio")
    ])

    return df


def calculate_quality_count(
    df: pl.DataFrame,
    window_days: int = 180,
    sigma_threshold: float = 2.0,
    segment_col: str = "segment",
    date_col: str = "datekey",
) -> pl.DataFrame:
    """
    Calculate count-based quality score.

    Counts how many quality metrics are outliers (beyond threshold).
    For quality: counts metrics with |z-score| > sigma_threshold in the
    "good" direction:
    - ROE, ROIC, current_ratio, interest_coverage: z > +threshold (high is good)
    - debt_to_equity, debt_to_assets: z < -threshold (low is good)

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
        - quality_count: number of metrics beyond threshold in "good" direction
        - quality_available: number of metrics with valid z-scores
        - quality_ratio: count / available (0-1)
    """
    # Define quality metrics with direction
    # direction: "higher" means high z-score is good, "lower" means low z-score is good
    quality_metrics = [
        ("roe_calculated", "higher"),
        ("roic_calculated", "higher"),
        ("current_ratio", "higher"),
        ("interest_coverage", "higher"),
        ("debt_to_equity", "lower"),  # Low debt is good
        ("debt_to_assets", "lower"),  # Low debt is good
    ]

    metric_names = [m[0] for m in quality_metrics]

    # Calculate rolling z-scores for quality metrics
    # All quality metrics require positive values for meaningful statistics
    df = _calculate_rolling_z_scores(
        df,
        metrics=metric_names,
        window_days=window_days,
        segment_col=segment_col,
        date_col=date_col,
        positive_only_metrics=metric_names,  # All require positive values
    )

    # Count outliers based on direction
    zscore_cols = [(f"{m[0]}_zscore", m[1]) for m in quality_metrics]

    # Build conditions for each metric
    outlier_conditions = []
    available_conditions = []

    for col, direction in zscore_cols:
        if direction == "higher":
            # High is good: count if z > +threshold
            outlier_conditions.append(
                pl.when(
                    pl.col(col).is_not_null() &
                    pl.col(col).is_finite() &
                    (pl.col(col) > sigma_threshold)
                )
                .then(1)
                .otherwise(0)
            )
        else:  # direction == "lower"
            # Low is good: count if z < -threshold
            outlier_conditions.append(
                pl.when(
                    pl.col(col).is_not_null() &
                    pl.col(col).is_finite() &
                    (pl.col(col) < -sigma_threshold)
                )
                .then(1)
                .otherwise(0)
            )

        available_conditions.append(
            pl.when(pl.col(col).is_not_null() & pl.col(col).is_finite())
            .then(1)
            .otherwise(0)
        )

    df = df.with_columns([
        # Count outliers in "good" direction
        pl.sum_horizontal(outlier_conditions).alias("quality_count"),

        # Count available metrics
        pl.sum_horizontal(available_conditions).alias("quality_available"),
    ])

    # Calculate ratio
    df = df.with_columns([
        pl.when(pl.col("quality_available") > 0)
        .then(pl.col("quality_count") / pl.col("quality_available"))
        .otherwise(None)
        .alias("quality_ratio")
    ])

    return df


def calculate_combined_counts(
    df: pl.DataFrame,
    window_days: int = 180,
    sigma_threshold: float = 2.0,
    segment_col: str = "segment",
    date_col: str = "datekey",
) -> pl.DataFrame:
    """
    Calculate both count-based scores.

    This is a convenience function that calculates both undervaluation
    and quality counts in one call.

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
        Original dataframe with added columns from both scoring functions
    """
    df = calculate_undervaluation_count(
        df,
        window_days=window_days,
        sigma_threshold=sigma_threshold,
        segment_col=segment_col,
        date_col=date_col,
    )

    df = calculate_quality_count(
        df,
        window_days=window_days,
        sigma_threshold=sigma_threshold,
        segment_col=segment_col,
        date_col=date_col,
    )

    return df
