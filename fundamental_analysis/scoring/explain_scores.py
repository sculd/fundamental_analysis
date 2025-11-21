"""Explanation and drill-down functions for outlier scores."""

import polars as pl

from .count_based_scores import ALL_METRICS


def get_outlier_details(
    df: pl.DataFrame,
    ticker: str | None = None,
    sigma_threshold: float = 2.0,
) -> pl.DataFrame:
    """
    Reshape scored data from wide to long format for detailed analysis.

    Transforms each stock-quarter row from wide format (one column per metric)
    to long format (one row per metric), making it easy to see which specific
    metrics are outliers and their values.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with z-scores for fundamental metrics. Typically output from
        calculate_metric_z_scores() or calculate_signal_counts().
    ticker : str | None, default None
        If provided, filter to only this ticker
    sigma_threshold : float, default 2.0
        Z-score threshold used to determine outliers

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with columns:
        - ticker: Stock ticker
        - datekey: Report date
        - segment: Sector segment
        - metric_name: Name of the metric (e.g., "pe_ratio")
        - metric_direction: "lower" or "higher" (what's favorable)
        - raw_value: Actual metric value
        - zscore: Z-score within segment
        - segment_mean: Mean value in segment
        - segment_std: Standard deviation in segment
        - is_outlier: Boolean, True if beyond threshold
        - outlier_direction: "favorable", "unfavorable", or None
    """
    # Filter to specific ticker if requested
    if ticker is not None:
        df = df.filter(pl.col("ticker") == ticker)

    # Build list of rows for long format
    rows = []

    for stock_row in df.iter_rows(named=True):
        for metric_name, metric_direction in ALL_METRICS:
            # Get values for this metric
            raw_value = stock_row.get(metric_name)
            zscore = stock_row.get(f"{metric_name}_zscore")
            segment_mean = stock_row.get(f"{metric_name}_mean")
            segment_std = stock_row.get(f"{metric_name}_std")

            # Determine if this is an outlier and in which direction
            is_outlier = False
            outlier_direction = None

            if zscore is not None and abs(zscore) >= sigma_threshold:
                is_outlier = True

                if metric_direction == "lower":
                    # Lower is better (valuation, leverage)
                    if zscore < -sigma_threshold:
                        outlier_direction = "favorable"
                    elif zscore > sigma_threshold:
                        outlier_direction = "unfavorable"
                else:  # metric_direction == "higher"
                    # Higher is better (profitability, liquidity)
                    if zscore > sigma_threshold:
                        outlier_direction = "favorable"
                    elif zscore < -sigma_threshold:
                        outlier_direction = "unfavorable"

            rows.append({
                "ticker": stock_row["ticker"],
                "datekey": stock_row["datekey"],
                "segment": stock_row.get("segment"),
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "raw_value": raw_value,
                "zscore": zscore,
                "segment_mean": segment_mean,
                "segment_std": segment_std,
                "is_outlier": is_outlier,
                "outlier_direction": outlier_direction,
            })

    return pl.DataFrame(rows)


def get_outlier_summary(
    df: pl.DataFrame,
    ticker: str | None = None,
    sigma_threshold: float = 2.0,
    direction_filter: str | None = None,
) -> pl.DataFrame:
    """
    Get only the outlier metrics (filtered version of get_outlier_details).

    This is a convenience function that returns only rows where is_outlier=True,
    making it easy to see just the extreme metrics without all the noise.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with z-scores for fundamental metrics. Typically output from
        calculate_metric_z_scores() or calculate_signal_counts().
    ticker : str | None, default None
        If provided, filter to only this ticker
    sigma_threshold : float, default 2.0
        Z-score threshold used to determine outliers
    direction_filter : str | None, default None
        If provided, filter to only this direction ("favorable" or "unfavorable")

    Returns
    -------
    pl.DataFrame
        Filtered long-format DataFrame showing only outlier metrics
    """
    # Get full details
    details = get_outlier_details(df, ticker=ticker, sigma_threshold=sigma_threshold)

    # Filter to outliers only
    result = details.filter(pl.col("is_outlier") == True)

    # Apply direction filter if requested
    if direction_filter is not None:
        result = result.filter(pl.col("outlier_direction") == direction_filter)

    return result


def get_stocks_with_metric_outlier(
    df: pl.DataFrame,
    metric_name: str,
    direction: str = "favorable",
    sigma_threshold: float = 2.0,
    min_stocks: int = 5,
) -> pl.DataFrame:
    """
    Find all stocks with an outlier in a specific metric.

    Useful for questions like:
    - "Which stocks have exceptionally high ROE?"
    - "Which stocks have very low P/E ratios?"

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with z-scores for fundamental metrics. Typically output from
        calculate_metric_z_scores() or calculate_signal_counts().
    metric_name : str
        Name of metric to filter on (e.g., "pe_ratio", "roe_calculated")
    direction : str, default "favorable"
        "favorable" or "unfavorable"
    sigma_threshold : float, default 2.0
        Z-score threshold used to determine outliers
    min_stocks : int, default 5
        Minimum number of stocks to return (returns top N by z-score in specified direction)

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame filtered to the specified metric outliers,
        sorted by z-score in the direction that matches the request
    """
    # Get all outlier details
    details = get_outlier_details(df, sigma_threshold=sigma_threshold)

    # Filter to specific metric and direction
    result = details.filter(
        (pl.col("metric_name") == metric_name) &
        (pl.col("outlier_direction") == direction)
    )

    # Sort by absolute z-score (most extreme first)
    result = result.with_columns(
        pl.col("zscore").abs().alias("abs_zscore")
    ).sort("abs_zscore", descending=True)

    # Ensure we return at least min_stocks (if available)
    if len(result) < min_stocks:
        # Relax to just the metric, regardless of outlier status
        # But still exclude null z-scores and sort by direction

        # Look up metric direction from config
        metric_config = {name: dir for name, dir in ALL_METRICS}
        metric_direction = metric_config.get(metric_name)

        # Determine sort direction based on favorable/unfavorable and metric direction
        # For "favorable" + "lower" metric → want most negative z-scores
        # For "favorable" + "higher" metric → want most positive z-scores
        # For "unfavorable" + "lower" metric → want most positive z-scores
        # For "unfavorable" + "higher" metric → want most negative z-scores
        if direction == "favorable":
            sort_ascending = (metric_direction == "lower")
        else:  # unfavorable
            sort_ascending = (metric_direction == "higher")

        result = details.filter(
            (pl.col("metric_name") == metric_name) &
            pl.col("zscore").is_not_null() &
            pl.col("zscore").is_finite()
        ).sort("zscore", descending=not sort_ascending).head(min_stocks)

    return result.drop("abs_zscore", strict=False)
