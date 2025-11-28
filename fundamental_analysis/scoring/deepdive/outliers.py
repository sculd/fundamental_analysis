"""Drill-down functions for finding stocks by specific metric outliers."""

import polars as pl

from fundamental_analysis.scoring.melt import melt_and_classify_metrics
from fundamental_analysis.scoring.z_score import ALL_METRICS, ZScoreOption


def get_stocks_with_metric_outlier(
    df: pl.DataFrame,
    metric_name: str,
    option: ZScoreOption | None = None,
    sigma_threshold: float = 2.0,
    direction: str = "favorable",
    min_stocks: int = 5,
) -> pl.DataFrame:
    """
    Find all stocks with an outlier in a specific metric.

    Useful for questions like:
    - "Which stocks have exceptionally high ROE?"
    - "Which stocks have very low P/E ratios?"

    Example query:
        # Find stocks with exceptionally high ROE (profitability outliers)
        high_roe = get_stocks_with_metric_outlier(
            df,
            metric_name="roe_calculated",
            direction="favorable",
        )

        Result:
        | ticker | segment    | metric_name    | raw_value | zscore | is_outlier |
        |--------|------------|----------------|-----------|--------|------------|
        | NVDA   | Technology | roe_calculated | 0.85      | 4.2    | True       |
        | AAPL   | Technology | roe_calculated | 0.72      | 3.1    | True       |
        | MSFT   | Technology | roe_calculated | 0.68      | 2.8    | True       |

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with z-scores for fundamental metrics. Typically output from
        calculate_metric_z_scores() or calculate_signal_counts().
    metric_name : str
        Name of metric to filter on (e.g., "pe_ratio", "roe_calculated")
    option : ZScoreOption | None, default None
        Configuration for z-score calculation. If None, uses default values.
    sigma_threshold : float, default 2.0
        Z-score threshold for outlier detection
    direction : str, default "favorable"
        "favorable" or "unfavorable"
    min_stocks : int, default 5
        Minimum number of stocks to return (returns top N by z-score in specified direction)

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame filtered to the specified metric outliers,
        sorted by z-score in the direction that matches the request
    """
    if option is None:
        option = ZScoreOption()

    # Get all outlier details
    details = melt_and_classify_metrics(df, sigma_threshold=sigma_threshold)

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
