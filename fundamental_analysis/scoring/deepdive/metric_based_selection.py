"""Drill-down functions for finding stocks by specific metric outliers."""

import polars as pl

from fundamental_analysis.scoring.common import ALL_METRICS, ScoreOption
from fundamental_analysis.scoring.melt import melt_and_classify_metrics
from fundamental_analysis.scoring.percentile_score import calculate_metric_percentiles


def get_stocks_with_metric_outlier(
    df: pl.DataFrame,
    metric_name: str,
    option: ScoreOption | None = None,
    percentile_threshold: float = 90.0,
    direction: str = "favorable",
    min_stocks: int = 5,
    melt: bool = True,
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

        Result (melt=True, default):
        | ticker | segment    | metric_name    | raw_value | percentile | is_outlier |
        |--------|------------|----------------|-----------|------------|------------|
        | NVDA   | Technology | roe_calculated | 0.85      | 98.5       | True       |
        | AAPL   | Technology | roe_calculated | 0.72      | 95.2       | True       |

        Result (melt=False):
        | ticker | segment    | roe_calculated | roe_calculated_percentile | ... |
        |--------|------------|----------------|---------------------------|-----|
        | NVDA   | Technology | 0.85           | 98.5                      | ... |
        | AAPL   | Technology | 0.72           | 95.2                      | ... |

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with fundamental metrics. Percentiles will be calculated if not present.
    metric_name : str
        Name of metric to filter on (e.g., "pe_ratio", "roe_calculated")
    option : ScoreOption | None, default None
        Configuration for percentile calculation. If None, uses default values.
    percentile_threshold : float, default 90.0
        Percentile threshold for outlier detection (0-100).
        90 means top/bottom 10% are outliers.
        For "lower is better" metrics: favorable if <= (100 - threshold)
        For "higher is better" metrics: favorable if >= threshold
    direction : str, default "favorable"
        "favorable" or "unfavorable"
    min_stocks : int, default 5
        Minimum number of stocks to return (returns top N by percentile in specified direction)
    melt : bool, default True
        If True, return long-format DataFrame with one row per metric.
        If False, return wide-format DataFrame with original columns.

    Returns
    -------
    pl.DataFrame
        DataFrame filtered to the specified metric outliers,
        sorted by percentile in the direction that matches the request.
        Format depends on melt parameter.
    """
    if option is None:
        option = ScoreOption()

    # Look up metric direction from config
    metric_config = {name: dir for name, dir in ALL_METRICS}
    metric_direction = metric_config.get(metric_name)

    percentile_col = f"{metric_name}_percentile"

    # Calculate percentiles if not already present
    if percentile_col not in df.columns:
        df = calculate_metric_percentiles(df, option=option)

    # Determine sort direction and outlier condition based on favorable/unfavorable and metric direction
    # percentile_threshold=90 means top/bottom 10% are outliers
    #
    # For "lower is better" metrics (valuation, leverage):
    #   - Favorable = low percentile (cheap/low leverage) = <= (100 - threshold)
    #   - Unfavorable = high percentile (expensive/high leverage) = >= threshold
    # For "higher is better" metrics (profitability, liquidity):
    #   - Favorable = high percentile (good profitability) = >= threshold
    #   - Unfavorable = low percentile (poor profitability) = <= (100 - threshold)

    if direction == "favorable":
        if metric_direction == "lower":
            # Lower is better, favorable = low percentile (e.g., <= 10 when threshold=90)
            outlier_condition = pl.col(percentile_col) <= (100 - percentile_threshold)
            sort_descending = False  # Lowest percentile first
        else:
            # Higher is better, favorable = high percentile (e.g., >= 90 when threshold=90)
            outlier_condition = pl.col(percentile_col) >= percentile_threshold
            sort_descending = True  # Highest percentile first
    else:  # unfavorable
        if metric_direction == "lower":
            # Lower is better, unfavorable = high percentile (e.g., >= 90 when threshold=90)
            outlier_condition = pl.col(percentile_col) >= percentile_threshold
            sort_descending = True  # Highest percentile first
        else:
            # Higher is better, unfavorable = low percentile (e.g., <= 10 when threshold=90)
            outlier_condition = pl.col(percentile_col) <= (100 - percentile_threshold)
            sort_descending = False  # Lowest percentile first

    # Filter to outliers
    result = df.filter(
        pl.col(percentile_col).is_not_null() &
        outlier_condition
    )

    # Sort by percentile
    result = result.sort(percentile_col, descending=sort_descending)

    # Ensure we return at least min_stocks (if available)
    if len(result) < min_stocks:
        result = df.filter(
            pl.col(percentile_col).is_not_null()
        ).sort(percentile_col, descending=sort_descending).head(min_stocks)

    # Melt if requested
    if melt:
        result = melt_and_classify_metrics(result, percentile_threshold=percentile_threshold)
        # Filter to only the requested metric
        result = result.filter(pl.col("metric_name") == metric_name)

    return result
