"""Drill-down functions for finding stocks by specific metric outliers."""

import polars as pl

from fundamental_analysis.scoring.melt import melt_and_classify_metrics
from fundamental_analysis.scoring.z_score import ALL_METRICS, ZScoreOption, calculate_metric_z_scores


def get_stocks_with_metric_outlier(
    df: pl.DataFrame,
    metric_name: str,
    option: ZScoreOption | None = None,
    sigma_threshold: float = 2.0,
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
        | ticker | segment    | metric_name    | raw_value | zscore | is_outlier |
        |--------|------------|----------------|-----------|--------|------------|
        | NVDA   | Technology | roe_calculated | 0.85      | 4.2    | True       |
        | AAPL   | Technology | roe_calculated | 0.72      | 3.1    | True       |

        Result (melt=False):
        | ticker | segment    | roe_calculated | roe_calculated_zscore | ... |
        |--------|------------|----------------|----------------------|-----|
        | NVDA   | Technology | 0.85           | 4.2                   | ... |
        | AAPL   | Technology | 0.72           | 3.1                   | ... |

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
    melt : bool, default True
        If True, return long-format DataFrame with one row per metric.
        If False, return wide-format DataFrame with original columns.

    Returns
    -------
    pl.DataFrame
        DataFrame filtered to the specified metric outliers,
        sorted by z-score in the direction that matches the request.
        Format depends on melt parameter.
    """
    if option is None:
        option = ZScoreOption()

    # Look up metric direction from config
    metric_config = {name: dir for name, dir in ALL_METRICS}
    metric_direction = metric_config.get(metric_name)

    zscore_col = f"{metric_name}_zscore"

    # Calculate z-scores if not already present
    if zscore_col not in df.columns:
        df = calculate_metric_z_scores(df, option=option)

    # Determine sort direction based on favorable/unfavorable and metric direction
    if direction == "favorable":
        sort_ascending = (metric_direction == "lower")
    else:  # unfavorable
        sort_ascending = (metric_direction == "higher")

    # Determine outlier condition based on direction and metric
    if direction == "favorable":
        if metric_direction == "lower":
            outlier_condition = pl.col(zscore_col) < -sigma_threshold
        else:
            outlier_condition = pl.col(zscore_col) > sigma_threshold
    else:  # unfavorable
        if metric_direction == "lower":
            outlier_condition = pl.col(zscore_col) > sigma_threshold
        else:
            outlier_condition = pl.col(zscore_col) < -sigma_threshold

    # Filter to outliers
    result = df.filter(
        pl.col(zscore_col).is_not_null() &
        pl.col(zscore_col).is_finite() &
        outlier_condition
    )

    # Sort by z-score
    result = result.sort(zscore_col, descending=not sort_ascending)

    # Ensure we return at least min_stocks (if available)
    if len(result) < min_stocks:
        result = df.filter(
            pl.col(zscore_col).is_not_null() &
            pl.col(zscore_col).is_finite()
        ).sort(zscore_col, descending=not sort_ascending).head(min_stocks)

    # Melt if requested
    if melt:
        result = melt_and_classify_metrics(result, sigma_threshold=sigma_threshold)
        # Filter to only the requested metric
        result = result.filter(pl.col("metric_name") == metric_name)

    return result
