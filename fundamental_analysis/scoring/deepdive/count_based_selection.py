"""Count-based scoring: count metrics beyond sigma threshold."""

import polars as pl

from fundamental_analysis.scoring.z_score import ALL_METRICS, ZScoreOption, calculate_metric_z_scores


def calculate_signal_counts(
    df: pl.DataFrame,
    option: ZScoreOption | None = None,
    sigma_threshold: float = 2.0,
    min_total_signal_count: int = 0,
) -> pl.DataFrame:
    """
    Calculate z-scores and bidirectional outlier counts for all fundamental metrics.

    Convenience function that combines z-score calculation (calculate_metric_z_scores)
    with outlier counting logic. Treats all metrics equally, counting outliers in
    both favorable and unfavorable directions.

    Each metric has a defined "good" direction:
    - Valuation ratios: lower is better (cheap)
    - Profitability: higher is better
    - Liquidity: higher is better
    - Leverage: lower is better

    Example:
        df = calculate_signal_counts(df, sigma_threshold=2.0, min_total_signal_count=1)

        Result:
        | ticker | favorable_count | unfavorable_count | total_signal_count | net_signal |
        |--------|-----------------|-------------------|--------------------| -----------|
        | AAPL   | 3               | 1                 | 4                  | 2          |
        | NVDA   | 5               | 0                 | 5                  | 5          |

    Parameters
    ----------
    df : pl.DataFrame
        Input data with fundamental metrics
    option : ZScoreOption | None, default None
        Configuration for z-score calculation. If None, uses default values.
    sigma_threshold : float, default 2.0
        Z-score threshold for outlier detection
    min_total_signal_count : int, default 0
        Minimum total_signal_count to include in results. Rows with fewer signals are filtered out.

    Returns
    -------
    pl.DataFrame
        Original dataframe with added columns:
        - favorable_count: number of metrics beyond threshold in "good" direction
        - unfavorable_count: number of metrics beyond threshold in "bad" direction
        - total_signal_count: favorable + unfavorable (magnitude of extremeness)
        - net_signal: favorable - unfavorable (overall direction)
        - metrics_available: number of metrics with valid z-scores
    """
    if option is None:
        option = ZScoreOption()

    # Calculate z-scores for all standard metrics
    df = calculate_metric_z_scores(df, option=option)

    # Build conditions for favorable and unfavorable outliers
    favorable_conditions = []
    unfavorable_conditions = []
    available_conditions = []

    for metric_name, direction in ALL_METRICS:
        zscore_col = f"{metric_name}_zscore"

        # Determine comparison based on metric direction
        if direction == "lower":
            # Lower is better (valuation, leverage)
            favorable_condition = pl.col(zscore_col) < -sigma_threshold
            unfavorable_condition = pl.col(zscore_col) > sigma_threshold
        else:  # direction == "higher"
            # Higher is better (profitability, liquidity)
            favorable_condition = pl.col(zscore_col) > sigma_threshold
            unfavorable_condition = pl.col(zscore_col) < -sigma_threshold

        # Add favorable condition with null/finite checks
        favorable_conditions.append(
            pl.when(
                pl.col(zscore_col).is_not_null() &
                pl.col(zscore_col).is_finite() &
                favorable_condition
            )
            .then(1)
            .otherwise(0)
        )

        # Add unfavorable condition with null/finite checks
        unfavorable_conditions.append(
            pl.when(
                pl.col(zscore_col).is_not_null() &
                pl.col(zscore_col).is_finite() &
                unfavorable_condition
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

    # Filter by minimum total signal count
    if min_total_signal_count > 0:
        df = df.filter(pl.col("total_signal_count") >= min_total_signal_count)

    return df
