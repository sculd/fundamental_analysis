"""MAD (Median Absolute Deviation) score calculation for fundamental metrics.

MAD score is more robust than z-score for skewed distributions because it uses
median instead of mean, making it less sensitive to outliers.

Modified Z-score = 0.6745 * (x - median) / MAD
where MAD = median(|x - median|)

The constant 0.6745 makes the MAD score comparable to z-score for normal distributions.
"""

import polars as pl

from fundamental_analysis.scoring.common import ALL_METRICS, ScoreOption

# Constant to make MAD score comparable to z-score for normal distributions
MAD_CONSTANT = 0.6745


def _calculate_mad_scores(
    df: pl.DataFrame,
    metrics: list[str],
    option: ScoreOption,
    positive_only_metrics: list[str] | None = None,
) -> pl.DataFrame:
    """
    Calculate MAD scores within each segment.

    MAD score = 0.6745 * (value - median) / MAD
    where MAD = median(|value - median|)
    """
    if positive_only_metrics is None:
        positive_only_metrics = []

    segment_col = option.segment_col

    # Ensure sorted by segment
    df = df.sort(segment_col)

    for metric in metrics:
        # Build filter condition
        filter_cond = pl.col(metric).is_not_null() & pl.col(metric).is_finite()
        if metric in positive_only_metrics:
            filter_cond = filter_cond & (pl.col(metric) > 0)

        # Calculate median within segment
        df = df.with_columns([
            pl.when(filter_cond)
            .then(pl.col(metric).median().over(segment_col))
            .otherwise(None)
            .alias(f"{metric}_median"),
        ])

        # Calculate MAD = median(|value - median|) within segment
        df = df.with_columns([
            pl.when(filter_cond)
            .then(
                (pl.col(metric) - pl.col(f"{metric}_median")).abs()
                .median().over(segment_col)
            )
            .otherwise(None)
            .alias(f"{metric}_mad"),
        ])

        # Calculate MAD score = 0.6745 * (value - median) / MAD
        df = df.with_columns([
            pl.when(filter_cond & (pl.col(f"{metric}_mad") > 0))
            .then(
                MAD_CONSTANT * (pl.col(metric) - pl.col(f"{metric}_median")) /
                pl.col(f"{metric}_mad")
            )
            .otherwise(None)
            .alias(f"{metric}_mad_score"),

            pl.when(filter_cond)
            .then(pl.col(metric).count().over(segment_col))
            .otherwise(None)
            .cast(pl.Int64)
            .alias(f"{metric}_population"),
        ])

    return df


def calculate_metric_mad_scores(
    df: pl.DataFrame,
    option: ScoreOption | None = None,
) -> pl.DataFrame:
    """
    Calculate MAD scores for all standard fundamental metrics.

    MAD (Median Absolute Deviation) score is more robust than z-score for
    skewed distributions like financial ratios. It uses median instead of mean,
    making it less sensitive to extreme outliers.

    Formula: MAD score = 0.6745 * (value - median) / MAD
    where MAD = median(|value - median|)

    Interpretation (similar to z-score):
    - |MAD score| > 2.0: Outlier
    - |MAD score| > 3.0: Extreme outlier

    Example:
        df = calculate_metric_mad_scores(df)

        Result includes columns like:
        - pe_ratio_median: segment median
        - pe_ratio_mad: segment MAD
        - pe_ratio_mad_score: modified z-score

    Parameters
    ----------
    df : pl.DataFrame
        Input data with fundamental metrics and segment columns
    option : ScoreOption | None, default None
        Configuration for MAD calculation. If None, uses default values.

    Returns
    -------
    pl.DataFrame
        Original dataframe with added MAD score columns:
        - {metric}_median: median within segment
        - {metric}_mad: MAD within segment
        - {metric}_mad_score: modified z-score
        - {metric}_population: count of valid values in segment

        For all 11 metrics: pe_ratio, pb_ratio, ps_ratio, pc_ratio,
        ev_ebitda_ratio, roe_calculated, roic_calculated, current_ratio,
        interest_coverage, debt_to_equity, debt_to_assets
    """
    if option is None:
        option = ScoreOption()

    metric_names = [m[0] for m in ALL_METRICS]

    # All metrics require positive values for meaningful statistics
    return _calculate_mad_scores(
        df,
        metrics=metric_names,
        option=option,
        positive_only_metrics=metric_names,
    )
