"""Sigma-based (z-score) outlier screening."""

from typing import Literal

import polars as pl

from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


def calculate_z_scores(
    df: pl.DataFrame,
    metrics: list[str],
    segment_col: str = "segment",
) -> pl.DataFrame:
    """
    Calculate z-scores for specified metrics within each segment.

    For each metric, calculates:
    - Segment mean
    - Segment standard deviation
    - Z-score: (value - mean) / std

    Adds columns: {metric}_zscore, {metric}_mean, {metric}_std
    """
    # Calculate segment-level statistics
    segment_stats = df.group_by(segment_col).agg(
        [
            pl.col(metric).mean().alias(f"{metric}_mean")
            for metric in metrics
        ] + [
            pl.col(metric).std().alias(f"{metric}_std")
            for metric in metrics
        ]
    )

    # Join stats back to original dataframe
    df = df.join(segment_stats, on=segment_col, how="left")

    # Calculate z-scores
    z_score_exprs = []
    for metric in metrics:
        z_score_exprs.append(
            (
                (pl.col(metric) - pl.col(f"{metric}_mean")) / pl.col(f"{metric}_std")
            ).alias(f"{metric}_zscore")
        )

    df = df.with_columns(z_score_exprs)

    return df


def screen_by_sigma(
    df: pl.DataFrame,
    metric: str,
    sigma_threshold: float = 2.0,
    direction: Literal["lower", "higher", "both"] = "both",
    segment_col: str = "segment",
) -> pl.DataFrame:
    """
    Screen stocks based on sigma (z-score) outliers for a specific metric.

    Args:
        df: DataFrame with metrics and segments
        metric: Metric column name to screen on
        sigma_threshold: Number of standard deviations from mean (default: 2.0)
        direction:
            - "lower": Select stocks with z-score < -sigma_threshold (undervalued)
            - "higher": Select stocks with z-score > sigma_threshold (overperforming)
            - "both": Select stocks with |z-score| > sigma_threshold (any outlier)
        segment_col: Column name containing segment labels

    Returns:
        Filtered DataFrame containing only outlier stocks
    """
    # Ensure z-scores are calculated
    if f"{metric}_zscore" not in df.columns:
        df = calculate_z_scores(df, [metric], segment_col=segment_col)

    # Apply filtering based on direction
    if direction == "lower":
        df_filtered = df.filter(pl.col(f"{metric}_zscore") < -sigma_threshold)
    elif direction == "higher":
        df_filtered = df.filter(pl.col(f"{metric}_zscore") > sigma_threshold)
    elif direction == "both":
        df_filtered = df.filter(pl.col(f"{metric}_zscore").abs() > sigma_threshold)
    else:
        raise ValueError(f"Invalid direction: {direction}. Must be 'lower', 'higher', or 'both'")

    logger.info(
        f"Screened {len(df_filtered)} stocks (out of {len(df)}) "
        f"with {metric} z-score {direction} {sigma_threshold} sigma"
    )

    return df_filtered


def screen_multi_metric_sigma(
    df: pl.DataFrame,
    metric_configs: list[dict],
    sigma_threshold: float = 2.0,
    require_all: bool = True,
    segment_col: str = "segment",
) -> pl.DataFrame:
    """
    Screen stocks based on multiple metric conditions.

    Args:
        df: DataFrame with metrics and segments
        metric_configs: List of dicts, each with:
            - "metric": metric column name
            - "direction": "lower", "higher", or "both"
            - "sigma_threshold": (optional) override default threshold
        sigma_threshold: Default sigma threshold if not specified per metric
        require_all: If True, stocks must satisfy ALL conditions (AND logic).
                    If False, stocks satisfy ANY condition (OR logic).
        segment_col: Column name containing segment labels

    Example:
        metric_configs = [
            {"metric": "pe_ratio", "direction": "lower"},  # Low P/E
            {"metric": "roe_calculated", "direction": "higher"},  # High ROE
        ]
    """
    # Collect all metrics and calculate z-scores in one pass
    all_metrics = [config["metric"] for config in metric_configs]
    df = calculate_z_scores(df, all_metrics, segment_col=segment_col)

    # Build filter conditions
    conditions = []
    for config in metric_configs:
        metric = config["metric"]
        direction = config["direction"]
        threshold = config.get("sigma_threshold", sigma_threshold)

        if direction == "lower":
            condition = pl.col(f"{metric}_zscore") < -threshold
        elif direction == "higher":
            condition = pl.col(f"{metric}_zscore") > threshold
        elif direction == "both":
            condition = pl.col(f"{metric}_zscore").abs() > threshold
        else:
            raise ValueError(f"Invalid direction: {direction}")

        conditions.append(condition)

    # Combine conditions
    if require_all:
        # AND logic: all conditions must be true
        combined_filter = conditions[0]
        for condition in conditions[1:]:
            combined_filter = combined_filter & condition
    else:
        # OR logic: any condition can be true
        combined_filter = conditions[0]
        for condition in conditions[1:]:
            combined_filter = combined_filter | condition

    df_filtered = df.filter(combined_filter)

    logic_str = "AND" if require_all else "OR"
    logger.info(
        f"Multi-metric screening ({logic_str} logic) found {len(df_filtered)} stocks "
        f"(out of {len(df)}) matching {len(metric_configs)} conditions"
    )

    return df_filtered
