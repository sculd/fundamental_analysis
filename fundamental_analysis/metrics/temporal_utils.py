"""Temporal feature utilities for calculating growth and change metrics."""

import polars as pl


def temporal_delta(base_expr: pl.Expr, shift: int) -> pl.Expr:
    """
    Calculate absolute change over time (current - previous).

    Used for metrics where absolute change is more meaningful (e.g., ratio deltas).

    Args:
        base_expr: The metric expression to calculate delta for
        shift: Number of periods to shift (1=QoQ, 4=YoY)
    """
    current = base_expr
    previous = base_expr.shift(shift).over("ticker", order_by="reportperiod")
    return current - previous


def temporal_change(
    base_expr: pl.Expr,
    shift: int,
    check_sign_crossing: bool = False
) -> pl.Expr:
    """
    Calculate percentage change over time: (current - previous) / |previous|.

    Args:
        base_expr: The metric expression to calculate percentage change for
        shift: Number of periods to shift (1=QoQ, 4=YoY)
        check_sign_crossing: If True, returns null when metric crosses zero
                             (avoids misleading change values for metrics that can be negative)

    Returns null when:
    - Previous value is 0
    - check_sign_crossing=True and current * previous < 0 (sign changed)
    """
    current = base_expr
    previous = base_expr.shift(shift).over("ticker", order_by="reportperiod")

    if check_sign_crossing:
        # For metrics that can be negative (debt ratios, ROE, ROIC)
        # Avoid misleading values when crossing zero
        return pl.when((previous != 0) & (current * previous > 0)).then(
            (current - previous) / previous.abs()
        ).otherwise(None)
    else:
        # For metrics that are always positive or where sign change is meaningful
        return pl.when(previous != 0).then(
            (current - previous) / previous.abs()
        ).otherwise(None)
