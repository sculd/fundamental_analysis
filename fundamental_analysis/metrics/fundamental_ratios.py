"""Fundamental valuation ratio calculations."""

import polars as pl

from fundamental_analysis.metrics.temporal_utils import temporal_delta


def _pe_ratio_expr() -> pl.Expr:
    """P/E ratio: price / diluted earnings per share."""
    return pl.when(pl.col("epsdil") != 0).then(
        pl.col("price") / pl.col("epsdil")
    ).otherwise(None)


def _pb_ratio_expr() -> pl.Expr:
    """P/B ratio: price / book value per share."""
    return pl.when(pl.col("bvps") != 0).then(
        pl.col("price") / pl.col("bvps")
    ).otherwise(None)


def _ps_ratio_expr() -> pl.Expr:
    """P/S ratio: price / sales per share."""
    return pl.when(pl.col("sps") != 0).then(
        pl.col("price") / pl.col("sps")
    ).otherwise(None)


def _pc_ratio_expr() -> pl.Expr:
    """P/C ratio: price / cash per share."""
    # Cash per share = cashneq / sharesbas
    cash_per_share = pl.when(pl.col("sharesbas") != 0).then(
        pl.col("cashneq") / pl.col("sharesbas")
    ).otherwise(None)

    return pl.when(cash_per_share != 0).then(
        pl.col("price") / cash_per_share
    ).otherwise(None)


def _ev_ebitda_ratio_expr() -> pl.Expr:
    """EV/EBITDA ratio: enterprise value / EBITDA."""
    return pl.when(pl.col("ebitda") != 0).then(
        pl.col("ev") / pl.col("ebitda")
    ).otherwise(None)


def get_fundamental_ratio_expressions() -> list[pl.Expr]:
    """
    Return list of fundamental ratio expressions for composing with other metrics.

    Use this for efficient batch calculation with other metric types.

    Includes snapshot values and temporal features (growth QoQ/YoY).
    Note: For valuation ratios, "growth" represents absolute change (delta).
    """
    return [
        # Snapshot values
        _pe_ratio_expr().alias("pe_ratio"),
        _pb_ratio_expr().alias("pb_ratio"),
        _ps_ratio_expr().alias("ps_ratio"),
        _pc_ratio_expr().alias("pc_ratio"),
        _ev_ebitda_ratio_expr().alias("ev_ebitda_ratio"),
        # Temporal features (growth - absolute change for ratios)
        temporal_delta(_pe_ratio_expr(), 1).alias("pe_ratio_growth_qoq"),
        temporal_delta(_pe_ratio_expr(), 4).alias("pe_ratio_growth_yoy"),
        temporal_delta(_pb_ratio_expr(), 1).alias("pb_ratio_growth_qoq"),
        temporal_delta(_pb_ratio_expr(), 4).alias("pb_ratio_growth_yoy"),
        temporal_delta(_ps_ratio_expr(), 1).alias("ps_ratio_growth_qoq"),
        temporal_delta(_ps_ratio_expr(), 4).alias("ps_ratio_growth_yoy"),
        temporal_delta(_pc_ratio_expr(), 1).alias("pc_ratio_growth_qoq"),
        temporal_delta(_pc_ratio_expr(), 4).alias("pc_ratio_growth_yoy"),
        temporal_delta(_ev_ebitda_ratio_expr(), 1).alias("ev_ebitda_ratio_growth_qoq"),
        temporal_delta(_ev_ebitda_ratio_expr(), 4).alias("ev_ebitda_ratio_growth_yoy"),
    ]
