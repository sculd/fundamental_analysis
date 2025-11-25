"""Simple sector-based segmentation."""

import polars as pl


def add_sector_segmentation(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add sector-based segmentation column to DataFrame.

    The sector information should already be present in the 'sector' column
    (typically joined from TICKERS metadata).

    This function simply creates a 'segment' column that equals the sector,
    with null sectors mapped to 'Unknown'.
    """
    df = df.with_columns(
        pl.when(pl.col("sector").is_null())
        .then(pl.lit("Unknown"))
        .otherwise(pl.col("sector"))
        .alias("segment")
    )

    return df
