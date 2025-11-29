from dataclasses import dataclass

@dataclass
class ScoreOption:
    """Configuration for scoring.

    Parameters
    ----------
    window_days : int, default 180
        Rolling window size in days for calculating segment statistics
    segment_col : str, default "segment"
        Column name for segmentation
    date_col : str, default "datekey"
        Column name for date
    """

    window_days: int = 180
    segment_col: str = "segment"
    date_col: str = "datekey"


# Define all metrics with their favorable direction
# Format: (metric_name, direction)
# direction: "lower" = low values are good, "higher" = high values are good
ALL_METRICS = [
    # Valuation metrics (lower is better - cheaper)
    ("pe_ratio", "lower"),
    ("pb_ratio", "lower"),
    ("ps_ratio", "lower"),
    ("pc_ratio", "lower"),
    ("ev_ebitda_ratio", "lower"),
    # Profitability metrics (higher is better)
    ("roe_calculated", "higher"),
    ("roic_calculated", "higher"),
    # Liquidity metrics (higher is better)
    ("current_ratio", "higher"),
    ("interest_coverage", "higher"),
    # Leverage metrics (lower is better)
    ("debt_to_equity", "lower"),
    ("debt_to_assets", "lower"),
]
