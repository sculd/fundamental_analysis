# Fundamental Analysis

Stock screening tool based on fundamental metrics from Sharadar SF1 data.

## Setup

```bash
pip install -r requirements.txt
```

Set environment variables:
```bash
export NASDAQ_DATA_API_KEY=your_key      # Required for data fetching
export ANTHROPIC_API_KEY=your_key        # Optional, for LLM analysis
```

## Usage

### 1. Fetch Data

Download SF1 fundamental data and tickers metadata:

```bash
python main_fetch.py --end-date 2025-11-30
```

### 2. Analyze Single Stock

View fundamental metrics and percentile rankings for a stock:

```bash
python main_analyze_single_stock.py --ticker AAPL
python main_analyze_single_stock.py --ticker AAPL --as-of-date 2025-09-01
python main_analyze_single_stock.py --ticker AAPL --llm  # Include Claude's analysis
```

### 3. Metric-Based Selection

Find stocks with outliers in a specific metric:

```bash
python main_metric_based_selection.py --metric roe_calculated
python main_metric_based_selection.py --metric pe_ratio --direction favorable
python main_metric_based_selection.py --metric debt_to_equity --direction unfavorable
```

Available metrics: `pe_ratio`, `pb_ratio`, `ps_ratio`, `pc_ratio`, `ev_ebitda_ratio`, `roe_calculated`, `roic_calculated`, `current_ratio`, `interest_coverage`, `debt_to_equity`, `debt_to_assets`

### 4. Count-Based Selection

Find stocks with multiple favorable/unfavorable metric outliers:

```bash
python main_count_based_selection.py --min-signals 3
python main_count_based_selection.py --min-signals 3 --max-signals 5
python main_count_based_selection.py --sort-by unfavorable_count  # Find risky stocks
```
