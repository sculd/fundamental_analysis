#!/usr/bin/env python3
"""Test script for options data fetching."""
import argparse
from fundamental_analysis.optionbook.optionbook import get_options_summary


def main():
    parser = argparse.ArgumentParser(description="Fetch options chain for a ticker")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument(
        "--min-days", type=int, default=150,
        help="Minimum days to expiration (default: 150 for 90-day hold + buffer)"
    )
    parser.add_argument(
        "--max-days", type=int, default=400,
        help="Maximum days to expiration (default: 400)"
    )
    parser.add_argument(
        "--type", choices=["call", "put"], default="call",
        help="Option type (default: call)"
    )
    args = parser.parse_args()

    print(f"\nFetching {args.type}s for {args.ticker}...")
    print(f"Expiration range: {args.min_days} - {args.max_days} days\n")

    try:
        summary = get_options_summary(
            ticker=args.ticker.upper(),
            min_days=args.min_days,
            max_days=args.max_days,
            option_type=args.type,
        )

        if not summary.expirations:
            print(f"No options found for {args.ticker} in the specified range.")
            return

        print(summary.format_table())

    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
