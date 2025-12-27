"""Options data fetcher using Polygon.io (Massive) API."""
import os
from datetime import datetime, timedelta
from dataclasses import dataclass
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("MASSIVE_API_KEY")
BASE_URL = "https://api.polygon.io"


@dataclass
class OptionQuote:
    ticker: str
    underlying: str
    strike: float
    expiration: str
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    mid: float
    last: float
    volume: int
    open_interest: int
    delta: float | None
    gamma: float | None
    theta: float | None
    vega: float | None
    iv: float | None
    underlying_price: float

    @property
    def break_even(self) -> float:
        if self.option_type == "call":
            return self.strike + self.mid
        return self.strike - self.mid

    @property
    def break_even_pct(self) -> float:
        return (self.break_even / self.underlying_price - 1) * 100

    @property
    def moneyness(self) -> str:
        if self.option_type == "call":
            diff = (self.strike / self.underlying_price - 1) * 100
        else:
            diff = (self.underlying_price / self.strike - 1) * 100
        if abs(diff) < 2:
            return "ATM"
        return f"{diff:+.0f}% OTM" if diff > 0 else f"{-diff:.0f}% ITM"


@dataclass
class OptionsSummary:
    ticker: str
    underlying_price: float | None
    expirations: dict[str, list["OptionQuote"]]

    def format_table(self) -> str:
        """Format options summary as a readable table."""
        lines = []

        lines.append(f"\n{'='*70}")
        lines.append(f"  {self.ticker} Options Chain - Current Price: ${self.underlying_price:.2f}")
        lines.append(f"{'='*70}\n")

        for exp, quotes in sorted(self.expirations.items()):
            days_to_exp = (datetime.strptime(exp, "%Y-%m-%d").date() - datetime.now().date()).days
            lines.append(f"Expiration: {exp} ({days_to_exp} days)")
            lines.append("-" * 70)
            lines.append(
                f"{'Strike':>8} {'Type':>8} {'Price':>8} {'Delta':>7} {'Theta':>7} "
                f"{'IV':>6} {'B/E':>8} {'B/E %':>7}"
            )
            lines.append("-" * 70)

            for q in quotes:
                delta_str = f"{q.delta:.2f}" if q.delta else "N/A"
                theta_str = f"{q.theta:.2f}" if q.theta else "N/A"
                iv_str = f"{q.iv*100:.0f}%" if q.iv else "N/A"

                lines.append(
                    f"${q.strike:>7.2f} {q.moneyness:>8} ${q.mid:>6.2f} {delta_str:>7} "
                    f"{theta_str:>7} {iv_str:>6} ${q.break_even:>6.2f} {q.break_even_pct:>+6.1f}%"
                )

            lines.append("")

        return "\n".join(lines)


def _get_underlying_price(ticker: str) -> float | None:
    """Get current price for underlying stock."""
    url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/prev"
    params = {"apiKey": API_KEY}

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return None

    data = response.json()
    results = data.get("results", [])
    if results:
        return results[0].get("c")  # closing price
    return None


def _get_options_chain(
    ticker: str,
    min_days: int = 90,
    max_days: int = 400,
    option_type: str = "call",
    limit: int = 100,
) -> list[OptionQuote]:
    """Fetch options chain for a ticker within expiration range."""
    today = datetime.now().date()
    min_exp = today + timedelta(days=min_days)
    max_exp = today + timedelta(days=max_days)

    # First get underlying price
    underlying_price = _get_underlying_price(ticker)
    if not underlying_price:
        raise ValueError(f"Could not fetch underlying price for {ticker}")

    # Fetch options snapshot
    url = f"{BASE_URL}/v3/snapshot/options/{ticker}"
    params = {
        "apiKey": API_KEY,
        "limit": limit,
        "expiration_date.gte": min_exp.isoformat(),
        "expiration_date.lte": max_exp.isoformat(),
        "contract_type": option_type,
        "order": "asc",
        "sort": "expiration_date",
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "OK":
        raise ValueError(f"API error: {data}")

    quotes = []
    for result in data.get("results", []):
        details = result.get("details", {})
        day = result.get("day", {})
        greeks = result.get("greeks", {})
        underlying = result.get("underlying_asset", {})

        quote = OptionQuote(
            ticker=details.get("ticker", ""),
            underlying=ticker,
            strike=details.get("strike_price", 0),
            expiration=details.get("expiration_date", ""),
            option_type=details.get("contract_type", "").lower(),
            bid=day.get("close", 0) or 0,  # Use close as proxy if no bid
            ask=day.get("close", 0) or 0,
            mid=day.get("close", 0) or 0,
            last=day.get("close", 0) or 0,
            volume=day.get("volume", 0) or 0,
            open_interest=day.get("open_interest", 0) or 0,
            delta=greeks.get("delta"),
            gamma=greeks.get("gamma"),
            theta=greeks.get("theta"),
            vega=greeks.get("vega"),
            iv=result.get("implied_volatility"),
            underlying_price=underlying.get("price") or underlying_price,
        )
        quotes.append(quote)

    return quotes


def get_options_summary(
    ticker: str,
    min_days: int = 150,  # 90-day hold + buffer
    max_days: int = 400,
    option_type: str = "call",
) -> OptionsSummary:
    """Get a summary of interesting options for a ticker.

    Returns options grouped by expiration with ATM and OTM strikes.
    """
    quotes = _get_options_chain(ticker, min_days, max_days, option_type)
    if not quotes:
        return OptionsSummary(ticker=ticker, underlying_price=None, expirations={})

    underlying_price = quotes[0].underlying_price

    # Group by expiration
    by_expiration: dict[str, list[OptionQuote]] = {}
    for q in quotes:
        if q.expiration not in by_expiration:
            by_expiration[q.expiration] = []
        by_expiration[q.expiration].append(q)

    # For each expiration, find ATM and key OTM strikes
    expirations: dict[str, list[OptionQuote]] = {}

    for exp, exp_quotes in by_expiration.items():
        # Sort by distance from ATM
        exp_quotes.sort(key=lambda q: abs(q.strike - underlying_price))

        selected = []
        # ATM (closest to current price)
        if exp_quotes:
            selected.append(exp_quotes[0])

        # Find ~10% OTM, ~20% OTM, ~30% OTM
        for target_pct in [10, 20, 30]:
            if option_type == "call":
                target_strike = underlying_price * (1 + target_pct / 100)
            else:
                target_strike = underlying_price * (1 - target_pct / 100)

            closest = min(exp_quotes, key=lambda q: abs(q.strike - target_strike), default=None)
            if closest and closest not in selected:
                selected.append(closest)

        selected.sort(key=lambda q: q.strike)
        expirations[exp] = selected

    return OptionsSummary(
        ticker=ticker,
        underlying_price=underlying_price,
        expirations=expirations,
    )


