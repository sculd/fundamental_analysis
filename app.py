"""Flask app to display stock verdicts with expandable metrics and analysis."""
import csv
import json
from pathlib import Path
from flask import Flask, render_template

app = Flask(__name__)

DATA_DIR = Path(__file__).parent
VERDICTS_FILE = DATA_DIR / "verdicts.csv"
CACHE_FILE = DATA_DIR / "llm_readable_cache" / "cached.jsonl"


def load_data():
    """Load verdicts and cached analysis, joining by ticker."""
    # Load cache into dict keyed by ticker
    cache = {}
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    cache[entry["ticker"]] = {
                        "metrics_str": entry.get("metrics_str", ""),
                        "analysis": entry.get("analysis", ""),
                    }

    # Load verdicts and join with cache
    verdicts = []
    if VERDICTS_FILE.exists():
        with open(VERDICTS_FILE, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ticker = row["ticker"]
                cached = cache.get(ticker, {})
                verdicts.append({
                    "ticker": ticker,
                    "date": row.get("date", ""),
                    "comment": row.get("comment", ""),
                    "metrics_str": cached.get("metrics_str", "No metrics available"),
                    "analysis": cached.get("analysis", "No analysis available"),
                })
    return verdicts


@app.route("/")
def index():
    verdicts = load_data()
    return render_template("index.html", verdicts=verdicts)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
