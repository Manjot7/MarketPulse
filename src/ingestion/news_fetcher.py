import logging
import time
from datetime import datetime, timedelta

import requests
import pandas as pd

from config.settings import NEWSAPI_KEY, NEWSAPI_BASE, NEWS_PER_DAY, TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Maps ticker symbols to human-readable company names for better news queries
TICKER_QUERY_MAP = {
    "NDX":   "Nasdaq 100 index technology stocks",
    "SPY":   "S&P 500 stock market economy",
    "AAPL":  "Apple Inc earnings revenue iPhone",
    "MSFT":  "Microsoft earnings cloud Azure",
    "GOOGL": "Google Alphabet earnings search advertising",
    "NVDA":  "Nvidia GPU AI chips semiconductors",
    "TSLA":  "Tesla electric vehicles Elon Musk",
    "META":  "Meta Facebook Instagram advertising",
    "JPM":   "JPMorgan Chase bank earnings interest rates",
    "AMZN":  "Amazon AWS cloud ecommerce earnings"
}


def fetch_headlines(ticker, date_str, num_articles=NEWS_PER_DAY):
    """
    Fetch top financial headlines for a ticker on a specific date from NewsAPI.
    Returns a list of headline strings. Returns empty list on failure.
    """
    if not NEWSAPI_KEY:
        logger.error("NEWSAPI_KEY not set in environment")
        return []

    query = TICKER_QUERY_MAP.get(ticker, ticker)

    params = {
        "q": query,
        "from": date_str,
        "to": date_str,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": num_articles,
        "apiKey": NEWSAPI_KEY
    }

    try:
        response = requests.get(NEWSAPI_BASE, params=params, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        headlines = [a["title"] for a in articles if a.get("title")]
        logger.debug(f"{ticker} {date_str}: {len(headlines)} headlines fetched")
        return headlines

    except requests.exceptions.RequestException as e:
        logger.warning(f"NewsAPI request failed for {ticker} on {date_str}: {e}")
        return []


def fetch_headlines_range(ticker, start_date, end_date):
    """
    Fetch headlines for a ticker across a date range.
    Returns a DataFrame with columns: Date, Ticker, headlines (list), headline_count.
    """
    records = []
    dates = pd.date_range(start=start_date, end=end_date, freq="B")

    for date in dates:
        date_str = date.strftime("%Y-%m-%d")
        headlines = fetch_headlines(ticker, date_str)

        records.append({
            "Date": date_str,
            "Ticker": ticker,
            "headlines": headlines,
            "headline_count": len(headlines)
        })

        time.sleep(0.3)

    df = pd.DataFrame(records)
    logger.info(f"{ticker}: headlines fetched for {len(df)} trading days")
    return df


def fetch_latest_headlines(ticker, num_articles=NEWS_PER_DAY):
    """
    Fetch today's headlines for a ticker.
    Used by the Kafka producer for live streaming.
    Returns a list of headline strings.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")

    headlines = fetch_headlines(ticker, today, num_articles)

    if not headlines:
        logger.info(f"No headlines for {ticker} today, trying yesterday")
        headlines = fetch_headlines(ticker, yesterday, num_articles)

    return headlines
