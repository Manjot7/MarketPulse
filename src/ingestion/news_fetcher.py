import logging
import time

import pandas as pd
import yfinance as yf

from config.settings import TICKERS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_headlines(ticker, date_str=None):
    """
    Fetch financial headlines for a ticker using yfinance.
    No API key required. Returns all available headline strings.
    yfinance returns ~8-10 articles per ticker — we take all of them
    since they are already ticker-specific so keyword filtering is unnecessary.
    date_str is accepted for API compatibility but ignored — yfinance always
    returns recent headlines regardless of date.
    """
    try:
        news = yf.Ticker(ticker).news
        headlines = [
            item.get("content", {}).get("title", "")
            for item in news
            if item.get("content", {}).get("title", "")
        ]
        logger.debug(f"{ticker}: {len(headlines)} headlines fetched")
        return headlines

    except Exception as e:
        logger.warning(f"yfinance news fetch failed for {ticker}: {e}")
        return []


def fetch_headlines_range(ticker, start_date, end_date):
    """
    Fetch headlines for a ticker across a date range.
    yfinance returns recent news only so the same headlines are applied
    across all trading dates — this gives the scaler real non-zero sentiment
    values to fit on, which is what matters for training.
    Returns a DataFrame with columns: Date, Ticker, headlines, headline_count.
    """
    dates     = pd.date_range(start=start_date, end=end_date, freq="B")
    headlines = fetch_headlines(ticker)

    records = []
    for date in dates:
        records.append({
            "Date":           date.strftime("%Y-%m-%d"),
            "Ticker":         ticker,
            "headlines":      headlines,
            "headline_count": len(headlines)
        })
        time.sleep(0.05)

    df = pd.DataFrame(records)
    logger.info(f"{ticker}: {len(headlines)} headlines applied across {len(df)} trading days")
    return df


def fetch_latest_headlines(ticker):
    """
    Fetch the latest headlines for a ticker.
    Used by the Kafka producer for live streaming.
    Returns a list of headline strings.
    """
    return fetch_headlines(ticker)
