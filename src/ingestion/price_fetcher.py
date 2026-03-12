import logging
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from config.settings import TICKERS, TRAIN_START_DATE, TRAIN_END_DATE, LIVE_INTERVAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fetch_historical(ticker, start=TRAIN_START_DATE, end=TRAIN_END_DATE):
    """
    Download full OHLCV history for a single ticker from Yahoo Finance.
    Returns a cleaned DataFrame with Date as a column (not index).
    """
    logger.info(f"Fetching historical data for {ticker} from {start} to {end}")

    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        logger.warning(f"No data returned for {ticker}")
        return pd.DataFrame()

    raw = raw.reset_index()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw["Ticker"] = ticker
    raw["Date"] = pd.to_datetime(raw["Date"]).dt.date.astype(str)

    raw = raw[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]]
    raw = raw.dropna()

    logger.info(f"{ticker}: {len(raw)} trading days fetched")
    return raw


def fetch_all_tickers(tickers=TICKERS, start=TRAIN_START_DATE, end=TRAIN_END_DATE):
    """
    Fetch historical data for all tickers and return as a combined DataFrame.
    """
    frames = []
    for ticker in tickers:
        df = fetch_historical(ticker, start, end)
        if not df.empty:
            frames.append(df)
        time.sleep(0.5)

    if not frames:
        logger.error("No data fetched for any ticker")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total rows fetched across all tickers: {len(combined)}")
    return combined


def fetch_latest_tick(ticker):
    """
    Fetch the most recent trading day's OHLCV data for a single ticker.
    Used by the Kafka producer for live streaming.
    Returns a single-row dictionary.
    """
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=5)).strftime("%Y-%m-%d")

    raw = yf.download(ticker, start=start, end=end, progress=False)

    if raw.empty:
        logger.warning(f"No recent tick data for {ticker}")
        return None

    raw = raw.reset_index()
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    latest = raw.iloc[-1]

    return {
        "ticker": ticker,
        "date": str(latest["Date"].date()),
        "open": round(float(latest["Open"]), 4),
        "high": round(float(latest["High"]), 4),
        "low": round(float(latest["Low"]), 4),
        "close": round(float(latest["Close"]), 4),
        "volume": int(latest["Volume"]),
        "fetched_at": datetime.utcnow().isoformat()
    }
