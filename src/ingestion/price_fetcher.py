import logging
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import TICKERS, TRAIN_START_DATE, TRAIN_END_DATE, LIVE_INTERVAL

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Indicator parameters — must match training_notebook.py exactly
RSI_PERIOD  = 14
MACD_FAST   = 12
MACD_SLOW   = 26
MACD_SIGNAL = 9
BB_PERIOD   = 20


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


def _compute_indicators(df):
    """
    Compute RSI, MACD, Bollinger Band width, price_change_1d, and volatility_10d
    on a DataFrame with Open, High, Low, Close, Volume columns.
    Must match training_notebook.py compute_indicators() exactly.
    Returns df with indicator columns added.
    """
    close = df["Close"]

    # RSI (Wilder EMA method)
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_fast   = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow   = close.ewm(span=MACD_SLOW, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow

    # Bollinger Bands width
    bb_mid         = close.rolling(BB_PERIOD).mean()
    bb_std         = close.rolling(BB_PERIOD).std()
    bb_upper       = bb_mid + 2 * bb_std
    bb_lower       = bb_mid - 2 * bb_std
    df["bb_width"] = (bb_upper - bb_lower) / bb_mid.replace(0, np.nan)

    # Price change and volatility
    df["price_change_1d"] = close.pct_change(1)
    df["volatility_10d"]  = close.pct_change().rolling(10).std()

    return df


def fetch_latest_tick(ticker):
    """
    Fetch the most recent trading day's OHLCV data for a single ticker,
    with all technical indicators computed from recent history.

    Fetches 90 days of history so that MACD (slow=26), Bollinger Bands (20),
    and volatility_10d are fully warmed up before reading the latest row.

    Returns a single-row dictionary with OHLCV + all SEQUENCE_FEATURES indicators,
    ready to be published directly to Kafka and used for inference.
    Returns None if price data is unavailable.
    """
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")

    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw is None or raw.empty:
        logger.warning(f"No recent tick data for {ticker}")
        return None

    raw = raw.reset_index()
    # Flatten MultiIndex columns (yfinance >=0.2 behaviour)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]

    df = raw[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna()

    if len(df) < MACD_SLOW + BB_PERIOD:
        logger.warning(f"Not enough history for {ticker} to compute indicators: {len(df)} rows")
        return None

    df = _compute_indicators(df)

    # Take the most recent fully-computed row
    latest = df.dropna(subset=["rsi", "macd", "bb_width", "price_change_1d", "volatility_10d"]).iloc[-1]

    return {
        "ticker":          ticker,
        "date":            str(latest["Date"].date()) if hasattr(latest["Date"], "date") else str(latest["Date"]),
        "open":            round(float(latest["Open"]),          4),
        "high":            round(float(latest["High"]),          4),
        "low":             round(float(latest["Low"]),           4),
        "close":           round(float(latest["Close"]),         4),
        "volume":          int(latest["Volume"]),
        # Technical indicators — computed from 90-day history, matching training exactly
        "rsi":             round(float(latest["rsi"]),           6),
        "macd":            round(float(latest["macd"]),          6),
        "bb_width":        round(float(latest["bb_width"]),      6),
        "price_change_1d": round(float(latest["price_change_1d"]), 6),
        "volatility_10d":  round(float(latest["volatility_10d"]),  6),
        "fetched_at":      datetime.utcnow().isoformat()
    }
