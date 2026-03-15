import logging

import numpy as np
import pandas as pd

from config.settings import (
    RSI_PERIOD,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BB_PERIOD,
    EMA_PERIODS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def _rsi(series, period):
    delta  = series.diff()
    gain   = delta.clip(lower=0)
    loss   = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(series, fast, slow, signal):
    ema_fast   = _ema(series, fast)
    ema_slow   = _ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    hist       = macd_line - signal_line
    return macd_line, signal_line, hist


def _bbands(series, period):
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return upper, mid, lower, width


def _obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def compute_indicators(df):
    """
    Add all technical indicator columns to a price DataFrame.
    Input DataFrame must have columns: Open, High, Low, Close, Volume.
    Returns the same DataFrame with additional indicator columns appended.
    NaN rows at the start due to lookback periods are dropped.
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # RSI
    df["rsi"] = _rsi(close, RSI_PERIOD)

    # MACD
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(
        close, MACD_FAST, MACD_SLOW, MACD_SIGNAL
    )

    # Bollinger Bands
    df["bb_upper"], df["bb_mid"], df["bb_lower"], df["bb_width"] = _bbands(
        close, BB_PERIOD
    )

    # EMA
    for period in EMA_PERIODS:
        df[f"ema_{period}"] = _ema(close, period)

    # OBV
    df["obv"] = _obv(close, volume)

    # ATR
    df["atr"] = _atr(high, low, close, period=14)

    # Price momentum
    df["price_change_1d"]  = close.pct_change(1)
    df["price_change_5d"]  = close.pct_change(5)
    df["price_change_10d"] = close.pct_change(10)

    # Rolling volatility
    df["volatility_10d"] = close.pct_change().rolling(10).std()
    df["volatility_20d"] = close.pct_change().rolling(20).std()

    # Volume features
    df["volume_ma_10"] = volume.rolling(10).mean()
    df["volume_ratio"] = volume / df["volume_ma_10"].replace(0, np.nan)

    # Direction label for classification models
    df["direction"] = (close.shift(-1) > close).astype(int)

    rows_before = len(df)
    df          = df.dropna()
    rows_after  = len(df)

    logger.info(
        f"Technical indicators computed. "
        f"Dropped {rows_before - rows_after} NaN rows from lookback period."
    )
    return df


def compute_all_tickers(combined_df):
    """
    Compute technical indicators for each ticker in a combined DataFrame.
    Processes each ticker separately to avoid cross-contamination of rolling windows.
    Returns a single combined DataFrame with all indicators.
    """
    frames = []

    for ticker in combined_df["Ticker"].unique():
        ticker_df = combined_df[combined_df["Ticker"] == ticker].copy()
        ticker_df = compute_indicators(ticker_df)
        frames.append(ticker_df)
        logger.info(f"{ticker}: {len(ticker_df)} rows after indicator computation")

    result = pd.concat(frames, ignore_index=True)
    logger.info(f"Technical indicators complete. Total rows: {len(result)}")
    return result
