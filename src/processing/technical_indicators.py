"""
Technical Indicators
Computes RSI, MACD, Bollinger Bands, EMA, OBV, and ATR from OHLCV price data.
These features feed the XGBoost and LightGBM models.
All computations use pandas-ta for consistency and speed.
"""

import logging

import pandas as pd
import pandas_ta as ta

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


def compute_indicators(df):
    """
    Add all technical indicator columns to a price DataFrame.
    Input DataFrame must have columns: Open, High, Low, Close, Volume.
    Returns the same DataFrame with additional indicator columns appended.
    NaN rows at the start (due to lookback periods) are dropped.
    """
    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)

    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # RSI: momentum oscillator measuring speed and change of price movements
    df["rsi"] = ta.rsi(close, length=RSI_PERIOD)

    # MACD: trend-following momentum indicator
    macd = ta.macd(close, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL)
    if macd is not None:
        df["macd"]        = macd[f"MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
        df["macd_signal"] = macd[f"MACDs_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]
        df["macd_hist"]   = macd[f"MACDh_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}"]

    # Bollinger Bands: volatility bands around a simple moving average
    bb = ta.bbands(close, length=BB_PERIOD)
    if bb is not None:
        df["bb_upper"]  = bb[f"BBU_{BB_PERIOD}_2.0"]
        df["bb_mid"]    = bb[f"BBM_{BB_PERIOD}_2.0"]
        df["bb_lower"]  = bb[f"BBL_{BB_PERIOD}_2.0"]
        df["bb_width"]  = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # EMA: exponential moving average giving more weight to recent prices
    for period in EMA_PERIODS:
        df[f"ema_{period}"] = ta.ema(close, length=period)

    # OBV: on-balance volume measures buying and selling pressure
    df["obv"] = ta.obv(close, volume)

    # ATR: average true range measures market volatility
    df["atr"] = ta.atr(high, low, close, length=14)

    # Price momentum features
    df["price_change_1d"]  = close.pct_change(1)
    df["price_change_5d"]  = close.pct_change(5)
    df["price_change_10d"] = close.pct_change(10)

    # Rolling volatility
    df["volatility_10d"] = close.pct_change().rolling(10).std()
    df["volatility_20d"] = close.pct_change().rolling(20).std()

    # Volume features
    df["volume_ma_10"]   = volume.rolling(10).mean()
    df["volume_ratio"]   = volume / df["volume_ma_10"]

    # Direction label for classification models (1 = price went up, 0 = price went down)
    df["direction"] = (close.shift(-1) > close).astype(int)

    rows_before = len(df)
    df = df.dropna()
    rows_after  = len(df)

    logger.info(f"Technical indicators computed. Dropped {rows_before - rows_after} NaN rows from lookback period.")
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
