"""
Feature Builder
Merges price data, sentiment scores, and technical indicators into a unified feature matrix.
Handles missing sentiment on non-trading days by forward-filling.
Produces both sequence format (LSTM/GRU) and tabular format (XGBoost/LightGBM).
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config.settings import (
    SEQUENCE_LENGTH,
    TRAIN_SPLIT,
    PROCESSED_DIR
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def merge_features(price_df, sentiment_df, indicators_df):
    """
    Merge price, sentiment, and technical indicator DataFrames on Date and Ticker.
    Missing sentiment values are forward-filled then backfilled.
    Returns a single unified DataFrame sorted by Ticker and Date.
    """
    df = indicators_df.copy()

    if not sentiment_df.empty:
        sentiment_cols = sentiment_df[["Date", "Ticker", "finbert_score", "vader_score"]]
        df = df.merge(sentiment_cols, on=["Date", "Ticker"], how="left")
        df["finbert_score"] = df.groupby("Ticker")["finbert_score"].ffill().bfill().fillna(0.0)
        df["vader_score"]   = df.groupby("Ticker")["vader_score"].ffill().bfill().fillna(0.0)
    else:
        df["finbert_score"] = 0.0
        df["vader_score"]   = 0.0

    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    logger.info(f"Feature matrix built: {len(df)} rows, {len(df.columns)} columns")
    return df


def build_sequences(ticker_df, feature_cols, target_col="Close", seq_len=SEQUENCE_LENGTH):
    """
    Convert a flat DataFrame into 3D sequences for LSTM/GRU models.
    Returns X (samples, seq_len, features), y (samples,), and the fitted scaler.
    """
    data   = ticker_df[feature_cols + [target_col]].values.astype(float)
    scaler = MinMaxScaler()
    data   = scaler.fit_transform(data)

    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len, :-1])
        y.append(data[i + seq_len, -1])

    return np.array(X), np.array(y), scaler


def train_test_split_sequences(X, y, split=TRAIN_SPLIT):
    """
    Split sequence arrays into train and test sets using a time-based split.
    No shuffling to preserve temporal order.
    """
    split_idx = int(len(X) * split)
    return X[:split_idx], X[split_idx:], y[:split_idx], y[split_idx:]


def build_tabular(ticker_df, feature_cols, target_col="direction", split=TRAIN_SPLIT):
    """
    Build flat tabular feature matrix for XGBoost/LightGBM/RandomForest.
    Returns train/test splits without scaling (tree models don't require it).
    """
    df = ticker_df.dropna(subset=feature_cols + [target_col])
    X  = df[feature_cols].values
    y  = df[target_col].values

    split_idx = int(len(X) * split)
    return (
        X[:split_idx], X[split_idx:],
        y[:split_idx], y[split_idx:]
    )


def save_features(df, ticker):
    """
    Save the full feature DataFrame for a ticker to the processed data directory.
    Used for caching between runs and for Cloudflare R2 archival.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    df.to_csv(path, index=False)
    logger.info(f"Features saved to {path}")
    return path


def load_features(ticker):
    """
    Load cached feature DataFrame for a ticker from the processed data directory.
    Returns None if file does not exist.
    """
    path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(path):
        logger.warning(f"No cached features found for {ticker} at {path}")
        return None

    df = pd.read_csv(path)
    logger.info(f"Loaded cached features for {ticker}: {len(df)} rows")
    return df


# Columns used for LSTM/GRU sequence models (price + sentiment)
SEQUENCE_FEATURE_COLS = [
    "Open", "High", "Low", "Volume",
    "finbert_score", "vader_score",
    "rsi", "macd", "bb_width",
    "price_change_1d", "volatility_10d"
]

# Columns used for tabular gradient boosting models
TABULAR_FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "rsi", "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_lower", "bb_width",
    "ema_9", "ema_21", "ema_50",
    "obv", "atr",
    "price_change_1d", "price_change_5d", "price_change_10d",
    "volatility_10d", "volatility_20d",
    "volume_ratio",
    "finbert_score", "vader_score"
]
