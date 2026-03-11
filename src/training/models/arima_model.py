"""
ARIMA Model
Classical AutoRegressive Integrated Moving Average baseline.
No machine learning, no sentiment. Pure statistical time series forecasting.
Establishes the floor: how much does adding ML and sentiment actually improve results?
"""

import logging
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from config.settings import TRAIN_SPLIT

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Standard ARIMA order. p=5 AR lags, d=1 differencing, q=0 MA terms
ARIMA_ORDER = (5, 1, 0)


def check_stationarity(series):
    """
    Run Augmented Dickey-Fuller test to check if the series is stationary.
    Returns True if stationary (p-value < 0.05), False otherwise.
    """
    result = adfuller(series.dropna())
    p_value = result[1]
    is_stationary = p_value < 0.05
    logger.info(f"ADF p-value: {p_value:.4f} | Stationary: {is_stationary}")
    return is_stationary


def train_and_predict(ticker_df):
    """
    Fit ARIMA on the training portion and generate one-step-ahead predictions
    on the test portion using a rolling window approach.

    Rolling prediction is more realistic than fitting once and forecasting ahead
    because ARIMA degrades significantly over long forecast horizons.

    Returns a DataFrame with columns: Date, actual, predicted.
    """
    close = ticker_df["Close"].values
    dates = ticker_df["Date"].values

    split_idx  = int(len(close) * TRAIN_SPLIT)
    train_data = list(close[:split_idx])
    test_data  = close[split_idx:]
    test_dates = dates[split_idx:]

    logger.info(f"ARIMA training on {len(train_data)} points, predicting {len(test_data)} steps")

    predictions = []

    for i in range(len(test_data)):
        try:
            model  = ARIMA(train_data, order=ARIMA_ORDER)
            fitted = model.fit()
            yhat   = fitted.forecast(steps=1)[0]
        except Exception as e:
            logger.warning(f"ARIMA failed at step {i}: {e}. Using last known value.")
            yhat = train_data[-1]

        predictions.append(yhat)
        train_data.append(test_data[i])

        if i % 20 == 0:
            logger.info(f"  ARIMA rolling prediction: {i}/{len(test_data)}")

    result = pd.DataFrame({
        "Date":      test_dates,
        "actual":    test_data,
        "predicted": predictions
    })

    logger.info("ARIMA prediction complete")
    return result
