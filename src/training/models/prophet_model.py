import logging

import pandas as pd
from prophet import Prophet

from config.settings import TRAIN_SPLIT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prepare_prophet_df(ticker_df, use_sentiment=True):
    """
    Reformat a ticker DataFrame into Prophet's required format.
    Prophet expects columns named 'ds' (date) and 'y' (target value).
    Sentiment score is added as an external regressor if available.
    """
    df = ticker_df[["Date", "Close"]].copy()
    df = df.rename(columns={"Date": "ds", "Close": "y"})
    df["ds"] = pd.to_datetime(df["ds"])

    if use_sentiment and "finbert_score" in ticker_df.columns:
        df["finbert_score"] = ticker_df["finbert_score"].values

    return df


def train(ticker_df, use_sentiment=True):
    """
    Train a Prophet model on the given ticker DataFrame.
    Returns the trained model and train/test DataFrames.
    """
    df = prepare_prophet_df(ticker_df, use_sentiment)
    split_idx = int(len(df) * TRAIN_SPLIT)

    train_df = df.iloc[:split_idx]
    test_df  = df.iloc[split_idx:]

    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )

    if use_sentiment and "finbert_score" in df.columns:
        model.add_regressor("finbert_score")

    model.fit(train_df)
    logger.info(f"Prophet model trained on {len(train_df)} rows")

    return model, train_df, test_df


def predict(model, test_df):
    """
    Generate predictions on the test DataFrame using a trained Prophet model.
    Returns a DataFrame with columns: ds, y (actual), yhat (predicted).
    """
    future    = model.make_future_dataframe(periods=len(test_df), freq="B")

    if "finbert_score" in test_df.columns:
        all_sentiment = pd.concat([
            model.history[["ds", "finbert_score"]],
            test_df[["ds", "finbert_score"]]
        ]).drop_duplicates("ds")
        future = future.merge(all_sentiment, on="ds", how="left")
        future["finbert_score"] = future["finbert_score"].fillna(0.0)

    forecast    = model.predict(future)
    test_preds  = forecast.tail(len(test_df))[["ds", "yhat"]].reset_index(drop=True)
    test_actual = test_df[["ds", "y"]].reset_index(drop=True)

    result = test_actual.merge(test_preds, on="ds", how="inner")
    return result
