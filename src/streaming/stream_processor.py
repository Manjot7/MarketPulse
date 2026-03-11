"""
Stream Processor
Kafka consumer running on the Oracle VM.
Reads enriched tick messages from Kafka, runs live inference using the
production model loaded from MLflow, writes predictions to Neon PostgreSQL
and caches the latest prediction in Redis.

Every DRIFT_CHECK_INTERVAL ticks, a drift check is run comparing recent
incoming data against the training reference distribution. If drift severity
exceeds the emergency threshold, retraining is triggered immediately in a
background thread without interrupting the stream.
"""

import json
import logging
import time
from collections import defaultdict

import mlflow.pyfunc
import numpy as np
import pandas as pd
import psycopg2
import redis
from kafka import KafkaConsumer
from kafka.errors import KafkaError

from config.settings import (
    KAFKA_BROKER,
    KAFKA_TOPIC_RAW,
    KAFKA_GROUP_ID,
    REDIS_HOST,
    REDIS_PORT,
    REDIS_TTL,
    DATABASE_URL,
    DRIFT_CHECK_INTERVAL,
    PROCESSED_DIR
)
from src.mlops.experiment_tracker import get_production_model_uri
from src.mlops.drift_monitor import check_and_handle_drift

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

tick_counter = 0

# Rolling buffer of recent ticks per ticker used as the current distribution
# in drift checks. Stores the last DRIFT_CHECK_INTERVAL ticks per ticker.
recent_ticks = defaultdict(list)

# Features we track for drift detection — must match training feature names
DRIFT_FEATURES = [
    "close", "finbert_score", "vader_score",
    "price_change_1d", "volatility"
]


def build_consumer():
    """
    Initialize and return a Kafka consumer with retry logic.
    """
    while True:
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC_RAW,
                bootstrap_servers=KAFKA_BROKER,
                group_id=KAFKA_GROUP_ID,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=60000
            )
            logger.info(f"Kafka consumer connected to {KAFKA_BROKER}")
            return consumer
        except KafkaError as e:
            logger.warning(f"Kafka not ready: {e}. Retrying in 5s...")
            time.sleep(5)


def build_redis_client():
    """
    Initialize and return a Redis client.
    """
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    client.ping()
    logger.info(f"Redis connected at {REDIS_HOST}:{REDIS_PORT}")
    return client


def load_production_model(ticker):
    """
    Load the current production model for a given ticker from MLflow registry.
    Returns None if no production model exists yet.
    """
    try:
        model_name = f"FinBERT-LSTM-{ticker}"
        model_uri  = get_production_model_uri(model_name)
        model      = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Production model loaded for {ticker}")
        return model
    except Exception as e:
        logger.warning(f"Could not load production model for {ticker}: {e}")
        return None


def write_prediction_to_db(conn, ticker, date, predicted_close, sentiment_score, model_used):
    """
    Insert a prediction record into the Neon PostgreSQL predictions table.
    """
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions
                (ticker, date, predicted_close, sentiment_score, model_used, created_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (ticker, date) DO UPDATE
                SET predicted_close = EXCLUDED.predicted_close,
                    sentiment_score = EXCLUDED.sentiment_score,
                    updated_at      = NOW()
            """,
            (ticker, date, predicted_close, sentiment_score, model_used)
        )
        conn.commit()
    except Exception as e:
        logger.warning(f"DB write failed for {ticker} {date}: {e}")
        conn.rollback()


def cache_prediction_in_redis(redis_client, ticker, payload):
    """
    Cache the latest prediction for a ticker in Redis as a JSON string.
    TTL is set to REDIS_TTL seconds (default 24 hours).
    """
    key = f"prediction:{ticker}:latest"
    redis_client.setex(key, REDIS_TTL, json.dumps(payload))


def load_reference_data(ticker):
    """
    Load the training feature CSV for a ticker from the processed data directory.
    This is the reference distribution Evidently compares live data against.
    Returns None if no reference data exists yet.
    """
    import os
    path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(path):
        logger.warning(f"No reference data found for {ticker} at {path}. Drift check skipped.")
        return None
    df = pd.read_csv(path)
    available = [c for c in DRIFT_FEATURES if c in df.columns]
    return df[available].dropna()


def build_current_df(ticker):
    """
    Build a DataFrame from the recent ticks buffer for a ticker.
    Used as the current distribution in drift checks.
    Returns None if not enough ticks have been collected yet.
    """
    ticks = recent_ticks.get(ticker, [])
    if len(ticks) < 20:
        return None

    rows = []
    prev_close = None
    for t in ticks:
        close = t.get("close", 0)
        row   = {
            "close":          close,
            "finbert_score":  t.get("finbert_score", 0.0),
            "vader_score":    t.get("vader_score", 0.0),
            "price_change_1d": ((close - prev_close) / prev_close) if prev_close else 0.0,
            "volatility":     abs((close - prev_close) / prev_close) if prev_close else 0.0
        }
        rows.append(row)
        prev_close = close

    df        = pd.DataFrame(rows)
    available = [c for c in DRIFT_FEATURES if c in df.columns]
    return df[available].dropna()


def run_drift_check(ticker):
    """
    Run drift check for a ticker if enough recent ticks and reference data exist.
    Calls check_and_handle_drift which handles reporting and emergency retraining.
    """
    reference_df = load_reference_data(ticker)
    current_df   = build_current_df(ticker)

    if reference_df is None or current_df is None:
        logger.info(f"Drift check skipped for {ticker}: insufficient data")
        return

    logger.info(f"Running drift check for {ticker} ({len(current_df)} recent ticks vs {len(reference_df)} reference rows)")
    drift_fraction = check_and_handle_drift(reference_df, current_df, ticker)
    logger.info(f"Drift check complete for {ticker}: fraction={drift_fraction:.1%}")


def run_processor():
    """
    Main stream processor loop.
    Consumes tick messages from Kafka, runs inference, writes outputs.
    Runs a drift check every DRIFT_CHECK_INTERVAL ticks per ticker and
    triggers emergency retraining if significant drift is detected.
    """
    global tick_counter

    consumer     = build_consumer()
    redis_client = build_redis_client()
    db_conn      = psycopg2.connect(DATABASE_URL)

    models            = {}
    ticker_tick_counts = defaultdict(int)

    logger.info("Stream processor started. Waiting for messages...")

    for message in consumer:
        try:
            tick      = message.value
            ticker    = tick.get("ticker")
            date      = tick.get("date")
            close     = tick.get("close")
            sentiment = tick.get("finbert_score", 0.0)

            recent_ticks[ticker].append(tick)
            if len(recent_ticks[ticker]) > DRIFT_CHECK_INTERVAL:
                recent_ticks[ticker].pop(0)

            if ticker not in models:
                models[ticker] = load_production_model(ticker)

            model = models.get(ticker)

            predicted_close = None
            model_used      = "none"

            if model is not None:
                features        = np.array([[close, sentiment]])
                predicted_close = float(model.predict(features)[0])
                model_used      = f"FinBERT-LSTM-{ticker}"

            write_prediction_to_db(
                db_conn, ticker, date,
                predicted_close, sentiment, model_used
            )

            cache_prediction_in_redis(redis_client, ticker, {
                "ticker":          ticker,
                "date":            date,
                "close":           close,
                "predicted_close": predicted_close,
                "sentiment":       sentiment,
                "model":           model_used
            })

            tick_counter                += 1
            ticker_tick_counts[ticker]  += 1

            logger.info(f"Processed tick #{tick_counter}: {ticker} close={close} pred={predicted_close}")

            if ticker_tick_counts[ticker] % DRIFT_CHECK_INTERVAL == 0:
                logger.info(f"Drift check interval reached for {ticker} ({ticker_tick_counts[ticker]} ticks processed)")
                run_drift_check(ticker)

                # Reload the production model after a drift check in case
                # emergency retraining has promoted a new model during this cycle
                models[ticker] = load_production_model(ticker)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            continue


if __name__ == "__main__":
    run_processor()
