import json
import logging
import os
import shutil
import time
from collections import defaultdict

import boto3
import joblib
import numpy as np
import pandas as pd
import psycopg2
import redis
import tensorflow as tf
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
    PROCESSED_DIR,
    R2_ENDPOINT_URL,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME
)
from src.mlops.drift_monitor import check_and_handle_drift
from src.processing.sentiment_scorer import score_headlines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

for noisy in ("kafka", "boto3", "botocore", "urllib3", "s3transfer"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

tick_counter = 0

# Rolling buffer of recent ticks per ticker used for drift detection and inference.
# Populated from Redis on startup so restarts don't lose buffered ticks.
recent_ticks = defaultdict(list)

# Redis key for persisting the tick buffer across restarts
TICK_BUFFER_KEY = "tick_buffer:{ticker}"

# Features we track for drift detection
DRIFT_FEATURES = [
    "close", "finbert_score", "vader_score",
    "price_change_1d", "volatility"
]

# Must match SEQUENCE_FEATURES from training_notebook.py exactly
SEQUENCE_FEATURES = [
    "Open", "High", "Low", "Volume",
    "finbert_score", "vader_score",
    "rsi", "macd", "bb_width",
    "price_change_1d", "volatility_10d"
]

SEQUENCE_LEN = 10   # Must match training SEQUENCE_LEN
N_FEATURES   = len(SEQUENCE_FEATURES) + 1  # +1 for Close target column

TICKERS = ["^NDX", "SPY", "AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "JPM", "AMZN"]


# ── S3 client ─────────────────────────────────────────────────────────────────

def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )


# ── Kafka / Redis ──────────────────────────────────────────────────────────────

def build_consumer():
    while True:
        try:
            consumer = KafkaConsumer(
                KAFKA_TOPIC_RAW,
                bootstrap_servers=KAFKA_BROKER,
                group_id=KAFKA_GROUP_ID,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                auto_offset_reset="latest",
                enable_auto_commit=True,
                consumer_timeout_ms=-1
            )
            logger.info(f"Kafka consumer connected to {KAFKA_BROKER}")
            return consumer
        except KafkaError as e:
            logger.warning(f"Kafka not ready: {e}. Retrying in 5s...")
            time.sleep(5)


def build_redis_client():
    client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    client.ping()
    logger.info(f"Redis connected at {REDIS_HOST}:{REDIS_PORT}")
    return client


# ── Tick buffer persistence ───────────────────────────────────────────────────

def save_tick_buffer(redis_client, ticker):
    """
    Persist the in-memory tick buffer for a ticker to Redis as a JSON list.
    Uses a separate key from prediction cache with a long TTL (7 days) so
    the buffer survives processor restarts without growing unbounded.
    """
    key = TICK_BUFFER_KEY.format(ticker=ticker)
    try:
        redis_client.setex(
            key,
            60 * 60 * 24 * 7,   # 7 days — covers weekends and short outages
            json.dumps(recent_ticks[ticker])
        )
    except Exception as e:
        logger.warning(f"Failed to persist tick buffer for {ticker}: {e}")


def restore_tick_buffers(redis_client):
    """
    On startup, reload tick buffers from Redis into the in-memory defaultdict.
    This means a restarted processor can produce predictions immediately
    rather than waiting for SEQUENCE_LEN new ticks to accumulate.
    Logs how many ticks were restored per ticker.
    """
    logger.info("Restoring tick buffers from Redis...")
    restored = 0

    for ticker in TICKERS:
        key = TICK_BUFFER_KEY.format(ticker=ticker)
        try:
            raw = redis_client.get(key)
            if raw:
                ticks = json.loads(raw)
                # Only keep the last max(DRIFT_CHECK_INTERVAL, SEQUENCE_LEN) ticks
                max_buffer = max(DRIFT_CHECK_INTERVAL, SEQUENCE_LEN)
                recent_ticks[ticker] = ticks[-max_buffer:]
                logger.info(f"  {ticker}: restored {len(recent_ticks[ticker])} ticks from Redis")
                restored += 1
            else:
                logger.info(f"  {ticker}: no buffer in Redis, starting fresh")
        except Exception as e:
            logger.warning(f"  {ticker}: buffer restore failed ({e}), starting fresh")

    logger.info(f"Tick buffer restore complete ({restored}/{len(TICKERS)} tickers had saved state)")


# ── DB ────────────────────────────────────────────────────────────────────────

def get_production_model_name(ticker):
    """
    Query production_models table for the promoted model name for this ticker.
    Falls back to GRU if no record exists.
    """
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT model_name FROM production_models WHERE ticker = %s", (ticker,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else "GRU"
    except Exception as e:
        logger.warning(f"Could not fetch production model name for {ticker}: {e}. Defaulting to GRU.")
        return "GRU"


def write_prediction_to_db(conn, ticker, date, predicted_close, sentiment_score, model_used):
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


# ── Model + Scaler loading ────────────────────────────────────────────────────

def load_production_model(ticker):
    """
    Download the .keras model zip from R2 and load it with tf.keras.models.load_model.
    Also downloads the scaler .pkl from R2.
    Returns (model, scaler) or (None, None) on failure.

    R2 key format:
      model:  models/{ticker}/{model_name}/{model_name}_{ticker}
      scaler: models/{ticker}/{model_name}/{model_name}_{ticker}_scaler
    """
    try:
        model_name  = get_production_model_name(ticker)
        safe_ticker = ticker.replace("^", "")   # ^NDX → NDX for local paths
        s3          = get_s3_client()

        # ── Model ──
        model_r2_key = f"models/{ticker}/{model_name}/{model_name}_{ticker}"
        zip_path     = f"/tmp/{model_name}_{safe_ticker}.zip"
        keras_path   = f"/tmp/{model_name}_{safe_ticker}.keras"

        s3.download_file(R2_BUCKET_NAME, model_r2_key, zip_path)
        shutil.copy(zip_path, keras_path)   # zip IS a .keras archive
        model = tf.keras.models.load_model(keras_path, compile=False)
        logger.info(f"Loaded {model_name} for {ticker} — input shape: {model.input_shape}")

        # ── Scaler ──
        scaler_r2_key = f"models/{ticker}/{model_name}/{model_name}_{ticker}_scaler"
        scaler_path   = f"/tmp/{model_name}_{safe_ticker}_scaler.pkl"

        s3.download_file(R2_BUCKET_NAME, scaler_r2_key, scaler_path)
        scaler = joblib.load(scaler_path)
        logger.info(f"Loaded scaler for {ticker}")

        return model, scaler

    except Exception as e:
        logger.warning(f"Could not load production model for {ticker}: {e}")
        return None, None


# ── Inference ─────────────────────────────────────────────────────────────────

def inverse_transform_prediction(pred_scaled, scaler):
    """
    Inverse transform a scaled model output back to USD price.
    The scaler was fit on [SEQUENCE_FEATURES + Close], so Close is the last column.
    """
    dummy        = np.zeros((1, N_FEATURES))
    dummy[0, -1] = pred_scaled
    return float(scaler.inverse_transform(dummy)[0, -1])


# ── Redis cache ───────────────────────────────────────────────────────────────

def cache_prediction_in_redis(redis_client, ticker, payload):
    key = f"prediction:{ticker}:latest"
    redis_client.setex(key, REDIS_TTL, json.dumps(payload))


# ── Drift detection ───────────────────────────────────────────────────────────

def load_reference_data(ticker):
    path = os.path.join(PROCESSED_DIR, f"{ticker}_features.csv")
    if not os.path.exists(path):
        logger.warning(f"No reference data for {ticker} at {path}. Drift check skipped.")
        return None
    df        = pd.read_csv(path)
    available = [c for c in DRIFT_FEATURES if c in df.columns]
    return df[available].dropna()


def build_current_df(ticker):
    ticks = recent_ticks.get(ticker, [])
    if len(ticks) < 20:
        return None

    rows       = []
    prev_close = None
    for t in ticks:
        close = t.get("close", 0)
        row   = {
            "close":           close,
            "finbert_score":   t.get("finbert_score", 0.0),
            "vader_score":     t.get("vader_score", 0.0),
            "price_change_1d": ((close - prev_close) / prev_close) if prev_close else 0.0,
            "volatility":      abs((close - prev_close) / prev_close) if prev_close else 0.0
        }
        rows.append(row)
        prev_close = close

    df        = pd.DataFrame(rows)
    available = [c for c in DRIFT_FEATURES if c in df.columns]
    return df[available].dropna()


def backfill_actual_closes(db_conn):
    """
    After market close each day, fetch the actual closing price from yfinance
    and write it back to any predictions rows that are missing actual_close.
    Called once per processor cycle — yfinance calls are only made for dates
    that actually have NULL actual_close to avoid unnecessary API hits.
    """
    import yfinance as yf
    from datetime import date, timedelta

    try:
        cursor = db_conn.cursor()

        # Find all ticker/date pairs that still need actual_close filled in
        cursor.execute(
            """
            SELECT DISTINCT ticker, date
            FROM predictions
            WHERE actual_close IS NULL
              AND date < CURRENT_DATE
            ORDER BY date DESC
            LIMIT 50
            """
        )
        rows = cursor.fetchall()

        if not rows:
            return

        logger.info(f"Backfilling actual closes for {len(rows)} prediction rows...")

        # Group by ticker to minimise yfinance calls
        from collections import defaultdict
        ticker_dates = defaultdict(list)
        for ticker, dt in rows:
            ticker_dates[ticker].append(dt)

        for ticker, dates in ticker_dates.items():
            try:
                min_date = min(dates) - timedelta(days=1)
                max_date = max(dates) + timedelta(days=1)

                df = yf.download(
                    ticker,
                    start=str(min_date),
                    end=str(max_date),
                    progress=False,
                    auto_adjust=True
                )

                if df.empty:
                    continue

                # Flatten MultiIndex columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                df.index = pd.to_datetime(df.index).date

                for dt in dates:
                    if dt in df.index:
                        actual = float(df.loc[dt, "Close"])
                        cursor.execute(
                            """
                            UPDATE predictions
                            SET actual_close = %s, updated_at = NOW()
                            WHERE ticker = %s AND date = %s
                              AND actual_close IS NULL
                            """,
                            (actual, ticker, dt)
                        )
                        logger.info(f"Backfilled actual close for {ticker} {dt}: ${actual:.2f}")

            except Exception as e:
                logger.warning(f"Backfill failed for {ticker}: {e}")

        db_conn.commit()

    except Exception as e:
        logger.warning(f"Backfill query failed: {e}")
        try:
            db_conn.rollback()
        except Exception:
            pass


def run_drift_check(ticker):
    reference_df = load_reference_data(ticker)
    current_df   = build_current_df(ticker)

    if reference_df is None or current_df is None:
        logger.info(f"Drift check skipped for {ticker}: insufficient data")
        return

    logger.info(f"Running drift check for {ticker} ({len(current_df)} recent vs {len(reference_df)} reference rows)")
    drift_fraction = check_and_handle_drift(reference_df, current_df, ticker)
    logger.info(f"Drift check complete for {ticker}: fraction={drift_fraction:.1%}")


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_processor():
    global tick_counter

    consumer           = build_consumer()
    redis_client       = build_redis_client()
    db_conn            = psycopg2.connect(DATABASE_URL)
    models             = {}   # ticker -> (model, scaler)
    ticker_tick_counts = defaultdict(int)
    last_backfill_date = None   # track so we only backfill once per calendar day

    # Restore tick buffers from Redis before processing any messages.
    # This means a restarted processor can produce predictions on the very
    # first tick rather than waiting for SEQUENCE_LEN cycles to accumulate.
    restore_tick_buffers(redis_client)

    # Pre-load all production models at startup
    logger.info("Pre-loading production models for all tickers...")
    for t in TICKERS:
        model, scaler = load_production_model(t)
        models[t]     = (model, scaler)
        status        = "OK" if model is not None else "FAILED"
        logger.info(f"  {t}: {status}")
    logger.info("Model pre-load complete. Waiting for messages...")

    for message in consumer:
        try:
            tick      = message.value
            ticker    = tick.get("ticker")
            date      = tick.get("date")
            close     = tick.get("close")
            headlines = tick.get("headlines", [])

            # Score sentiment in the processor (not the producer)
            finbert_score, _ = score_headlines(headlines)
            sentiment         = finbert_score

            tick["finbert_score"] = sentiment
            tick["vader_score"]   = 0.0

            # Cap buffer at max(DRIFT_CHECK_INTERVAL, SEQUENCE_LEN) to ensure
            # inference always has enough ticks regardless of config values
            max_buffer = max(DRIFT_CHECK_INTERVAL, SEQUENCE_LEN)
            recent_ticks[ticker].append(tick)
            if len(recent_ticks[ticker]) > max_buffer:
                recent_ticks[ticker].pop(0)

            # Persist updated buffer to Redis so restarts don't lose state
            save_tick_buffer(redis_client, ticker)

            # Lazy-load if ticker wasn't in pre-load list
            if ticker not in models:
                model, scaler = load_production_model(ticker)
                models[ticker] = (model, scaler)

            model, scaler   = models.get(ticker, (None, None))
            predicted_close = None
            model_used      = "none"

            if model is not None and scaler is not None:
                ticker_ticks = recent_ticks.get(ticker, [])

                if len(ticker_ticks) >= SEQUENCE_LEN:
                    # Build sequence from last SEQUENCE_LEN ticks
                    window = ticker_ticks[-SEQUENCE_LEN:]
                    rows   = []

                    for t in window:
                        c = t.get("close", 0)
                        rows.append([
                            t.get("open",            c),
                            t.get("high",            c),
                            t.get("low",             c),
                            t.get("volume",          0),
                            t.get("finbert_score",   0.0),
                            t.get("vader_score",     0.0),
                            t.get("rsi",             0.0),
                            t.get("macd",            0.0),
                            t.get("bb_width",        0.0),
                            t.get("price_change_1d", 0.0),
                            t.get("volatility_10d",  0.0),
                        ])

                    # Scale using the training scaler (features only, no target col)
                    feature_arr = np.array(rows, dtype=np.float64)
                    # Scaler was fit on [SEQUENCE_FEATURES + Close]; we pad Close
                    # with zeros, transform, then drop the last column
                    padded      = np.hstack([feature_arr, np.zeros((SEQUENCE_LEN, 1))])
                    scaled      = scaler.transform(padded)
                    X_seq       = scaled[:, :-1].reshape(1, SEQUENCE_LEN, len(SEQUENCE_FEATURES)).astype(np.float32)

                    output = model(X_seq)
                    if isinstance(output, dict):
                        output = list(output.values())[0]
                    pred_scaled     = float(np.array(output).flatten()[0])
                    predicted_close = inverse_transform_prediction(pred_scaled, scaler)
                    model_used      = get_production_model_name(ticker)
                    logger.info(f"  scaled_pred={pred_scaled:.4f} → ${predicted_close:.2f}")
                else:
                    logger.info(f"  {ticker}: buffering ticks ({len(ticker_ticks)}/{SEQUENCE_LEN} needed)")

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

            tick_counter               += 1
            ticker_tick_counts[ticker] += 1

            logger.info(f"Processed tick #{tick_counter}: {ticker} close={close} pred={predicted_close}")

            # Backfill actual closes once per calendar day after market close
            from datetime import date as _date
            today = _date.today()
            if last_backfill_date != today:
                backfill_actual_closes(db_conn)
                last_backfill_date = today

            if ticker_tick_counts[ticker] % DRIFT_CHECK_INTERVAL == 0:
                logger.info(f"Drift check interval reached for {ticker} ({ticker_tick_counts[ticker]} ticks)")
                run_drift_check(ticker)
                # Reload in case a new model was promoted after drift
                model, scaler  = load_production_model(ticker)
                models[ticker] = (model, scaler)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            continue


if __name__ == "__main__":
    run_processor()
