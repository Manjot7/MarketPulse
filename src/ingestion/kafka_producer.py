"""
Kafka Producer
Runs continuously on the Oracle VM.
Every market day at configurable intervals it fetches the latest price tick
and news headlines, scores sentiment, and publishes an enriched JSON message
to the Kafka raw ticks topic.
"""

import json
import logging
import os
import time
from datetime import datetime

from kafka import KafkaProducer
from kafka.errors import KafkaError

from config.settings import (
    KAFKA_BROKER,
    KAFKA_TOPIC_RAW,
    TICKERS
)
from src.ingestion.price_fetcher import fetch_latest_tick
from src.ingestion.news_fetcher import fetch_latest_headlines
from src.processing.sentiment_scorer import score_headlines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", 300))


def build_producer():
    """
    Initialize and return a Kafka producer with retry logic.
    Retries every 5 seconds until the broker is reachable.
    """
    while True:
        try:
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8"),
                retries=3,
                retry_backoff_ms=1000,
                request_timeout_ms=30000
            )
            logger.info(f"Kafka producer connected to {KAFKA_BROKER}")
            return producer
        except KafkaError as e:
            logger.warning(f"Kafka not ready yet: {e}. Retrying in 5s...")
            time.sleep(5)


def build_tick_payload(ticker):
    """
    Build a single enriched tick payload for a given ticker.
    Combines latest price data + sentiment score into one JSON message.
    Returns None if price data is unavailable.
    """
    price = fetch_latest_tick(ticker)
    if price is None:
        return None

    headlines = fetch_latest_headlines(ticker)
    finbert_score, vader_score = score_headlines(headlines)

    payload = {
        **price,
        "headlines": headlines,
        "finbert_score": finbert_score,
        "vader_score": vader_score,
        "ingested_at": datetime.utcnow().isoformat()
    }
    return payload


def run_producer():
    """
    Main producer loop.
    Iterates over all tickers, builds enriched payloads, publishes to Kafka.
    Sleeps for POLL_INTERVAL_SECONDS between cycles.
    """
    producer = build_producer()

    logger.info(f"Producer started. Polling {len(TICKERS)} tickers every {POLL_INTERVAL_SECONDS}s")

    while True:
        cycle_start = time.time()

        for ticker in TICKERS:
            try:
                payload = build_tick_payload(ticker)

                if payload is None:
                    logger.warning(f"Skipping {ticker}: no price data available")
                    continue

                producer.send(
                    KAFKA_TOPIC_RAW,
                    key=ticker,
                    value=payload
                )
                logger.info(f"Published tick for {ticker}: close={payload['close']} sentiment={payload['finbert_score']:.3f}")

            except Exception as e:
                logger.error(f"Failed to publish tick for {ticker}: {e}")

        producer.flush()

        elapsed = time.time() - cycle_start
        sleep_time = max(0, POLL_INTERVAL_SECONDS - elapsed)
        logger.info(f"Cycle complete in {elapsed:.1f}s. Sleeping {sleep_time:.1f}s")
        time.sleep(sleep_time)


if __name__ == "__main__":
    run_producer()
