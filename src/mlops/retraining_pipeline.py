import logging
import os

import pandas as pd
import psycopg2

from config.settings import (
    DATABASE_URL,
    TICKERS,
    RETRAIN_MIN_NEW_ROWS,
    RETRAIN_IMPROVEMENT_THRESHOLD
)
from src.ingestion.price_fetcher import fetch_all_tickers
from src.ingestion.news_fetcher import fetch_headlines_range
from src.processing.sentiment_scorer import score_dataframe
from src.processing.technical_indicators import compute_all_tickers
from src.processing.feature_builder import merge_features, save_features
from src.mlops.experiment_tracker import setup_mlflow, log_run, promote_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_last_training_date():
    """
    Query Neon PostgreSQL to find the most recent date used in training.
    """
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(trained_at) FROM retraining_log")
        result = cursor.fetchone()[0]
        conn.close()

        if result:
            return str(result.date())
        return "2020-01-01"

    except Exception as e:
        logger.warning(f"Could not query last training date: {e}. Using default.")
        return "2020-01-01"


def get_new_row_count(since_date):
    """
    Count how many new prediction rows have been inserted since the last training run.
    """
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM predictions WHERE date > %s AND actual_close IS NOT NULL",
            (since_date,)
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count

    except Exception as e:
        logger.warning(f"Could not count new rows: {e}")
        return 0


def log_retraining_run(run_id, old_mae, new_mae, promoted, triggered_by="github_actions"):
    """
    Insert a record into the retraining_log table in Neon PostgreSQL.
    """
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO retraining_log (run_id, triggered_by, old_mae, new_mae, promoted, trained_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            """,
            (run_id, triggered_by, old_mae, new_mae, promoted)
        )
        conn.commit()
        conn.close()
        logger.info(f"Retraining log saved: old_mae={old_mae}, new_mae={new_mae}, promoted={promoted}")

    except Exception as e:
        logger.warning(f"Could not write retraining log: {e}")


def fetch_sentiment_for_all_tickers(tickers, start_date, end_date):
    """
    Fetch and score headlines for all tickers across the training date range.
    Returns a single combined sentiment DataFrame with finbert_score and vader_score columns.
    """
    all_sentiment = []

    for ticker in tickers:
        logger.info(f"Fetching headlines for {ticker}...")
        try:
            headlines_df = fetch_headlines_range(ticker, start_date, end_date)
            scored_df    = score_dataframe(headlines_df)
            all_sentiment.append(scored_df)
            logger.info(f"{ticker}: sentiment scoring complete")
        except Exception as e:
            logger.warning(f"Sentiment fetch/score failed for {ticker}: {e}. Using zero sentiment.")
            # Build a zeroed-out fallback so this ticker doesn't block training
            dates = pd.date_range(start=start_date, end=end_date, freq="B")
            fallback = pd.DataFrame({
                "Date":          [d.strftime("%Y-%m-%d") for d in dates],
                "Ticker":        ticker,
                "headlines":     [[] for _ in dates],
                "headline_count": 0,
                "finbert_score": 0.0,
                "vader_score":   0.0
            })
            all_sentiment.append(fallback)

    if not all_sentiment:
        return pd.DataFrame()

    combined = pd.concat(all_sentiment, ignore_index=True)
    logger.info(f"Sentiment ready: {len(combined)} rows across {len(tickers)} tickers")
    return combined


def run_retraining(triggered_by="github_actions"):
    """
    Main retraining entrypoint.
    Checks if enough new data exists, retrains if so, promotes best model if improved.
    """
    setup_mlflow()

    logger.info(f"Retraining triggered by: {triggered_by}")

    last_date = get_last_training_date()
    new_rows  = get_new_row_count(last_date)

    if new_rows < RETRAIN_MIN_NEW_ROWS and triggered_by != "drift_monitor_emergency":
        logger.info(
            f"Only {new_rows} new rows since {last_date}. "
            f"Minimum is {RETRAIN_MIN_NEW_ROWS}. Skipping retraining."
        )
        return

    if triggered_by == "drift_monitor_emergency":
        logger.warning(f"Emergency retrain: bypassing minimum row check ({new_rows} new rows available)")
    else:
        logger.info(f"Sufficient new data found ({new_rows} rows). Starting retraining...")

    from datetime import date
    start_date = "2020-01-01"
    end_date   = str(date.today())

    # ── Price + technical indicators ──────────────────────────────────────────
    logger.info("Fetching price data and computing technical indicators...")
    price_df      = fetch_all_tickers(TICKERS, start=start_date, end=end_date)
    indicators_df = compute_all_tickers(price_df)

    # ── Sentiment ─────────────────────────────────────────────────────────────
    logger.info("Fetching and scoring sentiment for all tickers...")
    sentiment_df = fetch_sentiment_for_all_tickers(TICKERS, start_date, end_date)

    # ── Merge all features ────────────────────────────────────────────────────
    logger.info("Merging price, indicators, and sentiment features...")
    features_df = merge_features(price_df, sentiment_df, indicators_df)

    # Save per-ticker feature files (used by drift monitor)
    for ticker in TICKERS:
        ticker_df = features_df[features_df["Ticker"] == ticker]
        if not ticker_df.empty:
            save_features(ticker_df, ticker)

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Feature preparation complete. Retraining models...")
    from src.training.trainer import train_all_models
    from src.mlops.experiment_tracker import promote_best_model_for_ticker
    results = train_all_models(features_df)

    if results:
        best_result = min(results, key=lambda r: r.get("mae", float("inf")))
        logger.info(f"Best model this run: {best_result['model']} | MAE={best_result['mae']}")

        # Promote best sentiment-capable model per ticker to production
        # Groups results by ticker and promotes the best one
        logger.info("Promoting best models per ticker...")
        from itertools import groupby
        dl_model_names = {"LSTM-Baseline", "FinBERT-LSTM", "GRU", "BiLSTM", "CNN-LSTM"}
        dl_results_only = [r for r in results if r.get("model") in dl_model_names]

        ticker_results = {}
        for r in dl_results_only:
            ticker = r.get("ticker")
            if ticker not in ticker_results:
                ticker_results[ticker] = []
            ticker_results[ticker].append(r)

        for ticker, ticker_dl_results in ticker_results.items():
            promote_best_model_for_ticker(ticker, ticker_dl_results)

    else:
        logger.warning("No results returned from retraining")


if __name__ == "__main__":
    run_retraining()
