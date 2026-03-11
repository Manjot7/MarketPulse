"""
SentimentEdge
Main entry point for the offline training pipeline.
Fetches data, builds features, trains all models, logs to MLflow.

Usage:
    python main.py
    python main.py --tickers AAPL MSFT GOOGL
    python main.py --start 2021-01-01 --end 2024-01-01
"""

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sentimentedge_training.log")
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="SentimentEdge Training Pipeline")
    parser.add_argument(
        "--tickers", nargs="+",
        default=None,
        help="List of tickers to train on. Defaults to all configured tickers."
    )
    parser.add_argument(
        "--start", type=str,
        default="2020-01-01",
        help="Training data start date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end", type=str,
        default="2024-01-01",
        help="Training data end date in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--skip-news", action="store_true",
        help="Skip news fetching and use zero sentiment scores. Useful for quick testing."
    )
    return parser.parse_args()


def run_pipeline(tickers, start, end, skip_news):
    from config.settings import TICKERS
    from src.ingestion.price_fetcher import fetch_all_tickers
    from src.ingestion.news_fetcher import fetch_headlines_range
    from src.processing.sentiment_scorer import score_dataframe
    from src.processing.technical_indicators import compute_all_tickers
    from src.processing.feature_builder import merge_features, save_features
    from src.training.trainer import train_all_models
    from src.training.evaluator import build_results_table, save_results_markdown
    import pandas as pd

    target_tickers = tickers or TICKERS

    logger.info(f"Starting SentimentEdge training pipeline")
    logger.info(f"Tickers: {target_tickers}")
    logger.info(f"Date range: {start} to {end}")

    logger.info("Step 1: Fetching price data from Yahoo Finance...")
    price_df = fetch_all_tickers(target_tickers, start=start, end=end)

    if price_df.empty:
        logger.error("No price data fetched. Aborting.")
        return

    logger.info("Step 2: Computing technical indicators...")
    indicators_df = compute_all_tickers(price_df)

    sentiment_df = pd.DataFrame()

    if not skip_news:
        logger.info("Step 3: Fetching news headlines and scoring sentiment...")
        sentiment_frames = []
        for ticker in target_tickers:
            news_df = fetch_headlines_range(ticker, start, end)
            news_df = score_dataframe(news_df)
            sentiment_frames.append(news_df)
        if sentiment_frames:
            sentiment_df = pd.concat(sentiment_frames, ignore_index=True)
    else:
        logger.info("Step 3: Skipping news fetch (--skip-news flag set)")

    logger.info("Step 4: Merging features...")
    features_df = merge_features(price_df, sentiment_df, indicators_df)

    for ticker in target_tickers:
        ticker_features = features_df[features_df["Ticker"] == ticker]
        if not ticker_features.empty:
            save_features(ticker_features, ticker)

    logger.info("Step 5: Training all models...")
    all_results = train_all_models(features_df)

    if all_results:
        results_df = build_results_table(all_results)
        logger.info("\nFinal Results Summary:")
        logger.info(f"\n{results_df.to_string()}")

        for ticker in target_tickers:
            ticker_results = results_df[results_df["ticker"] == ticker]
            if not ticker_results.empty:
                save_results_markdown(ticker_results, ticker)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        tickers=args.tickers,
        start=args.start,
        end=args.end,
        skip_news=args.skip_news
    )
