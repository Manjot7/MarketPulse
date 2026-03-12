import json
import logging
import os
from datetime import datetime, date

import psycopg2
import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config.settings import (
    DATABASE_URL,
    REDIS_HOST,
    REDIS_PORT,
    TICKERS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MarketPulse API",
    description="Real-time stock price prediction with FinBERT sentiment analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)


class PredictionRequest(BaseModel):
    ticker: str
    date:   str = str(date.today())


def get_redis_client():
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


@app.get("/health")
def health_check():
    """
    Health check endpoint. Returns service status and timestamp.
    """
    return {
        "status":    "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "tickers":   TICKERS
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    """
    Return the latest prediction for a given ticker and date.
    Checks Redis cache first, falls back to PostgreSQL.
    """
    ticker = request.ticker.upper()

    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not supported. Supported: {TICKERS}")

    redis_client = get_redis_client()
    if redis_client:
        cached = redis_client.get(f"prediction:{ticker}:latest")
        if cached:
            return {"source": "cache", "data": json.loads(cached)}

    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT ticker, date, predicted_close, actual_close, sentiment_score, model_used, created_at
            FROM predictions
            WHERE ticker = %s AND date = %s
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (ticker, request.date)
        )
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"No prediction found for {ticker} on {request.date}")

        return {
            "source": "database",
            "data": {
                "ticker":           row[0],
                "date":             str(row[1]),
                "predicted_close":  row[2],
                "actual_close":     row[3],
                "sentiment_score":  row[4],
                "model_used":       row[5],
                "created_at":       str(row[6])
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/metrics")
def get_metrics():
    """
    Return current model performance metrics for all tickers from PostgreSQL.
    """
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT model_name, ticker, mae, mape, rmse, dir_accuracy, trained_at
            FROM model_metrics
            ORDER BY mae ASC
            """
        )
        rows = cursor.fetchall()
        conn.close()

        metrics = [
            {
                "model_name":    r[0],
                "ticker":        r[1],
                "mae":           r[2],
                "mape":          r[3],
                "rmse":          r[4],
                "dir_accuracy":  r[5],
                "trained_at":    str(r[6])
            }
            for r in rows
        ]

        return {"metrics": metrics, "count": len(metrics)}

    except Exception as e:
        logger.error(f"Metrics query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/predictions/{ticker}/history")
def prediction_history(ticker: str, days: int = 30):
    """
    Return recent prediction history for a ticker including actual vs predicted prices.
    """
    ticker = ticker.upper()
    if ticker not in TICKERS:
        raise HTTPException(status_code=400, detail=f"Ticker {ticker} not supported")

    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT date, predicted_close, actual_close, sentiment_score
            FROM predictions
            WHERE ticker = %s AND actual_close IS NOT NULL
            ORDER BY date DESC
            LIMIT %s
            """,
            (ticker, days)
        )
        rows = cursor.fetchall()
        conn.close()

        history = [
            {
                "date":             str(r[0]),
                "predicted_close":  r[1],
                "actual_close":     r[2],
                "sentiment_score":  r[3]
            }
            for r in rows
        ]

        return {"ticker": ticker, "history": history, "days": len(history)}

    except Exception as e:
        logger.error(f"History query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
