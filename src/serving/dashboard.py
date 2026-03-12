import json
import logging
from datetime import datetime, timedelta, date

import gradio as gr
import pandas as pd
import psycopg2
import plotly.graph_objects as go
import redis

from config.settings import (
    DATABASE_URL,
    REDIS_HOST,
    REDIS_PORT,
    TICKERS
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_redis_client():
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        client.ping()
        return client
    except Exception:
        return None


def get_db_connection():
    return psycopg2.connect(DATABASE_URL)


def fetch_latest_predictions():
    """
    Fetch the latest prediction for each ticker from Redis or PostgreSQL.
    Returns a list of dicts.
    """
    results  = []
    redis_cl = get_redis_client()

    for ticker in TICKERS:
        try:
            if redis_cl:
                cached = redis_cl.get(f"prediction:{ticker}:latest")
                if cached:
                    results.append(json.loads(cached))
                    continue

            conn   = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT ticker, date, predicted_close, actual_close, sentiment_score
                FROM predictions
                WHERE ticker = %s
                ORDER BY created_at DESC LIMIT 1
                """,
                (ticker,)
            )
            row = cursor.fetchone()
            conn.close()

            if row:
                results.append({
                    "ticker":          row[0],
                    "date":            str(row[1]),
                    "predicted_close": row[2],
                    "actual_close":    row[3],
                    "sentiment":       row[4]
                })

        except Exception as e:
            logger.warning(f"Failed to fetch prediction for {ticker}: {e}")

    return results


def fetch_prediction_history(ticker, days=60):
    """
    Fetch prediction vs actual history for a ticker from PostgreSQL.
    Returns a DataFrame with Date, predicted_close, actual_close columns.
    """
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

        df = pd.DataFrame(rows, columns=["Date", "Predicted", "Actual", "Sentiment"])
        df = df.sort_values("Date")
        return df

    except Exception as e:
        logger.warning(f"History fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_model_metrics():
    """
    Fetch model comparison metrics from PostgreSQL.
    Returns a DataFrame.
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

        df = pd.DataFrame(rows, columns=["Model", "Ticker", "MAE", "MAPE", "RMSE", "Dir Accuracy", "Trained At"])
        return df

    except Exception as e:
        logger.warning(f"Metrics fetch failed: {e}")
        return pd.DataFrame()


def build_prediction_chart(ticker):
    """
    Build a Plotly chart comparing actual vs predicted prices for a ticker.
    """
    df = fetch_prediction_history(ticker)

    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available yet", showarrow=False, font=dict(size=16))
        return fig

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Actual"],
        name="Actual Price",
        line=dict(color="#2c3e50", width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Predicted"],
        name="Predicted Price",
        line=dict(color="#e74c3c", width=1.8, dash="dash")
    ))

    fig.update_layout(
        title=f"{ticker} Actual vs Predicted Closing Price",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        legend=dict(x=0, y=1),
        template="plotly_white",
        height=450
    )

    return fig


def build_sentiment_bar():
    """
    Build a bar chart showing the latest FinBERT sentiment score per ticker.
    """
    predictions = fetch_latest_predictions()

    if not predictions:
        fig = go.Figure()
        fig.add_annotation(text="No sentiment data available", showarrow=False)
        return fig

    tickers    = [p["ticker"] for p in predictions]
    sentiments = [p.get("sentiment", 0.0) for p in predictions]
    colors     = ["#2ecc71" if s >= 0 else "#e74c3c" for s in sentiments]

    fig = go.Figure(go.Bar(
        x=tickers, y=sentiments,
        marker_color=colors,
        text=[f"{s:.3f}" for s in sentiments],
        textposition="outside"
    ))

    fig.update_layout(
        title="Current FinBERT Sentiment Score by Ticker",
        yaxis_title="Sentiment Score (-1 to 1)",
        yaxis_range=[-1.1, 1.1],
        template="plotly_white",
        height=350
    )

    return fig


def get_summary_table():
    """
    Return a summary DataFrame of the latest predictions for display.
    """
    predictions = fetch_latest_predictions()
    if not predictions:
        return pd.DataFrame(columns=["Ticker", "Date", "Predicted Close", "Actual Close", "Sentiment"])

    rows = []
    for p in predictions:
        rows.append({
            "Ticker":          p.get("ticker", ""),
            "Date":            p.get("date", ""),
            "Predicted Close": f"${p.get('predicted_close', 'N/A')}",
            "Actual Close":    f"${p.get('actual_close', 'N/A')}",
            "Sentiment":       round(p.get("sentiment", 0.0), 4)
        })

    return pd.DataFrame(rows)


with gr.Blocks(title="SentimentEdge Dashboard", theme=gr.themes.Soft()) as dashboard:

    gr.Markdown("# SentimentEdge — Real-Time Stock Prediction Dashboard")
    gr.Markdown("Live price predictions using FinBERT sentiment analysis + LSTM. Refreshes every 60 seconds.")

    with gr.Row():
        ticker_selector = gr.Dropdown(
            choices=TICKERS,
            value="AAPL",
            label="Select Ticker"
        )
        refresh_btn = gr.Button("Refresh Data", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            prediction_chart = gr.Plot(label="Actual vs Predicted Price")
        with gr.Column(scale=1):
            sentiment_chart  = gr.Plot(label="Sentiment Scores")

    with gr.Row():
        summary_table = gr.Dataframe(label="Latest Predictions", interactive=False)

    with gr.Row():
        metrics_table = gr.Dataframe(label="Model Comparison Metrics", interactive=False)

    def refresh(ticker):
        return (
            build_prediction_chart(ticker),
            build_sentiment_bar(),
            get_summary_table(),
            fetch_model_metrics()
        )

    ticker_selector.change(
        fn=refresh,
        inputs=[ticker_selector],
        outputs=[prediction_chart, sentiment_chart, summary_table, metrics_table]
    )

    refresh_btn.click(
        fn=refresh,
        inputs=[ticker_selector],
        outputs=[prediction_chart, sentiment_chart, summary_table, metrics_table]
    )

    dashboard.load(
        fn=refresh,
        inputs=[ticker_selector],
        outputs=[prediction_chart, sentiment_chart, summary_table, metrics_table]
    )


if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=7860)
