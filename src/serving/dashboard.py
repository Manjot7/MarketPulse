import json
import logging
from datetime import datetime, date

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

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg-primary:    #080c14;
    --bg-card:       #0d1424;
    --bg-card2:      #111827;
    --border:        rgba(99, 179, 237, 0.12);
    --border-bright: rgba(99, 179, 237, 0.35);
    --accent:        #63b3ed;
    --accent2:       #76e4c4;
    --text-primary:  #e2e8f0;
    --text-muted:    #718096;
    --text-dim:      #4a5568;
    --green:         #68d391;
}

body, .gradio-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 0 24px 48px !important;
}

.mp-header {
    padding: 32px 0 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 24px;
}

.mp-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #63b3ed 0%, #76e4c4 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 4px;
    line-height: 1.1;
}

.mp-header p {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    letter-spacing: 0.05em;
    margin: 0;
}

.mp-live-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(104, 211, 145, 0.08);
    border: 1px solid rgba(104, 211, 145, 0.25);
    border-radius: 20px;
    padding: 4px 12px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    color: var(--green);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 10px;
    width: fit-content;
}

.mp-live-badge::before {
    content: '';
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}

.mp-stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
    margin-bottom: 24px;
}

.mp-stat-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.mp-stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    opacity: 0.6;
}

.mp-stat-card:hover { border-color: var(--border-bright); }

.mp-stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}

.mp-stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1;
}

.mp-stat-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    margin-top: 4px;
}

.mp-section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 10px;
    padding-bottom: 6px;
    border-bottom: 1px solid var(--border);
}

label, .block-label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

button.primary {
    background: linear-gradient(135deg, #2b6cb0, #2c7a7b) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    color: #fff !important;
    transition: opacity 0.2s !important;
}

button.primary:hover { opacity: 0.85 !important; }

.plot-container, [data-testid="Plot"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.dataframe-container, table {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.76rem !important;
}

th {
    background: var(--bg-card2) !important;
    color: var(--accent) !important;
    font-size: 0.65rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    border-bottom: 1px solid var(--border-bright) !important;
    padding: 9px 12px !important;
}

td {
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border) !important;
    padding: 8px 12px !important;
}

tr:hover td { background: rgba(99, 179, 237, 0.04) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-bright); border-radius: 3px; }
"""

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(13,20,36,0)",
    plot_bgcolor="rgba(13,20,36,0)",
    font=dict(family="DM Mono, monospace", color="#718096", size=11),
    xaxis=dict(
        gridcolor="rgba(99,179,237,0.06)",
        linecolor="rgba(99,179,237,0.15)",
        tickfont=dict(color="#4a5568", size=10),
        zerolinecolor="rgba(99,179,237,0.1)",
    ),
    yaxis=dict(
        gridcolor="rgba(99,179,237,0.06)",
        linecolor="rgba(99,179,237,0.15)",
        tickfont=dict(color="#4a5568", size=10),
        zerolinecolor="rgba(99,179,237,0.1)",
    ),
    legend=dict(
        bgcolor="rgba(13,20,36,0.7)",
        bordercolor="rgba(99,179,237,0.2)",
        borderwidth=1,
        font=dict(color="#a0aec0", size=11),
    ),
    autosize=False,
    margin=dict(l=48, r=24, t=44, b=36),
    hoverlabel=dict(
        bgcolor="#0d1424",
        bordercolor="rgba(99,179,237,0.3)",
        font=dict(family="DM Mono, monospace", color="#e2e8f0", size=11),
    ),
)


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
                FROM predictions WHERE ticker = %s
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
                    "sentiment":       row[4],
                })
        except Exception as e:
            logger.warning(f"Failed to fetch prediction for {ticker}: {e}")
    return results


def fetch_prediction_history(ticker, days=60):
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT date, predicted_close, actual_close, sentiment_score
            FROM predictions
            WHERE ticker = %s AND actual_close IS NOT NULL
            ORDER BY date DESC LIMIT %s
            """,
            (ticker, days)
        )
        rows = cursor.fetchall()
        conn.close()
        df = pd.DataFrame(rows, columns=["Date", "Predicted", "Actual", "Sentiment"])
        return df.sort_values("Date")
    except Exception as e:
        logger.warning(f"History fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_model_metrics(filter_by="All Tickers", filter_value="ALL"):
    try:
        conn   = get_db_connection()
        cursor = conn.cursor()
        if filter_by == "Ticker" and filter_value != "ALL":
            cursor.execute(
                """
                SELECT model_name, ticker, mae, mape, rmse, dir_accuracy, trained_at
                FROM model_metrics WHERE ticker = %s ORDER BY mape ASC
                """,
                (filter_value,)
            )
        elif filter_by == "Model" and filter_value != "ALL":
            cursor.execute(
                """
                SELECT model_name, ticker, mae, mape, rmse, dir_accuracy, trained_at
                FROM model_metrics WHERE model_name = %s ORDER BY mape ASC
                """,
                (filter_value,)
            )
        else:
            cursor.execute(
                "SELECT model_name, ticker, mae, mape, rmse, dir_accuracy, trained_at FROM model_metrics ORDER BY mape ASC"
            )
        rows = cursor.fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=["Model", "Ticker", "MAE", "MAPE %", "RMSE", "Dir Acc", "Trained At"])
    except Exception as e:
        logger.warning(f"Metrics fetch failed: {e}")
        return pd.DataFrame()


def build_prediction_chart(ticker):
    df  = fetch_prediction_history(ticker)
    fig = go.Figure()

    if df.empty:
        fig.add_annotation(
            text="Awaiting market data…",
            showarrow=False,
            font=dict(family="DM Mono, monospace", size=13, color="#4a5568"),
            xref="paper", yref="paper", x=0.5, y=0.5
        )
    else:
        fig.add_trace(go.Scatter(
            x=pd.concat([df["Date"], df["Date"][::-1]]),
            y=pd.concat([df["Actual"], df["Predicted"][::-1]]),
            fill="toself",
            fillcolor="rgba(99,179,237,0.04)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Actual"],
            name="Actual",
            line=dict(color="#63b3ed", width=2),
            mode="lines",
        ))
        fig.add_trace(go.Scatter(
            x=df["Date"], y=df["Predicted"],
            name="Predicted",
            line=dict(color="#76e4c4", width=1.5, dash="dot"),
            mode="lines",
        ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text=f"<b>{ticker}</b>  Actual vs Predicted Close",
            font=dict(family="Syne, sans-serif", size=14, color="#a0aec0"),
            x=0.01, xanchor="left",
        ),
        height=300,
    )
    return fig


def build_sentiment_bar():
    predictions = fetch_latest_predictions()
    fig = go.Figure()

    if not predictions:
        fig.add_annotation(
            text="Awaiting sentiment data…",
            showarrow=False,
            font=dict(family="DM Mono, monospace", size=13, color="#4a5568"),
            xref="paper", yref="paper", x=0.5, y=0.5
        )
    else:
        tickers    = [p["ticker"] for p in predictions]
        sentiments = [p.get("sentiment", 0.0) for p in predictions]
        colors     = ["#68d391" if s >= 0 else "#fc8181" for s in sentiments]

        fig.add_trace(go.Bar(
            x=tickers,
            y=sentiments,
            marker=dict(color=colors, line=dict(color="rgba(0,0,0,0)", width=0), opacity=0.85),
            text=[f"{s:+.3f}" for s in sentiments],
            textposition="outside",
            textfont=dict(family="DM Mono, monospace", size=10, color="#a0aec0"),
        ))
        fig.add_hline(y=0, line=dict(color="rgba(99,179,237,0.3)", width=1, dash="dot"))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(
            text="<b>FinBERT Sentiment</b>  by Ticker",
            font=dict(family="Syne, sans-serif", size=14, color="#a0aec0"),
            x=0.01, xanchor="left",
        ),
        yaxis_range=[-1.15, 1.15],
        height=300,
        showlegend=False,
    )
    return fig


def get_summary_table():
    predictions = fetch_latest_predictions()
    if not predictions:
        return pd.DataFrame(columns=["Ticker", "Date", "Predicted $", "Actual $", "Sentiment", "Signal"])

    rows = []
    for p in predictions:
        pred   = p.get("predicted_close")
        act    = p.get("actual_close")
        sent   = p.get("sentiment", 0.0) or 0.0
        signal = "▲ BULLISH" if sent > 0.1 else ("▼ BEARISH" if sent < -0.1 else "◆ NEUTRAL")
        rows.append({
            "Ticker":      p.get("ticker", ""),
            "Date":        p.get("date", ""),
            "Predicted $": f"${pred:.2f}" if pred else "—",
            "Actual $":    f"${act:.2f}"  if act  else "—",
            "Sentiment":   f"{sent:+.4f}",
            "Signal":      signal,
        })
    return pd.DataFrame(rows)


MODEL_NAMES = [
    "ALL", "LSTM-Baseline", "FinBERT-LSTM", "GRU", "BiLSTM", "CNN-LSTM",
    "XGBoost-Regression", "LightGBM-Regression", "XGBoost-Direction",
    "LightGBM-Direction", "RandomForest-Direction", "ARIMA", "Prophet"
]
TICKER_OPTIONS = ["ALL"] + TICKERS


with gr.Blocks(
    title="MarketPulse",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    ),
) as dashboard:

    gr.HTML("""
    <div class="mp-header">
        <h1>MarketPulse</h1>
        <p>REAL-TIME STOCK PREDICTION · FINBERT SENTIMENT · LSTM MODELS</p>
        <div class="mp-live-badge">Live Pipeline</div>
    </div>
    <div class="mp-stats-row">
        <div class="mp-stat-card">
            <div class="mp-stat-label">Tickers Tracked</div>
            <div class="mp-stat-value">10</div>
            <div class="mp-stat-sub">^NDX SPY AAPL MSFT + 6 more</div>
        </div>
        <div class="mp-stat-card">
            <div class="mp-stat-label">Models Trained</div>
            <div class="mp-stat-value">120</div>
            <div class="mp-stat-sub">12 models × 10 tickers</div>
        </div>
        <div class="mp-stat-card">
            <div class="mp-stat-label">Best MAPE</div>
            <div class="mp-stat-value">2.0%</div>
            <div class="mp-stat-sub">GRU on AAPL</div>
        </div>
        <div class="mp-stat-card">
            <div class="mp-stat-label">Poll Interval</div>
            <div class="mp-stat-value">5m</div>
            <div class="mp-stat-sub">Kafka → Redis → Neon</div>
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=3):
            ticker_selector = gr.Dropdown(
                choices=TICKERS, value="AAPL",
                label="Select Ticker", interactive=True,
            )
        with gr.Column(scale=1):
            refresh_btn = gr.Button("⟳  Refresh", variant="primary")

    with gr.Row():
        with gr.Column(scale=2):
            prediction_chart = gr.Plot(label="Price Chart", min_width=0)
        with gr.Column(scale=1):
            sentiment_chart = gr.Plot(label="Sentiment", min_width=0)

    gr.HTML('<div class="mp-section-label" style="margin-top:16px;">Latest Predictions</div>')
    summary_table = gr.Dataframe(interactive=False, wrap=True)

    gr.HTML('<div class="mp-section-label" style="margin-top:20px;">Model Performance Metrics</div>')
    with gr.Row():
        with gr.Column(scale=1):
            metrics_filter_type = gr.Radio(
                choices=["All Tickers", "Ticker", "Model"],
                value="All Tickers",
                label="Filter By",
                interactive=True,
            )
        with gr.Column(scale=2):
            metrics_filter_ticker = gr.Dropdown(
                choices=TICKER_OPTIONS, value="ALL",
                label="Ticker Filter", interactive=True, visible=False,
            )
            metrics_filter_model = gr.Dropdown(
                choices=MODEL_NAMES, value="ALL",
                label="Model Filter", interactive=True, visible=False,
            )

    metrics_table = gr.Dataframe(interactive=False, wrap=True)

    gr.HTML("""
    <div style="margin-top:28px; padding-top:14px; border-top:1px solid rgba(99,179,237,0.1);
                font-family:'DM Mono',monospace; font-size:0.65rem; color:#4a5568;
                display:flex; justify-content:space-between;">
        <span>MarketPulse · FinBERT + LSTM · Kafka · Redis · Neon PostgreSQL · Cloudflare R2</span>
        <span>Predictions update every market day</span>
    </div>
    """)

    def refresh(ticker):
        return (
            build_prediction_chart(ticker),
            build_sentiment_bar(),
            get_summary_table(),
            fetch_model_metrics(),
        )

    def update_metrics_filter_visibility(filter_type):
        return (
            gr.update(visible=(filter_type == "Ticker")),
            gr.update(visible=(filter_type == "Model")),
        )

    def update_metrics_table(filter_type, ticker_val, model_val):
        if filter_type == "Ticker":
            return fetch_model_metrics("Ticker", ticker_val)
        elif filter_type == "Model":
            return fetch_model_metrics("Model", model_val)
        return fetch_model_metrics()

    ticker_selector.change(
        fn=refresh,
        inputs=[ticker_selector],
        outputs=[prediction_chart, sentiment_chart, summary_table, metrics_table],
    )
    refresh_btn.click(
        fn=refresh,
        inputs=[ticker_selector],
        outputs=[prediction_chart, sentiment_chart, summary_table, metrics_table],
    )
    dashboard.load(
        fn=refresh,
        inputs=[ticker_selector],
        outputs=[prediction_chart, sentiment_chart, summary_table, metrics_table],
    )
    metrics_filter_type.change(
        fn=update_metrics_filter_visibility,
        inputs=[metrics_filter_type],
        outputs=[metrics_filter_ticker, metrics_filter_model],
    )
    metrics_filter_ticker.change(
        fn=update_metrics_table,
        inputs=[metrics_filter_type, metrics_filter_ticker, metrics_filter_model],
        outputs=[metrics_table],
    )
    metrics_filter_model.change(
        fn=update_metrics_table,
        inputs=[metrics_filter_type, metrics_filter_ticker, metrics_filter_model],
        outputs=[metrics_table],
    )
    metrics_filter_type.change(
        fn=update_metrics_table,
        inputs=[metrics_filter_type, metrics_filter_ticker, metrics_filter_model],
        outputs=[metrics_table],
    )

if __name__ == "__main__":
    dashboard.launch(server_name="0.0.0.0", server_port=7860)
