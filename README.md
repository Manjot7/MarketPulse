# MarketPulse

**Real-time stock price prediction using FinBERT sentiment analysis and deep learning.**

MarketPulse is a full end-to-end MLOps project that ingests live market data and financial news, scores sentiment using FinBERT, generates price predictions with trained deep learning models, and serves them through a live REST API and Gradio dashboard — all on a continuously running pipeline.

[![Live API](https://img.shields.io/badge/API-HuggingFace%20Spaces-blue?style=flat-square)](https://Manjot7-marketpulse-api.hf.space/docs)
[![Dashboard](https://img.shields.io/badge/Dashboard-HuggingFace%20Spaces-green?style=flat-square)](https://Manjot7-marketpulse-dashboard.hf.space)
[![MLflow](https://img.shields.io/badge/Experiments-DagsHub-orange?style=flat-square)](https://dagshub.com/Manjot7/marketpulse)

---

## Overview

MarketPulse tracks 10 tickers across US equity markets and produces daily closing price predictions by combining technical indicators with real-time FinBERT sentiment scores extracted from financial news headlines. Predictions are served through a FastAPI backend with Redis caching and persisted to a Neon PostgreSQL database.

**Tickers:** `^NDX` `SPY` `AAPL` `MSFT` `GOOGL` `NVDA` `TSLA` `META` `JPM` `AMZN`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Ingestion                           │
│   yfinance (OHLCV)  +  NewsAPI (headlines)  →  Kafka Producer  │
└────────────────────────────┬────────────────────────────────────┘
                             │ Kafka Topic: market.ticks.raw
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Stream Processor                           │
│   FinBERT Sentiment  +  Technical Indicators  →  Inference      │
│   → Redis Cache  +  Neon PostgreSQL (predictions table)         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                  ┌──────────┴──────────┐
                  ▼                     ▼
          FastAPI (HF Spaces)   Gradio Dashboard (HF Spaces)
          /predict /metrics     Live charts + sentiment viz
          /health  /history
```

**Infrastructure:**
- **Kafka VM** — Oracle Cloud `VM.Standard.E2.1.Micro` (Ubuntu 22.04) running Apache Kafka 3.7 in KRaft mode
- **Redis VM** — Oracle Cloud `VM.Standard.E2.1.Micro` running Redis 7.2 + stream processor container
- **Model Storage** — Cloudflare R2 (S3-compatible object store)
- **Database** — Neon PostgreSQL (serverless, connection pooling)
- **Experiment Tracking** — MLflow on DagsHub
- **Serving** — HuggingFace Spaces (Docker + Gradio SDK)

---

## Models

12 models are trained per ticker (120 total). At deployment, the best-performing sequence model per ticker is promoted to production and stored in the `production_models` table. Only sequence-based models are eligible for promotion since they can consume the live sentiment signal.

| Category | Models |
|---|---|
| Deep Learning (production eligible) | LSTM-Baseline, FinBERT-LSTM, GRU, BiLSTM, CNN-LSTM |
| Tabular | XGBoost-Regression, LightGBM-Regression |
| Classification | XGBoost-Direction, LightGBM-Direction, RandomForest-Direction |
| Statistical | ARIMA, Prophet |

**Training results (best MAPE per ticker):**

| Ticker | Production Model | MAPE |
|---|---|---|
| AAPL | GRU | 2.04% |
| TSLA | LSTM-Baseline | 2.79% |
| SPY | GRU | 2.45% |
| MSFT | GRU | 3.00% |
| AMZN | LSTM-Baseline | 3.51% |
| META | GRU | 4.11% |
| ^NDX | LSTM-Baseline | 4.02% |
| JPM | GRU | 5.04% |
| GOOGL | GRU | 7.03% |
| NVDA | GRU | 10.95% |

Models are trained on Kaggle (GPU T4 x2) and uploaded to Cloudflare R2. The stream processor downloads the correct model per ticker from R2 at inference time.

---

## Features

**Real-time pipeline**
- Producer polls yfinance and NewsAPI every 5 minutes and publishes to Kafka
- Stream processor consumes ticks, runs FinBERT on news headlines, builds technical features, runs inference, and writes predictions to PostgreSQL
- Redis caches the latest prediction per ticker for fast API reads

**Technical indicators computed per tick**
- RSI (14), MACD (12/26/9), Bollinger Bands (20), EMA (9, 21, 50)
- BB Width, MACD Signal, price momentum features

**Sentiment analysis**
- FinBERT (`ProsusAI/finbert`) scores financial news headlines as positive/negative/neutral
- Sentiment score fed as an additional input feature to sequence models at inference

**MLOps**
- All training runs logged to MLflow on DagsHub (params, metrics, artifacts)
- Model promotion logic selects best sequence model per ticker by MAPE
- Drift monitoring via Evidently (triggers retraining when feature distributions shift)
- Weekly retraining via GitHub Actions (`retrain.yml`) — runs Kaggle notebook, promotes new models if MAPE improves by >1%

**API endpoints**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service status + supported tickers |
| POST | `/predict` | Latest prediction for a ticker + date |
| GET | `/metrics` | All model performance metrics |
| GET | `/predictions/{ticker}/history` | Prediction vs actual history |

---

## Project Structure

```
marketpulse/
├── config/
│   └── settings.py               # All config + env vars
├── docker/
│   ├── Dockerfile.producer       # Kafka producer image
│   ├── Dockerfile.processor      # Stream processor image
│   ├── Dockerfile.api            # FastAPI image
│   └── Dockerfile.dashboard      # Gradio dashboard image
├── docker-compose.yml            # Kafka + Redis + producer + processor
├── notebooks/
│   └── training-notebook.ipynb   # Kaggle training notebook (12 models × 10 tickers)
├── sql/
│   └── schema.sql                # Neon PostgreSQL schema
├── src/
│   ├── ingestion/
│   │   ├── kafka_producer.py     # yfinance + NewsAPI → Kafka
│   │   ├── price_fetcher.py      # OHLCV data fetch
│   │   └── news_fetcher.py       # NewsAPI headline fetch
│   ├── processing/
│   │   ├── feature_builder.py    # Feature engineering pipeline
│   │   ├── technical_indicators.py # RSI, MACD, BB, EMA
│   │   └── sentiment_scorer.py   # FinBERT inference
│   ├── streaming/
│   │   └── stream_processor.py   # Kafka consumer → inference → DB write
│   ├── training/
│   │   ├── trainer.py            # Training orchestrator
│   │   ├── evaluator.py          # MAPE, MAE, RMSE, direction accuracy
│   │   └── models/               # One file per model architecture
│   ├── mlops/
│   │   ├── experiment_tracker.py # MLflow logging + model promotion
│   │   ├── drift_monitor.py      # Evidently drift detection
│   │   └── retraining_pipeline.py # Retraining trigger logic
│   └── serving/
│       ├── api.py                # FastAPI app
│       └── dashboard.py          # Gradio dashboard
├── .github/
│   └── workflows/
│       ├── ci.yml                # Lint + test on push
│       └── retrain.yml           # Weekly retraining schedule
└── requirements.txt
```

---

## Setup

### Prerequisites

- Oracle Cloud account (2 free `VM.Standard.E2.1.Micro` instances)
- Neon PostgreSQL account
- Cloudflare R2 bucket
- DagsHub account + MLflow tracking URI
- NewsAPI key
- HuggingFace account + token
- Kaggle account (for training)

### 1. Environment variables

Create a `.env` file in the project root:

```env
# Kafka
KAFKA_BROKER=<kafka-vm-ip>:9092

# Redis
REDIS_HOST=<redis-vm-ip>
REDIS_PORT=6379

# PostgreSQL
DATABASE_URL=postgresql://<user>:<pass>@<host>/neondb?sslmode=require

# NewsAPI
NEWSAPI_KEY=<your-key>

# DagsHub / MLflow
MLFLOW_TRACKING_URI=https://dagshub.com/<user>/marketpulse.mlflow
DAGSHUB_USERNAME=<username>
DAGSHUB_TOKEN=<token>

# Cloudflare R2
R2_ENDPOINT_URL=https://<account-id>.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=<key>
R2_SECRET_ACCESS_KEY=<secret>
R2_BUCKET_NAME=marketpulse-artifacts

# HuggingFace
HF_TOKEN=<token>
```

### 2. Initialize the database

Run `sql/schema.sql` in the Neon SQL editor to create the four tables: `predictions`, `model_metrics`, `production_models`, and `market_data`.

### 3. Train models

Open `notebooks/training-notebook.ipynb` on Kaggle with GPU T4 x2 accelerator. Add all environment variables as Kaggle secrets. Run all cells — training takes approximately 2–3 hours for all 10 tickers. Models are automatically uploaded to R2 and metrics written to PostgreSQL on completion.

### 4. Deploy infrastructure

**Kafka VM:**
```bash
git clone https://github.com/Manjot7/MarketPulse.git
cd marketpulse
cp .env.example .env  # fill in values
docker-compose up -d kafka
docker-compose up -d producer
```

**Redis VM:**
```bash
git clone https://github.com/Manjot7/MarketPulse.git
cd marketpulse
cp .env.example .env
docker-compose up -d redis
docker-compose up -d processor
```

Add `@reboot docker-compose -f ~/marketpulse/docker-compose.yml up -d` to crontab on both VMs to auto-restart on reboot.

### 5. Deploy API and dashboard

Create two HuggingFace Spaces:
- `marketpulse-api` — Docker SDK
- `marketpulse-dashboard` — Gradio SDK

Upload the contents of `docker/Dockerfile.api` + `src/serving/api.py` + `config/` to the API space. Upload `src/serving/dashboard.py` + `config/` + `requirements.txt` to the dashboard space. Add `DATABASE_URL`, `REDIS_HOST`, and `REDIS_PORT` as Space secrets in both.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Data ingestion | yfinance, NewsAPI |
| Message queue | Apache Kafka 3.7 (KRaft) |
| Cache | Redis 7.2 |
| Sentiment model | ProsusAI/FinBERT (HuggingFace Transformers) |
| Deep learning | TensorFlow / Keras (LSTM, GRU, BiLSTM, CNN-LSTM) |
| Tabular ML | XGBoost, LightGBM, scikit-learn |
| Statistical | ARIMA (statsmodels), Prophet |
| Feature engineering | pandas, NumPy, TA indicators |
| Drift detection | Evidently |
| Experiment tracking | MLflow + DagsHub |
| Model storage | Cloudflare R2 (boto3) |
| Database | Neon PostgreSQL (psycopg2) |
| API | FastAPI + Uvicorn |
| Dashboard | Gradio 5 |
| Infrastructure | Oracle Cloud (Docker + Docker Compose) |
| CI/CD | GitHub Actions |
| Training | Kaggle Notebooks (GPU T4 x2) |

---

## Live Demo

- **API docs:** https://Manjot7-marketpulse-api.hf.space/docs
- **Dashboard:** https://Manjot7-marketpulse-dashboard.hf.space
- **Experiment tracking:** https://dagshub.com/Manjot7/marketpulse

Predictions are generated on each market day. The dashboard shows actual vs predicted closing prices, FinBERT sentiment scores per ticker, and full model performance metrics filterable by ticker or model type.

---

## Author

**Manjot Singh**
MS Applied Data Science — San Jose State University

[HuggingFace](https://huggingface.co/Manjot7) · [GitHub](https://github.com/Manjot7)
