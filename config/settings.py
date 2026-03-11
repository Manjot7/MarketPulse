"""
SentimentEdge Configuration
All environment variables and constants are defined here.
Import this module wherever configuration is needed.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------
TICKERS = [
    "NDX", "SPY", "AAPL", "MSFT", "GOOGL",
    "NVDA", "TSLA", "META", "JPM", "AMZN"
]

# ---------------------------------------------------------------
# Date ranges
# ---------------------------------------------------------------
TRAIN_START_DATE = "2020-01-01"
TRAIN_END_DATE   = "2024-01-01"
LIVE_INTERVAL    = "1d"

# ---------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------
SEQUENCE_LENGTH  = 10
TRAIN_SPLIT      = 0.85
EPOCHS           = 100
BATCH_SIZE       = 32
LEARNING_RATE    = 0.02
RANDOM_STATE     = 42

# ---------------------------------------------------------------
# Sentiment
# ---------------------------------------------------------------
FINBERT_MODEL    = "ProsusAI/finbert"
NEWS_PER_DAY     = 5

# ---------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------
RSI_PERIOD       = 14
MACD_FAST        = 12
MACD_SLOW        = 26
MACD_SIGNAL      = 9
BB_PERIOD        = 20
EMA_PERIODS      = [9, 21, 50]

# ---------------------------------------------------------------
# Kafka
# ---------------------------------------------------------------
KAFKA_BROKER          = os.getenv("KAFKA_BROKER", "localhost:9092")
KAFKA_TOPIC_RAW       = "market.ticks.raw"
KAFKA_TOPIC_ENRICHED  = "market.ticks.enriched"
KAFKA_TOPIC_PREDICTIONS = "market.predictions"
KAFKA_GROUP_ID        = "sentimentedge-processor"

# ---------------------------------------------------------------
# Redis
# ---------------------------------------------------------------
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL      = 86400

# ---------------------------------------------------------------
# PostgreSQL (Neon)
# ---------------------------------------------------------------
DATABASE_URL   = os.getenv("DATABASE_URL", "")

# ---------------------------------------------------------------
# NewsAPI
# ---------------------------------------------------------------
NEWSAPI_KEY    = os.getenv("NEWSAPI_KEY", "")
NEWSAPI_BASE   = "https://newsapi.org/v2/everything"

# ---------------------------------------------------------------
# MLflow / DagsHub
# ---------------------------------------------------------------
MLFLOW_TRACKING_URI      = os.getenv("MLFLOW_TRACKING_URI", "")
MLFLOW_EXPERIMENT_NAME   = "sentimentedge"
DAGSHUB_USERNAME         = os.getenv("DAGSHUB_USERNAME", "")
DAGSHUB_TOKEN            = os.getenv("DAGSHUB_TOKEN", "")

# ---------------------------------------------------------------
# Cloudflare R2
# ---------------------------------------------------------------
R2_ENDPOINT_URL      = os.getenv("R2_ENDPOINT_URL", "")
R2_ACCESS_KEY_ID     = os.getenv("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET_NAME       = "sentimentedge-artifacts"

# ---------------------------------------------------------------
# HuggingFace
# ---------------------------------------------------------------
HF_TOKEN         = os.getenv("HF_TOKEN", "")
HF_API_SPACE     = "sentimentedge-api"
HF_DASH_SPACE    = "sentimentedge-dashboard"

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR         = os.path.join(BASE_DIR, "data")
RAW_DIR          = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR    = os.path.join(DATA_DIR, "processed")
MODELS_DIR       = os.path.join(DATA_DIR, "models")

# ---------------------------------------------------------------
# Drift monitoring
# ---------------------------------------------------------------
DRIFT_CHECK_INTERVAL   = 100
DRIFT_REPORT_DIR       = os.path.join(BASE_DIR, "reports", "drift")

# ---------------------------------------------------------------
# Retraining
# ---------------------------------------------------------------
RETRAIN_MIN_NEW_ROWS   = 30
RETRAIN_IMPROVEMENT_THRESHOLD = 0.01
