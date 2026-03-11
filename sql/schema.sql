-- SentimentEdge PostgreSQL Schema
-- Run this once against your Neon PostgreSQL database to initialize all tables.
-- Connect via: psql "your_neon_connection_string" -f schema.sql

-- Stores all model predictions including live stream predictions
CREATE TABLE IF NOT EXISTS predictions (
    id              SERIAL PRIMARY KEY,
    ticker          VARCHAR(10)    NOT NULL,
    date            DATE           NOT NULL,
    predicted_close NUMERIC(12, 4),
    actual_close    NUMERIC(12, 4),
    sentiment_score NUMERIC(8, 6),
    model_used      VARCHAR(100),
    created_at      TIMESTAMP      DEFAULT NOW(),
    updated_at      TIMESTAMP,
    UNIQUE (ticker, date)
);

-- Stores model evaluation metrics after each training run
CREATE TABLE IF NOT EXISTS model_metrics (
    id            SERIAL PRIMARY KEY,
    model_name    VARCHAR(100)   NOT NULL,
    ticker        VARCHAR(10)    NOT NULL,
    mae           NUMERIC(12, 4),
    mape          NUMERIC(10, 8),
    rmse          NUMERIC(12, 4),
    dir_accuracy  NUMERIC(6, 4),
    trained_at    TIMESTAMP      DEFAULT NOW(),
    UNIQUE (model_name, ticker)
);

-- Stores drift detection report metadata including severity and whether
-- an emergency retraining run was triggered as a result
CREATE TABLE IF NOT EXISTS drift_reports (
    id                          SERIAL PRIMARY KEY,
    ticker                      VARCHAR(10)    NOT NULL,
    report_date                 DATE           NOT NULL,
    drift_detected              BOOLEAN        NOT NULL,
    drift_fraction              NUMERIC(6, 4),
    emergency_retrain_triggered BOOLEAN        DEFAULT FALSE,
    report_url                  TEXT,
    created_at                  TIMESTAMP      DEFAULT NOW()
);

-- Stores a log entry for each retraining run triggered by GitHub Actions
CREATE TABLE IF NOT EXISTS retraining_log (
    id            SERIAL PRIMARY KEY,
    run_id        VARCHAR(100),
    triggered_by  VARCHAR(50)    DEFAULT 'github_actions',
    old_mae       NUMERIC(12, 4),
    new_mae       NUMERIC(12, 4),
    promoted      BOOLEAN        DEFAULT FALSE,
    trained_at    TIMESTAMP      DEFAULT NOW()
);

-- Indexes to speed up the most common dashboard and API queries
CREATE INDEX IF NOT EXISTS idx_predictions_ticker_date
    ON predictions (ticker, date DESC);

CREATE INDEX IF NOT EXISTS idx_predictions_ticker_created
    ON predictions (ticker, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_model_metrics_mae
    ON model_metrics (mae ASC);

CREATE INDEX IF NOT EXISTS idx_drift_reports_ticker_date
    ON drift_reports (ticker, report_date DESC);
