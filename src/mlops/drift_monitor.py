import logging
import os
import threading
from datetime import datetime

import boto3
import pandas as pd
import psycopg2
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from evidently.report import Report

from config.settings import (
    R2_ENDPOINT_URL,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME,
    DRIFT_REPORT_DIR,
    DATABASE_URL
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Fraction of features that must show drift before triggering emergency retrain.
DRIFT_EMERGENCY_THRESHOLD = 0.3

# Cooldown period in seconds between emergency retrains to avoid retraining repeatedly
EMERGENCY_RETRAIN_COOLDOWN = 21600

# Tracks the timestamp of the last emergency retrain to enforce the cooldown
_last_emergency_retrain_time = 0


def build_r2_client():
    """
    Build and return a boto3 S3 client configured for Cloudflare R2.
    """
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )


def run_data_drift_report(reference_df, current_df, ticker):
    """
    Generate an Evidently data drift report comparing current vs reference distributions.
    Returns the local file path of the saved report.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)
    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename    = f"{ticker}_drift_{timestamp}.html"
    local_path  = os.path.join(DRIFT_REPORT_DIR, filename)

    report.save_html(local_path)
    logger.info(f"Drift report saved: {local_path}")

    upload_to_r2(local_path, f"drift/{filename}")
    return local_path


def run_regression_report(reference_df, current_df, ticker):
    """
    Generate an Evidently regression performance report.
    """
    report = Report(metrics=[RegressionPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs(DRIFT_REPORT_DIR, exist_ok=True)
    timestamp   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename    = f"{ticker}_regression_{timestamp}.html"
    local_path  = os.path.join(DRIFT_REPORT_DIR, filename)

    report.save_html(local_path)
    logger.info(f"Regression report saved: {local_path}")

    upload_to_r2(local_path, f"regression/{filename}")
    return local_path


def upload_to_r2(local_path, r2_key):
    """
    Upload a local file to Cloudflare R2 under the given key.
    Logs a warning on failure rather than raising so the main pipeline continues.
    """
    try:
        client = build_r2_client()
        client.upload_file(local_path, R2_BUCKET_NAME, r2_key)
        logger.info(f"Uploaded to R2: {r2_key}")
    except Exception as e:
        logger.warning(f"R2 upload failed for {r2_key}: {e}")


def check_drift_severity(reference_df, current_df):
    """
    Run a drift check and return the fraction of features that have drifted.
    Returns a float between 0.0 and 1.0.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    result          = report.as_dict()
    drift_metrics   = result["metrics"][0]["result"]

    total_features   = drift_metrics.get("number_of_columns", 1)
    drifted_features = drift_metrics.get("number_of_drifted_columns", 0)
    drift_fraction   = drifted_features / max(total_features, 1)

    logger.info(f"Drift check: {drifted_features}/{total_features} features drifted ({drift_fraction:.1%})")
    return drift_fraction


def log_drift_event_to_db(ticker, drift_fraction, emergency_triggered):
    """
    Write a drift detection event to the drift_reports table in Neon PostgreSQL.
    Records whether an emergency retrain was triggered so the dashboard can surface it.
    """
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO drift_reports (ticker, report_date, drift_detected, drift_fraction, emergency_retrain_triggered, created_at)
            VALUES (%s, NOW()::date, %s, %s, %s, NOW())
            """,
            (ticker, drift_fraction > 0, round(drift_fraction, 4), emergency_triggered)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Could not write drift event to DB: {e}")


def trigger_emergency_retrain(ticker, drift_fraction):
    """
    Trigger an immediate out-of-schedule retraining run in a background thread.
    This is called when drift severity exceeds DRIFT_EMERGENCY_THRESHOLD.
    A cooldown period prevents repeated triggers during prolonged volatility.
    """
    global _last_emergency_retrain_time
    import time

    now = time.time()
    time_since_last = now - _last_emergency_retrain_time

    if time_since_last < EMERGENCY_RETRAIN_COOLDOWN:
        remaining = int((EMERGENCY_RETRAIN_COOLDOWN - time_since_last) / 60)
        logger.info(f"Emergency retrain cooldown active. {remaining} minutes remaining. Skipping.")
        return

    _last_emergency_retrain_time = now

    logger.warning(
        f"EMERGENCY RETRAIN TRIGGERED for {ticker}. "
        f"Drift fraction: {drift_fraction:.1%} exceeds threshold {DRIFT_EMERGENCY_THRESHOLD:.1%}"
    )

    def retrain_job():
        try:
            logger.info("Emergency retraining started in background thread...")
            from src.mlops.retraining_pipeline import run_retraining
            run_retraining(triggered_by="drift_monitor_emergency")
            logger.info("Emergency retraining complete.")
        except Exception as e:
            logger.error(f"Emergency retraining failed: {e}")

    retrain_thread = threading.Thread(target=retrain_job, daemon=True)
    retrain_thread.start()


def check_and_handle_drift(reference_df, current_df, ticker):
    """
    Main drift handling function called by the stream processor.
    Runs drift severity check, saves a full report, and triggers emergency
    retraining if the drift fraction exceeds DRIFT_EMERGENCY_THRESHOLD.
    Returns the drift fraction so the caller can log or display it.
    """
    drift_fraction      = check_drift_severity(reference_df, current_df)
    emergency_triggered = False

    run_data_drift_report(reference_df, current_df, ticker)

    if drift_fraction >= DRIFT_EMERGENCY_THRESHOLD:
        trigger_emergency_retrain(ticker, drift_fraction)
        emergency_triggered = True
    else:
        logger.info(f"{ticker}: Drift fraction {drift_fraction:.1%} is below threshold. No emergency retrain needed.")

    log_drift_event_to_db(ticker, drift_fraction, emergency_triggered)
    return drift_fraction
