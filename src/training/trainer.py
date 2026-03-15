import logging
import os

import boto3
import joblib
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import numpy as np
import psycopg2
from sklearn.preprocessing import MinMaxScaler

from config.settings import (
    TICKERS,
    SEQUENCE_LENGTH,
    TRAIN_SPLIT,
    MODELS_DIR,
    R2_ENDPOINT_URL,
    R2_ACCESS_KEY_ID,
    R2_SECRET_ACCESS_KEY,
    R2_BUCKET_NAME,
    DATABASE_URL
)
from src.mlops.experiment_tracker import setup_mlflow, log_run
from src.processing.feature_builder import (
    build_sequences,
    build_tabular,
    train_test_split_sequences,
    SEQUENCE_FEATURE_COLS,
    TABULAR_FEATURE_COLS
)
from src.training.evaluator import regression_metrics, classification_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def get_r2_client():
    return boto3.client(
        "s3",
        endpoint_url=R2_ENDPOINT_URL,
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY
    )


def upload_to_r2(local_path, r2_key):
    """
    Upload a file to Cloudflare R2 using the exact key provided.
    Key format must match what stream_processor.py expects:
      model:  models/{ticker}/{model_name}/{model_name}_{ticker}
      scaler: models/{ticker}/{model_name}/{model_name}_{ticker}_scaler
    """
    if not local_path or not os.path.exists(local_path):
        logger.warning(f"File not found, skipping R2 upload: {local_path}")
        return

    if not all([R2_ENDPOINT_URL, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY]):
        logger.warning("R2 credentials not set. Skipping upload.")
        return

    try:
        client = get_r2_client()
        client.upload_file(local_path, R2_BUCKET_NAME, r2_key)
        logger.info(f"Uploaded to R2: {r2_key}")
    except Exception as e:
        logger.warning(f"R2 upload failed for {r2_key}: {e}")


def save_and_upload_model(model, model_name, ticker, scaler=None):
    """
    Save model and scaler locally then upload both to R2.
    R2 key format matches exactly what stream_processor.py expects:
      model:  models/{ticker}/{model_name}/{model_name}_{ticker}
      scaler: models/{ticker}/{model_name}/{model_name}_{ticker}_scaler
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save and upload model
    try:
        if hasattr(model, "save"):
            local_path = os.path.join(MODELS_DIR, f"{model_name}_{ticker}.keras")
            model.save(local_path)
        else:
            local_path = os.path.join(MODELS_DIR, f"{model_name}_{ticker}.pkl")
            joblib.dump(model, local_path)

        upload_to_r2(local_path, f"models/{ticker}/{model_name}/{model_name}_{ticker}")

    except Exception as e:
        logger.warning(f"Model save/upload failed for {model_name}_{ticker}: {e}")
        return

    # Save and upload scaler
    if scaler is not None:
        try:
            scaler_path = os.path.join(MODELS_DIR, f"{model_name}_{ticker}_scaler.pkl")
            joblib.dump(scaler, scaler_path)
            upload_to_r2(scaler_path, f"models/{ticker}/{model_name}/{model_name}_{ticker}_scaler")
            logger.info(f"Scaler uploaded to R2 for {model_name}/{ticker}")
        except Exception as e:
            logger.warning(f"Scaler save/upload failed for {model_name}_{ticker}: {e}")


def write_metrics_to_db(metrics):
    """
    Insert model performance metrics into the model_metrics table in Neon PostgreSQL.
    """
    try:
        conn   = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO model_metrics
                (model_name, ticker, mae, mape, rmse, dir_accuracy, trained_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (model_name, ticker) DO UPDATE
                SET mae=EXCLUDED.mae, mape=EXCLUDED.mape,
                    rmse=EXCLUDED.rmse, dir_accuracy=EXCLUDED.dir_accuracy,
                    trained_at=NOW()
            """,
            (
                metrics["model"], metrics["ticker"],
                metrics.get("mae"), metrics.get("mape"),
                metrics.get("rmse"), metrics.get("dir_accuracy")
            )
        )
        conn.commit()
        conn.close()

    except Exception as e:
        logger.warning(f"DB metrics write failed: {e}")


def train_deep_learning_models(ticker_df, ticker):
    """
    Train all sequence-based deep learning models for a given ticker.
    Returns a list of metrics dicts.
    """
    from src.training.models import (
        lstm_baseline,
        lstm_sentiment,
        gru_model,
        bilstm_model,
        cnn_lstm_model
    )

    results = []

    X, y, scaler = build_sequences(ticker_df, SEQUENCE_FEATURE_COLS)
    X_train, X_test, y_train, y_test = train_test_split_sequences(X, y)

    split_val  = int(len(X_train) * 0.9)
    X_val      = X_train[split_val:]
    y_val      = y_train[split_val:]
    X_train    = X_train[:split_val]
    y_train_tr = y_train[:split_val]

    dl_models = [
        ("LSTM-Baseline",   lstm_baseline),
        ("FinBERT-LSTM",    lstm_sentiment),
        ("GRU",             gru_model),
        ("BiLSTM",          bilstm_model),
        ("CNN-LSTM",        cnn_lstm_model)
    ]

    for model_name, module in dl_models:
        logger.info(f"Training {model_name} for {ticker}...")
        try:
            model, history = module.train(X_train, y_train_tr, X_val, y_val)

            y_pred_scaled = model.predict(X_test).flatten()
            dummy         = np.zeros((len(y_pred_scaled), len(SEQUENCE_FEATURE_COLS) + 1))
            dummy[:, -1]  = y_pred_scaled
            y_pred        = scaler.inverse_transform(dummy)[:, -1]

            dummy2        = np.zeros((len(y_test), len(SEQUENCE_FEATURE_COLS) + 1))
            dummy2[:, -1] = y_test
            y_true        = scaler.inverse_transform(dummy2)[:, -1]

            metrics = regression_metrics(y_true, y_pred, model_name, ticker)

            save_and_upload_model(model, model_name, ticker, scaler=scaler)
            write_metrics_to_db(metrics)

            params = {
                "sequence_length": SEQUENCE_LENGTH,
                "train_split":     TRAIN_SPLIT,
                "ticker":          ticker
            }
            log_run(model_name, ticker, params, {
                "mae":          metrics["mae"],
                "mape":         metrics["mape"],
                "rmse":         metrics["rmse"],
                "dir_accuracy": metrics["dir_accuracy"]
            })

            results.append(metrics)
            logger.info(f"{model_name} ({ticker}) complete: MAE={metrics['mae']}")

        except Exception as e:
            logger.error(f"{model_name} training failed for {ticker}: {e}")

    return results


def train_tabular_models(ticker_df, ticker):
    """
    Train all tabular models (XGBoost, LightGBM, Random Forest) for a given ticker.
    Returns a list of metrics dicts.
    """
    from src.training.models import xgboost_model, lightgbm_model, direction_classifier

    results = []

    X_train, X_test, y_train, y_test = build_tabular(
        ticker_df, TABULAR_FEATURE_COLS, target_col="Close"
    )

    X_train_c, X_test_c, y_train_c, y_test_c = build_tabular(
        ticker_df, TABULAR_FEATURE_COLS, target_col="direction"
    )

    tabular_reg = [
        ("XGBoost-Regression",  xgboost_model,  "regressor"),
        ("LightGBM-Regression", lightgbm_model, "regressor")
    ]

    for model_name, module, mode in tabular_reg:
        logger.info(f"Training {model_name} for {ticker}...")
        try:
            split = int(len(X_train) * 0.9)
            model = module.train_regressor(
                X_train[:split], y_train[:split],
                X_train[split:], y_train[split:]
            )
            y_pred  = model.predict(X_test)
            metrics = regression_metrics(y_test, y_pred, model_name, ticker)

            save_and_upload_model(model, model_name, ticker)
            write_metrics_to_db(metrics)

            log_run(model_name, ticker, {"ticker": ticker}, {
                "mae":  metrics["mae"],
                "mape": metrics["mape"],
                "rmse": metrics["rmse"]
            })

            results.append(metrics)

        except Exception as e:
            logger.error(f"{model_name} training failed for {ticker}: {e}")

    tabular_clf = [
        ("XGBoost-Direction",  xgboost_model,       "classifier"),
        ("LightGBM-Direction", lightgbm_model,       "classifier"),
        ("RF-Direction",       direction_classifier, "classifier")
    ]

    for model_name, module, mode in tabular_clf:
        logger.info(f"Training {model_name} for {ticker}...")
        try:
            split = int(len(X_train_c) * 0.9)

            if module == direction_classifier:
                model  = module.train(X_train_c[:split], y_train_c[:split])
            else:
                model  = module.train_classifier(
                    X_train_c[:split], y_train_c[:split],
                    X_train_c[split:], y_train_c[split:]
                )

            y_pred  = model.predict(X_test_c)
            metrics = classification_metrics(y_test_c, y_pred, model_name, ticker)

            save_and_upload_model(model, model_name, ticker)

            log_run(model_name, ticker, {"ticker": ticker}, {
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"]
            })

            results.append(metrics)

        except Exception as e:
            logger.error(f"{model_name} training failed for {ticker}: {e}")

    return results


def train_statistical_models(ticker_df, ticker):
    """
    Train ARIMA and Prophet models for a given ticker.
    Returns a list of metrics dicts.
    """
    from src.training.models import arima_model, prophet_model
    from src.training.evaluator import regression_metrics

    results = []

    logger.info(f"Training ARIMA for {ticker}...")
    try:
        arima_result = arima_model.train_and_predict(ticker_df)
        metrics      = regression_metrics(
            arima_result["actual"].values,
            arima_result["predicted"].values,
            "ARIMA", ticker
        )
        write_metrics_to_db(metrics)
        log_run("ARIMA", ticker, {"order": "5,1,0"}, {"mae": metrics["mae"], "rmse": metrics["rmse"]})
        results.append(metrics)
    except Exception as e:
        logger.error(f"ARIMA training failed for {ticker}: {e}")

    logger.info(f"Training Prophet for {ticker}...")
    try:
        model, train_df, test_df = prophet_model.train(ticker_df)
        prophet_result           = prophet_model.predict(model, test_df)
        metrics                  = regression_metrics(
            prophet_result["y"].values,
            prophet_result["yhat"].values,
            "Prophet", ticker
        )
        write_metrics_to_db(metrics)
        log_run("Prophet", ticker, {"ticker": ticker}, {"mae": metrics["mae"], "rmse": metrics["rmse"]})
        results.append(metrics)
    except Exception as e:
        logger.error(f"Prophet training failed for {ticker}: {e}")

    return results


def train_all_models(features_df):
    """
    Train all models for all tickers.
    Iterates over each ticker, trains deep learning + tabular + statistical models.
    Returns a flat list of all metrics dicts across all tickers and models.
    """
    setup_mlflow()
    all_results = []

    for ticker in features_df["Ticker"].unique():
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting training for {ticker}")
        logger.info(f"{'='*60}")

        ticker_df = features_df[features_df["Ticker"] == ticker].copy()
        ticker_df = ticker_df.reset_index(drop=True)

        if len(ticker_df) < 100:
            logger.warning(f"Insufficient data for {ticker} ({len(ticker_df)} rows). Skipping.")
            continue

        dl_results   = train_deep_learning_models(ticker_df, ticker)
        tab_results  = train_tabular_models(ticker_df, ticker)
        stat_results = train_statistical_models(ticker_df, ticker)

        ticker_results = dl_results + tab_results + stat_results
        all_results.extend(ticker_results)

        logger.info(f"{ticker} complete: {len(ticker_results)} models trained")

    logger.info(f"\nAll training complete. Total runs: {len(all_results)}")
    return all_results
