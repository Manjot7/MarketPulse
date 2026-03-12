import logging
import os

import mlflow
import mlflow.sklearn
import mlflow.tensorflow

from config.settings import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    DAGSHUB_USERNAME,
    DAGSHUB_TOKEN
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SENTIMENT_CAPABLE_MODELS = [
    "LSTM-Baseline",
    "FinBERT-LSTM",
    "GRU",
    "BiLSTM",
    "CNN-LSTM"
]


def setup_mlflow():
    """
    Configure MLflow to use DagsHub as the remote tracking server.
    Sets credentials via environment variables for the DagsHub HTTP auth.
    """
    if not MLFLOW_TRACKING_URI:
        logger.warning("MLFLOW_TRACKING_URI not set. Using local tracking.")
        return

    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"]  = DAGSHUB_TOKEN

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    logger.info(f"MLflow tracking configured: {MLFLOW_TRACKING_URI}")


def log_run(model_name, ticker, params, metrics, artifact_paths=None):
    """
    Log a single training run to MLflow.
    Returns the MLflow run_id string.
    """
    run_name = f"{model_name}_{ticker}"

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("ticker", ticker)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        if artifact_paths:
            for path in artifact_paths:
                if os.path.exists(path):
                    mlflow.log_artifact(path)
                else:
                    logger.warning(f"Artifact path not found: {path}")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run logged: {run_name} | run_id={run_id}")
        return run_id


def register_model(model_name, run_id, artifact_path="model"):
    """
    Register a trained model in the MLflow Model Registry.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result    = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"Model registered: {model_name} v{result.version}")
    return result


def promote_model(model_name, version, stage="Production"):
    """
    Promote a registered model version to a lifecycle stage.
    """
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage
    )
    logger.info(f"Model {model_name} v{version} promoted to {stage}")


def get_production_model_uri(model_name):
    """
    Return the URI of the current Production model for a given model name.
    Used by the stream processor and retraining pipeline to load the best model.
    """
    return f"models:/{model_name}/Production"


def promote_best_model_for_ticker(ticker, dl_results):
    """
    After all models finish training for a ticker, find the sentiment-capable
    sequence model with the lowest MAPE and register it as the Production model.
    Only considers SENTIMENT_CAPABLE_MODELS — ARIMA, Prophet, XGBoost, LightGBM,
    and RandomForest are excluded because they cannot consume the live sentiment
    signal from the Kafka stream.\
    """
    candidates = [
        r for r in dl_results
        if r.get("model") in SENTIMENT_CAPABLE_MODELS and r.get("mape_pct") is not None
    ]

    if not candidates:
        logger.warning(f"No sentiment-capable model results found for {ticker}. Skipping promotion.")
        return

    best = min(candidates, key=lambda r: r["mape_pct"])
    best_model_name = best["model"]
    best_mape       = best["mape_pct"]

    logger.info(
        f"Best sentiment-capable model for {ticker}: {best_model_name} "
        f"(MAPE={best_mape:.3f}%) — promoting to Production"
    )

    client      = mlflow.tracking.MlflowClient()
    experiment  = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)

    if experiment is None:
        logger.warning(f"MLflow experiment '{MLFLOW_EXPERIMENT_NAME}' not found. Skipping promotion.")
        return

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.model = '{best_model_name}' AND tags.ticker = '{ticker}'",
        order_by=["metrics.mape ASC"],
        max_results=1
    )

    if not runs:
        logger.warning(f"No MLflow run found for {best_model_name} / {ticker}. Skipping promotion.")
        return

    run_id        = runs[0].info.run_id
    registry_name = f"Production-{ticker}"

    try:
        result = mlflow.register_model(
            model_uri=f"runs:/{run_id}/model",
            name=registry_name
        )
        client.transition_model_version_stage(
            name=registry_name,
            version=result.version,
            stage="Production"
        )
        logger.info(
            f"Registered and promoted {best_model_name} for {ticker} "
            f"as {registry_name} v{result.version} (Production)"
        )
    except Exception as e:
        logger.warning(f"Model promotion failed for {ticker}: {e}")
