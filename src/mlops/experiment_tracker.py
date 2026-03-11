"""
Experiment Tracker
Thin MLflow wrapper for logging runs, metrics, parameters, and artifacts to DagsHub.
Keeps MLflow calls centralized so training code stays clean.
"""

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


def setup_mlflow():
    """
    Configure MLflow to use DagsHub as the remote tracking server.
    Sets credentials via environment variables for the DagsHub HTTP auth.
    Must be called once before any logging operations.
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

    Parameters
    ----------
    model_name      : string identifier for the model e.g. "FinBERT-LSTM"
    ticker          : stock ticker this run was trained on e.g. "AAPL"
    params          : dict of hyperparameters
    metrics         : dict of evaluation metrics
    artifact_paths  : list of local file paths to log as artifacts
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
    The model will appear in the DagsHub registry under model_name.
    """
    model_uri = f"runs:/{run_id}/{artifact_path}"
    result    = mlflow.register_model(model_uri=model_uri, name=model_name)
    logger.info(f"Model registered: {model_name} v{result.version}")
    return result


def promote_model(model_name, version, stage="Production"):
    """
    Promote a registered model version to a lifecycle stage.
    Stages: Staging, Production, Archived.
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
