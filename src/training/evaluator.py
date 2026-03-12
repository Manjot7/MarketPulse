import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    accuracy_score,
    f1_score
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "reports")


def regression_metrics(y_true, y_pred, model_name="model", ticker=""):
    """
    Compute MAE, MAPE, RMSE, and directional accuracy for a regression model.
    Returns a metrics dictionary.
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()

    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    actual_dir    = np.sign(np.diff(y_true))
    predicted_dir = np.sign(np.diff(y_pred))
    dir_accuracy  = np.mean(actual_dir == predicted_dir)

    metrics = {
        "model":         model_name,
        "ticker":        ticker,
        "mae":           round(mae, 4),
        "mape":          round(mape, 6),
        "mape_pct":      round(mape * 100, 4),
        "rmse":          round(rmse, 4),
        "accuracy":      round(1 - mape, 6),
        "dir_accuracy":  round(dir_accuracy, 4)
    }

    logger.info(f"{model_name} ({ticker}): MAE={mae:.2f}, MAPE={mape*100:.3f}%, RMSE={rmse:.2f}, DirAcc={dir_accuracy:.3f}")
    return metrics


def classification_metrics(y_true, y_pred, model_name="model", ticker=""):
    """
    Compute accuracy and F1 score for a direction classification model.
    Returns a metrics dictionary.
    """
    metrics = {
        "model":    model_name,
        "ticker":   ticker,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred, average="weighted"), 4)
    }

    logger.info(f"{model_name} ({ticker}): Accuracy={metrics['accuracy']}, F1={metrics['f1_score']}")
    return metrics


def plot_predictions(y_true, predictions_dict, ticker, save=True):
    """
    Plot actual vs predicted prices for multiple models on the same chart.
    predictions_dict: {model_name: y_pred_array}
    """
    plt.figure(figsize=(14, 7))
    plt.plot(y_true, linewidth=2.5, color="black", label="Actual Price")

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for idx, (model_name, y_pred) in enumerate(predictions_dict.items()):
        color = colors[idx % len(colors)]
        plt.plot(y_pred, linewidth=1.5, color=color, label=model_name, alpha=0.85)

    plt.xlabel("Trading Days", fontsize=12, labelpad=10)
    plt.ylabel("Closing Price (USD)", fontsize=12, labelpad=10)
    plt.title(f"{ticker} Price Prediction Comparison", fontsize=16, pad=15)
    plt.legend(loc="upper right", fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        plots_dir = os.path.join(REPORTS_DIR, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        path = os.path.join(plots_dir, f"{ticker}_predictions.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {path}")
        plt.close()

    return plt


def build_results_table(all_metrics):
    """
    Build a summary DataFrame from a list of metrics dictionaries.
    Sorts by MAE ascending.
    """
    df = pd.DataFrame(all_metrics)
    df = df.sort_values("mae").reset_index(drop=True)
    return df


def save_results_markdown(results_df, ticker):
    """
    Save a formatted Markdown results table for a given ticker.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    path = os.path.join(REPORTS_DIR, f"{ticker}_results.md")

    with open(path, "w") as f:
        f.write(f"# {ticker} Model Comparison Results\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n")

    logger.info(f"Results table saved to {path}")
    return path
