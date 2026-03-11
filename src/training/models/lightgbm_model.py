"""
LightGBM Model
Gradient boosted trees using leaf-wise growth strategy.
Generally faster than XGBoost and performs similarly on financial tabular data.
Included for direct head-to-head comparison with XGBoost.
"""

import logging

import lightgbm as lgb

from config.settings import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGRESSION_PARAMS = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "objective":        "regression",
    "metric":           "mae",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
    "verbose":          -1
}

CLASSIFICATION_PARAMS = {
    **REGRESSION_PARAMS,
    "objective":        "binary",
    "metric":           "binary_logloss"
}


def train_regressor(X_train, y_train, X_val, y_val):
    """
    Train a LightGBM regressor for price prediction.
    Returns the trained model.
    """
    model = lgb.LGBMRegressor(**REGRESSION_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
    )
    logger.info("LightGBM regressor training complete")
    return model


def train_classifier(X_train, y_train, X_val, y_val):
    """
    Train a LightGBM classifier for direction prediction (up/down).
    Returns the trained model.
    """
    model = lgb.LGBMClassifier(**CLASSIFICATION_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)]
    )
    logger.info("LightGBM classifier training complete")
    return model
