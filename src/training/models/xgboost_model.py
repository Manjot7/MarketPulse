import logging

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from config.settings import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REGRESSION_PARAMS = {
    "n_estimators":      500,
    "max_depth":         6,
    "learning_rate":     0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "min_child_weight":  3,
    "reg_alpha":         0.1,
    "reg_lambda":        1.0,
    "objective":         "reg:squarederror",
    "tree_method":       "hist",
    "random_state":      RANDOM_STATE,
    "n_jobs":            -1,
    "early_stopping_rounds": 20
}

CLASSIFICATION_PARAMS = {
    **REGRESSION_PARAMS,
    "objective":         "binary:logistic",
    "eval_metric":       "logloss",
    "scale_pos_weight":  1.0
}


def train_regressor(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost regressor for price prediction.
    Returns the trained model.
    """
    model = xgb.XGBRegressor(**REGRESSION_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    logger.info("XGBoost regressor training complete")
    return model


def train_classifier(X_train, y_train, X_val, y_val):
    """
    Train an XGBoost classifier for direction prediction (up/down).
    Returns the trained model.
    """
    model = xgb.XGBClassifier(**CLASSIFICATION_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    logger.info("XGBoost classifier training complete")
    return model
