import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

from config.settings import RANDOM_STATE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RF_PARAMS = {
    "n_estimators":  300,
    "max_depth":     10,
    "min_samples_split": 5,
    "min_samples_leaf":  2,
    "max_features":  "sqrt",
    "class_weight":  "balanced",
    "random_state":  RANDOM_STATE,
    "n_jobs":        -1
}


def train(X_train, y_train):
    """
    Train the Random Forest direction classifier.
    Returns the trained model.
    """
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    logger.info(f"Random Forest classifier trained on {len(X_train)} samples")
    return model


def evaluate(model, X_test, y_test):
    """
    Evaluate the classifier and return a metrics dictionary.
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred, average="weighted"), 4)
    }

    logger.info(f"Direction Classifier Results:")
    logger.info(f"  Accuracy: {metrics['accuracy']}")
    logger.info(f"  F1 Score: {metrics['f1_score']}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['Down', 'Up'])}")

    return metrics


def get_feature_importance(model, feature_names):
    """
    Return a sorted list of (feature_name, importance_score) tuples.
    Useful for SHAP analysis and reporting.
    """
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return pairs
