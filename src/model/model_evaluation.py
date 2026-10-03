import json
import logging
import os
import pickle

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =========================================================
# 1. Load environment variables
# =========================================================

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set.")


# =========================================================
# 2. DagsHub configuration
# =========================================================

repo_owner = "arshpreetsingh-01"
repo_name = "ue"

tracking_uri = f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"

# DagsHub authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set MLflow tracking URI
mlflow.set_tracking_uri(tracking_uri)

print("MLflow version:", mlflow.__version__)
print("MLflow tracking URI:", mlflow.get_tracking_uri())


# =========================================================
# 3. Logging configuration
# =========================================================

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("model_evaluation_errors.log")
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =========================================================
# 4. Load model
# =========================================================

def load_model(file_path: str):
    """Load the trained model from a pickle file."""
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)

        logger.info("Model loaded from %s", file_path)
        return model

    except FileNotFoundError:
        logger.error("Model file not found: %s", file_path)
        raise
    except Exception as exc:
        logger.error("Error loading model: %s", exc)
        raise


# =========================================================
# 5. Load test data
# =========================================================

def load_data(file_path: str) -> pd.DataFrame:
    """Load test data from CSV."""
    try:
        df = pd.read_csv(file_path)
        logger.info("Test data loaded from %s", file_path)
        return df

    except Exception as exc:
        logger.error("Error loading test data: %s", exc)
        raise


# =========================================================
# 6. Evaluate model
# =========================================================

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    pos_label=1
) -> dict:
    """Evaluate model and return metrics."""
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)

        # Identify positive class index dynamically to prevent index errors
        if hasattr(clf, "classes_"):
            classes = clf.classes_
            
            # Check for exact match (e.g. integer 1)
            if pos_label in classes:
                resolved_label = pos_label
            # Check for string match (e.g. string '1')
            elif str(pos_label) in classes:
                resolved_label = str(pos_label)
            # Fallback to the second class if the provided label isn't found
            else:
                resolved_label = classes[1] if len(classes) > 1 else classes[0]
                logger.warning(
                    "pos_label '%s' not found in %s. Automatically defaulting to '%s'.", 
                    pos_label, classes, resolved_label
                )
            
            pos_idx = np.where(classes == resolved_label)[0][0]
            pos_proba = y_pred_proba[:, pos_idx]
        else:
            resolved_label = pos_label
            pos_proba = y_pred_proba[:, 1]

        accuracy = accuracy_score(y_test, y_pred)

        precision = precision_score(
            y_test,
            y_pred,
            pos_label=resolved_label,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            pos_label=resolved_label,
            zero_division=0
        )

        auc = roc_auc_score(
            y_test == resolved_label,
            pos_proba
        )

        metrics = {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "auc": float(auc),
        }

        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    except Exception as exc:
        logger.error("Error during model evaluation: %s", exc)
        raise


# =========================================================
# 7. Save metrics
# =========================================================

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save metrics to JSON."""
    try:
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.info("Metrics saved to %s", file_path)

    except Exception as exc:
        logger.error("Error saving metrics: %s", exc)
        raise


# =========================================================
# 8. Save model information
# =========================================================

def save_model_info(
    run_id: str,
    model_id: str,
    file_path: str
) -> None:

    directory = os.path.dirname(file_path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    model_info = {
        "run_id": run_id,
        "model_id": model_id
    }

    with open(file_path, "w") as file:
        json.dump(
            model_info,
            file,
            indent=4
        )

    logger.info(
        "Model information saved to %s",
        file_path
    )

# =========================================================
# 9. Main
# =========================================================

# =========================================================
# 9. Main
# =========================================================

def main():
    try:
        # Set experiment name
        mlflow.set_experiment("dvc-pipeline")

        # Start MLflow run
        with mlflow.start_run() as run:
            logger.info("MLflow run started: %s", run.info.run_id)

            # Load model
            clf = load_model("./models/model.pkl")

            # Load test data
            test_data = load_data("./data/processed/test_bow.csv")

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate model
            metrics = evaluate_model(clf, X_test, y_test, pos_label=1)

            # Save metrics locally
            save_metrics(metrics, "reports/metrics.json")

            # Log metrics to MLflow
            mlflow.log_metrics(metrics)

            # Log model parameters
            if hasattr(clf, "get_params"):
                params = clf.get_params()
                clean_params = {key: str(value) for key, value in params.items()}
                mlflow.log_params(clean_params)

            # ✅ FIXED: Changed 'name' to 'artifact_path'
            logged_model = mlflow.sklearn.log_model(
                sk_model=clf,
                artifact_path="model"
            )

            # Get model_id safely from ModelInfo or construct fallback
            model_id = getattr(logged_model, "model_id", f"runs:/{run.info.run_id}/model")

            logger.info(
                "Logged model ID: %s",
                model_id
            )
            
            # Save experiment info
            save_model_info(
                run_id=run.info.run_id,
                model_id=model_id,
                file_path="reports/experiment_info.json"
            )

            # Log artifacts
            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("reports/experiment_info.json")

            if os.path.exists("model_evaluation_errors.log"):
                mlflow.log_artifact("model_evaluation_errors.log")

            logger.info("MLflow run completed successfully.")
            logger.info("Run ID: %s", run.info.run_id)
            logger.info("Model artifact path: model")
            logger.info("Tracking URI: %s", mlflow.get_tracking_uri())

    except Exception as exc:
        logger.error("Failed to complete model evaluation: %s", exc)
        print(f"Error: {exc}")
        raise


if __name__ == '__main__':
    main()