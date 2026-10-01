import json
import logging
import os

import mlflow
from dotenv import load_dotenv


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# DAGSHUB CONFIGURATION
# ============================================================

DAGSHUB_USER = "arshpreetsingh-01"
DAGSHUB_REPO = "ue"

DAGSHUB_TOKEN = os.getenv("DAGSHUB_PAT")

if not DAGSHUB_TOKEN:
    raise EnvironmentError(
        "DAGSHUB_PAT environment variable is not set."
    )


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

TRACKING_URI = (
    f"https://dagshub.com/"
    f"{DAGSHUB_USER}/"
    f"{DAGSHUB_REPO}.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_tracking_uri(TRACKING_URI)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("model_registration")
logger.setLevel(logging.DEBUG)

if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        "model_registration_errors.log"
    )
    file_handler.setLevel(logging.ERROR)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - "
        "%(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# ============================================================
# LOAD MODEL INFORMATION
# ============================================================

def load_model_info(file_path: str) -> dict:

    try:

        with open(file_path, "r") as file:
            model_info = json.load(file)

        logger.info(
            "Model information loaded from %s",
            file_path
        )

        return model_info

    except FileNotFoundError:

        logger.error(
            "Model information file not found: %s",
            file_path
        )

        raise

    except json.JSONDecodeError:

        logger.error(
            "Invalid JSON file: %s",
            file_path
        )

        raise

    except Exception as exc:

        logger.error(
            "Error loading model information: %s",
            exc
        )

        raise


# ============================================================
# GET LOGGED MODEL
# ============================================================

def get_logged_model(model_id: str):

    try:

        logger.info(
            "Retrieving LoggedModel from MLflow..."
        )

        logger.info(
            "Model ID: %s",
            model_id
        )

        logged_model = mlflow.get_logged_model(
            model_id
        )

        logger.info(
            "LoggedModel successfully retrieved."
        )

        logger.info(
            "Logged model ID: %s",
            logged_model.model_id
        )

        logger.info(
            "Logged model name: %s",
            logged_model.name
        )

        logger.info(
            "Logged model artifact URI: %s",
            logged_model.artifact_location
        )

        logger.info(
            "Source run ID: %s",
            logged_model.source_run_id
        )

        return logged_model

    except Exception as exc:

        logger.error(
            "Unable to retrieve LoggedModel: %s",
            exc
        )

        raise


# ============================================================
# REGISTER LOGGED MODEL
# ============================================================

def register_model(
    model_name: str,
    model_info: dict
):

    # --------------------------------------------------------
    # Get model_id from experiment_info.json
    # --------------------------------------------------------

    model_id = model_info.get("model_id")

    if not model_id:

        raise ValueError(
            "model_id is missing from "
            "reports/experiment_info.json"
        )

    logger.info(
        "=========================================="
    )

    logger.info(
        "MLflow version: %s",
        mlflow.__version__
    )

    logger.info(
        "MLflow tracking URI: %s",
        mlflow.get_tracking_uri()
    )

    logger.info(
        "DagsHub repository: %s",
        DAGSHUB_REPO
    )

    logger.info(
        "Model ID: %s",
        model_id
    )

    logger.info(
        "=========================================="
    )

    # --------------------------------------------------------
    # Retrieve the actual MLflow 3.x LoggedModel
    # --------------------------------------------------------

    logged_model = get_logged_model(
        model_id
    )

    # --------------------------------------------------------
    # Verify that the model belongs to the expected run
    # --------------------------------------------------------

    logger.info(
        "Verifying source run..."
    )

    logger.info(
        "Source run ID: %s",
        logged_model.source_run_id
    )

    # --------------------------------------------------------
    # Get the actual artifact URI from LoggedModel
    # --------------------------------------------------------

    model_uri = logged_model.artifact_location

    if not model_uri:

        raise ValueError(
            "LoggedModel does not contain an "
            "artifact location."
        )

    logger.info(
        "Using LoggedModel artifact URI: %s",
        model_uri
    )

    # --------------------------------------------------------
    # Register model
    # --------------------------------------------------------

    logger.info(
        "Registering model as '%s'...",
        model_name
    )

    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    logger.info(
        "=========================================="
    )

    logger.info(
        "MODEL REGISTERED SUCCESSFULLY"
    )

    logger.info(
        "Model name: %s",
        model_version.name
    )

    logger.info(
        "Model version: %s",
        model_version.version
    )

    logger.info(
        "Source: %s",
        model_version.source
    )

    logger.info(
        "=========================================="
    )

    return model_version


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # experiment_info.json created by model_evaluation.py
        # ----------------------------------------------------

        model_info_path = (
            "reports/experiment_info.json"
        )

        model_info = load_model_info(
            model_info_path
        )

        # ----------------------------------------------------
        # Model Registry name
        # ----------------------------------------------------

        model_name = "my_model"

        # ----------------------------------------------------
        # Register
        # ----------------------------------------------------

        model_version = register_model(
            model_name=model_name,
            model_info=model_info
        )

        print(
            f"Model '{model_version.name}' "
            f"version '{model_version.version}' "
            f"registered successfully."
        )

    except Exception as exc:

        logger.error(
            "Failed to complete model registration: %s",
            exc
        )

        print(
            f"Model registration failed: {exc}"
        )

        raise


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()