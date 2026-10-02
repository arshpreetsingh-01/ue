# promote model

import os
import mlflow
from mlflow import MlflowClient

def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "arshpreetsingh-01"
    repo_name = "ue"

    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

    client = MlflowClient()
    model_name = "my_model"

    # --------------------------------------------------------------------------
    # 1. Fetch the target model version to promote
    # --------------------------------------------------------------------------
    # Modern approach: Retrieve model version by 'candidate' / 'staging' alias,
    # or fall back to getting the latest created version number.
    try:
        staging_version_obj = client.get_model_version_by_alias(model_name, "staging")
        target_version = staging_version_obj.version
    except Exception:
        # Fallback: Find the latest created model version if no alias is set
        all_versions = client.search_model_versions(f"name='{model_name}'")
        if not all_versions:
            raise RuntimeError(f"No registered model versions found for model '{model_name}'")
        
        latest_version = max(all_versions, key=lambda v: int(v.version))
        target_version = latest_version.version

    # --------------------------------------------------------------------------
    # 2. Promote model version to Production using Model Aliases
    # --------------------------------------------------------------------------
    # Assigning the 'champion' (or 'production') alias automatically transfers 
    # it from any previous version to the target_version.
    target_alias = "champion"  # or "production"
    
    client.set_registered_model_alias(
        name=model_name,
        alias=target_alias,
        version=str(target_version)
    )

    print(f"Model version {target_version} successfully promoted with alias '{target_alias}'.")

if __name__ == "__main__":
    promote_model()