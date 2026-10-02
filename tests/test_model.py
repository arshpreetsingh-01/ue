```python
import os
import pickle
import unittest

import dagshub
import mlflow
import pandas as pd


class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # ---------------------------------------------------------
        # 1. Read DagsHub credentials from GitHub Actions secrets
        # ---------------------------------------------------------
        repo_owner = os.getenv("DAGSHUB_USER")
        repo_name = "ue"
        dagshub_token = os.getenv("DAGSHUB_PAT")

        if not repo_owner:
            raise RuntimeError("DAGSHUB_USER is not set")

        if not dagshub_token:
            raise RuntimeError("DAGSHUB_PAT is not set")

        # ---------------------------------------------------------
        # 2. Configure MLflow authentication explicitly
        # ---------------------------------------------------------
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        tracking_uri = (
            f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        )

        os.environ["MLFLOW_TRACKING_URI"] = tracking_uri

        # ---------------------------------------------------------
        # 3. Authenticate with DagsHub
        # ---------------------------------------------------------
        dagshub.auth.add_app_token(dagshub_token)

        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True,
        )

        # Make tracking / registry URI explicit
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)

        print(f"MLflow tracking URI: {tracking_uri}")

        # ---------------------------------------------------------
        # 4. Create MLflow client
        # ---------------------------------------------------------
        client = mlflow.MlflowClient(
            tracking_uri=tracking_uri,
            registry_uri=tracking_uri,
        )

        cls.new_model_name = "my_model"

        # ---------------------------------------------------------
        # 5. Find model version
        #
        # First try the "champion" alias.
        # If DagsHub does not support / expose that alias,
        # fall back to the latest READY version.
        # ---------------------------------------------------------
        mv = None

        try:
            mv = client.get_model_version_by_alias(
                cls.new_model_name,
                "champion",
            )

            print(
                f"Using champion model version: {mv.version}"
            )

        except Exception as alias_error:
            print(
                "Champion alias could not be used. "
                f"Falling back to latest READY version. "
                f"Reason: {alias_error}"
            )

        # ---------------------------------------------------------
        # 6. Fallback: find latest READY model version
        # ---------------------------------------------------------
        if mv is None:
            versions = list(
                client.search_model_versions(
                    filter_string=f"name = '{cls.new_model_name}'"
                )
            )

            if not versions:
                raise RuntimeError(
                    f"No model versions found for "
                    f"'{cls.new_model_name}'."
                )

            ready_versions = [
                version
                for version in versions
                if str(version.status).upper() == "READY"
            ]

            if not ready_versions:
                raise RuntimeError(
                    f"No READY model versions found for "
                    f"'{cls.new_model_name}'."
                )

            mv = max(
                ready_versions,
                key=lambda version: int(version.version),
            )

            print(
                f"Using latest READY model version: {mv.version}"
            )

        # ---------------------------------------------------------
        # 7. Store model metadata
        # ---------------------------------------------------------
        cls.new_model_version = str(mv.version)

        print(f"Model name: {mv.name}")
        print(f"Model version: {mv.version}")
        print(f"Model status: {mv.status}")
        print(f"Model source: {mv.source}")
        print(f"Model run_id: {mv.run_id}")
        print(f"Model model_id: {mv.model_id}")

        if str(mv.status).upper() != "READY":
            raise RuntimeError(
                f"Model '{cls.new_model_name}' version "
                f"{cls.new_model_version} is not READY. "
                f"Current status: {mv.status}"
            )

        if not mv.source:
            raise RuntimeError(
                f"Model '{cls.new_model_name}' version "
                f"{cls.new_model_version} has no source URI."
            )

        # ---------------------------------------------------------
        # 8. Download using the MODEL SOURCE URI
        #
        # IMPORTANT:
        #
        # Do NOT use:
        #
        # models:/my_model/14
        #
        # because MLflow then calls:
        #
        # get-model-version-download-uri
        #
        # which was returning HTTP 500 from DagsHub.
        #
        # Your MLflow 3.x model has this source instead:
        #
        # mlflow-artifacts:/...
        #
        # That is a valid MLflow artifact URI.
        # ---------------------------------------------------------
        model_source_uri = mv.source

        print(
            f"Downloading model from source URI: "
            f"{model_source_uri}"
        )

        try:
            local_model_dir = mlflow.artifacts.download_artifacts(
                artifact_uri=model_source_uri,
                tracking_uri=tracking_uri,
                registry_uri=tracking_uri,
            )

        except Exception as download_error:
            raise RuntimeError(
                "Failed to download the registered model "
                "using its source URI.\n"
                f"Source URI: {model_source_uri}\n"
                f"Tracking URI: {tracking_uri}\n"
                f"Original error: {download_error}"
            ) from download_error

        print(
            f"Model downloaded successfully to: "
            f"{local_model_dir}"
        )

        # ---------------------------------------------------------
        # 9. Load the downloaded model locally
        # ---------------------------------------------------------
        try:
            cls.new_model = mlflow.pyfunc.load_model(
                local_model_dir
            )

        except Exception as load_error:
            raise RuntimeError(
                "Model artifacts were downloaded, but MLflow "
                "could not load the model.\n"
                f"Local model directory: {local_model_dir}\n"
                f"Original error: {load_error}"
            ) from load_error

        print("MLflow model loaded successfully.")

        # ---------------------------------------------------------
        # 10. Load vectorizer
        # ---------------------------------------------------------
        vectorizer_path = "models/vectorizer.pkl"

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer not found: {vectorizer_path}"
            )

        with open(vectorizer_path, "rb") as file:
            cls.vectorizer = pickle.load(file)

        print("Vectorizer loaded successfully.")

        # ---------------------------------------------------------
        # 11. Load processed test data
        # ---------------------------------------------------------
        test_data_path = "data/processed/test_bow.csv"

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"Test data not found: {test_data_path}"
            )

        cls.holdout_data = pd.read_csv(test_data_path)

        print("Holdout test data loaded successfully.")

    # -------------------------------------------------------------
    # Test 1: Check model object
    # -------------------------------------------------------------
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    # -------------------------------------------------------------
    # Test 2: Check model input/output signature
    # -------------------------------------------------------------
    def test_model_signature(self):
        input_text = "hi how are you"

        # Transform text using the same vectorizer
        input_data = self.vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[
                str(i)
                for i in range(input_data.shape[1])
            ],
        )

        # Predict
        prediction = self.new_model.predict(input_df)

        # Number of features must match the vectorizer
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out()),
        )

        # One input row must produce one prediction
        self.assertEqual(
            len(prediction),
            input_df.shape[0],
        )

        # Prediction must be one-dimensional
        self.assertEqual(
            getattr(prediction, "ndim", 1),
            1,
        )


if __name__ == "__main__":
    unittest.main()
```
