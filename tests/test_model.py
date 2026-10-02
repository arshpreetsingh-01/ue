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
        # 1. DagsHub / MLflow authentication
        # ---------------------------------------------------------
        repo_owner = os.getenv("DAGSHUB_USER")
        repo_name = "ue"
        dagshub_token = os.getenv("DAGSHUB_PAT")

        if not repo_owner:
            raise RuntimeError("DAGSHUB_USER is not set")

        if not dagshub_token:
            raise RuntimeError("DAGSHUB_PAT is not set")

        # Explicit MLflow credentials for CI/CD environments
        os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Authenticate DagsHub
        dagshub.auth.add_app_token(dagshub_token)

        # Initialize DagsHub MLflow
        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True,
        )

        # Make tracking URI explicit
        tracking_uri = (
            f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow"
        )
        mlflow.set_tracking_uri(tracking_uri)

        # ---------------------------------------------------------
        # 2. Find the model version
        # ---------------------------------------------------------
        cls.new_model_name = "my_model"

        client = mlflow.MlflowClient()

        try:
            # Preferred: MLflow alias
            mv = client.get_model_version_by_alias(
                cls.new_model_name,
                "champion",
            )

            print(
                f"Using champion model: "
                f"{cls.new_model_name} v{mv.version}"
            )

        except Exception as alias_error:
            print(
                f"Champion alias was not available: {alias_error}"
            )
            print("Searching for the latest model version...")

            versions = list(
                client.search_model_versions(
                    filter_string=f"name = '{cls.new_model_name}'"
                )
            )

            if not versions:
                raise RuntimeError(
                    f"No versions found for registered model "
                    f"'{cls.new_model_name}'."
                )

            # Only consider READY versions
            ready_versions = [
                version
                for version in versions
                if version.status == "READY"
            ]

            if not ready_versions:
                raise RuntimeError(
                    f"No READY versions found for "
                    f"'{cls.new_model_name}'."
                )

            mv = max(
                ready_versions,
                key=lambda version: int(version.version),
            )

            print(
                f"Using latest READY model: "
                f"{cls.new_model_name} v{mv.version}"
            )

        # ---------------------------------------------------------
        # 3. Validate model version metadata
        # ---------------------------------------------------------
        cls.new_model_version = str(mv.version)

        print(f"Model name: {mv.name}")
        print(f"Model version: {mv.version}")
        print(f"Model status: {mv.status}")
        print(f"Model source: {mv.source}")
        print(f"Model run_id: {mv.run_id}")
        print(f"Model model_id: {mv.model_id}")

        if mv.status != "READY":
            raise RuntimeError(
                f"Model {cls.new_model_name} version "
                f"{cls.new_model_version} is not READY. "
                f"Current status: {mv.status}"
            )

        # ---------------------------------------------------------
        # 4. Load model WITHOUT models:/name/version
        #
        # IMPORTANT:
        # DagsHub is currently returning HTTP 500 for:
        #
        #   /api/2.0/mlflow/model-versions/get-download-uri
        #
        # Therefore we use the registered model's SOURCE URI.
        # For normally registered models this is typically:
        #
        #   runs:/<run_id>/<artifact_path>
        #
        # This bypasses the failing registry download-uri request.
        # ---------------------------------------------------------
        model_source_uri = mv.source

        if not model_source_uri:
            raise RuntimeError(
                f"Model {cls.new_model_name} version "
                f"{cls.new_model_version} does not have a source URI."
            )

        print(f"Loading model from source URI: {model_source_uri}")

        # Standard MLflow registration from a run normally produces
        # a runs:/ URI. This is the path we intentionally want here.
        if model_source_uri.startswith("runs:/"):
            cls.new_model = mlflow.pyfunc.load_model(
                model_source_uri
            )

        else:
            # MLflow 3.x can have model versions linked to a
            # LoggedModel using model_id.
            #
            # Try to resolve the LoggedModel artifact URI first.
            if mv.model_id:
                print(
                    "Model source is not a runs:/ URI. "
                    "Trying MLflow LoggedModel metadata..."
                )

                try:
                    logged_model = mlflow.get_logged_model(
                        mv.model_id
                    )

                    logged_model_uri = logged_model.artifact_uri

                    if not logged_model_uri:
                        raise RuntimeError(
                            "LoggedModel has no artifact_uri."
                        )

                    print(
                        f"LoggedModel artifact URI: "
                        f"{logged_model_uri}"
                    )

                    if logged_model_uri.startswith("runs:/"):
                        cls.new_model = mlflow.pyfunc.load_model(
                            logged_model_uri
                        )
                    else:
                        raise RuntimeError(
                            "The LoggedModel artifact URI is not a "
                            f"runs:/ URI: {logged_model_uri}"
                        )

                except Exception as logged_model_error:
                    raise RuntimeError(
                        "Unable to load the registered model without "
                        "using DagsHub's failing "
                        "get-download-uri endpoint.\n"
                        f"Model source: {model_source_uri}\n"
                        f"Model run_id: {mv.run_id}\n"
                        f"Model model_id: {mv.model_id}\n"
                        f"LoggedModel error: {logged_model_error}"
                    ) from logged_model_error

            else:
                raise RuntimeError(
                    "The registered model does not expose a usable "
                    "runs:/ source URI or model_id.\n"
                    f"Model source: {model_source_uri}\n"
                    f"Model run_id: {mv.run_id}"
                )

        # ---------------------------------------------------------
        # 5. Load vectorizer
        # ---------------------------------------------------------
        vectorizer_path = "models/vectorizer.pkl"

        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Vectorizer not found: {vectorizer_path}"
            )

        with open(vectorizer_path, "rb") as f:
            cls.vectorizer = pickle.load(f)

        # ---------------------------------------------------------
        # 6. Load holdout test data
        # ---------------------------------------------------------
        test_data_path = "data/processed/test_bow.csv"

        if not os.path.exists(test_data_path):
            raise FileNotFoundError(
                f"Test data not found: {test_data_path}"
            )

        cls.holdout_data = pd.read_csv(test_data_path)

        print("Model, vectorizer, and holdout data loaded successfully.")

    # -------------------------------------------------------------
    # Test 1: Model loaded
    # -------------------------------------------------------------
    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    # -------------------------------------------------------------
    # Test 2: Model input/output signature
    # -------------------------------------------------------------
    def test_model_signature(self):
        input_text = "hi how are you"

        input_data = self.vectorizer.transform([input_text])

        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[
                str(i)
                for i in range(input_data.shape[1])
            ],
        )

        prediction = self.new_model.predict(input_df)

        # Number of features must match vectorizer
        self.assertEqual(
            input_df.shape[1],
            len(self.vectorizer.get_feature_names_out()),
        )

        # One input row -> one prediction
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