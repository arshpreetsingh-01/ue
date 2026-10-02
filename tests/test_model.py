# load test + signature test + performance test

import os
import pickle
import unittest
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Suppress Pydantic protected namespace warnings during test runs
warnings.filterwarnings(
    "ignore", 
    category=UserWarning, 
    module="pydantic.*"
)

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # 1. Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "arshpreetsingh-01"
        repo_name = "ue"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # 2. Get latest model version via Alias (compatible with MLflow 3.x)
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name, alias="champion")

        # Fallback to loading direct version or uri if alias isn't set
        if cls.new_model_version:
            cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        else:
            cls.new_model_uri = f"models:/{cls.new_model_name}@champion"

        # Load model using PyFunc
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # 3. Load local artifacts and test data
        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")

    @staticmethod
    def get_latest_model_version(model_name: str, alias: str = "champion"):
        """
        Fetches model version using MLflow Aliases (MLflow 3.x modern approach).
        """
        client = mlflow.MlflowClient()
        try:
            # get_model_version_by_alias returns a single ModelVersion object
            mv = client.get_model_version_by_alias(model_name, alias)
            return mv.version
        except Exception:
            # Fallback: Search for the latest created model version if no alias exists
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                return latest.version
            return None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input based on expected vectorizer output
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])],
        )

        # Predict using the loaded MLflow model
        prediction = self.new_model.predict(input_df)

        # Verify input feature dimensions match vectorizer vocabulary
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify prediction shape and dimensions
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(prediction.ndim, 1)

    def test_model_performance(self):
        """Uncomment and run when verifying model threshold performance."""
        X_holdout = self.holdout_data.iloc[:, 0:-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        y_pred_new = self.new_model.predict(X_holdout)

        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new, zero_division=0)
        recall_new = recall_score(y_holdout, y_pred_new, zero_division=0)
        f1_new = f1_score(y_holdout, y_pred_new, zero_division=0)

        expected_threshold = 0.40

        self.assertGreaterEqual(
            accuracy_new, expected_threshold, f"Accuracy should be >= {expected_threshold}"
        )
        self.assertGreaterEqual(
            precision_new, expected_threshold, f"Precision should be >= {expected_threshold}"
        )
        self.assertGreaterEqual(
            recall_new, expected_threshold, f"Recall should be >= {expected_threshold}"
        )
        self.assertGreaterEqual(
            f1_new, expected_threshold, f"F1 score should be >= {expected_threshold}"
        )


if __name__ == "__main__":
    unittest.main()