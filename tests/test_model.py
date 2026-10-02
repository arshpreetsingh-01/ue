import os
import pickle
import unittest
import dagshub
import mlflow
import pandas as pd

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        repo_owner = os.getenv("DAGSHUB_USER")
        repo_name = "ue"
        dagshub_token = os.getenv("DAGSHUB_PAT")

        if not repo_owner:
            raise RuntimeError("DAGSHUB_USER is not set")

        if not dagshub_token:
            raise RuntimeError("DAGSHUB_PAT is not set")

        dagshub.auth.add_app_token(dagshub_token)

        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True
        )

        cls.new_model_name = "my_model"
        
        # 2. Get latest model version via Alias (MLflow 3.x feature)
        client = mlflow.MlflowClient()
        try:
            mv = client.get_model_version_by_alias(cls.new_model_name, "champion")
            cls.new_model_version = mv.version
        except Exception:
            versions = client.search_model_versions(f"name='{cls.new_model_name}'")
            cls.new_model_version = max(versions, key=lambda v: int(v.version)).version if versions else None

        # 3. Load model using PyFunc or fallback local artifact download
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        
        try:
            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
        except Exception as e:
            # Fallback for DagsHub MLflow 3.x proxy compatibility:
            # Download artifacts explicitly using MLflow 3.x artifact module
            local_dir = mlflow.artifacts.download_artifacts(artifact_uri=cls.new_model_uri)
            cls.new_model = mlflow.pyfunc.load_model(local_dir)

        # 4. Load vectorizer & holdout data
        with open("models/vectorizer.pkl", "rb") as f:
            cls.vectorizer = pickle.load(f)

        cls.holdout_data = pd.read_csv("data/processed/test_bow.csv")

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(
            input_data.toarray(),
            columns=[str(i) for i in range(input_data.shape[1])],
        )

        prediction = self.new_model.predict(input_df)

        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(prediction.ndim, 1)

if __name__ == "__main__":
    unittest.main()