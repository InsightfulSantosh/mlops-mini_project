import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up MLflow tracking and load the latest model."""

        # Load Dagshub credentials from environment variables
        dagshub_token = os.getenv("DAGSHUB_PAT")
        if not dagshub_token:
            raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        # Set MLflow Tracking URI
        dagshub_url = "https://dagshub.com"
        repo_owner = "InsightfulSantosh"
        repo_name = "mlops-mini_project"
        mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

        # Fetch the latest model version efficiently
        cls.new_model_name = "uma"
        client = mlflow.MlflowClient()
        latest_versions = client.search_model_versions(f"name='{cls.new_model_name}'")
        
        if not latest_versions:
            raise ValueError(f"No valid model version found for {cls.new_model_name}")
        
        cls.new_model_version = max(latest_versions, key=lambda v: int(v.version)).version
        cls.new_model_uri = f"models:/{cls.new_model_name}/{cls.new_model_version}"
        logging.info(f"Loading model from URI: {cls.new_model_uri}")

        # Load model
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load vectorizer
        try:
            vectorizer_path = "models/vectorizer.pkl"
            with open(vectorizer_path, "rb") as f:
                cls.vectorizer = pickle.load(f)
            logging.info(f"Successfully loaded vectorizer from {vectorizer_path}")
        except Exception as e:
            raise FileNotFoundError(f"Error loading vectorizer: {e}")

        # Load test data
        try:
            test_data_path = "./data/processed/test_bow.csv"
            cls.holdout_data = pd.read_csv(test_data_path)
            logging.info(f"Successfully loaded test data from {test_data_path}")
        except Exception as e:
            raise FileNotFoundError(f"Error loading test data: {e}")

    def test_model_loaded_properly(self):
        """Check if the model loaded correctly."""
        self.assertIsNotNone(self.new_model, "Model failed to load")

    def test_model_signature(self):
        """Test if the model signature matches expected input and output shapes."""
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the model
        prediction = self.new_model.predict(input_df)

        # Verify input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()), "Input shape mismatch")

        # Verify output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0], "Output length mismatch")
        self.assertEqual(len(prediction.shape), 1, "Output should be 1D")

    def test_model_performance(self):
        """Evaluate model performance on holdout test data."""
        X_holdout = self.holdout_data.iloc[:, :-1]
        y_holdout = self.holdout_data.iloc[:, -1]

        # Predict using the model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Expected thresholds
        expected_thresholds = {
            "accuracy": 0.90,
            "precision": 0.40,
            "recall": 0.40,
            "f1_score": 0.40,
        }

        # Check if performance meets expectations
        self.assertGreaterEqual(accuracy_new, expected_thresholds["accuracy"], "Accuracy below threshold")
        self.assertGreaterEqual(precision_new, expected_thresholds["precision"], "Precision below threshold")
        self.assertGreaterEqual(recall_new, expected_thresholds["recall"], "Recall below threshold")
        self.assertGreaterEqual(f1_new, expected_thresholds["f1_score"], "F1 score below threshold")

if __name__ == "__main__":
    unittest.main()