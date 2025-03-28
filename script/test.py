import os

import dagshub
import joblib
import mlflow
import pandas as pd  # Import missing pandas

# Set up MLflow tracking and load the latest model
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "InsightfulSantosh"
repo_name = "mlops-mini_project"
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")


def get_latest_model_version(model_name):
    """Fetch the latest version of a registered model."""
    client = mlflow.MlflowClient()
    latest_versions = client.search_model_versions(f"name='{model_name}'")
    return max(latest_versions, key=lambda v: int(v.version)).version


# Example usage
model_name = "uma"
latest_version = get_latest_model_version(model_name)
model_uri = f"models:/{model_name}/{latest_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Load vectorizer
with open("./models/vectorizer.pkl", "rb") as vect:
    vector = joblib.load(vect)

text = "i am sad disgusting disaster worst bad"

# Fix: `transform()` needs a list
x = vector.transform([text])
features_df = pd.DataFrame(x.toarray())

# Make prediction
print(model.predict(features_df)[0])
