import os
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

def setup_mlflow():
    """Set up MLflow tracking with DagsHub credentials."""
    dagshub_token = os.getenv("DAGSHUB_PAT")
    if not dagshub_token:
        raise EnvironmentError(" DAGSHUB_PAT environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "InsightfulSantosh"
    repo_name = "mlops-mini_project"

    mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")
    return MlflowClient()


from pprint import pprint

client = MlflowClient()
uma(client.search_registered_models())
