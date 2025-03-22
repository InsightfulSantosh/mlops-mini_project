import os
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# Retrieve DagsHub credentials from environment variables
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Initialize DagsHub MLflow tracking
dagshub.init(repo_owner="InsightfulSantosh", repo_name="mlops-mini_project", mlflow=True)

from mlflow.tracking import MlflowClient

# Initialize client
client = MlflowClient()

client.delete_experiment(experiment_id=3)