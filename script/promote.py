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
dagshub_url = "https://dagshub.com"
repo_owner = "InsightfulSantosh"
repo_name = "mlops-mini_project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

def promote_model_with_alias(model_name: str, model_version: str, alias: str):
    """
    Promote an MLflow registered model version by assigning it an alias.

    Parameters:
    model_name (str): Name of the registered model.
    model_version (str): Version number of the registered model.
    alias (str): Alias to assign to the model version (e.g., 'production').
    """
    client = MlflowClient()

    # Set alias for the specified model version
    client.set_registered_model_alias(
        name=model_name,
        alias=alias,
        version=model_version
    )
    print(f"Model '{model_name}' version '{model_version}' is now aliased as '{alias}'.")

# Example usage
if __name__ == "__main__":
    model_name = "randomforest"
    model_version = "1"
    alias = "production"  # Define your alias (e.g., 'staging', 'production')
    promote_model_with_alias(model_name, model_version, alias)
