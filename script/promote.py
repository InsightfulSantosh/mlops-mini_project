import os

import dagshub
import mlflow
from mlflow.tracking import MlflowClient


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


def get_latest_model_version(model_name):
    """Fetch the latest registered model version."""
    client = setup_mlflow()

    try:
        latest_versions = client.get_latest_versions(model_name)

        if latest_versions:
            latest_version = latest_versions[0].version
            print(f" Latest version of '{model_name}': {latest_version}")
            return latest_version
        else:
            print(f" No versions found for model '{model_name}'.")
            return None

    except Exception as e:
        print(f" Error fetching model: {e}")
        return None


def promote_model_to_production(model_name):
    """Promote the latest version of a model to 'production' alias."""
    client = setup_mlflow()

    latest_version = get_latest_model_version(model_name)
    if latest_version is None:
        print(f" No version available to promote for model '{model_name}'")
        return

    try:
        client.set_registered_model_alias(
            name=model_name, alias="production", version=latest_version
        )
        print(
            f" Model '{model_name}' version '{latest_version}' is now in 'production'."
        )
    except Exception as e:
        print(f"Error promoting model: {e}")


if __name__ == "__main__":
    model_name = "uma"
    promote_model_to_production(model_name)
