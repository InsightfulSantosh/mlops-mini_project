import os
import json
import logging
import mlflow
from mlflow.tracking import MlflowClient
import dagshub

# ---------------------- SETUP DAGSHUB MLflow TRACKING ----------------------
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("❌ DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(repo_owner="InsightfulSantosh", repo_name="mlops-mini_project", mlflow=True)

# ---------------------- SETUP LOGGING ----------------------
logger = logging.getLogger("model_registry")
logger.setLevel(logging.DEBUG)

# Clear existing handlers to prevent duplicate logs
if logger.hasHandlers():
    logger.handlers.clear()

log_dir = "pipeline-logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "6.model_registry.log")

# Remove previous log file if exists
if os.path.exists(log_file_path):
    os.remove(log_file_path)

# Setup logging handlers
handler = logging.FileHandler(log_file_path)
handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())  # Print logs to console

logger.info("✅ Logging setup completed.")


def load_model_info(path):
    """Load model info from a JSON file."""
    try:
        logger.info(f"🔍 Loading model info from {path}")
        with open(path, "r") as file:
            model_info = json.load(file)
        logger.info(f"✅ Model info loaded successfully: {model_info}")
        return model_info
    except FileNotFoundError:
        logger.error(f"❌ Model info file not found: {path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"❌ Error decoding JSON from file: {path}")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error while loading model info: {str(e)}")
        raise


def model_registry(model_name, model_info):
    """Register the model in MLflow using aliases."""
    try:
        logger.info("🚀 Starting model registration...")

        model_uri = f"runs:/{model_info['run_id']}/model"
        logger.info(f"🔗 Model URI: {model_uri}")

        # Register the model
        client = MlflowClient()
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"✅ Model registered successfully: {model_name} (Version {result.version})")

        # Wait for model registration to complete
        import time
        time.sleep(5)

        # Log additional model metadata
        client.update_model_version(
            name=model_name,
            version=result.version,
            description="Tweet Emotion Classification Model"
        )
        client.set_model_version_tag(model_name, result.version, "author", "Santosh")
        client.set_model_version_tag(model_name, result.version, "use_case", "NLP")
        client.set_model_version_tag(model_name, result.version, "version_notes", "Initial model deployment")


        # Assign alias to model version
        alias = "staging"  # Change to "staging", "testing", etc., as needed
        client.set_registered_model_alias(model_name, alias, result.version)
        logger.info(f"✅ Model '{model_name}' assigned alias '{alias}' (Version {result.version})")

    except Exception as e:
        logger.error(f"❌ Error during model registration: {str(e)}")
        raise


def main():
    """Main function to execute model registration pipeline."""
    try:
        model_info_path = "reports/model_info.json"
        logger.info("🚀 Starting model registry pipeline...")

        model_info = load_model_info(model_info_path)
        model_name = "tweet_emotion_classifier"

        model_registry(model_name, model_info)
        logger.info("✅ Model registry pipeline completed successfully.")

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {str(e)}")


if __name__ == '__main__':
    main()
