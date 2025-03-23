import os
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("❌ DAGSHUB_PAT environment variable is missing!")

os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/InsightfulSantosh/mlops-mini_project.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(repo_owner="InsightfulSantosh", repo_name="mlops-mini_project", mlflow=True)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

print(f"✅ MLflow is tracking at: {mlflow.get_tracking_uri()}")
