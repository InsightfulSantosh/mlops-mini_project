import numpy as np
import pandas as pd
import pickle
import json
import os
import logging
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from mlflow.models.signature import infer_signature

# ---------------------- SETUP DAGSHUB MLflow TRACKING ----------------------

dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("❌ DAGSHUB_PAT environment variable is missing!")
else:
    print("✅ DAGSHUB_PAT is set correctly.")


os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(repo_owner='InsightfulSantosh', repo_name='mlops-mini_project', mlflow=True)

# ---------------------- SETUP LOGGING ----------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

log_dir = "pipeline-logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "5.model_evaluation.log")

if os.path.exists(log_file_path):
    os.remove(log_file_path)

handler = logging.FileHandler(log_file_path)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

# ---------------------- UTILITY FUNCTIONS ----------------------
def load_model(file_path):
    """Loads the model from a pickle file."""
    try:
        logger.info(f'Loading model from {file_path}')
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.info(f'Model loaded successfully: {model}')
        return model
    except FileNotFoundError:
        logger.error(f"Model file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def load_data(path):
    """Loads data from a CSV file."""
    try:
        logger.info(f'Loading data from {path}')
        data = pd.read_csv(path)
        logger.info(f'Data loaded successfully, shape: {data.shape}')
        return data
    except FileNotFoundError:
        logger.error(f"Data file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def evaluate(model, x_test: np.ndarray, y_test: np.ndarray):
    """Evaluates the model and logs performance metrics."""
    try:
        logger.info("Starting model evaluation...")
        y_pred = model.predict(x_test)
        y_pred_proba = model.predict_proba(x_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        logger.info("Evaluation metrics:")
        logger.info(json.dumps(metrics, indent=4))
        
        return metrics
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logger.debug('Metrics saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

# ---------------------- MAIN EXECUTION PIPELINE ----------------------
def main():
    mlflow.set_experiment("experiment:1")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            model_path = "./models/model.pkl"
            x_path = "./data/processed/test_bow.csv"
            y_path = "./data/raw/test/test.csv"
            
            logger.info("Starting model evaluation pipeline...")
            
            model = load_model(model_path)
            x = load_data(x_path)
            y = load_data(y_path)
            
            x_test = x.iloc[:, :-1].values  # Exclude last column if it's the label
            y_test = y.iloc[:, 0].values.ravel()
            
            metrics = evaluate(model, x_test, y_test)
            save_metrics(metrics, './reports/metrics.json')

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters to MLflow
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log the model with signature and input example
            input_example = x_test[:5]  # First 5 test samples
            signature = infer_signature(x_test, model.predict(x_test))
            
            artifact_path = "model"
            mlflow.sklearn.log_model(
                model, artifact_path,
                signature=signature,
                input_example=input_example
            )

            # Save model info
            save_model_info(run.info.run_id, artifact_path, 'reports/model_info.json')

            # Log artifacts (metrics, model info, logs)
            mlflow.log_artifact('reports/metrics.json')
            mlflow.log_artifact('reports/model_info.json')
            mlflow.log_artifacts(log_dir)

            logger.info("✅ Model evaluation pipeline completed successfully.")
            logger.info(f"🔗 View MLflow Run: {mlflow.get_tracking_uri()}/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")
        except Exception as e:
            logger.error(f"❌ Pipeline execution failed: {str(e)}")

# ---------------------- EXECUTE SCRIPT ----------------------
if __name__ == '__main__':
    main()
