import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from mlflow.models.signature import infer_signature
import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub.init(repo_owner='InsightfulSantosh', repo_name='mlops-mini_project', mlflow=True)
experiment_name = "tweet_emotion_classification"
experiment = mlflow.get_experiment_by_name(experiment_name)

# Set up logging
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

# Define the log file path
os.makedirs("pipeline-logs", exist_ok=True)
log_file_path = "pipeline-logs/5.model_evaluation.log"

# Delete the log file if it exists
if os.path.exists(log_file_path):
    os.remove(log_file_path)

handler = logging.FileHandler(log_file_path)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler())

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

def main():
    mlflow.set_experiment("dvc-pipeline")
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
            
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/model_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

            # Log the model info file to MLflow
            mlflow.log_artifact('reports/model_info.json')

            # Log the logging  file to MLflow
            mlflow.log_artifacts('./pipeline-logs')
            
            logger.info("Model evaluation pipeline completed successfully.")
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")

if __name__ == '__main__':
    main()
