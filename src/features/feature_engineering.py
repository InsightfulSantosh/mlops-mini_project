import pickle
import pandas as pd
import os
import logging
import yaml
from sklearn.feature_extraction.text import CountVectorizer

# Set up logger
logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

# Remove existing handlers to avoid duplicates
if logger.hasHandlers():
    logger.handlers.clear()

# Define log file path
os.makedirs("pipeline-logs", exist_ok=True)
log_file_path = "pipeline-logs/3.feature_engineering.log"

# Delete the log file if it exists
if os.path.exists(log_file_path):
    os.remove(log_file_path)
    
# FileHandler to log messages to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# ConsoleHandler to log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and attach it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
            if params is None:
                raise ValueError(f"YAML file {params_path} is empty or invalid.")
        logger.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logger.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logger.error("YAML error: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error: %s", e)
        raise

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(path)
        df.fillna("", inplace=True)
        logger.debug("Data loaded successfully from %s", path)
        return df
    except FileNotFoundError:
        logger.error("File not found: %s", path)
        raise
    except Exception as e:
        logger.error("Error loading file %s: %s", path, e)
        raise

def bow(train: pd.DataFrame, test: pd.DataFrame, max_feature: int):
    """Perform Bag-of-Words transformation."""
    try:
        x_train = train["content"].values
        y_train = train["sentiment"].values
        x_test = test["content"].values
        y_test = test["sentiment"].values

        vectorizer = CountVectorizer(max_features=max_feature)
        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df["label"] = y_train
        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df["label"] = y_test

        # Ensure the models directory exists
        os.makedirs("models", exist_ok=True)
        with open("models/vectorizer.pkl", 'wb') as file:
            pickle.dump(vectorizer, file)

        logger.debug(f"Vectorizer with {max_feature} features saved successfully to models/vectorizer.pkl")
        
        return train_df, test_df
    except Exception as e:
        logger.error("Error in Bag-of-Words transformation: %s", e)
        raise

def save_data(df: pd.DataFrame, path: str):
    """Save data to a CSV file."""
    try:
        df.to_csv(path, index=False)
        logger.debug("Data saved successfully to %s", path)
    except Exception as e:
        logger.error("Error saving data to %s: %s", path, e)
        raise

def main():
    try:
        logger.info("Starting feature engineering process.")
        params = load_params("params.yaml")
        max_features = params["feature_engineering"]["max_features"]
    except Exception as e:
        logger.error("Error loading parameters: %s", e)
        return

    try:
        train = load_data(os.path.join("data", "interim", "train_processed.csv"))
        test = load_data(os.path.join("data", "interim", "test_processed.csv"))
    except Exception as e:
        logger.error("Error loading data: %s", e)
        return

    try:
        train_df, test_df = bow(train, test, max_features)
        logger.info("Bag-of-Words transformation completed successfully.")
    except Exception as e:
        logger.error("Error in Bag-of-Words transformation: %s", e)
        return
    
    try:
        os.makedirs("data/processed", exist_ok=True)
        save_data(train_df, os.path.join("data", "processed", "train_bow.csv"))
        save_data(test_df, os.path.join("data", "processed", "test_bow.csv"))
    except Exception as e:
        logger.error("Error saving data: %s", e)
        return

if __name__ == "__main__":
    main()