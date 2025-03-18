import pandas as pd
import numpy as np
import os
import logging
import yaml  # Added for loading YAML files
from sklearn.model_selection import train_test_split
from requests.exceptions import RequestException

# Set up logger
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Define the log file path
os.makedirs("pipeline_logs", exist_ok=True)

log_file_path = "pipeline_logs/1.data_ingestion.log"

# Delete the log file if it exists
if os.path.exists(log_file_path):
    os.remove(log_file_path)
    logger.debug(f"Previous log file '{log_file_path}' deleted.")

# FileHandler to log messages to a file
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

# ConsoleHandler to log messages to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a formatter and attach it to the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Define BASE_FOLDER globally
BASE_FOLDER = "data"

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        import yaml  # Ensure YAML is available

        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

def make_dir():
    """Creates required directories if they don't exist."""
    try:
        sub_dirs = ["raw", "raw/train", "raw/test"]
        for sub in sub_dirs:
            dir_path = os.path.join(BASE_FOLDER, sub)
            os.makedirs(dir_path, exist_ok=True)
            logger.debug(f"Directory created or already exists: {dir_path}")
    except Exception as e:
        logger.error(f"Error in making directories: {e}")
        raise  # Re-raise exception to propagate it

def load_data(url):
    """Loads data from the provided URL."""
    try:
        logger.debug(f"Loading data from URL: {url}")
        df = pd.read_csv(url)
        logger.info(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    except RequestException as e:
        logger.error(f"Request exception occurred while loading data from URL: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def processing(df):
    """Filters, maps sentiment values, and shuffles data."""
    try:
        logger.debug("Processing the data (filtering, mapping, shuffling).")
        processed_df = (
            df[df["sentiment"].isin(["happiness", "sadness"])]
            .drop(columns=["tweet_id"], errors="ignore")
            .assign(sentiment=lambda x: x["sentiment"].map({"happiness": 1, "sadness": 0}))
            .sample(frac=1, random_state=42)  # Shuffle the data
        )
        logger.info(f"Data processed successfully. {processed_df.shape[0]} rows remaining.")
        return processed_df
    except KeyError as e:
        logger.error(f"Key error during processing: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

def save_data(final_df, test_size):
    """Splits data into train and test sets and saves them."""
    try:
        logger.debug(f"Splitting data into train and test sets with test_size={test_size}.")
        train_df, test_df = train_test_split(final_df, test_size=test_size, random_state=42)

        # Log the shapes of the train and test datasets
        logger.info(f"Train dataset shape: {train_df.shape}")
        logger.info(f"Test dataset shape: {test_df.shape}")

        train_path = os.path.join(BASE_FOLDER, "raw/train", "train.csv")
        test_path = os.path.join(BASE_FOLDER, "raw/test", "test.csv")

        logger.debug(f"Saving train data to: {train_path}")
        train_df.to_csv(train_path, index=False)
        
        logger.debug(f"Saving test data to: {test_path}")
        test_df.to_csv(test_path, index=False)

        logger.info(f"Data saved successfully: {train_path} and {test_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise
def main():
    try:
        logger.info("Starting the ETL process.")
        make_dir()
        
        # Example of how to load parameters (you'll need to provide a path)
        params_path = "params.yaml"  # Change this to the actual path to your YAML file
        params = load_params(params_path)
        
        # Debug: Log the loaded params to check the structure
        logger.debug(f"Loaded params: {params}")

        # Safe access to the test_size value
        try:
            test_size = params.get('data_ingestion', {}).get('test_size', 0.2)  # Default to 0.2 if not found
            logger.debug(f"Test size parameter: {test_size}")
        except KeyError as e:
            logger.error(f"KeyError while accessing test_size: {e}")
            raise
        
        url = "https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv"
        
        df = load_data(url)
        final_df = processing(df)
        save_data(final_df, test_size)
        logger.info("ETL process completed successfully.")
    except Exception as e:
        logger.critical(f"ETL process failed: {e}")
        

if __name__ == "__main__":
    main()