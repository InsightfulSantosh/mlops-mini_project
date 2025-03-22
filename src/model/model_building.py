import pandas as pd
import pickle
import os
import logging
from sklearn.tree import DecisionTreeClassifier

# Configure logger
logger = logging.getLogger("Model building")
logger.setLevel(logging.DEBUG)

if logger.hasHandlers():
    logger.handlers.clear()

# Define the log file path
os.makedirs("pipeline-logs", exist_ok=True)

log_file_path = "pipeline-logs/4.model_buildeing.log"

# Delete the log file if it exists
if os.path.exists(log_file_path):
    os.remove(log_file_path)
    logger.debug(f"Previous log file '{log_file_path}' deleted.")

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


def load_data(file_path):
    try:
        logger.info('Process started to load the data')
        df = pd.read_csv(file_path)
        logger.debug('Train_df loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise
    

def train_model(X_train, y_train):
    try:
        logger.info("Training model with Decision Tree Classifier")
        params = {'max_depth': 30, 'random_state': 42, 'criterion': 'entropy'}

        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)
        
        logger.debug("Model training completed successfully")
        return model
    except Exception as e:
        logger.error('Error during model training: %s', e)
        raise

def save_model(model, model_dir):
    try:
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, 'model.pkl')
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        logger.info(f'Model saved at {model_path}')
    except Exception as e:
        logger.error('Error during saving the model: %s', e)
        raise


def main():
    try:
        # Load data
        train_path = "./data/processed/train_bow.csv"
        train_df = load_data(train_path)
        
        # Split data    
        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values

        logger.info('Data loaded successfully')
        logger.debug("X_train shape: %s", X_train.shape)
    except Exception as e:
        logger.error('Error during data loading: %s', e)
        return  # Stop execution if data loading fails
    
    try:
        # Train model
        model = train_model(X_train, y_train)
        # Save model
        save_model(model, 'models')
    except Exception as e:
        logger.error('Error during model training or saving: %s', e)


if __name__ == '__main__':
    main()