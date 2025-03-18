import re
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords    # For stopwords
from nltk.stem import WordNetLemmatizer # For stemming and lemmatization
import logging

# Downloading NLTK data
nltk.download('stopwords')   # Downloading stopwords data
nltk.download('wordnet')     # Downloading WordNet data for lemmatization

# Set up logger
logger = logging.getLogger("data_cleaning")
logger.setLevel(logging.DEBUG)

# Path to the log file
log_file_path = "pipeline_logs/2.data_cleaning.log"

# Delete the previous log file if it exists
if os.path.exists(log_file_path):
    os.remove(log_file_path)
    logger.debug(f"Deleted the previous log file: {log_file_path}")

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

def data_cleaning(text_series):
    """Cleans the text data by removing URLs, emails, numbers, and punctuation."""
    number_pattern = r"(?<=\D)\d+|\d+(?=\D)"  # Removes numbers but keeps letters
    url_pattern = r"https?://\S+|www\.\S+"
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    punctuation_pattern = r"[^\w\s]"

    try:
        cleaned_text = (
            text_series.astype(str)  # Ensure text is string
            .str.lower()
            .str.replace(url_pattern, " ", regex=True)
            .str.replace(email_pattern, " ", regex=True)
            .str.replace(number_pattern, " ", regex=True)
            .str.replace(punctuation_pattern, " ", regex=True)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)  # Normalize spaces
        )
        logger.debug("Data cleaning completed successfully.")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")
        raise

def remove_short_words(text_series, min_length=3):
    """Removes words shorter than `min_length` characters."""
    try:
        cleaned_text = text_series.apply(lambda x: " ".join([word for word in x.split() if len(word) >= min_length]))
        logger.debug(f"Removed short words (length < {min_length}) successfully.")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error removing short words: {e}")
        raise

def lemmatization(text_series):
    """Lemmatizes words using WordNetLemmatizer."""
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatized_text = text_series.apply(lambda x: " ".join([lemmatizer.lemmatize(word, pos="v") for word in x.split()]))
        logger.debug("Lemmatization completed successfully.")
        return lemmatized_text
    except Exception as e:
        logger.error(f"Error during lemmatization: {e}")
        raise

def remove_stopwords(text_series):
    """Removes stopwords from text."""
    try:
        stop_words = frozenset(stopwords.words("english"))  # Faster lookup
        cleaned_text = text_series.apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))
        logger.debug("Stopwords removed successfully.")
        return cleaned_text
    except Exception as e:
        logger.error(f"Error removing stopwords: {e}")
        raise

def normalize(df):
    """Applies text preprocessing steps."""
    try:
        logger.debug("Starting the data preprocessing pipeline.")
        df["content"] = data_cleaning(df["content"])
        df["content"] = remove_short_words(df["content"])
        df["content"] = lemmatization(df["content"])
        df["content"] = remove_stopwords(df["content"])
        logger.debug("Text preprocessing completed successfully.")
        return df
    except Exception as e:
        logger.error(f"Error during data normalization: {e}")
        raise

def make_dir():
    """Creates required directories if they don't exist."""
    try:
        base_folder = os.path.abspath("./data")
        
        # Ensure base folder exists
        if not os.path.exists(base_folder):
            os.makedirs(base_folder)
            logger.debug(f"Base directory created: {base_folder}")
        
        data_path = os.path.join(base_folder, "interim")
        
        # Create interim folder
        if not os.path.exists(data_path):
            os.makedirs(data_path)
            logger.debug(f"Created directory: {data_path}")
        else:
            logger.debug(f"Directory already exists: {data_path}")
    except Exception as e:
        logger.error(f"Error in making directories: {e}")
        raise  # Re-raise exception to propagate it

def main():
    try:
        logger.info("🚀 Starting the data cleaning process.")

        # Ensure directories exist
        logger.debug("🔍 Ensuring necessary directories exist.")
        make_dir()

        # Load training and testing data
        train_file_path = "data/raw/train/train.csv"
        test_file_path = "data/raw/test/test.csv"

        logger.info(f"📥 Loading training data from: {train_file_path}")
        logger.info(f"📥 Loading testing data from: {test_file_path}")

        if not os.path.exists(train_file_path) or not os.path.exists(test_file_path):
            logger.error("❌ One or both dataset files are missing!")
            return  # Exit early if files are not found

        train_data = pd.read_csv(train_file_path)
        test_data = pd.read_csv(test_file_path)

        logger.info("✅ Data successfully loaded.")

        # Normalize (clean) the data
        logger.debug("🧼 Starting data normalization.")
        train_processed_data = normalize(train_data)
        test_processed_data = normalize(test_data)

        logger.info("✅ Data normalization completed.")

        # Save the processed data
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_file = os.path.join(data_path, "train_processed.csv")
        test_processed_file = os.path.join(data_path, "test_processed.csv")

        logger.info(f"💾 Saving processed training data to: {train_processed_file}")
        train_processed_data.to_csv(train_processed_file, index=False)

        logger.info(f"💾 Saving processed testing data to: {test_processed_file}")
        test_processed_data.to_csv(test_processed_file, index=False)

        logger.info("🎉 Data cleaning process completed successfully!")

    except Exception as e:
        logger.critical("🔥 Data cleaning process failed!", exc_info=True)

if __name__ == "__main__":
    main()