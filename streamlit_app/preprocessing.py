import re
import string

import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK resources are downloaded before use
try:
    nltk.data.find("corpora/wordnet.zip")
    nltk.data.find("corpora/stopwords.zip")
except LookupError:
    nltk.download("wordnet")
    nltk.download("stopwords")
    nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def lemmatization(text):
    """Lemmatize the input text."""
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])


def remove_stop_words(text):
    """Faster stopword removal using NumPy."""
    words = text.split()
    return " ".join(np.array(words)[~np.isin(words, list(stop_words))])


def remove_numbers(text):
    """Remove numbers from text."""
    return "".join([char for char in text if not char.isdigit()])


def lower_case(text):
    """Convert text to lowercase."""
    return text.lower()


def remove_punctuation(text):
    """Remove punctuation from text."""
    return re.sub(f"[{re.escape(string.punctuation)}]", " ", text)


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def normalize_text(text):
    """Apply full text preprocessing pipeline."""
    text = lower_case(text)
    text = remove_stop_words(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_urls(text)
    text = lemmatization(text)
    return text
