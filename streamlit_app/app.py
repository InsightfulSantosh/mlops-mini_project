import os

import dagshub
import mlflow
import streamlit as st
from model_utils import (get_latest_model_version, load_model, load_vectorizer,
                         predict_sentiment)
from preprocessing import normalize_text

# Retrieve DagsHub credentials from environment variables
dagshub_token = os.getenv("DAGSHUB_PAT")

if not dagshub_token:
    st.error("DAGSHUB_PAT environment variable is not set")
    st.stop()

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Initialize DagsHub MLflow tracking
dagshub_url = "https://dagshub.com"
repo_owner = "InsightfulSantosh"
repo_name = "mlops-mini_project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f"{dagshub_url}/{repo_owner}/{repo_name}.mlflow")

# ✅ Get Model Name from Environment Variable
MODEL_NAME = os.getenv("MODEL_NAME", "uma")

# ✅ Fetch the latest production model using alias
MODEL_VERSION = get_latest_model_version(MODEL_NAME)

VECTOR_PATH = "models/vectorizer.pkl"


@st.cache_resource
def get_model():
    return load_model(MODEL_NAME, MODEL_VERSION)


@st.cache_resource
def get_vectorizer():
    return load_vectorizer(VECTOR_PATH)


try:
    model = get_model()
    vectorizer = get_vectorizer()
except RuntimeError as e:
    st.error(f"🚨 {e}")
    st.stop()

# ✅ Streamlit UI
st.title("📊 Tweet Sentiment Analysis")
st.markdown(f"**Using model:** `{MODEL_NAME}` (Version: `{MODEL_VERSION}`) ✅")

# ✅ User Input
text_input = st.text_area(
    "✍️ Enter a tweet:", placeholder="E.g., I love this new movie!"
)

if st.button("🔍 Analyze Sentiment"):
    if text_input.strip():
        cleaned_text = normalize_text(text_input.strip())

        try:
            sentiment = predict_sentiment(model, vectorizer, [cleaned_text])
            sentiment_label = "😊 Happy" if sentiment == "Happy" else "😔 Sad"
            st.success(f"**Sentiment:** {sentiment_label}")
        except RuntimeError as e:
            st.error(f"🚨 Prediction failed: {e}")
    else:
        st.warning("⚠️ Please enter a tweet before analyzing.")
