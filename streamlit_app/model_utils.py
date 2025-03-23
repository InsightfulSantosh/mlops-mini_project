import mlflow
import mlflow.pyfunc
import joblib
import onnxruntime as ort
import pandas as pd
import os

def get_latest_model_version(model_name):
    """
    Retrieve the latest 'Production' version of a model using MLflow Aliases.
    If no alias exists, fallback to the latest available version.
    """
    client = mlflow.MlflowClient()
    try:
        # ✅ Fetch model version using alias
        alias_version = client.get_model_version_by_alias(model_name, "production")
        if alias_version:
            return alias_version.version  # Return the production version

        # ✅ Fallback: Get the latest registered version
        all_versions = client.search_model_versions(f"name='{model_name}'", order_by=["version DESC"])
        if all_versions:
            return all_versions[0].version  # Return the most recent version

        raise ValueError(f"No registered model found with name: {model_name}")

    except Exception as e:
        raise RuntimeError(f"Error retrieving model version: {e}")

def load_model(model_name: str, model_version: str):
    """Load MLflow model, with fallback to ONNX model if available."""
    try:
        model_uri = f"models:/{model_name}/{model_version}"
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        print(f"⚠️ MLflow model loading failed: {e}")
        onnx_path = f"models/{model_name}.onnx"
        if os.path.exists(onnx_path):
            print("🔄 Loading ONNX model for faster inference...")
            return ort.InferenceSession(onnx_path)
        raise RuntimeError(f"Error loading model: {e}")

def load_vectorizer(vectorizer_path: str):
    """Load vectorizer using joblib (faster than pickle)."""
    try:
        return joblib.load(vectorizer_path)
    except Exception as e:
        raise RuntimeError(f"Error loading vectorizer: {e}")

def predict_sentiment(model, vectorizer, text_list):
    """Transform text and make sentiment prediction (0 -> Sad, 1 -> Happy)."""
    try:
        features = vectorizer.transform(text_list)
        features_df = pd.DataFrame(features.toarray())

        if isinstance(model, ort.InferenceSession):  # If ONNX model
            input_name = model.get_inputs()[0].name
            result = model.run(None, {input_name: features_df.astype("float32").to_numpy()})[0]
        else:  # MLflow model
            result = model.predict(features_df)

        sentiment_labels = {0: "Sad", 1: "Happy"}
        return sentiment_labels.get(result[0], "Unknown")

    except Exception as e:
        raise RuntimeError(f"❌ Prediction failed: {e}")
