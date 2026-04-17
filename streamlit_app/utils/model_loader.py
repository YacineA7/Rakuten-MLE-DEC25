from __future__ import annotations

from pathlib import Path

import streamlit as st

try:
    import joblib
except Exception:
    joblib = None

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

TEXT_MODEL_FILES = {
    "KNN": "knn_text_model.joblib",
    "Naive Bayes": "nb_text_model.joblib",
    "SVM": "svm_text_model.joblib",
    "XGBoost": "xgb_text_model.joblib",
}

IMAGE_MODEL_FILE = "ENB_model.keras"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"
LABEL_ENCODER_FILE = "label_encoder.joblib"


def get_models_dir(current_file: str) -> Path:
    return Path(current_file).resolve().parent.parent / "models"


@st.cache_resource
def load_joblib_artifact(path: Path):
    if joblib is None:
        raise ImportError("joblib n'est pas installé.")
    return joblib.load(path)


@st.cache_resource
def load_keras_artifact(path: Path):
    if load_model is None:
        raise ImportError("TensorFlow/Keras n'est pas installé.")
    return load_model(path)


def decode_prediction(pred_idx, label_encoder):
    try:
        return label_encoder.inverse_transform([pred_idx])[0]
    except Exception:
        return pred_idx
