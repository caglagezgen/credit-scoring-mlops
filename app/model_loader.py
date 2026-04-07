"""
Model Loader Module

Implements the single most critical pattern in production ML APIs:
load the model once at startup, reuse it forever.
"""

import joblib
import os

_model = None
_feature_columns = None


def load_model():
    """Load model and feature list ONCE at startup."""
    global _model, _feature_columns
    
    model_path = os.environ.get("MODEL_PATH", "model/credit_model.pkl")
    features_path = os.environ.get("FEATURES_PATH", "model/feature_columns.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features not found at {features_path}")
    
    _model = joblib.load(model_path)
    _feature_columns = joblib.load(features_path)
    print(f"✓ Model loaded from {model_path}")
    print(f"✓ Features: {len(_feature_columns)} columns")
    return _model


def get_model():
    """Get the loaded model."""
    if _model is None:
        raise RuntimeError("Model not loaded! Call load_model() at startup.")
    return _model


def get_feature_columns():
    """Get feature names in training order."""
    if _feature_columns is None:
        raise RuntimeError("Features not loaded!")
    return _feature_columns