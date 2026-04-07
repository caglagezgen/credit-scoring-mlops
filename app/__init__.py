"""
Credit Scoring API Package

Production-ready FastAPI application for loan default probability prediction.
Provides REST endpoints for model serving with monitoring and validation.
"""

from app.main import app
from app.model_loader import load_model, get_model, get_feature_columns
from app.schemas import LoanApplication, PredictionResponse, HealthResponse

__version__ = "1.0.0"
__author__ = "ML Ops Team"

__all__ = [
    "app",
    "load_model",
    "get_model",
    "get_feature_columns",
    "LoanApplication",
    "PredictionResponse",
    "HealthResponse",
]