"""
Pytest Configuration

Ensures proper setup for running tests:
- Sets working directory to project root
- Configures environment variables with absolute paths BEFORE any imports
- Ensures model can be loaded during test session
- Provides fixture to manually load model if lifespan doesn't trigger properly

This fixes the issue where TestClient's lifespan startup
couldn't find model files due to relative path resolution.
"""

import os
import sys
import pytest

# Get the project root (parent of tests directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MUST set environment variables BEFORE test_api.py is imported
# This happens during conftest.py loading, before test modules are imported
os.environ["MODEL_PATH"] = os.path.join(PROJECT_ROOT, "model", "credit_model.pkl")
os.environ["FEATURES_PATH"] = os.path.join(PROJECT_ROOT, "model", "feature_columns.pkl")

# Change to project root for all tests
os.chdir(PROJECT_ROOT)

# Ensure project root is in Python path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print(f"\n{'='*80}")
print(f"Pytest Configuration (conftest.py module level):")
print(f"  Project Root: {PROJECT_ROOT}")
print(f"  MODEL_PATH: {os.environ['MODEL_PATH']}")
print(f"  FEATURES_PATH: {os.environ['FEATURES_PATH']}")
print(f"{'='*80}\n")


@pytest.fixture(scope="session", autouse=True)
def ensure_model_loaded():
    """
    Autouse session fixture that ensures model is loaded before any tests run.
    
    This is a safety net in case the lifespan startup doesn't trigger properly
    during TestClient creation. Manually loads the model if needed.
    """
    from app.model_loader import load_model, get_model
    
    try:
        # Try to get the model - if it fails, we need to load it
        get_model()
        print("✓ Model already loaded from lifespan")
    except RuntimeError:
        # Model wasn't loaded by lifespan, load it manually
        print("⚠ Model not loaded by lifespan, loading manually...")
        load_model()
        print("✓ Model loaded successfully")
    
    yield  # Tests run here
    
    # Cleanup (if needed)
