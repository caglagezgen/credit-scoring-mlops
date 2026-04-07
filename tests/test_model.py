"""
Model Tests: Test model artifact and loading

Tests verify:
- Model file exists and loads
- Model has required methods
- Feature columns saved correctly
- Model produces valid predictions
"""

import pytest
import joblib
import os
import numpy as np
import pandas as pd

from app import load_model, get_model, get_feature_columns


class TestModelArtifacts:
    """Test model files exist and are valid."""
    
    def test_model_file_exists(self):
        """Model pickle file must exist."""
        assert os.path.exists("model/credit_model.pkl"), \
            "model/credit_model.pkl not found. Run train_model.py first."
    
    def test_feature_columns_file_exists(self):
        """Feature columns pickle file must exist."""
        assert os.path.exists("model/feature_columns.pkl"), \
            "model/feature_columns.pkl not found. Run train_model.py first."
    
    def test_reference_data_exists(self):
        """Reference data for drift detection must exist."""
        assert os.path.exists("data/reference_data.csv"), \
            "data/reference_data.csv not found. Run train_model.py first."


class TestModelLoading:
    """Test model loading functionality."""
    
    def test_model_loads_successfully(self):
        """Model should load without errors."""
        model = joblib.load("model/credit_model.pkl")
        assert model is not None
    
    def test_model_has_predict_method(self):
        """Model must have predict method."""
        model = joblib.load("model/credit_model.pkl")
        assert hasattr(model, 'predict')
    
    def test_model_has_predict_proba_method(self):
        """Model must have predict_proba method."""
        model = joblib.load("model/credit_model.pkl")
        assert hasattr(model, 'predict_proba')
    
    def test_feature_columns_load_successfully(self):
        """Feature columns should load without errors."""
        columns = joblib.load("model/feature_columns.pkl")
        assert columns is not None
        assert isinstance(columns, list)
    
    def test_feature_columns_count(self):
        """Should have exactly 18 features."""
        columns = joblib.load("model/feature_columns.pkl")
        assert len(columns) == 18, f"Expected 18 features, got {len(columns)}"
    
    def test_feature_columns_have_expected_names(self):
        """Feature columns should have predictable names."""
        columns = joblib.load("model/feature_columns.pkl")
        required_features = {
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
            'AGE_YEARS', 'YEARS_EMPLOYED', 'EDUCATION_LEVEL',
        }
        for feature in required_features:
            assert feature in columns, f"Missing expected feature: {feature}"


class TestModelPredictions:
    """Test model prediction functionality."""
    
    def test_model_produces_valid_predictions(self):
        """Model should produce predictions on valid data."""
        model = joblib.load("model/credit_model.pkl")
        columns = joblib.load("model/feature_columns.pkl")
        
        # Load reference data
        ref_data = pd.read_csv("data/reference_data.csv")
        
        # Take a sample row
        test_row = ref_data[columns].iloc[[0]]
        
        # Get prediction
        pred = model.predict(test_row)
        
        # Should be binary (0 or 1)
        assert pred[0] in [0, 1]
    
    def test_model_produces_valid_probabilities(self):
        """Model probabilities should be between 0 and 1."""
        model = joblib.load("model/credit_model.pkl")
        columns = joblib.load("model/feature_columns.pkl")
        
        ref_data = pd.read_csv("data/reference_data.csv")
        test_row = ref_data[columns].iloc[[0]]
        
        proba = model.predict_proba(test_row)
        
        # Should be 2D array (samples, classes)
        assert proba.shape[1] == 2
        
        # Probabilities should sum to 1
        assert np.isclose(proba.sum(axis=1), 1.0).all()
        
        # All values should be between 0 and 1
        assert (proba >= 0.0).all() and (proba <= 1.0).all()
    
    def test_model_batch_predictions(self):
        """Model should handle batch predictions."""
        model = joblib.load("model/credit_model.pkl")
        columns = joblib.load("model/feature_columns.pkl")
        
        ref_data = pd.read_csv("data/reference_data.csv")
        test_rows = ref_data[columns].iloc[:10]
        
        preds = model.predict(test_rows)
        
        assert len(preds) == 10
        assert all(p in [0, 1] for p in preds)


class TestReferenceData:
    """Test reference data validity."""
    
    def test_reference_data_loads(self):
        """Reference data should load."""
        ref_data = pd.read_csv("data/reference_data.csv")
        assert ref_data is not None
    
    def test_reference_data_has_correct_shape(self):
        """Reference data should have correct number of features."""
        ref_data = pd.read_csv("data/reference_data.csv")
        assert ref_data.shape[1] == 18
    
    def test_reference_data_has_no_nan(self):
        """Reference data should have no missing values."""
        ref_data = pd.read_csv("data/reference_data.csv")
        assert not ref_data.isnull().any().any()
    
    def test_reference_data_has_reasonable_size(self):
        """Reference data should be large enough for statistics."""
        ref_data = pd.read_csv("data/reference_data.csv")
        assert ref_data.shape[0] > 1000, "Need at least 1000 reference samples"


class TestModelLoaderModule:
    """Test the model_loader module functions."""
    
    def test_load_model_function(self):
        """load_model() should successfully load."""
        model = load_model()
        assert model is not None
    
    def test_get_model_after_load(self):
        """get_model() should work after load_model()."""
        load_model()
        model = get_model()
        assert model is not None
    
    def test_get_feature_columns_after_load(self):
        """get_feature_columns() should work after load_model()."""
        load_model()
        columns = get_feature_columns()
        assert len(columns) == 18
