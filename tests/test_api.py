"""
API Tests: Comprehensive test suite for FastAPI endpoints

Tests cover:
- Valid requests (happy path)
- Invalid requests (error handling)
- Data validation
- Response schema validation
- Input constraints
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


# ===== TEST DATA =====
VALID_APPLICANT = {
    "ext_source_1": 0.5,
    "ext_source_2": 0.65,
    "ext_source_3": 0.48,
    "amt_income_total": 202500,
    "amt_credit": 406597,
    "amt_annuity": 24700,
    "amt_goods_price": 351000,
    "code_gender": 1,
    "flag_own_car": 0,
    "flag_own_realty": 1,
    "cnt_children": 0,
    "age_years": 39.9,
    "years_employed": 5.3,
    "years_id_publish": 8.5,
    "education_level": 1,
}


# ===== HEALTH CHECK TESTS =====
class TestHealthCheck:
    """Test the health check endpoint."""
    
    def test_health_returns_200(self):
        """Health endpoint should always return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200
    
    def test_health_reports_model_loaded(self):
        """After startup, model should be loaded."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


# ===== VALID PREDICTION TESTS =====
class TestValidPredictions:
    """Test successful predictions with valid data."""
    
    def test_valid_prediction_returns_200(self):
        """Valid request should return HTTP 200."""
        response = client.post("/predict", json=VALID_APPLICANT)
        assert response.status_code == 200
    
    def test_response_has_all_required_fields(self):
        """Response must include prediction, probability, and risk category."""
        response = client.post("/predict", json=VALID_APPLICANT)
        data = response.json()
        assert "prediction" in data
        assert "probability_of_default" in data
        assert "risk_category" in data
    
    def test_prediction_is_binary(self):
        """Prediction must be exactly 0 or 1."""
        response = client.post("/predict", json=VALID_APPLICANT)
        prediction = response.json()["prediction"]
        assert prediction in [0, 1], f"Prediction {prediction} is not binary"
    
    def test_probability_in_valid_range(self):
        """Probability must be between 0.0 and 1.0."""
        response = client.post("/predict", json=VALID_APPLICANT)
        prob = response.json()["probability_of_default"]
        assert 0.0 <= prob <= 1.0, f"Probability {prob} out of range [0, 1]"
    
    def test_risk_category_is_valid(self):
        """Risk category must be Low, Medium, or High."""
        response = client.post("/predict", json=VALID_APPLICANT)
        risk = response.json()["risk_category"]
        assert risk in ["Low", "Medium", "High"], f"Invalid risk category: {risk}"
    
    def test_probability_to_risk_mapping(self):
        """Verify risk category matches probability correctly."""
        # Low risk: probability < 0.3
        low_risk_app = {**VALID_APPLICANT, "ext_source_1": 0.95}  # High score → low risk
        response = client.post("/predict", json=low_risk_app)
        prob = response.json()["probability_of_default"]
        risk = response.json()["risk_category"]
        if prob < 0.3:
            assert risk == "Low"
        elif prob < 0.6:
            assert risk == "Medium"
        else:
            assert risk == "High"


# ===== INVALID INPUT TESTS =====
class TestInvalidInputs:
    """Test that invalid data is properly rejected."""
    
    def test_ext_source_above_1_rejected(self):
        """External scores must be 0-1, above 1 should be rejected."""
        bad = {**VALID_APPLICANT, "ext_source_1": 5.0}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422, "Should reject ext_source > 1"
    
    def test_ext_source_below_0_rejected(self):
        """External scores must be 0-1, below 0 should be rejected."""
        bad = {**VALID_APPLICANT, "ext_source_2": -0.1}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_negative_income_rejected(self):
        """Income must be positive."""
        bad = {**VALID_APPLICANT, "amt_income_total": -1000}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_zero_income_rejected(self):
        """Income must be greater than zero."""
        bad = {**VALID_APPLICANT, "amt_income_total": 0}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_zero_credit_rejected(self):
        """Credit amount must be greater than zero."""
        bad = {**VALID_APPLICANT, "amt_credit": 0}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_negative_credit_rejected(self):
        """Credit amount must be positive."""
        bad = {**VALID_APPLICANT, "amt_credit": -100}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_underage_applicant_rejected(self):
        """Applicants must be at least 18."""
        bad = {**VALID_APPLICANT, "age_years": 15}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_overage_applicant_rejected(self):
        """Applicants over 80 should be rejected."""
        bad = {**VALID_APPLICANT, "age_years": 150}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_invalid_gender_rejected(self):
        """Gender must be 0 or 1."""
        bad = {**VALID_APPLICANT, "code_gender": 2}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_invalid_education_rejected(self):
        """Education level must be 0-4."""
        bad = {**VALID_APPLICANT, "education_level": 5}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_too_many_children_rejected(self):
        """Number of children must be reasonable (<= 20)."""
        bad = {**VALID_APPLICANT, "cnt_children": 100}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


# ===== MISSING FIELD TESTS =====
class TestMissingFields:
    """Test that missing required fields are rejected."""
    
    def test_missing_ext_source_1_rejected(self):
        """Missing ext_source_1 should be rejected."""
        incomplete = {k: v for k, v in VALID_APPLICANT.items() if k != "ext_source_1"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422
    
    def test_missing_income_rejected(self):
        """Missing income field should be rejected."""
        incomplete = {k: v for k, v in VALID_APPLICANT.items() if k != "amt_income_total"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422
    
    def test_missing_age_rejected(self):
        """Missing age field should be rejected."""
        incomplete = {k: v for k, v in VALID_APPLICANT.items() if k != "age_years"}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422


# ===== WRONG DATA TYPE TESTS =====
class TestWrongDataTypes:
    """Test that wrong data types are rejected."""
    
    def test_string_age_rejected(self):
        """Age must be numeric, not string."""
        bad = {**VALID_APPLICANT, "age_years": "forty"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_string_income_rejected(self):
        """Income must be numeric, not string."""
        bad = {**VALID_APPLICANT, "amt_income_total": "two hundred"}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
    
    def test_float_gender_rejected(self):
        """Gender must be int, not float."""
        bad = {**VALID_APPLICANT, "code_gender": 1.5}
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


# ===== BUSINESS LOGIC TESTS =====
class TestBusinessLogicValidation:
    """Test cross-field business logic validation."""
    
    def test_unrealistic_credit_to_income_ratio_rejected(self):
        """Credit-to-income ratio > 100x should be rejected."""
        bad = {
            **VALID_APPLICANT,
            "amt_income_total": 1000,  # Very low income
            "amt_credit": 200000,       # Very high credit → 200x ratio
        }
        response = client.post("/predict", json=bad)
        assert response.status_code == 422


# ===== INTEGRATION TESTS =====
class TestIntegration:
    """End-to-end integration tests."""
    
    def test_multiple_predictions_consistent(self):
        """Multiple predictions on same data should be identical."""
        response1 = client.post("/predict", json=VALID_APPLICANT)
        response2 = client.post("/predict", json=VALID_APPLICANT)
        
        data1 = response1.json()
        data2 = response2.json()
        
        assert data1["prediction"] == data2["prediction"]
        assert data1["probability_of_default"] == data2["probability_of_default"]
    
    def test_api_interactive_docs_available(self):
        """Swagger UI should be available at /docs."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_openapi_schema_available(self):
        """OpenAPI schema should be available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/predict" in data["paths"]
