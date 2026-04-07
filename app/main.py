"""
FastAPI Application: Credit Scoring Model Prediction API

Serves predictions via HTTP endpoints with:
- Input validation (Pydantic)
- Structured JSON logging
- Health checks
- Interactive documentation (Swagger)
"""

import logging
import json
import time
import pandas as pd
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import LoanApplication, PredictionResponse, HealthResponse
from app.model_loader import load_model, get_model, get_feature_columns


# ===== LOGGING SETUP =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("credit_scoring_api")


# ===== APP LIFESPAN (startup/shutdown) =====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle.
    
    Code before yield: runs at startup
    Code after yield: runs at shutdown
    """
    # STARTUP
    logger.info("=" * 80)
    logger.info("STARTING UP - Loading model...")
    try:
        load_model()
        logger.info("✓ Model loaded successfully - Ready to serve predictions")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        raise
    logger.info("=" * 80)
    
    yield
    
    # SHUTDOWN
    logger.info("Shutting down...")


# ===== CREATE FASTAPI APP =====
app = FastAPI(
    title="Home Credit Scoring API",
    description="Predict loan default probability using gradient boosting on Home Credit dataset",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS (allows requests from other origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== ENDPOINTS =====

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
    description="Returns API health status. Used by load balancers and monitoring."
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
    - status: "healthy" or "unhealthy"
    - model_loaded: whether model is ready
    
    Used by:
    - Load balancers: determines if we're alive
    - Monitoring tools: uptime tracking
    - Deployment: verification after startup
    """
    try:
        model = get_model()
        return HealthResponse(status="healthy", model_loaded=True)
    except RuntimeError:
        return HealthResponse(status="unhealthy", model_loaded=False)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Predictions"],
    summary="Get loan default prediction",
    description="Accepts loan applicant data and returns default probability and risk category."
)
async def predict(application: LoanApplication) -> PredictionResponse:
    """
    Main prediction endpoint.
    
    Input: LoanApplication (18 features, all validated by Pydantic)
    Output: PredictionResponse (prediction, probability, risk category)
    
    Process:
    1. Pydantic validates input (automatic, returns 422 if invalid)
    2. Arrange features in training order
    3. Compute engineered features (ratios)
    4. Call model.predict_proba()
    5. Map probability to risk category
    6. Log everything for monitoring
    7. Return response
    
    Example request (curl):
        curl -X POST "http://localhost:8000/predict" \\
          -H "Content-Type: application/json" \\
          -d '{
            "ext_source_1": 0.5, "ext_source_2": 0.65, "ext_source_3": 0.48,
            "amt_income_total": 202500, "amt_credit": 406597, "amt_annuity": 24700,
            "amt_goods_price": 351000, "code_gender": 1, "flag_own_car": 0,
            "flag_own_realty": 1, "cnt_children": 0, "age_years": 39.9,
            "years_employed": 5.3, "years_id_publish": 8.5, "education_level": 1
          }'
    """
    
    start_time = time.time()
    
    try:
        model = get_model()
        feature_columns = get_feature_columns()
        
        # ===== PREPARE FEATURES =====
        # Create feature dict with all model features
        # Note: engineered features are computed here (API receives raw values)
        features_dict = {
            'EXT_SOURCE_1': application.ext_source_1,
            'EXT_SOURCE_2': application.ext_source_2,
            'EXT_SOURCE_3': application.ext_source_3,
            'AMT_INCOME_TOTAL': application.amt_income_total,
            'AMT_CREDIT': application.amt_credit,
            'AMT_ANNUITY': application.amt_annuity,
            'AMT_GOODS_PRICE': application.amt_goods_price,
            'CODE_GENDER': application.code_gender,
            'FLAG_OWN_CAR': application.flag_own_car,
            'FLAG_OWN_REALTY': application.flag_own_realty,
            'CNT_CHILDREN': application.cnt_children,
            'AGE_YEARS': application.age_years,
            'YEARS_EMPLOYED': application.years_employed,
            'YEARS_ID_PUBLISH': application.years_id_publish,
            'EDUCATION_LEVEL': application.education_level,
            # Engineered features (must match training data)
            'CREDIT_INCOME_RATIO': application.amt_credit / (application.amt_income_total + 1),
            'ANNUITY_INCOME_RATIO': application.amt_annuity / (application.amt_income_total + 1),
            'CREDIT_GOODS_RATIO': application.amt_credit / (application.amt_goods_price + 1),
        }
        
        # Create DataFrame with features in exact training order
        input_df = pd.DataFrame([features_dict])[feature_columns]
        
        # ===== GET PREDICTIONS =====
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        # ===== MAP TO RISK CATEGORY =====
        # Business rule: map probability to human-readable risk level
        if probability < 0.3:
            risk_category = "Low"
        elif probability < 0.6:
            risk_category = "Medium"
        else:
            risk_category = "High"
        
        # ===== LOG PREDICTION =====
        # Every prediction logged as JSON for monitoring and audit
        inference_time_ms = (time.time() - start_time) * 1000
        
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "prediction",
            "inputs": application.model_dump(),
            "outputs": {
                "prediction": prediction,
                "probability_of_default": round(probability, 4),
                "risk_category": risk_category,
            },
            "inference_time_ms": round(inference_time_ms, 2),
        }
        logger.info(json.dumps(log_entry))
        
        # ===== RETURN RESPONSE =====
        return PredictionResponse(
            prediction=prediction,
            probability_of_default=round(probability, 4),
            risk_category=risk_category,
        )
        
    except Exception as e:
        # ===== ERROR HANDLING =====
        inference_time_ms = (time.time() - start_time) * 1000
        
        error_log = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "prediction_error",
            "error": str(e),
            "error_type": type(e).__name__,
            "inputs": application.model_dump() if application else None,
            "inference_time_ms": round(inference_time_ms, 2),
        }
        logger.error(json.dumps(error_log))
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ===== ROOT ENDPOINT =====
@app.get(
    "/",
    tags=["System"],
    summary="API Information",
    description="Returns basic API information."
)
async def root():
    """Root endpoint - returns API information."""
    return {
        "name": "Home Credit Scoring API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)