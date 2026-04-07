"""
Pydantic Schemas for FastAPI Validation

Defines strict input/output validation rules.
These schemas serve dual purposes:
1. Automatic input validation (rejects invalid data with 422 error)
2. Interactive API documentation via Swagger
"""

from pydantic import BaseModel, Field, field_validator


class LoanApplication(BaseModel):
    """Input schema: loan applicant data from API client."""
    
    ext_source_1: float = Field(..., ge=0.0, le=1.0, description="External source score 1")
    ext_source_2: float = Field(..., ge=0.0, le=1.0, description="External source score 2")
    ext_source_3: float = Field(..., ge=0.0, le=1.0, description="External source score 3")
    amt_income_total: float = Field(..., gt=0, description="Total annual income")
    amt_credit: float = Field(..., gt=0, description="Credit amount")
    amt_annuity: float = Field(..., gt=0, description="Loan annuity (monthly payment)")
    amt_goods_price: float = Field(..., gt=0, description="Price of goods")
    code_gender: int = Field(..., ge=0, le=1, description="Gender (0=M, 1=F)")
    flag_own_car: int = Field(..., ge=0, le=1, description="Owns car (0/1)")
    flag_own_realty: int = Field(..., ge=0, le=1, description="Owns property (0/1)")
    cnt_children: int = Field(..., ge=0, le=20, description="Number of children")
    age_years: float = Field(..., ge=18, le=80, description="Age in years")
    years_employed: float = Field(..., ge=0, le=50, description="Years employed")
    years_id_publish: float = Field(..., ge=0, le=60, description="Years since ID published")
    education_level: int = Field(..., ge=0, le=4, description="Education (0-4)")
    
    @field_validator('amt_credit')
    @classmethod
    def credit_reasonable(cls, v, info):
        """Validate credit-to-income ratio is realistic."""
        if 'amt_income_total' in info.data:
            ratio = v / (info.data['amt_income_total'] + 1)
            if ratio > 100:
                raise ValueError(f"Credit-to-income ratio ({ratio:.0f}x) unrealistic")
        return v


class PredictionResponse(BaseModel):
    """Output schema: prediction response."""
    
    prediction: int = Field(..., description="0=No Default, 1=Default")
    probability_of_default: float = Field(..., ge=0.0, le=1.0, description="Probability 0.0-1.0")
    risk_category: str = Field(..., description="Low/Medium/High")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="healthy/unhealthy")
    model_loaded: bool = Field(..., description="Model ready?")