"""
Feature Definitions and Catalog
Centralizes metadata about all features used in the model
Enables feature versioning, discovery, and documentation
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class FeatureType(Enum):
    """Enumeration of feature types"""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    BINARY = "binary"
    ORDINAL = "ordinal"
    ENGINEERED = "engineered"


@dataclass
class Feature:
    """Feature metadata"""
    name: str
    description: str
    feature_type: FeatureType
    source_table: str
    data_type: str
    nullable: bool = False
    range_min: float = None
    range_max: float = None
    encoding: str = None
    unit: str = None
    importance: float = None   # Feature importance from model
    lineage: str = None


# Feature Catalog - All features used in the credit scoring model
FEATURE_CATALOG = {
    # ============ SOURCE FEATURES (from raw data) ============
    "ext_source_1": Feature(
        name="ext_source_1",
        description="External source 1 - normalized credit score from external agency",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=0.0,
        range_max=1.0,
        unit="normalized_score",
        lineage="Direct from Kaggle dataset"
    ),
    
    "ext_source_2": Feature(
        name="ext_source_2",
        description="External source 2 - normalized credit score from another external agency",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=0.0,
        range_max=1.0,
        unit="normalized_score",
        lineage="Direct from Kaggle dataset"
    ),
    
    "ext_source_3": Feature(
        name="ext_source_3",
        description="External source 3 - normalized credit score from third external agency",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=0.0,
        range_max=1.0,
        unit="normalized_score",
        lineage="Direct from Kaggle dataset"
    ),
    
    "amt_income_total": Feature(
        name="amt_income_total",
        description="Total annual household income (in currency units)",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=25000,
        range_max=500000,
        unit="currency",
        lineage="Direct from Kaggle dataset"
    ),
    
    "amt_credit": Feature(
        name="amt_credit",
        description="Credit amount requested by applicant",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=25000,
        range_max=4000000,
        unit="currency",
        lineage="Direct from Kaggle dataset"
    ),
    
    "amt_annuity": Feature(
        name="amt_annuity",
        description="Annual payment amount for the credit",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=1000,
        range_max=400000,
        unit="currency",
        lineage="Direct from Kaggle dataset"
    ),
    
    "amt_goods_price": Feature(
        name="amt_goods_price",
        description="Price of goods that the credit was used for",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=0,
        range_max=4000000,
        unit="currency",
        lineage="Direct from Kaggle dataset"
    ),
    
    "code_gender": Feature(
        name="code_gender",
        description="Applicant gender (binary encoded)",
        feature_type=FeatureType.BINARY,
        source_table="application_train",
        data_type="int",
        nullable=False,
        range_min=0,
        range_max=1,
        encoding="M→0, F→1",
        lineage="One-hot encoded from DAYS_BIRTH positive/negative"
    ),
    
    "flag_own_car": Feature(
        name="flag_own_car",
        description="Flag indicating if applicant owns a car",
        feature_type=FeatureType.BINARY,
        source_table="application_train",
        data_type="int",
        nullable=False,
        range_min=0,
        range_max=1,
        encoding="N→0, Y→1",
        lineage="Direct from Kaggle dataset, binary encoded"
    ),
    
    "flag_own_realty": Feature(
        name="flag_own_realty",
        description="Flag indicating if applicant owns real estate",
        feature_type=FeatureType.BINARY,
        source_table="application_train",
        data_type="int",
        nullable=False,
        range_min=0,
        range_max=1,
        encoding="N→0, Y→1",
        lineage="Direct from Kaggle dataset, binary encoded"
    ),
    
    "age_years": Feature(
        name="age_years",
        description="Applicant age in years (converted from DAYS_BIRTH)",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=18,
        range_max=80,
        unit="years",
        encoding="Converted from DAYS_BIRTH / 365",
        lineage="Transformation: DAYS_BIRTH (days) → years"
    ),
    
    "education_level": Feature(
        name="education_level",
        description="Applicant education level (ordinal encoded)",
        feature_type=FeatureType.ORDINAL,
        source_table="application_train",
        data_type="int",
        nullable=False,
        range_min=0,
        range_max=4,
        encoding="0=Academic degree, 1=Higher education, 2=Incomplete higher, 3=Secondary education, 4=Lower secondary",
        lineage="Ordinal encoding preserving education hierarchy"
    ),
    
    "days_employed_years": Feature(
        name="days_employed_years",
        description="Number of years the applicant has been employed (converted from DAYS_EMPLOYED)",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=-50,
        range_max=50,
        unit="years",
        lineage="Transformation: DAYS_EMPLOYED (anomaly fixed, converted to years)"
    ),
    
    "days_id_publish_years": Feature(
        name="days_id_publish_years",
        description="Years since ID was published (converted from DAYS_ID_PUBLISH)",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=0,
        range_max=25,
        unit="years",
        lineage="Transformation: DAYS_ID_PUBLISH / 365 (converted to years)"
    ),
    
    "days_birth_years": Feature(
        name="days_birth_years",
        description="Age in years based on birth date (converted from DAYS_BIRTH)",
        feature_type=FeatureType.NUMERIC,
        source_table="application_train",
        data_type="float",
        nullable=False,
        range_min=18,
        range_max=80,
        unit="years",
        lineage="Transformation: DAYS_BIRTH / 365 (converted to years)"
    ),
    
    # ============ ENGINEERED FEATURES ============
    "credit_income_ratio": Feature(
        name="credit_income_ratio",
        description="Debt-to-income ratio (credit amount / annual income)",
        feature_type=FeatureType.ENGINEERED,
        source_table="derived",
        data_type="float",
        nullable=False,
        range_min=0,
        range_max=100,
        unit="ratio",
        lineage="Engineering: amt_credit / amt_income_total",
        encoding="Ratio, higher = higher borrowing burden"
    ),
    
    "annuity_income_ratio": Feature(
        name="annuity_income_ratio",
        description="Payment-to-income ratio (annual payment / annual income)",
        feature_type=FeatureType.ENGINEERED,
        source_table="derived",
        data_type="float",
        nullable=False,
        range_min=0,
        range_max=1.0,
        unit="ratio",
        lineage="Engineering: amt_annuity / amt_income_total",
        encoding="Ratio, higher = higher payment strain"
    ),
    
    "credit_goods_ratio": Feature(
        name="credit_goods_ratio",
        description="Credit-to-goods ratio (credit amount / goods price)",
        feature_type=FeatureType.ENGINEERED,
        source_table="derived",
        data_type="float",
        nullable=False,
        range_min=0,
        range_max=3.0,
        unit="ratio",
        lineage="Engineering: amt_credit / amt_goods_price",
        encoding="Ratio, >1 = financing beyond goods price"
    ),
}


def get_feature_names() -> List[str]:
    """Get list of all feature names in catalog"""
    return list(FEATURE_CATALOG.keys())


def get_feature_description(feature_name: str) -> str:
    """Get detailed description for a feature"""
    if feature_name in FEATURE_CATALOG:
        return FEATURE_CATALOG[feature_name].description
    return f"Feature '{feature_name}' not found in catalog"


def get_feature_ranges() -> Dict[str, tuple]:
    """Get min/max ranges for all numeric features"""
    ranges = {}
    for name, feature in FEATURE_CATALOG.items():
        if feature.range_min is not None and feature.range_max is not None:
            ranges[name] = (feature.range_min, feature.range_max)
    return ranges


def get_engineered_features() -> List[str]:
    """Get list of engineered features only"""
    return [name for name, f in FEATURE_CATALOG.items() 
            if f.feature_type == FeatureType.ENGINEERED]


def get_source_features() -> List[str]:
    """Get list of source (non-engineered) features"""
    return [name for name, f in FEATURE_CATALOG.items() 
            if f.feature_type != FeatureType.ENGINEERED]


def validate_feature_exists(feature_name: str) -> bool:
    """Check if feature exists in catalog"""
    return feature_name in FEATURE_CATALOG


def print_feature_catalog():
    """Print formatted feature catalog"""
    print("\n" + "="*80)
    print("CREDIT SCORING MODEL - FEATURE CATALOG")
    print("="*80 + "\n")
    
    print(f"Total Features: {len(FEATURE_CATALOG)}")
    print(f"Source Features: {len(get_source_features())}")
    print(f"Engineered Features: {len(get_engineered_features())}\n")
    
    for name, feature in FEATURE_CATALOG.items():
        print(f"📊 {name}")
        print(f"   Type: {feature.feature_type.value}")
        print(f"   Description: {feature.description}")
        if feature.range_min is not None:
            print(f"   Range: [{feature.range_min}, {feature.range_max}]")
        if feature.unit:
            print(f"   Unit: {feature.unit}")
        print(f"   Lineage: {feature.lineage}\n")


if __name__ == "__main__":
    print_feature_catalog()
