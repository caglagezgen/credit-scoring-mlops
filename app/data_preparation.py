"""
Data Preparation Module for Credit Scoring Model

Handles the transformation of raw Home Credit dataset into clean, 
feature-engineered data ready for model training.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_and_prepare_data(filepath: str = 'data/application_train.csv') -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load raw data and apply comprehensive data preparation.
    
    Steps:
    1. Load 307,511 rows × 122 columns
    2. Select 15 most predictive features
    3. Fix data quality issues (anomalies, missing values)
    4. Encode categorical variables
    5. Engineer new features from domain knowledge
    6. Return clean data ready for training
    
    Args:
        filepath: Path to application_train.csv
        
    Returns:
        X: Feature matrix (307,511 × 18 features)
        y: Target vector (307,511 items)
    """
    
    print("=" * 80)
    print("LOADING AND PREPARING HOME CREDIT DATASET")
    print("=" * 80)
    
    # Load raw data
    df = pd.read_csv(filepath)
    print(f"\nRaw data shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Select features based on Kaggle research
    selected_features = [
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH',
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
        'CNT_CHILDREN', 'NAME_EDUCATION_TYPE',
    ]
    
    X = df[selected_features].copy()
    y = df['TARGET'].copy()
    
    print(f"\nSelected features: {len(selected_features)}")
    print(f"Target distribution:")
    print(f"  No Default (0): {(y == 0).sum():,} ({(y == 0).mean():.1%})")
    print(f"  Default (1): {(y == 1).sum():,} ({(y == 1).mean():.1%})")
    
    # ===== FIX ANOMALIES =====
    print("\n" + "-" * 80)
    print("FIXING DATA QUALITY ISSUES")
    print("-" * 80)
    
    # DAYS_EMPLOYED anomaly: 365243 is a placeholder for unemployed people (~1000 years)
    anomalous_count = (X['DAYS_EMPLOYED'] == 365243).sum()
    X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace(365243, np.nan)
    print(f"Fixed DAYS_EMPLOYED anomaly: {anomalous_count:,} rows set to NaN")
    
    # ===== CONVERT DAYS TO YEARS =====
    print("\nConverting days to years (more interpretable):")
    X['AGE_YEARS'] = (-X['DAYS_BIRTH'] / 365.25).round(1)
    X['YEARS_EMPLOYED'] = (-X['DAYS_EMPLOYED'] / 365.25).round(1)
    X['YEARS_ID_PUBLISH'] = (-X['DAYS_ID_PUBLISH'] / 365.25).round(1)
    X = X.drop(columns=['DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH'])
    print(f"  AGE_YEARS: {X['AGE_YEARS'].min():.1f} - {X['AGE_YEARS'].max():.1f} years")
    print(f"  YEARS_EMPLOYED: {X['YEARS_EMPLOYED'].min():.1f} - {X['YEARS_EMPLOYED'].max():.1f} years")
    
    # ===== ENCODE CATEGORICAL VARIABLES =====
    print("\nEncoding categorical variables:")
    
    # Binary encoding for gender and flags
    X['CODE_GENDER'] = X['CODE_GENDER'].map({'M': 0, 'F': 1}).fillna(0).astype(int)
    X['FLAG_OWN_CAR'] = X['FLAG_OWN_CAR'].map({'N': 0, 'Y': 1}).astype(int)
    X['FLAG_OWN_REALTY'] = X['FLAG_OWN_REALTY'].map({'N': 0, 'Y': 1}).astype(int)
    print("  CODE_GENDER: {M → 0, F → 1}")
    print("  FLAG_OWN_CAR: {N → 0, Y → 1}")
    print("  FLAG_OWN_REALTY: {N → 0, Y → 1}")
    
    # Ordinal encoding for education (preserves ordering)
    education_map = {
        'Lower secondary': 0,
        'Secondary / secondary special': 1,
        'Incomplete higher': 2,
        'Higher education': 3,
        'Academic degree': 4,
    }
    X['EDUCATION_LEVEL'] = X['NAME_EDUCATION_TYPE'].map(education_map).fillna(1).astype(int)
    X = X.drop(columns=['NAME_EDUCATION_TYPE'])
    print("  EDUCATION_LEVEL: ordinal encoding (0=Lower secondary, 4=Academic degree)")
    
    # ===== HANDLE MISSING VALUES =====
    print("\nHandling missing values:")
    missing_before = X.isnull().sum().sum()
    print(f"  Missing values before imputation: {missing_before:,}")
    
    # Impute with median (robust to outliers, especially in financial data)
    X = X.fillna(X.median())
    missing_after = X.isnull().sum().sum()
    print(f"  Missing values after imputation: {missing_after:,}")
    
    # ===== FEATURE ENGINEERING =====
    print("\nFeature engineering (domain knowledge):")
    
    # Ratios tell richer stories than absolute values
    X['CREDIT_INCOME_RATIO'] = X['AMT_CREDIT'] / (X['AMT_INCOME_TOTAL'] + 1)
    print(f"  CREDIT_INCOME_RATIO: how many years of income is the loan?")
    print(f"    Mean: {X['CREDIT_INCOME_RATIO'].mean():.2f}x, Max: {X['CREDIT_INCOME_RATIO'].max():.2f}x")
    
    X['ANNUITY_INCOME_RATIO'] = X['AMT_ANNUITY'] / (X['AMT_INCOME_TOTAL'] + 1)
    print(f"  ANNUITY_INCOME_RATIO: what % of income goes to monthly payments?")
    print(f"    Mean: {X['ANNUITY_INCOME_RATIO'].mean():.3f} ({X['ANNUITY_INCOME_RATIO'].mean()*100:.1f}%)")
    
    X['CREDIT_GOODS_RATIO'] = X['AMT_CREDIT'] / (X['AMT_GOODS_PRICE'] + 1)
    print(f"  CREDIT_GOODS_RATIO: how much of purchase is financed?")
    print(f"    Mean: {X['CREDIT_GOODS_RATIO'].mean():.2f}, Max: {X['CREDIT_GOODS_RATIO'].max():.2f}")
    
    # ===== FINAL VALIDATION =====
    print("\n" + "-" * 80)
    print("FINAL VALIDATION")
    print("-" * 80)
    
    print(f"\nFinal feature set:")
    print(f"  Shape: {X.shape[0]:,} rows × {X.shape[1]} columns")
    print(f"  Features: {list(X.columns)}")
    print(f"  Any remaining NaN: {X.isnull().any().any()}")
    
    # Verify all features are numeric
    assert X.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), \
        "All features must be numeric"
    
    # Verify no NaN values
    assert not X.isnull().any().any(), "No NaN values should remain"
    
    # Verify reasonable ranges
    assert (X['AGE_YEARS'] >= 18).all() and (X['AGE_YEARS'] <= 150).all(), \
        "Age should be 18-150"
    
    print("\n✓ Data preparation complete and validation passed!")
    print(f"✓ Ready for model training with target distribution: {y.mean():.1%} defaults")
    
    return X, y


if __name__ == '__main__':
    # Example usage
    X, y = load_and_prepare_data('data/application_train.csv')
    print(f"\nReturned: X{X.shape}, y{y.shape}")
