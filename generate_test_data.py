#!/usr/bin/env python3
"""
Generate minimal training dataset for CI/CD testing.
Creates a small dataset with the same schema as the real Kaggle dataset.

CRITICAL: DAYS_BIRTH range must produce ages 18-150 for validation.
  18 years = 6574.5 days → randint upper bound = -6575 (exclusive)
  80 years = 29220 days → randint lower bound = -29220
"""

import pandas as pd
import numpy as np
import os

def generate_test_dataset(n_samples=5000, output_path="data/application_train.csv"):
    """Generate minimal training dataset for CI/CD."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    
    data = {
        'EXT_SOURCE_1': np.random.uniform(0, 1, n_samples),
        'EXT_SOURCE_2': np.random.uniform(0, 1, n_samples),
        'EXT_SOURCE_3': np.random.uniform(0, 1, n_samples),
        'DAYS_BIRTH': np.random.randint(-29220, -6575, n_samples),  # Ages: 18-80 years
        'DAYS_EMPLOYED': np.random.randint(-10000, 1, n_samples),
        'DAYS_ID_PUBLISH': np.random.randint(-10000, 0, n_samples),
        'AMT_INCOME_TOTAL': np.random.uniform(25000, 500000, n_samples),
        'AMT_CREDIT': np.random.uniform(25000, 1000000, n_samples),
        'AMT_ANNUITY': np.random.uniform(1000, 50000, n_samples),
        'AMT_GOODS_PRICE': np.random.uniform(25000, 1000000, n_samples),
        'CODE_GENDER': np.random.randint(0, 2, n_samples),
        'FLAG_OWN_CAR': np.random.randint(0, 2, n_samples),
        'FLAG_OWN_REALTY': np.random.randint(0, 2, n_samples),
        'CNT_CHILDREN': np.random.randint(0, 5, n_samples),
        'NAME_EDUCATION_TYPE': np.random.randint(0, 5, n_samples),
        'TARGET': np.random.binomial(1, 0.08, n_samples)  # ~8% default rate
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"✓ Generated {n_samples:,} test samples at {output_path}")

if __name__ == "__main__":
    generate_test_dataset()
