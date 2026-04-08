#!/usr/bin/env python3
"""
Training wrapper script that ensures model is trained before tests.
Validates that model artifacts are created.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_model_artifacts():
    """Verify model files exist."""
    required_files = [
        "model/credit_model.pkl",
        "model/feature_columns.pkl",
        "data/reference_data.csv",
        "model/metadata.json"
    ]
    
    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)
    
    return missing

def run_training():
    """Run training and capture output."""
    print("\n" + "=" * 80)
    print("TRAINING MODEL FOR CI/CD")
    print("=" * 80 + "\n")
    
    result = subprocess.run(
        [sys.executable, "train_model.py"],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"\n✗ Training failed with exit code {result.returncode}")
        return False
    
    return True

def main():
    """Main wrapper logic."""
    
    # Check if we need to train
    missing = check_model_artifacts()
    if not missing:
        print("✓ Model artifacts already exist, skipping training")
        return 0
    
    print(f"✗ Missing model artifacts: {missing}")
    print("\nGenerating test data...")
    
    # Generate test data
    result = subprocess.run(
        [sys.executable, "generate_test_data.py"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"✗ Failed to generate test data: {result.stderr}")
        return 1
    
    # Train model
    if not run_training():
        return 1
    
    # Verify artifacts
    print("\n" + "-" * 80)
    print("VERIFYING MODEL ARTIFACTS")
    print("-" * 80)
    
    missing = check_model_artifacts()
    if missing:
        print(f"✗ Still missing artifacts: {missing}")
        return 1
    
    print("✓ All model artifacts created successfully:")
    for f in ["model/credit_model.pkl", "model/feature_columns.pkl", "data/reference_data.csv"]:
        size = os.path.getsize(f) / 1024  # KB
        print(f"  {f:40s} ({size:.1f} KB)")
    
    print("\n✓ Ready for testing!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
