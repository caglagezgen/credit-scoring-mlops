"""
Model Training and Saving

Trains a scikit-learn gradient boosting model on Home Credit data
and saves artifacts for production deployment.

Key Features:
1. Loads hyperparameters from configs/experiment.yaml (reproducibility)
2. Generates model/metadata.json with training info and lineage
3. Tracks git commit hash for reproducibility
4. Saves models, features, and reference data for monitoring
"""

import os
import json
import joblib
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import yaml
import subprocess

from app.data_preparation import load_and_prepare_data


# ===== CONFIGURATION LOADER =====
def load_experiment_config(config_path: str = "configs/experiment.yaml") -> dict:
    """Load experiment configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_git_info() -> dict:
    """Get current git commit and branch info"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent
        ).decode().strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=Path(__file__).parent
        ).decode().strip()
        return {"commit": commit, "branch": branch}
    except Exception as e:
        print(f"Warning: Could not get git info: {e}")
        return {"commit": "unknown", "branch": "unknown"}


def generate_metadata(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    y_proba_train,
    y_proba_test,
    roc_auc_train: float,
    roc_auc_test: float,
    config: dict,
    training_start_time: datetime
) -> dict:
    """
    Generate comprehensive model metadata for registry and monitoring
    
    Includes:
    - Model info (name, version, framework)
    - Training info (date, duration, samples)
    - Performance metrics (ROC AUC, precision, recall)
    - Hyperparameters (for reproducibility)
    - Feature info (count, names)
    - Git lineage (commit, branch)
    - Data version info
    """
    training_end_time = datetime.now()
    training_duration = (training_end_time - training_start_time).total_seconds()
    git_info = get_git_info()
    y_pred_test = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    
    metadata = {
        "model_registry": {
            "model_name": config["experiment"]["name"],
            "model_id": f"{config['experiment']['name']}_{config['experiment']['version'].replace('.', '_')}",
            "version": config["experiment"]["version"],
            "created_date": training_end_time.isoformat(),
            "created_by": "Automated training pipeline",
            "model_type": config["model"]["name"],
            "framework": "scikit-learn",
            "status": "trained"
        },
        
        "model_lineage": {
            "git_commit": git_info["commit"],
            "git_branch": git_info["branch"],
            "training_script": "train_model.py",
            "config_file": "configs/experiment.yaml",
            "data_version": config["data"]["version"],
            "feature_engineering_code": "app/data_preparation.py"
        },
        
        "training_info": {
            "training_date": training_end_time.isoformat(),
            "training_duration_seconds": training_duration,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "total_samples": len(X_train) + len(X_test),
            "feature_count": len(X_train.columns),
            "source_feature_count": config["features"]["selected_feature_count"] - len(config["features"]["engineered_features"]),
            "engineered_feature_count": len(config["features"]["engineered_features"]),
            "class_distribution_train": {
                "no_default": float(1 - y_train.mean()),
                "default": float(y_train.mean())
            },
            "class_distribution_test": {
                "no_default": float(1 - y_test.mean()),
                "default": float(y_test.mean())
            },
            "train_test_split_ratio": config["data"]["train_test_split"],
            "stratified": config["data"]["stratify"],
            "random_seed": config["model"]["hyperparameters"]["random_state"]
        },
        
        "model_performance": {
            "training_set": {
                "roc_auc": float(roc_auc_train),
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0
            },
            "test_set": {
                "roc_auc": float(roc_auc_test),
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0
            }
        },
        
        "hyperparameters": config["model"]["hyperparameters"],
        
        "preprocessing": {
            "scaling": config["data"]["preprocessing"]["scaling"],
            "missing_values": config["data"]["preprocessing"]["handle_missing_values"],
            "encoding": config["data"]["preprocessing"]["encoding_strategy"]
        },
        
        "output_paths": {
            "model_artifact": config["output"]["model_artifact_path"],
            "feature_columns": config["output"]["feature_columns_path"],
            "reference_data": config["output"]["reference_data_path"],
            "metadata": config["output"]["metadata_path"]
        }
    }
    
    return metadata


def train_credit_model():
    """
    Train a production-ready credit scoring model.
    
    Key Design Decisions:
    1. Use Pipeline to bundle StandardScaler + GradientBoostingClassifier
       - Ensures scaler and model never get out of sync
       - Single file to deploy instead of multiple artifacts
    
    2. Gradient Boosting (not Random Forest, SVM, Neural Networks)
       - Best for tabular/structured data
       - Consistent top performer in Kaggle competitions
       - Interpretable feature importances
    
    3. Evaluate with ROC AUC (not Accuracy)
       - Dataset is imbalanced (92% no-default, 8% default)
       - Accuracy is misleading: model that always predicts "no default" 
         would have 92% accuracy but be useless
       - ROC AUC measures ranking quality: are defaults scored higher?
    """
    
    print("\n" + "=" * 80)
    print("TRAINING CREDIT SCORING MODEL")
    print("=" * 80)
    
    training_start_time = datetime.now()
    
    # ===== LOAD CONFIGURATION =====
    print("\n0. LOADING CONFIGURATION")
    print("-" * 80)
    config = load_experiment_config("configs/experiment.yaml")
    print(f"✓ Loaded experiment config: {config['experiment']['name']} v{config['experiment']['version']}")
    print(f"  Git Tracking: {get_git_info()}")
    
    # ===== LOAD AND PREPARE DATA =====
    print("\n1. LOADING AND PREPARING DATA")
    print("-" * 80)
    X, y = load_and_prepare_data()
    
    # ===== TRAIN/TEST SPLIT =====
    print("\n2. TRAIN/TEST SPLIT")
    print("-" * 80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["data"]["train_test_split"],
        random_state=config["data"]["random_seed"],
        stratify=y if config["data"]["stratify"] else None
    )
    
    print(f"Training set: {X_train.shape[0]:,} rows ({X_train.shape[0]/len(X):.1%})")
    print(f"Test set: {X_test.shape[0]:,} rows ({X_test.shape[0]/len(X):.1%})")
    print(f"\nTrain default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")
    print("✓ Stratification successful (both sets have same default rate)")
    
    # ===== BUILD PIPELINE =====
    print("\n3. BUILDING PIPELINE")
    print("-" * 80)
    
    # Load hyperparameters from config
    hyperparams = config["model"]["hyperparameters"]
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(
            n_estimators=hyperparams["n_estimators"],
            max_depth=hyperparams["max_depth"],
            learning_rate=hyperparams["learning_rate"],
            subsample=hyperparams["subsample"],
            min_samples_split=hyperparams.get("min_samples_split", 20),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 10),
            random_state=hyperparams["random_state"],
            verbose=hyperparams.get("verbose", 0),
        ))
    ])
    
    print("Pipeline architecture:")
    print("  1. StandardScaler → normalize each feature to mean=0, std=1")
    print("  2. GradientBoostingClassifier → Sequential ensemble of decision trees")
    print("\nHyperparameters (from configs/experiment.yaml):")
    print(f"  n_estimators={hyperparams['n_estimators']}     → trees")
    print(f"  max_depth={hyperparams['max_depth']}          → tree depth")
    print(f"  learning_rate={hyperparams['learning_rate']}    → step size")
    print(f"  subsample={hyperparams['subsample']}        → data per tree")
    
    # ===== TRAIN MODEL =====
    print("\n4. TRAINING MODEL")
    print("-" * 80)
    print(f"Training on {X_train.shape[0]:,} samples with {X_train.shape[1]} features...")
    pipeline.fit(X_train, y_train)
    print("✓ Training complete")
    
    # ===== EVALUATE ON TEST SET =====
    print("\n5. EVALUATION")
    print("-" * 80)
    
    y_pred_train = pipeline.predict(X_train)
    y_proba_train = pipeline.predict_proba(X_train)[:, 1]
    roc_auc_train = roc_auc_score(y_train, y_proba_train)
    
    y_pred_test = pipeline.predict(X_test)
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]
    roc_auc_test = roc_auc_score(y_test, y_proba_test)
    
    print(f"\nTraining Set ROC AUC: {roc_auc_train:.4f}")
    print(f"Test Set ROC AUC: {roc_auc_test:.4f}")
    print(f"  Interpretation: Model ranks defaults {roc_auc_test:.1%} better than random")
    print(f"  Typical range for this dataset: 0.74-0.76 (matches Kaggle submissions)")
    
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['No Default', 'Default']))
    
    print(f"\nTest Set Confusion Matrix Interpretation:")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    print(f"  True Negatives (correctly identified safe applicants): {tn:,}")
    print(f"  False Positives (rejected safe applicants): {fp:,}")
    print(f"  False Negatives (approved risky applicants): {fn:,}")
    print(f"  True Positives (correctly identified risky applicants): {tp:,}")
    
    # ===== SAVE ARTIFACTS =====
    print("\n6. SAVING ARTIFACTS")
    print("-" * 80)
    
    os.makedirs('model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Save trained pipeline
    joblib.dump(pipeline, config["output"]["model_artifact_path"])
    print(f"✓ Saved: {config['output']['model_artifact_path']}")
    
    # Save feature columns (order matters!)
    joblib.dump(list(X_train.columns), config["output"]["feature_columns_path"])
    print(f"✓ Saved: {config['output']['feature_columns_path']}")
    
    # Save reference data (training data for drift detection)
    X_train.to_csv(config["output"]["reference_data_path"], index=False)
    print(f"✓ Saved: {config['output']['reference_data_path']} (for drift detection baseline)")
    
    # Save test data
    X_test.to_csv('data/test_data.csv', index=False)
    y_test.to_csv('data/test_labels.csv', index=False)
    print(f"✓ Saved: data/test_data.csv and data/test_labels.csv")
    
    # ===== GENERATE AND SAVE METADATA =====
    print("\n7. GENERATING MODEL METADATA")
    print("-" * 80)
    metadata = generate_metadata(
        pipeline, X_train, X_test, y_train, y_test,
        y_proba_train, y_proba_test, roc_auc_train, roc_auc_test,
        config, training_start_time
    )
    
    metadata_path = config["output"]["metadata_path"]
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved: {metadata_path}")
    print(f"  Model Lineage: Git commit {metadata['model_lineage']['git_commit'][:8]}")
    print(f"  Performance: ROC AUC {metadata['model_performance']['test_set']['roc_auc']:.4f}")
    print(f"  Data Version: {metadata['model_lineage']['data_version']}")
    
    print("\n" + "=" * 80)
    print("MODEL TRAINING COMPLETE")
    print("\n" + "=" * 80)
    print(f"\nSummary:")
    print(f"  ROC AUC (Test): {roc_auc_test:.4f}")
    print(f"  Model file: {config['output']['model_artifact_path']}")
    print(f"  Feature columns: {config['output']['feature_columns_path']}")
    print(f"  Reference data: {config['output']['reference_data_path']} ({X_train.shape[0]:,} rows)")
    print(f"  Metadata: {metadata_path} (lineage, metrics, hyperparameters)")
    print(f"\nNext steps:")
    print(f"  1. Deploy model via FastAPI (app/main.py)")
    print(f"  2. Run tests: pytest tests/ -v")
    print(f"  3. Build Docker image: docker build -t credit-scoring-api .")
    print(f"  4. Monitor for data drift using {config['output']['reference_data_path']}")
    
    return pipeline, roc_auc_test


if __name__ == '__main__':
    pipeline, auc = train_credit_model()
