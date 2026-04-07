"""
Monitoring and Drift Detection Module

Tracks:
1. Prediction logs (inputs, outputs, latency)
2. Data drift detection (KS test, Jensen-Shannon distance)
3. Prediction drift (output distribution changes)
4. Performance metrics (latency p99, error rate)
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon


# ===== SETUP LOGGING =====
def setup_logging(log_dir: str = "logs", log_file: str = "predictions.log"):
    """Setup JSON structured logging for monitoring"""
    Path(log_dir).mkdir(exist_ok=True)
    
    logger = logging.getLogger("credit_scoring")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # File handler (JSON format)
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    
    return logger


# Global logger
LOGGER = setup_logging()


class PredictionLogger:
    """
    Logs predictions with metadata for monitoring
    
    Tracks:
    - Input features
    - Output prediction and probability
    - Inference latency
    - Request timestamp
    """
    
    @staticmethod
    def log_prediction(
        features: Dict[str, Any],
        prediction: int,
        probability: float,
        inference_time_ms: float,
        risk_category: str
    ):
        """Log a single prediction as structured JSON"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": int(prediction),
            "probability": float(probability),
            "risk_category": risk_category,
            "inference_time_ms": float(inference_time_ms),
            "features_count": len(features),
            "features_sum": float(sum(v for v in features.values() if isinstance(v, (int, float)))),
        }
        
        # Log to file
        LOGGER.info(json.dumps(log_entry))
        return log_entry


class DriftDetector:
    """
    Detects data drift and prediction drift
    
    Methods:
    1. Kolmogorov-Smirnov test (KS test) for univariate drift
    2. Jensen-Shannon distance for distribution comparison
    3. Prediction distribution tracking
    """
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize drift detector
        
        Args:
            reference_data: Baseline data distribution (e.g., training set)
        """
        self.reference_data = reference_data
        self.reference_stats = self._calculate_stats(reference_data) if reference_data is not None else None
        self.recent_predictions = []
        self.prediction_history = []
    
    def _calculate_stats(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calculate statistics for each numeric column"""
        stats = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                "mean": float(data[col].mean()),
                "std": float(data[col].std()),
                "min": float(data[col].min()),
                "max": float(data[col].max()),
                "q25": float(data[col].quantile(0.25)),
                "median": float(data[col].quantile(0.5)),
                "q75": float(data[col].quantile(0.75)),
                "values": data[col].values  # For KS test
            }
        return stats
    
    def check_data_drift(
        self,
        new_data: pd.DataFrame,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detect data drift using Kolmogorov-Smirnov test
        
        Args:
            new_data: Recent data to compare to reference
            alpha: Significance level for drift detection (default 0.05)
        
        Returns:
            Dict with drift detection results per feature
        """
        if self.reference_stats is None:
            return {"status": "no_reference_data", "drift_detected": False}
        
        drift_report = {
            "timestamp": datetime.utcnow().isoformat(),
            "alpha": alpha,
            "drift_detected": False,
            "features_with_drift": [],
            "feature_statistics": {}
        }
        
        for col in new_data.select_dtypes(include=[np.number]).columns:
            if col not in self.reference_stats:
                continue
            
            # Kolmogorov-Smirnov test
            statistic, p_value = ks_2samp(
                self.reference_stats[col]["values"],
                new_data[col].values
            )
            
            is_drift = p_value < alpha
            
            drift_report["feature_statistics"][col] = {
                "ks_statistic": float(statistic),
                "ks_pvalue": float(p_value),
                "drift_detected": is_drift,
                "reference_mean": self.reference_stats[col]["mean"],
                "new_mean": float(new_data[col].mean()),
                "reference_std": self.reference_stats[col]["std"],
                "new_std": float(new_data[col].std())
            }
            
            if is_drift:
                drift_report["drift_detected"] = True
                drift_report["features_with_drift"].append({
                    "feature": col,
                    "p_value": float(p_value),
                    "reference_mean": self.reference_stats[col]["mean"],
                    "new_mean": float(new_data[col].mean())
                })
        
        return drift_report
    
    def track_prediction(self, prediction: int, probability: float):
        """Track prediction for drift detection"""
        self.recent_predictions.append({
            "prediction": prediction,
            "probability": probability,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.prediction_history.append(probability)
    
    def check_prediction_drift(self, window_size: int = 1000) -> Dict[str, Any]:
        """
        Detect prediction drift (output distribution change)
        
        Compares recent prediction distribution to earlier predictions
        
        Args:
            window_size: Number of recent predictions to compare
        
        Returns:
            Dict with prediction drift metrics
        """
        if len(self.prediction_history) < 2 * window_size:
            return {"status": "insufficient_data", "drift_detected": False}
        
        # Split into two periods
        earlier = self.prediction_history[:-window_size][-window_size:]
        recent = self.prediction_history[-window_size:]
        
        # Jensen-Shannon distance (better than KS for distributions)
        earlier_hist, bins = np.histogram(earlier, bins=10, range=(0, 1))
        recent_hist, _ = np.histogram(recent, bins=bins)
        
        # Normalize
        earlier_hist = earlier_hist / earlier_hist.sum()
        recent_hist = recent_hist / recent_hist.sum()
        
        distance = float(jensenshannon(earlier_hist, recent_hist))
        
        # Heuristic: JS distance > 0.1 suggests drift
        drift_detected = distance > 0.1
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "window_size": window_size,
            "jensen_shannon_distance": distance,
            "drift_detected": drift_detected,
            "earlier_mean_prob": float(np.mean(earlier)),
            "recent_mean_prob": float(np.mean(recent)),
            "earlier_default_rate": float(np.mean([p > 0.5 for p in earlier])),
            "recent_default_rate": float(np.mean([p > 0.5 for p in recent]))
        }


class PerformanceMonitor:
    """
    Monitors API performance metrics
    """
    
    def __init__(self):
        self.latencies = []
        self.error_count = 0
        self.total_requests = 0
    
    def record_latency(self, latency_ms: float):
        """Record API response latency"""
        self.latencies.append(latency_ms)
        self.total_requests += 1
    
    def record_error(self):
        """Record API error"""
        self.error_count += 1
        self.total_requests += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        if not self.latencies:
            return {"status": "no_data"}
        
        latencies_sorted = sorted(self.latencies)
        n = len(latencies_sorted)
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / self.total_requests if self.total_requests > 0 else 0,
            "latency_metrics": {
                "mean_ms": float(np.mean(self.latencies)),
                "p50_ms": float(latencies_sorted[int(n * 0.5)]),
                "p95_ms": float(latencies_sorted[int(n * 0.95)]),
                "p99_ms": float(latencies_sorted[int(n * 0.99)]),
                "min_ms": float(min(self.latencies)),
                "max_ms": float(max(self.latencies))
            }
        }


# ===== GLOBAL INSTANCES =====
prediction_logger = PredictionLogger()
drift_detector = DriftDetector()
performance_monitor = PerformanceMonitor()


def load_reference_data(reference_path: str = "data/reference_data.csv") -> pd.DataFrame:
    """Load reference data for drift detection"""
    if os.path.exists(reference_path):
        return pd.read_csv(reference_path)
    return None


# Initialize drift detector with reference data if available
_ref_data = load_reference_data()
if _ref_data is not None:
    drift_detector = DriftDetector(_ref_data)
