"""
Tests Package for Credit Scoring API

Comprehensive test suite covering:
- API endpoints (validation, errors, happy paths)
- Model loading and inference
- Data preparation pipelines
- Integration tests

Run with: pytest tests/ -v
"""

import os
import sys

# Ensure app module is importable during tests
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

__all__ = ["test_api", "test_model"]
