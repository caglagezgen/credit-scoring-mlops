# Credit Scoring MLOps Pipeline

A complete, production-ready MLOps pipeline for predicting loan default probability using the Home Credit Default Risk dataset from Kaggle.

**What You'll Learn:**
- Data preparation & feature engineering
- Model training with scikit-learn
- FastAPI for serving predictions
- Pydantic for input validation
- Docker containerization
- CI/CD with GitHub Actions
- Data drift detection
- Performance optimization with ONNX

---

## Quick Start

### 1. Setup Environment

```bash
# Clone and navigate
cd credit-scoring-mlops

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
# Option 1: Using Kaggle CLI (requires API token)
pip install kaggle
# See: https://www.kaggle.com/settings/account for API token setup
kaggle competitions download -c home-credit-default-risk
mkdir -p data
mv home-credit-default-risk/application_train.csv data/

# Option 2: Manual download from Kaggle (faster for one file)
# Download application_train.csv from: https://www.kaggle.com/c/home-credit-default-risk/data
# Place in: data/application_train.csv
```

### 3. Train Model

```bash
python train_model.py
```

**Output:**
- `model/credit_model.pkl` — trained pipeline
- `model/feature_columns.pkl` — feature order
- `data/reference_data.csv` — training data baseline

### 4. Start API Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Access:**
- API: http://localhost:8000
- Interactive Docs (Swagger): http://localhost:8000/docs
- OpenAPI Schema: http://localhost:8000/openapi.json

### 5. Test the API

**Via Swagger UI:**
1. Go to http://localhost:8000/docs
2. Click "Try it out" on the `/predict` endpoint
3. Fill in example data and execute

**Via curl:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
    "education_level": 1
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "probability_of_default": 0.1234,
  "risk_category": "Low"
}
```

---

## Project Structure

```
credit-scoring-mlops/
├── app/                          # API application
│   ├── __init__.py
│   ├── main.py                   # FastAPI application & endpoints
│   ├── schemas.py                # Pydantic validation models
│   ├── model_loader.py           # Model loading (load once pattern)
│   └── data_preparation.py       # Data processing pipeline
│
├── model/                        # Trained model artifacts
│   ├── credit_model.pkl          # Trained pipeline (scaler + model)
│   └── feature_columns.pkl       # Feature order (critical!)
│
├── data/                         # Dataset directory
│   ├── application_train.csv     # Original Kaggle dataset
│   ├── reference_data.csv        # Training data (drift baseline)
│   └── test_data.csv             # Test set
│
├── tests/                        # Automated tests
│   ├── __init__.py
│   ├── test_api.py               # FastAPI endpoint tests
│   └── test_model.py             # Model artifact tests
│
├── notebooks/                    # Jupyter notebooks
│   └── data_drift_analysis.ipynb # Drift detection analysis
│
├── monitoring/                   # Monitoring & observability
│   ├── logger.py                 # Logging utilities
│   └── drift_report.html         # Generated drift reports
│
├── .github/workflows/            # CI/CD configuration
│   └── ci-cd.yml                 # GitHub Actions workflow
│
├── Dockerfile                    # Docker image recipe
├── requirements.txt              # Python dependencies
├── train_model.py                # Model training script
├── MLOPS_DEVELOPMENT_GUIDE.md   # Detailed guide
└── README.md                     # This file
```

---

## Running Tests

Test the entire API:

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

**Expected Output:**
```
tests/test_api.py::TestHealthCheck::test_health_returns_200 PASSED
tests/test_api.py::TestValidPredictions::test_valid_prediction_returns_200 PASSED
tests/test_api.py::TestInvalidInputs::test_ext_source_above_1_rejected PASSED
... (40+ tests total, all should pass)
```

---

## Docker

### Build Image

```bash
docker build -t credit-scoring-api .
```

### Run Container

```bash
docker run -d -p 8000:8000 --name scoring-api credit-scoring-api
```

### Test Container

```bash
curl http://localhost:8000/health
# {"status":"healthy","model_loaded":true}
```

### Stop Container

```bash
docker stop scoring-api
docker rm scoring-api
```

---

## CI/CD with GitHub Actions

### Setup

1. Create `.github/workflows/ci-cd.yml` (template provided)
2. Add Docker secrets to GitHub:
   - Settings → Secrets → New repository secret
   - `DOCKER_USERNAME` — your Docker Hub username
   - `DOCKER_TOKEN` — Docker Hub access token

### Workflow

Every push to `main`:
1. **TEST** — Run pytest
2. **BUILD** — Build Docker image, smoke test
3. **DEPLOY** — Push to Docker Hub

If any stage fails, deployment stops.

---

## Data Drift Detection

Monitor for data distribution shifts over time:

```bash
# Create drift analysis
jupyter notebook notebooks/data_drift_analysis.ipynb
```

Key metrics:
- Are external credit scores shifting?
- Are applicant ages changing?
- Are financial amounts inflating?

Drift indicates model retraining needed.

---

## Performance Optimization

Measure and optimize inference latency:

```bash
# Create optimization analysis
jupyter notebook notebooks/performance_optimization.ipynb
```

Key findings:
- Baseline scikit-learn inference: ~5-10ms
- ONNX Runtime optimization: ~1-2ms (5x faster)
- Critical: always verify predictions don't change!

---

## Configuration

### Environment Variables

```bash
# Model paths (useful for Docker)
export MODEL_PATH=/app/model/credit_model.pkl
export FEATURES_PATH=/app/model/feature_columns.pkl

# API host (default 0.0.0.0:8000)
export API_HOST=0.0.0.0
export API_PORT=8000
```

### Feature Description

The API accepts 15 raw features:

| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `ext_source_1,2,3` | float | 0-1 | External credit bureau scores |
| `amt_income_total` | float | >0 | Annual income |
| `amt_credit` | float | >0 | Loan amount |
| `amt_annuity` | float | >0 | Monthly payment |
| `amt_goods_price` | float | >0 | Goods price being financed |
| `code_gender` | int | 0-1 | 0=Male, 1=Female |
| `flag_own_car` | int | 0-1 | Owns car |
| `flag_own_realty` | int | 0-1 | Owns real estate |
| `cnt_children` | int | 0-20 | Number of children |
| `age_years` | float | 18-80 | Age in years |
| `years_employed` | float | 0-50 | Years at current job |
| `years_id_publish` | float | 0-60 | Years since ID issued |
| `education_level` | int | 0-4 | Education (0=Lower secondary, 4=Academic) |

### Output Description

| Field | Type | Range | Meaning |
|-------|------|-------|---------|
| `prediction` | int | 0-1 | 0=Will repay, 1=Will default |
| `probability_of_default` | float | 0.0-1.0 | Confidence (0=safe, 1=risky) |
| `risk_category` | str | Low/Med/High | Business-friendly assessment |

---

## Troubleshooting

### Model Not Found
```
FileNotFoundError: Model not found at model/credit_model.pkl
```

**Solution:** Run `python train_model.py` first.

### Kaggle Dataset Download Failed
```
Check your Kaggle API token (~/.kaggle/kaggle.json)
Or download manually from: https://www.kaggle.com/c/home-credit-default-risk/data
```

### Tests Fail
```bash
# Ensure model is trained
python train_model.py

# Run tests with verbose output
pytest tests/ -v -s

# Check specific test
pytest tests/test_api.py::TestValidPredictions::test_valid_prediction_returns_200 -v
```

### Docker Build Issues
```bash
# Clean old images
docker system prune -a

# Rebuild
docker build --no-cache -t credit-scoring-api .
```

---

## Performance Benchmarks

| Component | Latency | Throughput |
|-----------|---------|-----------|
| API endpoint (uvicorn) | ~15-20ms | ~50-100 req/s |
| Model inference (sklearn) | ~5-10ms | — |
| Model inference (ONNX) | ~1-2ms | — |
| Input validation (Pydantic) | <1ms | — |

---

## Key Learnings (From Reference Article)

1. **Data preparation is 50% of the work** — real data is messy
2. **Load models once, never per request** — single biggest performance decision
3. **Validate everything at the boundary** — Pydantic catches edge cases
4. **Test invalid inputs** — silent failures are dangerous
5. **Docker eliminates environment surprises** — reproducibility matters
6. **CI/CD is your quality gate** — broken code doesn't reach production
7. **Monitor for drift** — models degrade silently without monitoring
8. **Profile before optimizing** — measure before you optimize
9. **Start simple, iterate** — don't build everything at once
10. **Understand the why** — not just the how

---

## Resources

- [HackerNoon Article](https://hackernoon.com/weekend-project-i-built-a-full-mlops-pipeline-for-a-credit-scoring-model-and-you-can-too) — Complete step-by-step guide
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — API framework
- [Pydantic Documentation](https://docs.pydantic.dev/) — Input validation
- [Docker Documentation](https://docs.docker.com/) — Containerization
- [Evidently AI](https://docs.evidentlyai.com/) — Drift detection
- [scikit-learn Pipelines](https://scikit-learn.org/stable/modules/pipeline.html) — Model pipelines
- [GitHub Actions](https://docs.github.com/en/actions) — CI/CD

---

## License

This project is educational. The Home Credit dataset is provided by Home Credit for educational/competition purposes.

---

## Next Steps

1. **Start Local Development**
   ```bash
   python train_model.py
   uvicorn app.main:app --reload
   # Visit http://localhost:8000/docs
   ```

2. **Run Full Test Suite**
   ```bash
   pytest tests/ -v
   ```

3. **Build Docker Image**
   ```bash
   docker build -t credit-scoring-api .
   docker run -p 8000:8000 credit-scoring-api
   ```

4. **Setup GitHub Actions**
   - Create `.github/workflows/ci-cd.yml`
   - Add Docker secrets
   - Push to trigger pipeline

5. **Monitor for Drift**
   - Run `notebooks/data_drift_analysis.ipynb`
   - Set up automated drift reports
   - Establish retraining triggers

6. **Optimize Performance**
   - Convert to ONNX format
   - Benchmark against baseline
   - Deploy optimized version

---
