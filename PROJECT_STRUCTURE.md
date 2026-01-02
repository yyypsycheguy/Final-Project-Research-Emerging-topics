# Project Structure Documentation

## Solar Emission Projection Pipeline - Complete File Reference

---

## Directory Structure

```
solar_emission_pipeline/
├── config/                      # Configuration files
│   └── config.yaml             # Main configuration (scenarios, parameters, settings)
│
├── data/                        # Data storage
│   ├── raw/                    # Original datasets
│   │   └── Global-Solar-Power-Tracker-February-2025.xlsx
│   ├── processed/              # Cleaned and processed data
│   │   ├── solar_data_processed.parquet
│   │   ├── solar_data_features.parquet
│   │   ├── scenario_projections.parquet
│   │   └── transition_risk.parquet
│   └── models/                 # Trained model artifacts
│       ├── emission_model_NZE.pkl
│       ├── emission_model_APS.pkl
│       ├── emission_model_STEPS.pkl
│       ├── metrics_NZE.yaml
│       ├── metrics_APS.yaml
│       └── metrics_STEPS.yaml
│
├── src/                         # Source code
│   ├── main.py                 # Main pipeline orchestrator
│   │
│   ├── ingestion/              # Data loading and validation
│   │   ├── __init__.py
│   │   └── data_loader.py     # SolarDataIngestion class
│   │
│   ├── analysis/               # Feature engineering
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # FeatureEngineering class
│   │
│   ├── modeling/               # Emission projection models
│   │   ├── __init__.py
│   │   └── emission_model.py  # EmissionProjectionModel class
│   │
│   └── api/                    # REST API
│       ├── __init__.py
│       └── app.py             # FastAPI application
│
├── notebooks/                   # Jupyter notebooks
│   └── 01_exploratory_analysis.ipynb  # Interactive analysis
│
├── scripts/                     # Utility scripts
│   ├── demo.py                 # Demo execution script
│   ├── scheduler.py            # Automation scheduler
│   └── visualize.py            # Visualization generation
│
├── tests/                       # Unit and integration tests
│   └── test_pipeline.py        # Comprehensive test suite
│
├── outputs/                     # Generated outputs
│   ├── reports/                # Analysis reports
│   │   └── pipeline_summary.json
│   └── visualizations/         # Charts and graphs
│       ├── scenario_comparison.html
│       ├── regional_breakdown_2030.html
│       ├── capacity_growth.html
│       ├── risk_heatmap.png
│       └── dashboard.html
│
├── logs/                        # Log files
│   ├── pipeline.log
│   ├── ingestion.log
│   ├── feature_engineering.log
│   ├── modeling.log
│   └── api.log
│
├── docs/                        # Documentation
│   └── api_reference.md        # API endpoint documentation
│
├── .github/                     # GitHub Actions
│   └── workflows/
│       └── ci-cd.yml           # CI/CD pipeline
│
├── README.md                    # Project overview
├── QUICKSTART.md               # Quick start guide
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker container definition
├── docker-compose.yml          # Multi-container orchestration
└── .gitignore                  # Git ignore rules
```

---

## Key Components

### 1. Configuration (`config/config.yaml`)

**Purpose**: Central configuration for all pipeline parameters

**Contents**:
- IEA scenario definitions (NZE, APS, STEPS)
- Emission factors and grid intensities
- Model hyperparameters
- Data validation rules
- Automation schedules
- Output settings

**Usage**:
```python
import yaml
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)
```

---

### 2. Data Ingestion (`src/ingestion/data_loader.py`)

**Class**: `SolarDataIngestion`

**Methods**:
- `load_data()` - Load Excel sheets
- `combine_datasets()` - Merge large/small scale data
- `validate_data()` - Apply quality filters
- `clean_data()` - Standardize and clean
- `run_pipeline()` - Execute full ingestion

**Output**: `data/processed/solar_data_processed.parquet`

---

### 3. Feature Engineering (`src/analysis/feature_engineering.py`)

**Class**: `FeatureEngineering`

**Feature Types**:
- **Temporal**: age, years operational, retirement timeline
- **Geographic**: region, climate zone, solar resource
- **Capacity**: size categories, log transforms
- **Risk**: stranded asset risk, policy risk, market risk
- **Emission**: lifecycle emissions, grid displacement
- **Scenario**: growth rates, carbon prices, competitive scores

**Output**: `data/processed/solar_data_features.parquet`

---

### 4. Emission Modeling (`src/modeling/emission_model.py`)

**Class**: `EmissionProjectionModel`

**Models**:
- XGBoost (primary)
- LightGBM (alternative)
- Random Forest (baseline)

**Methods**:
- `prepare_features()` - Feature selection and encoding
- `train_model()` - Model training with CV
- `project_emissions()` - Future year projections
- `create_scenario_comparison()` - Multi-scenario analysis
- `calculate_transition_risk()` - Risk metrics

**Outputs**:
- `data/processed/scenario_projections.parquet`
- `data/processed/transition_risk.parquet`
- Model files in `data/models/`

---

### 5. REST API (`src/api/app.py`)

**Framework**: FastAPI

**Endpoints**:
- `GET /health` - Health check
- `GET /api/v1/scenarios` - List scenarios
- `GET /api/v1/regions` - List regions
- `GET /api/v1/projections` - Get projections
- `GET /api/v1/projections/compare` - Compare scenarios
- `GET /api/v1/risk` - Get risk metrics
- `GET /api/v1/risk/summary` - Aggregated risk
- `GET /api/v1/timeline` - Emissions timeline

**Documentation**: http://localhost:8000/docs (when running)

---

### 6. Main Orchestrator (`src/main.py`)

**Class**: `EmissionPipeline`

**Phases**:
1. Data Ingestion
2. Feature Engineering
3. Emission Modeling
4. Report Generation

**Usage**:
```bash
# Full pipeline
python src/main.py --scenario all

# Specific scenario
python src/main.py --scenario NZE

# Specific phase
python src/main.py --phase ingestion
```

---

### 7. Automation (`scripts/scheduler.py`)

**Class**: `PipelineScheduler`

**Features**:
- Scheduled data refreshes
- Automated model retraining
- Health checks
- Email notifications
- Error handling

**Schedules** (configurable):
- Data refresh: Weekly (Sunday 2 AM)
- Model retrain: Monthly (1st day, 3 AM)
- Health check: Daily (midnight)

---

### 8. Visualization (`scripts/visualize.py`)

**Class**: `EmissionVisualizer`

**Visualizations**:
- Scenario comparison (line charts)
- Regional breakdown (bar charts)
- Capacity growth (area charts)
- Risk heatmap (heatmap)
- Stranded assets (area chart)
- Comprehensive dashboard (multi-plot)

**Output**: `outputs/visualizations/`

---

## Data Flow

```
Raw Data (Excel)
    ↓
[Data Ingestion]
    ↓
Processed Data (Parquet)
    ↓
[Feature Engineering]
    ↓
Featured Data (Parquet)
    ↓
[Emission Modeling]
    ↓
├─→ Projections (Parquet)
├─→ Risk Metrics (Parquet)
└─→ Trained Models (PKL)
    ↓
[API / Reports / Visualizations]
```

---

## Technology Stack

### Core
- **Python 3.9+**: Primary language
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing

### Machine Learning
- **Scikit-learn**: Model framework
- **XGBoost**: Gradient boosting
- **LightGBM**: Alternative boosting

### Data Storage
- **Parquet**: Efficient columnar storage
- **Pickle**: Model serialization

### API
- **FastAPI**: REST API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical graphics
- **Plotly**: Interactive charts

### Development
- **Pytest**: Testing framework
- **Jupyter**: Interactive notebooks
- **Docker**: Containerization

### Automation
- **Schedule**: Python scheduling
- **GitHub Actions**: CI/CD
- **Loguru**: Logging

---

## Environment Variables

```bash
# Optional environment variables
PYTHONPATH=/path/to/solar_emission_pipeline
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
```

---

## File Formats

### Parquet Files
- Efficient columnar storage
- Fast read/write
- Preserves data types
- Compression support

### YAML Configuration
- Human-readable
- Supports comments
- Hierarchical structure

### PKL Models
- Serialized Python objects
- Preserves model state
- Fast loading

---

## Best Practices

### Data Processing
1. Always validate input data
2. Use Parquet for intermediate storage
3. Preserve original data
4. Log all transformations

### Modeling
1. Version control models
2. Track metrics
3. Cross-validate
4. Test on holdout set

### API Development
1. Use Pydantic for validation
2. Document all endpoints
3. Implement error handling
4. Add rate limiting (production)

### Deployment
1. Use Docker containers
2. Implement health checks
3. Configure monitoring
4. Set up automated backups

---

## Maintenance

### Regular Tasks
- Weekly: Data refresh
- Monthly: Model retraining
- Quarterly: Configuration review
- Annually: Full pipeline audit

### Monitoring
- Check logs daily
- Monitor disk space
- Verify API health
- Track model performance

---

## Security Considerations

### Development
- No authentication required
- Local access only

### Production
- Implement API authentication
- Use HTTPS/TLS
- Set rate limits
- Configure firewalls
- Regular security audits

---

## Scalability

### Current Limits
- ~25,000 solar projects
- 3 scenarios
- 6 regions
- 5 projection years

### Scaling Options
1. **Horizontal**: Multiple worker nodes
2. **Vertical**: More RAM/CPU
3. **Distributed**: Spark/Dask
4. **Cloud**: AWS/GCP/Azure

---

## License

MIT License - See LICENSE file

---

## Support

- Documentation: `/docs` folder
- API Docs: http://localhost:8000/docs
- Issues: GitHub Issues
- Email: support@example.com
