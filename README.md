# Solar Power Emission Projection & Transition Risk Analysis Pipeline

A robust, automated, and reproducible data science workflow for estimating and projecting emissions from solar power infrastructure under different IEA climate scenarios.

## 📊 Project Overview

This pipeline provides end-to-end automation for:
- **Data Ingestion**: Automated loading and validation of Global Solar Power Tracker data
- **Feature Engineering**: Temporal, geographic, and capacity-based features for modeling
- **Emission Modeling**: Scenario-based emission projections (IEA NZE, APS, STEPS)
- **Transition Risk Analysis**: Stranded asset risk, policy impact, and scenario divergence metrics
- **Production Deployment**: RESTful API and reproducible notebooks

## 🏗️ Architecture

```
solar_emission_pipeline/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned and engineered features
│   └── models/           # Trained model artifacts
├── src/
│   ├── ingestion/        # Data loading and validation
│   ├── analysis/         # EDA and feature engineering
│   ├── modeling/         # Emission projection models
│   └── api/              # REST API deployment
├── notebooks/            # Jupyter notebooks for analysis
├── config/               # Configuration files
├── tests/                # Unit and integration tests
└── outputs/              # Reports and visualizations
```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd solar_emission_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Full pipeline execution
python src/main.py --scenario all

# Specific scenario
python src/main.py --scenario NZE

# API deployment
python src/api/app.py
```

### Interactive Analysis

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## 📈 Emission Scenarios

The pipeline implements three IEA scenarios:

1. **Net Zero Emissions (NZE)**: 1.5°C pathway with aggressive renewables deployment
2. **Announced Pledges Scenario (APS)**: Current policy commitments
3. **Stated Policies Scenario (STEPS)**: Conservative baseline

## 🔄 Workflow Automation

The pipeline supports automated updates via:
- **Scheduled Runs**: Cron jobs for daily/weekly updates
- **GitHub Actions**: CI/CD integration
- **Airflow DAGs**: Enterprise workflow orchestration

## 📊 Key Outputs

- **Emission Projections**: Annual CO2e estimates by country/region/technology
- **Transition Risk Metrics**: Stranded asset exposure, policy sensitivity
- **Interactive Dashboards**: Streamlit/Plotly visualizations
- **API Endpoints**: Real-time scenario queries

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

## 📝 Documentation

- [Data Dictionary](docs/data_dictionary.md)
- [Model Methodology](docs/methodology.md)
- [API Reference](docs/api_reference.md)

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
