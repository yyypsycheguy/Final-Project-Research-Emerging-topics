# Quick Start Guide

## Solar Emission Projection Pipeline - Getting Started

This guide will help you set up and run the emission projection pipeline in under 10 minutes.

---

## Prerequisites

- Python 3.9 or higher
- 4GB+ RAM
- 2GB free disk space
- (Optional) Docker for containerized deployment

---

## Installation Methods

### Method 1: Local Installation (Recommended for Development)

#### 1. Clone Repository
```bash
git clone <repository-url>
cd solar_emission_pipeline
```

#### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Configure
```bash
# Review and modify configuration if needed
nano config/config.yaml
```

#### 5. Add Data
Place your `Global-Solar-Power-Tracker-February-2025.xlsx` file in:
```
data/raw/Global-Solar-Power-Tracker-February-2025.xlsx
```

---

### Method 2: Docker Deployment (Recommended for Production)

#### 1. Build Containers
```bash
docker-compose build
```

#### 2. Start Services
```bash
docker-compose up -d
```

#### 3. Check Status
```bash
docker-compose ps
```

Services will be available at:
- API: http://localhost:8000
- Jupyter: http://localhost:8888
- Scheduler: Running in background

---

## Running the Pipeline

### Full Pipeline Execution

Run all phases (ingestion → features → modeling → reporting):

```bash
python src/main.py --scenario all
```

Expected output:
```
✓ Pipeline execution successful!

Results saved to:
  - Projections: data/processed/scenario_projections.parquet
  - Risk Analysis: data/processed/transition_risk.parquet
  - Summary Report: outputs/reports/pipeline_summary.json
```

### Run Specific Scenarios

```bash
# Net Zero Emissions scenario only
python src/main.py --scenario NZE

# Multiple scenarios
python src/main.py --scenario NZE APS
```

### Run Individual Phases

```bash
# Data ingestion only
python src/main.py --phase ingestion

# Feature engineering only
python src/main.py --phase features

# Modeling only
python src/main.py --phase modeling
```

---

## Accessing Results

### 1. View Summary Report
```bash
cat outputs/reports/pipeline_summary.json
```

### 2. Launch API
```bash
python src/api/app.py
```

Then visit: http://localhost:8000/docs for interactive API documentation

### 3. Interactive Analysis
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

### 4. View Processed Data
```python
import pandas as pd

# Load projections
df = pd.read_parquet('data/processed/scenario_projections.parquet')
print(df.head())

# Load risk metrics
risk = pd.read_parquet('data/processed/transition_risk.parquet')
print(risk.head())
```

---

## API Quick Test

### 1. Start API
```bash
python src/api/app.py
```

### 2. Test Endpoints
```bash
# Health check
curl http://localhost:8000/health

# Get 2030 NZE projections for Asia
curl "http://localhost:8000/api/v1/projections?scenario=NZE&year=2030&region=Asia"

# Compare scenarios for 2030
curl "http://localhost:8000/api/v1/projections/compare?year=2030"

# Get risk metrics
curl "http://localhost:8000/api/v1/risk?year=2030"
```

### 3. Interactive Documentation
Visit: http://localhost:8000/docs

---

## Automation Setup

### Schedule Regular Runs

#### Using Cron (Linux/Mac)
```bash
# Edit crontab
crontab -e

# Add weekly pipeline run (Sunday 2 AM)
0 2 * * 0 cd /path/to/solar_emission_pipeline && /path/to/venv/bin/python src/main.py --scenario all
```

#### Using Task Scheduler (Windows)
1. Open Task Scheduler
2. Create Basic Task
3. Set trigger: Weekly, Sunday, 2:00 AM
4. Set action: Start program
   - Program: `C:\path\to\venv\Scripts\python.exe`
   - Arguments: `src/main.py --scenario all`
   - Start in: `C:\path\to\solar_emission_pipeline`

#### Using Built-in Scheduler
```bash
python scripts/scheduler.py
```

---

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run with Coverage
```bash
pytest --cov=src tests/
```

### Test Specific Module
```bash
pytest tests/test_pipeline.py::TestDataIngestion -v
```

---

## Common Issues & Solutions

### Issue: Import errors
**Solution:** Make sure virtual environment is activated and all dependencies are installed
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Data file not found
**Solution:** Ensure data file is in correct location
```bash
ls -la data/raw/Global-Solar-Power-Tracker-February-2025.xlsx
```

### Issue: Out of memory
**Solution:** Reduce dataset size or increase available RAM
```python
# In config/config.yaml, adjust:
data:
  sample_size: 10000  # Process subset of data
```

### Issue: API won't start
**Solution:** Check if port 8000 is already in use
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
python src/api/app.py --port 8001
```

---

## Performance Tips

1. **Use Parquet format** - Faster than CSV, smaller file size
2. **Enable caching** - Reuse processed data between runs
3. **Parallel processing** - Set `n_jobs=-1` in model config
4. **Sample data** - Use subset for development
5. **Docker deployment** - Better resource isolation

---

## Next Steps

1. **Customize scenarios**: Edit `config/config.yaml` to add custom scenarios
2. **Add visualizations**: Create custom plots in notebooks
3. **Extend API**: Add new endpoints in `src/api/app.py`
4. **Schedule runs**: Set up automated pipeline execution
5. **Deploy**: Use Docker for production deployment

---

## Getting Help

- **Documentation**: Check `docs/` folder
- **Examples**: See `notebooks/` for interactive examples
- **API Docs**: Visit http://localhost:8000/docs
- **Issues**: Report bugs on GitHub
- **Logs**: Check `logs/` folder for error details

---

## Production Checklist

Before deploying to production:

- [ ] Configure authentication for API
- [ ] Set up monitoring and alerting
- [ ] Implement rate limiting
- [ ] Configure backup strategy
- [ ] Set up SSL/TLS certificates
- [ ] Review security settings
- [ ] Test disaster recovery
- [ ] Document deployment process
- [ ] Set up CI/CD pipeline
- [ ] Configure logging aggregation

---

## License

MIT License - See LICENSE file for details

---

## Support

For questions or issues:
- Email: support@example.com
- GitHub: [repository-url]/issues
- Documentation: [repository-url]/docs
