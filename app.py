"""
Emission Projection API
RESTful API for emission projections and transition risk analysis
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import yaml
from loguru import logger

# Initialize FastAPI app
app = FastAPI(
    title="Solar Emission Projection API",
    description="API for solar power emission projections and transition risk analysis",
    version="1.0.0"
)

# Load configuration
with open("config/config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Load models and data
MODELS = {}
DATA_CACHE = {}

def load_resources():
    """Load models and data at startup"""
    try:
        # Load models
        for scenario in ['NZE', 'APS', 'STEPS']:
            model_path = Path("data/models") / f"emission_model_{scenario}.pkl"
            if model_path.exists():
                MODELS[scenario] = joblib.load(model_path)
                logger.info(f"Loaded {scenario} model")
        
        # Load projection data
        proj_path = Path("data/processed/scenario_projections.parquet")
        if proj_path.exists():
            DATA_CACHE['projections'] = pd.read_parquet(proj_path)
            logger.info("Loaded projection data")
        
        # Load risk data
        risk_path = Path("data/processed/transition_risk.parquet")
        if risk_path.exists():
            DATA_CACHE['risk'] = pd.read_parquet(risk_path)
            logger.info("Loaded risk data")
        
    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")


# Pydantic models for request/response
class ProjectionRequest(BaseModel):
    scenario: str = Field(..., description="IEA scenario: NZE, APS, or STEPS")
    year: int = Field(..., ge=2025, le=2050, description="Projection year")
    region: Optional[str] = Field(None, description="Filter by region")

class ProjectionResponse(BaseModel):
    scenario: str
    year: int
    region: str
    capacity_mw: float
    generation_mwh: float
    emissions_avoided_tco2e: float

class RiskMetrics(BaseModel):
    year: int
    region: str
    transition_risk_score: float
    policy_risk_score: float
    stranded_asset_exposure: float
    nze_aps_gap: float
    aps_steps_gap: float

class HealthResponse(BaseModel):
    status: str
    models_loaded: int
    data_available: bool


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Load resources on startup"""
    load_resources()

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Solar Emission Projection API",
        "version": "1.0.0",
        "endpoints": {
            "projections": "/api/v1/projections",
            "risk": "/api/v1/risk",
            "scenarios": "/api/v1/scenarios",
            "regions": "/api/v1/regions",
            "health": "/health"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=len(MODELS),
        data_available='projections' in DATA_CACHE and 'risk' in DATA_CACHE
    )

@app.get("/api/v1/scenarios", response_model=List[Dict])
async def get_scenarios():
    """Get available scenarios and their parameters"""
    scenarios = []
    for name, params in config['scenarios'].items():
        scenarios.append({
            "scenario": name,
            "name": params['name'],
            "description": params['description'],
            "solar_growth_rate": params['solar_growth_rate'],
            "carbon_price_2030": params['carbon_price_2030']
        })
    return scenarios

@app.get("/api/v1/regions", response_model=List[str])
async def get_regions():
    """Get available regions"""
    if 'projections' not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Projection data not available")
    
    regions = DATA_CACHE['projections']['region'].unique().tolist()
    return sorted(regions)

@app.get("/api/v1/projections", response_model=List[ProjectionResponse])
async def get_projections(
    scenario: str = Query(..., description="Scenario: NZE, APS, or STEPS"),
    year: int = Query(..., ge=2025, le=2050, description="Projection year"),
    region: Optional[str] = Query(None, description="Filter by region")
):
    """Get emission projections for specified scenario and year"""
    
    if 'projections' not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Projection data not available")
    
    if scenario not in ['NZE', 'APS', 'STEPS']:
        raise HTTPException(status_code=400, detail="Invalid scenario. Use NZE, APS, or STEPS")
    
    df = DATA_CACHE['projections']
    
    # Filter data
    filtered = df[(df['scenario'] == scenario) & (df['year'] == year)]
    
    if region:
        filtered = filtered[filtered['region'] == region]
    
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No projections found for specified parameters")
    
    # Convert to response format
    results = []
    for _, row in filtered.iterrows():
        results.append(ProjectionResponse(
            scenario=row['scenario'],
            year=row['year'],
            region=row['region'],
            capacity_mw=round(row['capacity_mw'], 2),
            generation_mwh=round(row['generation_mwh'], 2),
            emissions_avoided_tco2e=round(row['emissions_avoided_tco2e'], 2)
        ))
    
    return results

@app.get("/api/v1/projections/compare")
async def compare_scenarios(
    year: int = Query(..., ge=2025, le=2050),
    region: Optional[str] = Query(None)
):
    """Compare all scenarios for a specific year and region"""
    
    if 'projections' not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Projection data not available")
    
    df = DATA_CACHE['projections']
    filtered = df[df['year'] == year]
    
    if region:
        filtered = filtered[filtered['region'] == region]
    
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Pivot for comparison
    comparison = filtered.pivot_table(
        index='region',
        columns='scenario',
        values='emissions_avoided_tco2e',
        aggfunc='sum'
    ).reset_index()
    
    return comparison.to_dict(orient='records')

@app.get("/api/v1/risk", response_model=List[RiskMetrics])
async def get_risk_metrics(
    year: int = Query(..., ge=2025, le=2050),
    region: Optional[str] = Query(None)
):
    """Get transition risk metrics"""
    
    if 'risk' not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Risk data not available")
    
    df = DATA_CACHE['risk']
    filtered = df[df['year'] == year]
    
    if region:
        filtered = filtered[filtered['region'] == region]
    
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No risk data found")
    
    results = []
    for _, row in filtered.iterrows():
        results.append(RiskMetrics(
            year=row['year'],
            region=row['region'],
            transition_risk_score=round(row['transition_risk_score'], 3),
            policy_risk_score=round(row['policy_risk_score'], 3),
            stranded_asset_exposure=round(row['stranded_asset_exposure'], 2),
            nze_aps_gap=round(row['nze_aps_gap'], 2),
            aps_steps_gap=round(row['aps_steps_gap'], 2)
        ))
    
    return results

@app.get("/api/v1/risk/summary")
async def get_risk_summary(year: int = Query(2030, ge=2025, le=2050)):
    """Get aggregated risk summary across all regions"""
    
    if 'risk' not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Risk data not available")
    
    df = DATA_CACHE['risk']
    filtered = df[df['year'] == year]
    
    summary = {
        "year": year,
        "global_metrics": {
            "avg_transition_risk": float(filtered['transition_risk_score'].mean()),
            "avg_policy_risk": float(filtered['policy_risk_score'].mean()),
            "total_stranded_asset_exposure": float(filtered['stranded_asset_exposure'].sum()),
            "max_nze_steps_gap": float(filtered['nze_steps_gap'].max())
        },
        "high_risk_regions": filtered.nlargest(5, 'transition_risk_score')[['region', 'transition_risk_score']].to_dict(orient='records')
    }
    
    return summary

@app.get("/api/v1/timeline")
async def get_timeline(
    scenario: str = Query("NZE"),
    region: str = Query("Asia")
):
    """Get emissions timeline for a scenario and region"""
    
    if 'projections' not in DATA_CACHE:
        raise HTTPException(status_code=503, detail="Projection data not available")
    
    df = DATA_CACHE['projections']
    filtered = df[(df['scenario'] == scenario) & (df['region'] == region)]
    
    if filtered.empty:
        raise HTTPException(status_code=404, detail="No data found")
    
    timeline = filtered.sort_values('year')[['year', 'emissions_avoided_tco2e']].to_dict(orient='records')
    
    return {
        "scenario": scenario,
        "region": region,
        "timeline": timeline
    }


if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logger.add("logs/api.log", rotation="10 MB")
    
    # Run server
    uvicorn.run(
        app,
        host=config['outputs']['api']['host'],
        port=config['outputs']['api']['port'],
        reload=config['outputs']['api']['reload']
    )
