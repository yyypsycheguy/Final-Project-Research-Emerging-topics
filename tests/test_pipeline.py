"""
Unit Tests for Emission Projection Pipeline
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import tempfile
import shutil


# Test Data Ingestion
class TestDataIngestion:
    """Test data loading and validation"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample solar data"""
        return pd.DataFrame({
            'Capacity (MW)': [50, 100, 200, 5, 1000],
            'Start year': [2020, 2021, 2019, 2022, 2018],
            'Country/Area': ['USA', 'China', 'India', 'Brazil', 'Australia'],
            'Status': ['Operating', 'Construction', 'Operating', 'Announced', 'Operating'],
            'Technology Type': ['Solar PV', 'Solar PV', 'Solar Thermal', 'Solar PV', 'Solar PV'],
            'Latitude': [40.0, 35.0, 20.0, -15.0, -25.0],
            'Longitude': [-100.0, 110.0, 77.0, -47.0, 135.0],
            'GEM phase ID': ['P1', 'P2', 'P3', 'P4', 'P5']
        })
    
    def test_data_validation(self, sample_data):
        """Test data validation logic"""
        from src.ingestion.data_loader import SolarDataIngestion
        
        ingestion = SolarDataIngestion()
        validated = ingestion.validate_data(sample_data)
        
        # Check capacity filtering
        assert all(validated['Capacity (MW)'] >= 1)
        assert all(validated['Capacity (MW)'] <= 10000)
        
        # Check year filtering
        assert all(validated['Start year'] >= 2000)
        assert all(validated['Start year'] <= 2050)
    
    def test_data_cleaning(self, sample_data):
        """Test data cleaning"""
        from src.ingestion.data_loader import SolarDataIngestion
        
        ingestion = SolarDataIngestion()
        cleaned = ingestion.clean_data(sample_data)
        
        # Check column name standardization
        assert 'capacity_(mw)' in cleaned.columns
        assert 'start_year' in cleaned.columns
        
        # Check no missing critical values
        assert cleaned['capacity_(mw)'].notna().all()


# Test Feature Engineering
class TestFeatureEngineering:
    """Test feature creation"""
    
    @pytest.fixture
    def featured_data(self):
        """Create sample data with cleaned column names (as produced by data_loader)"""
        return pd.DataFrame({
            'capacity_(mw)': [50, 100, 200],
            'start_year': [2010, 2015, 2020],
            'country_area': ['USA', 'China', 'India'],
            'latitude': [40.0, 35.0, 20.0],
            'longitude': [-100.0, 110.0, 77.0],
            'technology_type': ['Solar PV', 'Solar PV', 'Solar Thermal'],
            'status': ['Operational', 'Operational', 'Under Construction'],
            'project_name': ['Project A', 'Project B', 'Project C']
        })
    
    def test_temporal_features(self, featured_data):
        """Test temporal feature creation"""
        from src.analysis.feature_engineering import FeatureEngineering
        
        engineer = FeatureEngineering()
        result = engineer.create_temporal_features(featured_data)
        
        # Check new features exist
        assert 'years_operational' in result.columns
        assert 'age_category' in result.columns
        assert 'years_to_retirement' in result.columns
        
        # Validate calculations
        assert result['years_operational'].min() >= 0
    
    def test_capacity_features(self, featured_data):
        """Test capacity feature creation"""
        from src.analysis.feature_engineering import FeatureEngineering
        
        engineer = FeatureEngineering()
        result = engineer.create_capacity_features(featured_data)
        
        assert 'log_capacity' in result.columns
        assert 'capacity_category' in result.columns
        
        # Check log transformation
        assert (result['log_capacity'] >= 0).all()
    
    def test_emission_features(self, featured_data):
        """Test emission feature calculation"""
        from src.analysis.feature_engineering import FeatureEngineering
        
        engineer = FeatureEngineering()
        
        # Need geographic features first for region_group and solar_resource_proxy
        df_with_geo = engineer.create_geographic_features(featured_data)
        result = engineer.create_emission_features(df_with_geo)
        
        assert 'lifecycle_emissions_factor' in result.columns
        assert 'annual_emissions_avoided' in result.columns
        assert 'grid_intensity' in result.columns


# Test Modeling
class TestEmissionModel:
    """Test emission projection models"""
    
    @pytest.fixture
    def model_data(self):
        """Create sample data for modeling"""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'capacity_(mw)': np.random.uniform(10, 500, n),
            'log_capacity': np.random.uniform(2, 6, n),
            'years_operational': np.random.randint(0, 20, n),
            'latitude': np.random.uniform(-60, 60, n),
            'longitude': np.random.uniform(-180, 180, n),
            'solar_resource_proxy': np.random.uniform(0.5, 1.0, n),
            'age_risk_score': np.random.uniform(0, 1, n),
            'tech_risk_score': np.random.uniform(0, 1, n),
            'market_risk_score': np.random.uniform(0, 1, n),
            'lifecycle_emissions_factor': np.random.uniform(40, 60, n),
            'grid_intensity': np.random.uniform(300, 600, n),
            'capacity_factor': np.random.uniform(0.15, 0.30, n),
            'annual_generation_mwh': np.random.uniform(10000, 500000, n),
            'NZE_growth_rate': np.full(n, 0.15),
            'NZE_carbon_price_2030': np.full(n, 130),
            'NZE_competitive_score': np.random.uniform(0.5, 1.5, n),
            'annual_emissions_avoided': np.random.uniform(1000, 100000, n)
        })
    
    def test_model_training(self, model_data):
        """Test model training"""
        from src.modeling.emission_model import EmissionProjectionModel
        
        model_obj = EmissionProjectionModel()
        
        # Prepare features
        X = model_data.drop('annual_emissions_avoided', axis=1)
        y = model_data['annual_emissions_avoided']
        
        # Train model
        model, metrics = model_obj.train_model(X, y, model_type='random_forest')
        
        # Check model exists
        assert model is not None
        
        # Check metrics
        assert 'r2' in metrics
        assert 'rmse' in metrics
        assert isinstance(metrics['r2'], (int, float))  # R² should be a number
        assert metrics['rmse'] >= 0  # RMSE should be non-negative
    
    def test_projection_calculation(self, model_data):
        """Test emission projections"""
        from src.modeling.emission_model import EmissionProjectionModel
        
        model_obj = EmissionProjectionModel()
        
        # Add required columns
        model_data['region_group'] = 'Asia'
        model_data['retirement_probability'] = 0.1
        model_data['emission_reduction_factor'] = 400
        
        # Mock model
        class MockModel:
            def predict(self, X):
                return np.random.uniform(1000, 100000, len(X))
        
        model = MockModel()
        
        # Test projection
        projections = model_obj.project_emissions(model_data, model, 'NZE', 2030)
        
        assert 'projected_capacity' in projections.columns
        assert 'projected_emissions_avoided' in projections.columns
        assert len(projections) > 0


# Test Configuration
class TestConfiguration:
    """Test configuration loading and validation"""
    
    def test_config_loading(self):
        """Test configuration file loading"""
        config_path = "config/config.yaml"
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check main sections
        assert 'scenarios' in config
        assert 'data' in config
        assert 'modeling' in config
        
        # Check scenarios
        assert 'NZE' in config['scenarios']
        assert 'APS' in config['scenarios']
        assert 'STEPS' in config['scenarios']
    
    def test_scenario_parameters(self):
        """Test scenario parameter validity"""
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        for scenario_name, params in config['scenarios'].items():
            # Check required parameters exist
            assert 'solar_growth_rate' in params
            assert 'carbon_price_2030' in params
            
            # Validate ranges
            assert 0 <= params['solar_growth_rate'] <= 1
            assert params['carbon_price_2030'] > 0


# Test API
class TestAPI:
    """Test REST API endpoints"""
    
    @pytest.mark.skip(reason="TestClient compatibility issue with current httpx version")
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from src.api.app import app
        
        # Create a fresh app instance for testing without startup events
        from fastapi import FastAPI
        test_app = FastAPI()
        
        # Copy routes from the main app
        for route in app.routes:
            test_app.router.add_route(route.path, route.endpoint, methods=route.methods)
        
        return TestClient(test_app)
    
    @pytest.mark.skip(reason="TestClient compatibility issue with current httpx version")
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert 'status' in data
    
    @pytest.mark.skip(reason="TestClient compatibility issue with current httpx version")
    def test_scenarios_endpoint(self, client):
        """Test scenarios endpoint"""
        response = client.get("/api/v1/scenarios")
        assert response.status_code == 200
        
        scenarios = response.json()
        assert len(scenarios) > 0
        assert any(s['scenario'] == 'NZE' for s in scenarios)


# Integration Tests
class TestPipelineIntegration:
    """Test full pipeline integration"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)
    
    def test_full_pipeline_flow(self, temp_dir):
        """Test complete pipeline execution"""
        # This would require sample data files
        # Simplified test to check imports work
        from src.main import EmissionPipeline
        
        pipeline = EmissionPipeline()
        assert pipeline is not None


# Test Utilities
class TestDataQuality:
    """Test data quality checks"""
    
    def test_missing_value_detection(self):
        """Test missing value detection"""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4],
            'b': [5, 6, 7, 8]
        })
        
        missing_pct = df.isnull().sum() / len(df)
        assert missing_pct['a'] == 0.25
        assert missing_pct['b'] == 0.0
    
    def test_outlier_detection(self):
        """Test outlier detection logic"""
        data = np.array([1, 2, 3, 4, 5, 100])
        
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        outliers = (data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR))
        
        assert outliers.sum() == 1  # 100 is an outlier
        assert outliers[-1] == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
