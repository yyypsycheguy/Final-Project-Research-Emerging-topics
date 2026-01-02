"""
Feature Engineering Module
Creates features for emission modeling and transition risk analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from loguru import logger
import yaml


class FeatureEngineering:
    """Engineer features for emission projection models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_year = self.config['modeling']['base_year']
        self.projection_horizon = self.config['modeling']['projection_horizon']
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        logger.info("Creating temporal features...")
        
        df_feat = df.copy()
        
        # Years operational
        df_feat['years_operational'] = self.base_year - df_feat['start_year']
        df_feat['years_operational'] = df_feat['years_operational'].clip(lower=0)
        
        # Years until projection horizon
        df_feat['years_remaining'] = self.projection_horizon - self.base_year
        
        # Age categories
        df_feat['age_category'] = pd.cut(
            df_feat['years_operational'],
            bins=[-1, 5, 10, 15, 20, 100],
            labels=['New', 'Young', 'Mature', 'Aging', 'Old']
        )
        
        # Decade of construction
        df_feat['construction_decade'] = (df_feat['start_year'] // 10) * 10
        
        # Expected retirement year (assuming 25-year lifespan)
        df_feat['expected_retirement_year'] = df_feat['start_year'] + 25
        
        # Time to retirement
        df_feat['years_to_retirement'] = df_feat['expected_retirement_year'] - self.base_year
        df_feat['years_to_retirement'] = df_feat['years_to_retirement'].clip(lower=0)
        
        logger.info(f"Created {6} temporal features")
        return df_feat
    
    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create location-based features"""
        logger.info("Creating geographic features...")
        
        df_feat = df.copy()
        
        # Regional groupings
        region_mapping = {
            'China': 'Asia',
            'India': 'Asia',
            'United States': 'North America',
            'Brazil': 'South America',
            'Australia': 'Oceania'
        }
        df_feat['region_group'] = df_feat['country_area'].map(region_mapping).fillna('Other')
        
        # Climate zone (based on latitude)
        df_feat['climate_zone'] = pd.cut(
            df_feat['latitude'].abs(),
            bins=[0, 23.5, 35, 66.5, 90],
            labels=['Tropical', 'Subtropical', 'Temperate', 'Polar']
        )
        
        # Hemisphere
        df_feat['hemisphere'] = np.where(df_feat['latitude'] >= 0, 'Northern', 'Southern')
        
        # Solar irradiance proxy (simplified based on latitude)
        df_feat['solar_resource_proxy'] = 1 - (df_feat['latitude'].abs() / 90)
        
        # Coastal vs inland (simplified - within 100km of coast approximation)
        # This is a placeholder - would need actual coastal distance data
        df_feat['coastal_proximity'] = 'Unknown'
        
        logger.info(f"Created {5} geographic features")
        return df_feat
    
    def create_capacity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create capacity and scale features"""
        logger.info("Creating capacity features...")
        
        df_feat = df.copy()
        
        # Capacity categories
        df_feat['capacity_category'] = pd.cut(
            df_feat['capacity_(mw)'],
            bins=[0, 10, 50, 100, 500, 10000],
            labels=['Small', 'Medium', 'Large', 'Very Large', 'Utility Scale']
        )
        
        # Log capacity for modeling
        df_feat['log_capacity'] = np.log1p(df_feat['capacity_(mw)'])
        
        # Capacity per project (for projects with multiple phases)
        capacity_by_project = df_feat.groupby('project_name')['capacity_(mw)'].transform('sum')
        df_feat['total_project_capacity'] = capacity_by_project
        df_feat['phase_capacity_ratio'] = df_feat['capacity_(mw)'] / df_feat['total_project_capacity']
        
        # Technology efficiency proxy
        tech_efficiency = {
            'Solar PV': 1.0,
            'Solar Thermal': 0.85,
            'Concentrated Solar Power': 0.90,
            'Unknown': 0.95
        }
        df_feat['tech_efficiency_proxy'] = df_feat['technology_type'].map(tech_efficiency).fillna(0.95)
        
        logger.info(f"Created {5} capacity features")
        return df_feat
    
    def create_risk_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create transition risk indicators"""
        logger.info("Creating risk features...")
        
        df_feat = df.copy()
        risk_config = self.config['risk_assessment']['stranded_asset_risk']
        
        # Age-based risk
        age_threshold = risk_config['age_threshold_years']
        df_feat['age_risk_score'] = (df_feat['years_operational'] / age_threshold).clip(0, 1)
        
        # Policy risk (based on region and technology)
        # High policy risk for older, less efficient technologies in regulated markets
        policy_risk_map = {
            'North America': 0.3,
            'Europe': 0.2,
            'Asia': 0.4,
            'South America': 0.5,
            'Other': 0.6
        }
        df_feat['policy_risk_score'] = df_feat['region_group'].map(policy_risk_map).fillna(0.5)
        
        # Technology risk
        tech_risk_map = {
            'Solar PV': 0.1,  # Low risk - modern technology
            'Solar Thermal': 0.4,  # Higher risk - less common
            'Concentrated Solar Power': 0.3,
            'Unknown': 0.5
        }
        df_feat['tech_risk_score'] = df_feat['technology_type'].map(tech_risk_map).fillna(0.5)
        
        # Market risk (based on capacity utilization proxy)
        # Larger, newer projects assumed to have better utilization
        df_feat['market_risk_score'] = 1 - (
            (df_feat['log_capacity'] / df_feat['log_capacity'].max()) * 
            (1 - df_feat['age_risk_score'])
        ).clip(0, 1)
        
        # Composite stranded asset risk
        weights = risk_config
        df_feat['stranded_asset_risk'] = (
            df_feat['policy_risk_score'] * weights['policy_risk_weight'] +
            df_feat['tech_risk_score'] * weights['technology_risk_weight'] +
            df_feat['market_risk_score'] * weights['market_risk_weight']
        )
        
        # Retirement probability (based on age and risk)
        df_feat['retirement_probability'] = (
            df_feat['age_risk_score'] * 0.5 +
            df_feat['stranded_asset_risk'] * 0.5
        ).clip(0, 1)
        
        logger.info(f"Created {7} risk features")
        return df_feat
    
    def create_emission_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create emission-related features"""
        logger.info("Creating emission features...")
        
        df_feat = df.copy()
        emission_factors = self.config['emission_factors']
        
        # Lifecycle emissions (kgCO2e/MWh)
        tech_emissions = {
            'Solar PV': emission_factors['solar_pv_lifecycle'],
            'Solar Thermal': emission_factors['solar_thermal_lifecycle'],
            'Concentrated Solar Power': emission_factors['solar_thermal_lifecycle'],
            'Unknown': emission_factors['solar_pv_lifecycle']
        }
        df_feat['lifecycle_emissions_factor'] = df_feat['technology_type'].map(tech_emissions)
        
        # Grid displacement emissions (based on region)
        grid_intensity = emission_factors['grid_intensity']
        region_grid_map = {
            'Asia': grid_intensity['non_oecd'],
            'North America': grid_intensity['oecd'],
            'Europe': grid_intensity['eu'],
            'South America': grid_intensity['non_oecd'],
            'Other': grid_intensity['global_average']
        }
        df_feat['grid_intensity'] = df_feat['region_group'].map(region_grid_map)
        
        # Net emission reduction potential
        df_feat['emission_reduction_factor'] = (
            df_feat['grid_intensity'] - df_feat['lifecycle_emissions_factor']
        )
        
        # Annual generation estimate (MWh/year)
        # Assuming capacity factor based on solar resource
        df_feat['capacity_factor'] = 0.15 + (df_feat['solar_resource_proxy'] * 0.15)
        df_feat['annual_generation_mwh'] = (
            df_feat['capacity_(mw)'] * 8760 * df_feat['capacity_factor']
        )
        
        # Annual emissions avoided (tCO2e)
        df_feat['annual_emissions_avoided'] = (
            df_feat['annual_generation_mwh'] * 
            df_feat['emission_reduction_factor'] / 1000  # Convert kg to tonnes
        )
        
        logger.info(f"Created {6} emission features")
        return df_feat
    
    def create_scenario_features(self, df: pd.DataFrame, scenario: str = 'NZE') -> pd.DataFrame:
        """Create scenario-specific features"""
        logger.info(f"Creating {scenario} scenario features...")
        
        df_feat = df.copy()
        scenario_config = self.config['scenarios'][scenario]
        
        # Growth rate adjustments
        df_feat[f'{scenario}_growth_rate'] = scenario_config['solar_growth_rate']
        
        # Carbon price impact
        carbon_price_2030 = scenario_config['carbon_price_2030']
        df_feat[f'{scenario}_carbon_price_2030'] = carbon_price_2030
        
        # Retirement risk adjustment
        retirement_rate = (
            scenario_config.get('coal_retirement_rate', 0) +
            scenario_config.get('gas_retirement_rate', 0)
        ) / 2
        df_feat[f'{scenario}_retirement_adjustment'] = retirement_rate
        
        # Competitive advantage score
        df_feat[f'{scenario}_competitive_score'] = (
            df_feat['emission_reduction_factor'] / df_feat['grid_intensity'] *
            (1 + scenario_config['electrification_rate'])
        )
        
        logger.info(f"Created {4} {scenario} scenario features")
        return df_feat
    
    def engineer_all_features(self, df: pd.DataFrame, scenarios: List[str] = None) -> pd.DataFrame:
        """Apply all feature engineering steps"""
        logger.info("Starting comprehensive feature engineering...")
        
        if scenarios is None:
            scenarios = ['NZE', 'APS', 'STEPS']
        
        df_features = df.copy()
        
        # Apply feature engineering in sequence
        df_features = self.create_temporal_features(df_features)
        df_features = self.create_geographic_features(df_features)
        df_features = self.create_capacity_features(df_features)
        df_features = self.create_risk_features(df_features)
        df_features = self.create_emission_features(df_features)
        
        # Apply scenario-specific features
        for scenario in scenarios:
            df_features = self.create_scenario_features(df_features, scenario)
        
        logger.info(f"Feature engineering complete: {len(df_features.columns)} total features")
        
        return df_features
    
    def get_feature_importance_report(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate feature statistics for validation"""
        numeric_features = df.select_dtypes(include=[np.number]).columns
        
        report = pd.DataFrame({
            'feature': numeric_features,
            'missing_pct': [df[col].isna().mean() * 100 for col in numeric_features],
            'mean': [df[col].mean() for col in numeric_features],
            'std': [df[col].std() for col in numeric_features],
            'min': [df[col].min() for col in numeric_features],
            'max': [df[col].max() for col in numeric_features]
        })
        
        return report.sort_values('missing_pct', ascending=False)


if __name__ == "__main__":
    # Test feature engineering
    logger.add("logs/feature_engineering.log", rotation="10 MB")
    
    # Load processed data
    df = pd.read_parquet("data/processed/solar_data_processed.parquet")
    
    # Engineer features
    engineer = FeatureEngineering()
    df_features = engineer.engineer_all_features(df)
    
    # Save
    df_features.to_parquet("data/processed/solar_data_features.parquet")
    
    # Report
    report = engineer.get_feature_importance_report(df_features)
    print("\nFeature Engineering Summary:")
    print(f"Total features: {len(df_features.columns)}")
    print(f"\nTop features by completeness:")
    print(report.head(10))
