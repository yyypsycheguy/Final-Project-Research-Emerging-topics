"""
Emission Projection Model
Forecasts emissions under different IEA scenarios (NZE, APS, STEPS)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import joblib
from pathlib import Path
from loguru import logger
import yaml


class EmissionProjectionModel:
    """Build and deploy emission projection models"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize model with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['modeling']
        self.scenarios = self.config['scenarios']
        self.models = {}
        
    def prepare_features(self, df: pd.DataFrame, scenario: str = 'NZE') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        logger.info(f"Preparing features for {scenario} scenario...")
        
        # Define feature sets
        base_features = [
            'capacity_(mw)', 'log_capacity', 'years_operational',
            'latitude', 'longitude', 'solar_resource_proxy',
            'age_risk_score', 'tech_risk_score', 'market_risk_score',
            'lifecycle_emissions_factor', 'grid_intensity',
            'capacity_factor', 'annual_generation_mwh'
        ]
        
        # Add scenario-specific features
        scenario_features = [
            f'{scenario}_growth_rate',
            f'{scenario}_carbon_price_2030',
            f'{scenario}_competitive_score'
        ]
        
        # Categorical features to encode
        categorical_features = ['age_category', 'capacity_category', 'climate_zone', 'region_group']
        
        # Select features that exist
        available_features = [f for f in base_features + scenario_features if f in df.columns]
        available_categorical = [f for f in categorical_features if f in df.columns]
        
        # Create feature matrix
        X = df[available_features].copy()
        
        # One-hot encode categorical features
        for cat_feat in available_categorical:
            dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
        
        # Target: annual emissions avoided
        y = df['annual_emissions_avoided'].copy()
        
        # Remove rows with missing target
        valid_idx = ~y.isna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Features prepared: {X.shape[1]} features, {len(X):,} samples")
        
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'xgboost') -> object:
        """Train emission projection model"""
        logger.info(f"Training {model_type} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.model_config['test_size'],
            random_state=self.model_config['random_state']
        )
        
        # Initialize model
        if model_type == 'xgboost':
            params = self.model_config['xgboost']
            model = xgb.XGBRegressor(**params)
        
        elif model_type == 'lightgbm':
            if not LIGHTGBM_AVAILABLE:
                raise ValueError("LightGBM is not available. Install it or use 'xgboost' or 'random_forest'.")
            params = self.model_config['lightgbm']
            model = lgb.LGBMRegressor(**params)
        
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                random_state=self.model_config['random_state'],
                n_jobs=-1
            )
        
        elif model_type == 'gradient_boost':
            model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.05,
                random_state=self.model_config['random_state']
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=self.model_config['cv_folds'],
            scoring='r2',
            n_jobs=-1
        )
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
        
        logger.info(f"Model trained: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
        
        return model, metrics
    
    def evaluate_model(self, model: object, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        return metrics
    
    def project_emissions(
        self, 
        df: pd.DataFrame, 
        model: object, 
        scenario: str,
        year: int
    ) -> pd.DataFrame:
        """Project emissions for a specific year and scenario"""
        logger.info(f"Projecting emissions for {year} under {scenario} scenario...")
        
        df_proj = df.copy()
        scenario_config = self.scenarios[scenario]
        
        # Adjust capacity based on growth rate
        years_from_base = year - self.model_config['base_year']
        growth_rate = scenario_config['solar_growth_rate']
        
        # Apply growth adjustment
        df_proj['projected_capacity'] = (
            df_proj['capacity_(mw)'] * 
            (1 + growth_rate) ** years_from_base
        )
        
        # Adjust for retirement
        retirement_prob = df_proj['retirement_probability']
        retirement_adjustment = 1 - (retirement_prob * years_from_base / 25)
        df_proj['projected_capacity'] *= retirement_adjustment.clip(0, 1)
        
        # Recalculate annual generation
        df_proj['projected_generation'] = (
            df_proj['projected_capacity'] * 8760 * df_proj['capacity_factor']
        )
        
        # Project emissions avoided
        df_proj['projected_emissions_avoided'] = (
            df_proj['projected_generation'] * 
            df_proj['emission_reduction_factor'] / 1000
        )
        
        # Aggregate by region
        regional_projections = df_proj.groupby('region_group').agg({
            'projected_capacity': 'sum',
            'projected_generation': 'sum',
            'projected_emissions_avoided': 'sum'
        }).round(2)
        
        return regional_projections
    
    def create_scenario_comparison(
        self, 
        df: pd.DataFrame, 
        years: List[int] = None
    ) -> pd.DataFrame:
        """Compare emissions across all scenarios"""
        if years is None:
            years = list(range(2025, 2051, 5))
        
        logger.info("Creating scenario comparison...")
        
        results = []
        
        for scenario in ['NZE', 'APS', 'STEPS']:
            # Prepare features and train model
            X, y = self.prepare_features(df, scenario)
            model, metrics = self.train_model(X, y, model_type='xgboost')
            
            # Store model
            self.models[scenario] = {'model': model, 'metrics': metrics}
            
            # Project for each year
            for year in years:
                projections = self.project_emissions(df, model, scenario, year)
                
                for region, row in projections.iterrows():
                    results.append({
                        'scenario': scenario,
                        'year': year,
                        'region': region,
                        'capacity_mw': row['projected_capacity'],
                        'generation_mwh': row['projected_generation'],
                        'emissions_avoided_tco2e': row['projected_emissions_avoided']
                    })
        
        df_comparison = pd.DataFrame(results)
        
        logger.info(f"Scenario comparison created: {len(df_comparison):,} projections")
        
        return df_comparison
    
    def calculate_transition_risk(self, df_comparison: pd.DataFrame) -> pd.DataFrame:
        """Calculate transition risk metrics"""
        logger.info("Calculating transition risk metrics...")
        
        # Pivot to get scenarios side by side
        pivot = df_comparison.pivot_table(
            index=['year', 'region'],
            columns='scenario',
            values='emissions_avoided_tco2e'
        ).reset_index()
        
        # Calculate scenario divergence
        pivot['nze_aps_gap'] = pivot['NZE'] - pivot['APS']
        pivot['aps_steps_gap'] = pivot['APS'] - pivot['STEPS']
        pivot['nze_steps_gap'] = pivot['NZE'] - pivot['STEPS']
        
        # Risk scores (higher divergence = higher transition risk)
        pivot['transition_risk_score'] = (
            np.abs(pivot['nze_steps_gap']) / pivot['NZE']
        ).clip(0, 1)
        
        # Policy risk (based on gap between APS and STEPS)
        pivot['policy_risk_score'] = (
            np.abs(pivot['aps_steps_gap']) / pivot['APS']
        ).clip(0, 1)
        
        # Stranded asset exposure
        pivot['stranded_asset_exposure'] = (
            pivot['STEPS'] - pivot['NZE']
        ).clip(lower=0)
        
        logger.info("Transition risk metrics calculated")
        
        return pivot
    
    def save_models(self, output_dir: str = "data/models"):
        """Save trained models and metadata"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for scenario, model_data in self.models.items():
            # Save model
            model_file = output_path / f"emission_model_{scenario}.pkl"
            joblib.dump(model_data['model'], model_file)
            
            # Save metrics
            metrics_file = output_path / f"metrics_{scenario}.yaml"
            with open(metrics_file, 'w') as f:
                yaml.dump(model_data['metrics'], f)
            
            logger.info(f"Saved {scenario} model to {model_file}")
    
    def load_model(self, scenario: str, model_dir: str = "data/models") -> object:
        """Load a trained model"""
        model_path = Path(model_dir) / f"emission_model_{scenario}.pkl"
        model = joblib.load(model_path)
        logger.info(f"Loaded {scenario} model from {model_path}")
        return model


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/modeling.log", rotation="10 MB")
    
    # Load featured data
    df = pd.read_parquet("data/processed/solar_data_features.parquet")
    
    # Initialize model
    emission_model = EmissionProjectionModel()
    
    # Create scenario comparison
    df_comparison = emission_model.create_scenario_comparison(df)
    
    # Calculate transition risk
    df_risk = emission_model.calculate_transition_risk(df_comparison)
    
    # Save results
    df_comparison.to_parquet("data/processed/scenario_projections.parquet")
    df_risk.to_parquet("data/processed/transition_risk.parquet")
    
    # Save models
    emission_model.save_models()
    
    print("\n✓ Modeling complete")
    print(f"Scenarios analyzed: {df_comparison['scenario'].nunique()}")
    print(f"Regions covered: {df_comparison['region'].nunique()}")
    print(f"Years projected: {df_comparison['year'].nunique()}")
