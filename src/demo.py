"""
Demo Script - Solar Emission Projection Pipeline
Demonstrates pipeline capabilities with sample execution
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def demo_data_ingestion():
    """Demonstrate data ingestion"""
    print_banner("DEMO: Data Ingestion")
    
    from src.ingestion.data_loader import SolarDataIngestion
    
    print("Loading Global Solar Power Tracker data...")
    ingestion = SolarDataIngestion()
    
    # Load data
    df_large, df_small = ingestion.load_data()
    print(f"✓ Loaded {len(df_large):,} large-scale projects")
    print(f"✓ Loaded {len(df_small):,} small-scale projects")
    
    # Combine
    df_combined = ingestion.combine_datasets(df_large, df_small)
    print(f"✓ Combined: {len(df_combined):,} total projects")
    
    # Validate
    df_validated = ingestion.validate_data(df_combined)
    print(f"✓ Validated: {len(df_validated):,} projects passed filters")
    
    # Clean
    df_clean = ingestion.clean_data(df_validated)
    print(f"✓ Cleaned: {len(df_clean.columns)} columns ready")
    
    # Quick stats
    print("\nQuick Statistics:")
    print(f"  Total Capacity: {df_clean['capacity_(mw)'].sum():,.0f} MW")
    print(f"  Average Project Size: {df_clean['capacity_(mw)'].mean():.1f} MW")
    print(f"  Countries: {df_clean['country_area'].nunique()}")
    print(f"  Status Distribution:")
    for status, count in df_clean['status'].value_counts().head(5).items():
        print(f"    {status}: {count:,}")
    
    return df_clean


def demo_feature_engineering(df):
    """Demonstrate feature engineering"""
    print_banner("DEMO: Feature Engineering")
    
    from src.analysis.feature_engineering import FeatureEngineering
    
    print("Creating features...")
    engineer = FeatureEngineering()
    
    # Create features
    df_features = engineer.engineer_all_features(df)
    
    print(f"✓ Original columns: {len(df.columns)}")
    print(f"✓ After engineering: {len(df_features.columns)}")
    print(f"✓ New features created: {len(df_features.columns) - len(df.columns)}")
    
    # Show sample features
    print("\nSample Engineered Features:")
    feature_cols = [
        'years_operational', 'age_category', 'capacity_category',
        'stranded_asset_risk', 'annual_emissions_avoided', 'NZE_growth_rate'
    ]
    
    for col in feature_cols:
        if col in df_features.columns:
            if df_features[col].dtype in ['float64', 'int64']:
                print(f"  {col}: {df_features[col].mean():.2f} (avg)")
            else:
                print(f"  {col}: {df_features[col].value_counts().index[0]} (most common)")
    
    return df_features


def demo_modeling(df):
    """Demonstrate emission modeling"""
    print_banner("DEMO: Emission Modeling")
    
    from src.modeling.emission_model import EmissionProjectionModel
    
    print("Training models for all scenarios...")
    model = EmissionProjectionModel()
    
    # Create scenario comparison
    print("Running scenario analysis...")
    df_comparison = model.create_scenario_comparison(
        df, 
        years=[2025, 2030, 2040, 2050]
    )
    
    print(f"✓ Generated {len(df_comparison):,} projections")
    print(f"✓ Scenarios: {', '.join(df_comparison['scenario'].unique())}")
    print(f"✓ Regions: {df_comparison['region'].nunique()}")
    print(f"✓ Years: {', '.join(map(str, sorted(df_comparison['year'].unique())))}")
    
    # Show 2030 comparison
    print("\n2030 Global Emissions Avoided (tCO2e):")
    year_2030 = df_comparison[df_comparison['year'] == 2030]
    for scenario in ['NZE', 'APS', 'STEPS']:
        total = year_2030[year_2030['scenario'] == scenario]['emissions_avoided_tco2e'].sum()
        print(f"  {scenario:6s}: {total:>15,.0f}")
    
    # Calculate transition risk
    print("\nCalculating transition risk metrics...")
    df_risk = model.calculate_transition_risk(df_comparison)
    
    print(f"✓ Risk metrics calculated for {len(df_risk):,} region-year combinations")
    
    # Show high-risk regions
    print("\nTop 5 High-Risk Regions (2030):")
    high_risk_2030 = df_risk[df_risk['year'] == 2030].nlargest(5, 'transition_risk_score')
    for _, row in high_risk_2030.iterrows():
        print(f"  {row['region']:20s}: {row['transition_risk_score']:.3f}")
    
    return df_comparison, df_risk


def demo_api():
    """Demonstrate API capabilities"""
    print_banner("DEMO: API Usage")
    
    print("Starting API server...")
    print("API Documentation: http://localhost:8000/docs")
    print("\nExample API Calls:")
    
    examples = [
        ("Health Check", 
         "curl http://localhost:8000/health"),
        
        ("Get Scenarios", 
         "curl http://localhost:8000/api/v1/scenarios"),
        
        ("2030 NZE Projections", 
         'curl "http://localhost:8000/api/v1/projections?scenario=NZE&year=2030"'),
        
        ("Compare Scenarios", 
         'curl "http://localhost:8000/api/v1/projections/compare?year=2030"'),
        
        ("Risk Metrics", 
         'curl "http://localhost:8000/api/v1/risk?year=2030"'),
    ]
    
    for name, cmd in examples:
        print(f"\n{name}:")
        print(f"  {cmd}")


def demo_results():
    """Show final results"""
    print_banner("DEMO: Results Summary")
    
    # Load results
    print("Loading pipeline results...")
    
    try:
        df_proj = pd.read_parquet('data/processed/scenario_projections.parquet')
        df_risk = pd.read_parquet('data/processed/transition_risk.parquet')
        
        print("✓ Results loaded successfully\n")
        
        # Summary statistics
        print("Pipeline Execution Summary:")
        print(f"  Total projections: {len(df_proj):,}")
        print(f"  Scenarios analyzed: {df_proj['scenario'].nunique()}")
        print(f"  Regions covered: {df_proj['region'].nunique()}")
        print(f"  Years projected: {df_proj['year'].nunique()}")
        
        print("\nGlobal Totals by Scenario (2050):")
        year_2050 = df_proj[df_proj['year'] == 2050]
        for scenario in ['NZE', 'APS', 'STEPS']:
            scenario_data = year_2050[year_2050['scenario'] == scenario]
            capacity = scenario_data['capacity_mw'].sum()
            emissions = scenario_data['emissions_avoided_tco2e'].sum()
            print(f"  {scenario}:")
            print(f"    Capacity: {capacity:>15,.0f} MW")
            print(f"    Emissions Avoided: {emissions:>15,.0f} tCO2e")
        
        print("\nTransition Risk Overview:")
        print(f"  Average risk score: {df_risk['transition_risk_score'].mean():.3f}")
        print(f"  High-risk regions: {(df_risk['transition_risk_score'] > 0.5).sum()}")
        print(f"  Total stranded assets: {df_risk['stranded_asset_exposure'].sum():,.0f} tCO2e")
        
        print("\nOutput Files Generated:")
        output_files = [
            "data/processed/solar_data_processed.parquet",
            "data/processed/solar_data_features.parquet",
            "data/processed/scenario_projections.parquet",
            "data/processed/transition_risk.parquet",
            "data/models/emission_model_NZE.pkl",
            "data/models/emission_model_APS.pkl",
            "data/models/emission_model_STEPS.pkl",
        ]
        
        for file in output_files:
            if Path(file).exists():
                size = Path(file).stat().st_size / (1024**2)  # MB
                print(f"  ✓ {file} ({size:.2f} MB)")
            else:
                print(f"  ✗ {file} (not found)")
        
    except FileNotFoundError:
        print("⚠ Results not found. Please run the full pipeline first:")
        print("  python src/main.py --scenario all")


def main():
    """Run complete demo"""
    start_time = datetime.now()
    
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "SOLAR EMISSION PROJECTION PIPELINE DEMO" + " " * 24 + "║")
    print("╚" + "═" * 78 + "╝")
    
    try:
        # Demo each component
        df_clean = demo_data_ingestion()
        df_features = demo_feature_engineering(df_clean)
        df_comparison, df_risk = demo_modeling(df_features)
        demo_api()
        demo_results()
        
        # Final summary
        duration = (datetime.now() - start_time).total_seconds()
        
        print_banner("DEMO COMPLETE")
        print(f"Total execution time: {duration:.2f} seconds")
        print("\nNext Steps:")
        print("  1. Run full pipeline: python src/main.py --scenario all")
        print("  2. Start API server: python src/api/app.py")
        print("  3. Open Jupyter notebook: jupyter notebook notebooks/01_exploratory_analysis.ipynb")
        print("  4. View API docs: http://localhost:8000/docs")
        print("\nFor more information, see QUICKSTART.md\n")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        print("Please ensure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        raise


if __name__ == "__main__":
    main()
