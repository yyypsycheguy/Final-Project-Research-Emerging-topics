"""
Main Pipeline Orchestrator
Executes the complete emission projection and transition risk analysis workflow: ingestion → feature engineering → modeling → reports
"""

import sys
import argparse
from pathlib import Path
from loguru import logger
import yaml
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent))

from ingestion.data_loader import SolarDataIngestion
from analysis.feature_engineering import FeatureEngineering
from modeling.emission_model import EmissionProjectionModel


class EmissionPipeline:
    """Orchestrate the complete emission projection pipeline"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize pipeline"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize components
        self.ingestion = SolarDataIngestion(config_path)
        self.feature_eng = FeatureEngineering(config_path)
        self.model = EmissionProjectionModel(config_path)
        
        logger.info("Pipeline initialized")
    
    def _setup_logging(self):
        """Configure logging"""
        log_config = self.config.get('logging', {})
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Configure logger
        logger.add(
            log_config.get('file', 'logs/pipeline.log'),
            rotation=log_config.get('rotation', '100 MB'),
            retention=log_config.get('retention', '30 days'),
            level=log_config.get('level', 'INFO'),
            format=log_config.get('format', '{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}')
        )
    
    def run_ingestion(self) -> str:
        """Execute data ingestion phase"""
        logger.info("=" * 80)
        logger.info("PHASE 1: DATA INGESTION")
        logger.info("=" * 80)
        
        df = self.ingestion.run_pipeline()
        
        logger.info(f"✓ Ingestion complete: {len(df):,} records processed")
        return "data/processed/solar_data_processed.parquet"
    
    def run_feature_engineering(self, data_path: str) -> str:
        """Execute feature engineering phase"""
        logger.info("=" * 80)
        logger.info("PHASE 2: FEATURE ENGINEERING")
        logger.info("=" * 80)
        
        # Load processed data
        import pandas as pd
        df = pd.read_parquet(data_path)
        
        # Engineer features
        df_features = self.feature_eng.engineer_all_features(df)
        
        # Save
        output_path = "data/processed/solar_data_features.parquet"
        df_features.to_parquet(output_path, index=False)
        
        logger.info(f"✓ Feature engineering complete: {len(df_features.columns)} features created")
        return output_path
    
    def run_modeling(self, data_path: str, scenarios: list = None) -> dict:
        """Execute modeling phase"""
        logger.info("=" * 80)
        logger.info("PHASE 3: EMISSION MODELING")
        logger.info("=" * 80)
        
        if scenarios is None:
            scenarios = ['NZE', 'APS', 'STEPS']
        
        # Load featured data
        import pandas as pd
        df = pd.read_parquet(data_path)
        
        # Create scenario comparison
        df_comparison = self.model.create_scenario_comparison(df)
        
        # Calculate transition risk
        df_risk = self.model.calculate_transition_risk(df_comparison)
        
        # Save results
        proj_path = "data/processed/scenario_projections.parquet"
        risk_path = "data/processed/transition_risk.parquet"
        
        df_comparison.to_parquet(proj_path, index=False)
        df_risk.to_parquet(risk_path, index=False)
        
        # Save models
        self.model.save_models()
        
        logger.info(f"✓ Modeling complete: {len(scenarios)} scenarios analyzed")
        
        return {
            'projections': proj_path,
            'risk': risk_path,
            'scenarios': scenarios
        }
    
    def generate_reports(self, results: dict):
        """Generate summary reports"""
        logger.info("=" * 80)
        logger.info("PHASE 4: REPORT GENERATION")
        logger.info("=" * 80)
        
        import pandas as pd
        
        # Load results
        df_proj = pd.read_parquet(results['projections'])
        df_risk = pd.read_parquet(results['risk'])
        
        # Create summary report
        report = {
            'pipeline_run': datetime.now().isoformat(),
            'total_projections': len(df_proj),
            'scenarios_analyzed': results['scenarios'],
            'regions_covered': df_proj['region'].nunique(),
            'years_projected': sorted(df_proj['year'].unique().tolist()),
            
            'global_summary_2030': self._generate_year_summary(df_proj, 2030),
            'global_summary_2050': self._generate_year_summary(df_proj, 2050),
            
            'risk_summary': {
                'avg_transition_risk': float(df_risk['transition_risk_score'].mean()),
                'high_risk_regions': df_risk.nlargest(5, 'transition_risk_score')[
                    ['region', 'year', 'transition_risk_score']
                ].to_dict(orient='records')
            }
        }
        
        # Save report
        report_path = Path("outputs/reports")
        report_path.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(report_path / "pipeline_summary.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Reports generated and saved to {report_path}")
        
        return report
    
    def _generate_year_summary(self, df: pd.DataFrame, year: int) -> dict:
        """Generate summary for a specific year"""
        year_data = df[df['year'] == year]
        
        summary = {}
        for scenario in ['NZE', 'APS', 'STEPS']:
            scenario_data = year_data[year_data['scenario'] == scenario]
            summary[scenario] = {
                'total_capacity_mw': float(scenario_data['capacity_mw'].sum()),
                'total_generation_mwh': float(scenario_data['generation_mwh'].sum()),
                'total_emissions_avoided_tco2e': float(scenario_data['emissions_avoided_tco2e'].sum())
            }
        
        return summary
    
    def run_full_pipeline(self, scenarios: list = None):
        """Execute complete pipeline"""
        start_time = datetime.now()
        
        logger.info("╔" + "═" * 78 + "╗")
        logger.info("║" + " " * 15 + "SOLAR EMISSION PROJECTION PIPELINE" + " " * 29 + "║")
        logger.info("╚" + "═" * 78 + "╝")
        
        try:
            # Phase 1: Data Ingestion
            processed_data_path = self.run_ingestion()
            
            # Phase 2: Feature Engineering
            featured_data_path = self.run_feature_engineering(processed_data_path)
            
            # Phase 3: Modeling
            results = self.run_modeling(featured_data_path, scenarios)
            
            # Phase 4: Reports
            report = self.generate_reports(results)
            
            # Summary
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.info("=" * 80)
            logger.info("PIPELINE EXECUTION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Duration: {duration:.2f} seconds")
            logger.info(f"Scenarios: {', '.join(results['scenarios'])}")
            logger.info(f"Projections: {report['total_projections']:,}")
            logger.info(f"Regions: {report['regions_covered']}")
            logger.info("=" * 80)
            
            print("\n✓ Pipeline execution successful!")
            print(f"\nResults saved to:")
            print(f"  - Projections: {results['projections']}")
            print(f"  - Risk Analysis: {results['risk']}")
            print(f"  - Summary Report: outputs/reports/pipeline_summary.json")
            
            return results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point with CLI"""
    parser = argparse.ArgumentParser(
        description="Solar Emission Projection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with all scenarios
  python src/main.py --scenario all

  # Run specific scenario
  python src/main.py --scenario NZE

  # Run multiple scenarios
  python src/main.py --scenario NZE APS

  # Run only ingestion
  python src/main.py --phase ingestion
        """
    )
    
    parser.add_argument(
        '--scenario',
        nargs='+',
        choices=['all', 'NZE', 'APS', 'STEPS'],
        default=['all'],
        help='Scenarios to run (default: all)'
    )
    
    parser.add_argument(
        '--phase',
        choices=['ingestion', 'features', 'modeling', 'all'],
        default='all',
        help='Pipeline phase to run (default: all)'
    )
    
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Determine scenarios to run
    if 'all' in args.scenario:
        scenarios = ['NZE', 'APS', 'STEPS']
    else:
        scenarios = args.scenario
    
    # Initialize pipeline
    pipeline = EmissionPipeline(args.config)
    
    # Execute requested phase
    if args.phase == 'all':
        pipeline.run_full_pipeline(scenarios)
    elif args.phase == 'ingestion':
        pipeline.run_ingestion()
    elif args.phase == 'features':
        data_path = "data/processed/solar_data_processed.parquet"
        pipeline.run_feature_engineering(data_path)
    elif args.phase == 'modeling':
        data_path = "data/processed/solar_data_features.parquet"
        pipeline.run_modeling(data_path, scenarios)


if __name__ == "__main__":
    main()
