"""
Data Ingestion Module
Handles loading, validation, and initial processing of Global Solar Power Tracker data
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from loguru import logger
import yaml


class SolarDataIngestion:
    """Load and validate Global Solar Power Tracker data"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_data_path = self.config['data']['raw_data_path']
        self.processed_data_path = self.config['data']['processed_data_path']
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both large and small scale solar data"""
        logger.info(f"Loading data from {self.raw_data_path}")
        
        try:
            # Load large scale projects (>= 20 MW)
            df_large = pd.read_excel(
                self.raw_data_path,
                sheet_name=self.config['data']['sheets']['large_scale']
            )
            logger.info(f"Loaded {len(df_large):,} large-scale projects")
            
            # Load small scale projects (1-20 MW)
            df_small = pd.read_excel(
                self.raw_data_path,
                sheet_name=self.config['data']['sheets']['small_scale']
            )
            logger.info(f"Loaded {len(df_small):,} small-scale projects")
            
            return df_large, df_small
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def combine_datasets(self, df_large: pd.DataFrame, df_small: pd.DataFrame) -> pd.DataFrame:
        """Combine large and small scale datasets"""
        logger.info("Combining datasets...")
        
        # Add scale identifier
        df_large['scale'] = 'large'
        df_small['scale'] = 'small'
        
        # Combine
        df_combined = pd.concat([df_large, df_small], ignore_index=True)
        logger.info(f"Combined dataset: {len(df_combined):,} total projects")
        
        return df_combined
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality and apply filters"""
        logger.info("Validating data...")
        
        initial_rows = len(df)
        validation = self.config['data']['validation']
        
        # Check capacity range
        df = df[
            (df['Capacity (MW)'] >= validation['min_capacity_mw']) &
            (df['Capacity (MW)'] <= validation['max_capacity_mw'])
        ]
        logger.info(f"Capacity filter: {initial_rows - len(df):,} rows removed")
        
        # Check year range
        year_min, year_max = validation['valid_years_range']
        df = df[
            (df['Start year'].isna()) | 
            ((df['Start year'] >= year_min) & (df['Start year'] <= year_max))
        ]
        logger.info(f"Year filter: {initial_rows - len(df):,} rows removed")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['GEM phase ID'], keep='first')
        logger.info(f"Duplicate removal: {initial_rows - len(df):,} rows removed")
        
        # Log data quality metrics
        self._log_data_quality(df)
        
        return df
    
    def _log_data_quality(self, df: pd.DataFrame):
        """Log data quality metrics"""
        logger.info("Data Quality Report:")
        logger.info(f"  Total records: {len(df):,}")
        logger.info(f"  Missing capacity: {df['Capacity (MW)'].isna().sum():,}")
        logger.info(f"  Missing start year: {df['Start year'].isna().sum():,}")
        logger.info(f"  Missing country: {df['Country/Area'].isna().sum():,}")
        logger.info(f"  Missing coordinates: {df[['Latitude', 'Longitude']].isna().any(axis=1).sum():,}")
        
        # Status distribution
        logger.info("\nStatus Distribution:")
        status_counts = df['Status'].value_counts()
        for status, count in status_counts.items():
            logger.info(f"  {status}: {count:,} ({count/len(df)*100:.1f}%)")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        logger.info("Cleaning data...")
        
        df_clean = df.copy()
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('/', '_')
        
        # Handle missing values
        df_clean['start_year'] = df_clean['start_year'].fillna(df_clean['start_year'].median())
        df_clean['technology_type'] = df_clean['technology_type'].fillna('Unknown')
        
        # Standardize status categories
        status_mapping = {
            'operating': 'Operational',
            'operational': 'Operational',
            'construction': 'Under Construction',
            'announced': 'Planned',
            'pre-construction': 'Planned',
            'shelved': 'Delayed/Cancelled',
            'cancelled': 'Delayed/Cancelled',
            'retired': 'Retired'
        }
        df_clean['status'] = df_clean['status'].str.lower().map(status_mapping).fillna(df_clean['status'])
        
        # Parse dates
        if 'date_last_researched' in df_clean.columns:
            df_clean['date_last_researched'] = pd.to_datetime(df_clean['date_last_researched'], errors='coerce')
        
        logger.info(f"Data cleaned: {len(df_clean):,} rows, {len(df_clean.columns)} columns")
        
        return df_clean
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "solar_data_processed.parquet"):
        """Save processed data to parquet format for efficient storage"""
        output_path = Path(self.processed_data_path) / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Saved processed data to {output_path}")
        
        return str(output_path)
    
    def run_pipeline(self) -> pd.DataFrame:
        """Execute complete ingestion pipeline"""
        logger.info("Starting data ingestion pipeline...")
        
        # Load data
        df_large, df_small = self.load_data()
        
        # Combine datasets
        df_combined = self.combine_datasets(df_large, df_small)
        
        # Validate data
        df_validated = self.validate_data(df_combined)
        
        # Clean data
        df_clean = self.clean_data(df_validated)
        
        # Save processed data
        self.save_processed_data(df_clean)
        
        logger.info("Data ingestion pipeline completed successfully")
        
        return df_clean


if __name__ == "__main__":
    # Configure logging
    logger.add("logs/ingestion.log", rotation="10 MB")
    
    # Run ingestion
    ingestion = SolarDataIngestion()
    df = ingestion.run_pipeline()
    
    print(f"\n✓ Ingestion complete: {len(df):,} records processed")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
