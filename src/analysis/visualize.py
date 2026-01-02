"""
Visualization Script
Generate publication-quality visualizations for emission projections and risk analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import yaml


class EmissionVisualizer:
    """Create visualizations for emission projections"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize visualizer"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.output_path = Path(self.config['outputs']['visualizations']['path'])
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette('husl')
    
    def plot_scenario_comparison(self, df: pd.DataFrame):
        """Create scenario comparison visualization"""
        fig = go.Figure()
        
        for scenario in ['NZE', 'APS', 'STEPS']:
            scenario_data = df[df['scenario'] == scenario].groupby('year')['emissions_avoided_tco2e'].sum()
            
            fig.add_trace(go.Scatter(
                x=scenario_data.index,
                y=scenario_data.values,
                mode='lines+markers',
                name=scenario,
                line=dict(width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='Global Emissions Avoided by Scenario',
            xaxis_title='Year',
            yaxis_title='Emissions Avoided (tCO2e)',
            hovermode='x unified',
            template='plotly_white',
            font=dict(size=12),
            legend=dict(x=0.02, y=0.98)
        )
        
        output_file = self.output_path / 'scenario_comparison.html'
        fig.write_html(str(output_file))
        print(f"✓ Saved: {output_file}")
        
        return fig
    
    def plot_regional_breakdown(self, df: pd.DataFrame, year: int = 2030):
        """Create regional breakdown visualization"""
        year_data = df[df['year'] == year]
        
        fig = px.bar(
            year_data,
            x='region',
            y='emissions_avoided_tco2e',
            color='scenario',
            barmode='group',
            title=f'{year} Emissions Avoided by Region',
            labels={'emissions_avoided_tco2e': 'Emissions Avoided (tCO2e)'},
            template='plotly_white'
        )
        
        fig.update_layout(
            xaxis_title='Region',
            yaxis_title='Emissions Avoided (tCO2e)',
            font=dict(size=12)
        )
        
        output_file = self.output_path / f'regional_breakdown_{year}.html'
        fig.write_html(str(output_file))
        print(f"✓ Saved: {output_file}")
        
        return fig
    
    def plot_capacity_growth(self, df: pd.DataFrame):
        """Plot capacity growth over time"""
        capacity_data = df.groupby(['year', 'scenario'])['capacity_mw'].sum().reset_index()
        
        fig = px.area(
            capacity_data,
            x='year',
            y='capacity_mw',
            color='scenario',
            title='Projected Solar Capacity Growth',
            labels={'capacity_mw': 'Capacity (MW)'},
            template='plotly_white'
        )
        
        output_file = self.output_path / 'capacity_growth.html'
        fig.write_html(str(output_file))
        print(f"✓ Saved: {output_file}")
        
        return fig
    
    def plot_risk_heatmap(self, df_risk: pd.DataFrame):
        """Create risk heatmap"""
        # Pivot for heatmap
        heatmap_data = df_risk.pivot_table(
            index='region',
            columns='year',
            values='transition_risk_score'
        )
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn_r',
            cbar_kws={'label': 'Transition Risk Score'},
            ax=ax
        )
        
        ax.set_title('Transition Risk Score by Region and Year', fontsize=14, pad=20)
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Region', fontsize=12)
        
        plt.tight_layout()
        
        output_file = self.output_path / 'risk_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {output_file}")
        
        plt.close()
    
    def plot_stranded_assets(self, df_risk: pd.DataFrame):
        """Plot stranded asset exposure"""
        exposure_data = df_risk.groupby('year')['stranded_asset_exposure'].sum().reset_index()
        
        fig = px.area(
            exposure_data,
            x='year',
            y='stranded_asset_exposure',
            title='Global Stranded Asset Exposure Over Time',
            labels={'stranded_asset_exposure': 'Stranded Asset Exposure (tCO2e)'},
            color_discrete_sequence=['#d62728'],
            template='plotly_white'
        )
        
        output_file = self.output_path / 'stranded_assets.html'
        fig.write_html(str(output_file))
        print(f"✓ Saved: {output_file}")
        
        return fig
    
    def create_dashboard(self, df_proj: pd.DataFrame, df_risk: pd.DataFrame):
        """Create comprehensive dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Global Emissions Avoided',
                '2030 Regional Breakdown',
                'Capacity Growth',
                'Average Transition Risk'
            ),
            specs=[
                [{"type": "scatter"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Plot 1: Global emissions
        for scenario in ['NZE', 'APS', 'STEPS']:
            data = df_proj[df_proj['scenario'] == scenario].groupby('year')['emissions_avoided_tco2e'].sum()
            fig.add_trace(
                go.Scatter(x=data.index, y=data.values, name=scenario, mode='lines+markers'),
                row=1, col=1
            )
        
        # Plot 2: Regional breakdown 2030
        year_2030 = df_proj[df_proj['year'] == 2030]
        for scenario in ['NZE', 'APS', 'STEPS']:
            data = year_2030[year_2030['scenario'] == scenario]
            fig.add_trace(
                go.Bar(x=data['region'], y=data['emissions_avoided_tco2e'], name=scenario),
                row=1, col=2
            )
        
        # Plot 3: Capacity growth
        for scenario in ['NZE', 'APS', 'STEPS']:
            data = df_proj[df_proj['scenario'] == scenario].groupby('year')['capacity_mw'].sum()
            fig.add_trace(
                go.Scatter(x=data.index, y=data.values, name=scenario, mode='lines'),
                row=2, col=1
            )
        
        # Plot 4: Risk over time
        risk_data = df_risk.groupby('year')['transition_risk_score'].mean()
        fig.add_trace(
            go.Scatter(x=risk_data.index, y=risk_data.values, name='Avg Risk', mode='lines+markers', line=dict(color='red')),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Solar Emission Projection Dashboard",
            showlegend=True,
            template='plotly_white'
        )
        
        output_file = self.output_path / 'dashboard.html'
        fig.write_html(str(output_file))
        print(f"✓ Saved: {output_file}")
        
        return fig
    
    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Generating visualizations...")
        
        # Load data
        df_proj = pd.read_parquet('data/processed/scenario_projections.parquet')
        df_risk = pd.read_parquet('data/processed/transition_risk.parquet')
        
        # Generate plots
        self.plot_scenario_comparison(df_proj)
        self.plot_regional_breakdown(df_proj, 2030)
        self.plot_regional_breakdown(df_proj, 2050)
        self.plot_capacity_growth(df_proj)
        self.plot_risk_heatmap(df_risk)
        self.plot_stranded_assets(df_risk)
        self.create_dashboard(df_proj, df_risk)
        
        print(f"\n✓ All visualizations saved to {self.output_path}")


if __name__ == "__main__":
    visualizer = EmissionVisualizer()
    visualizer.generate_all_visualizations()
