"""
Comprehensive EDA Visualization Suite for EV Charging Research

This module provides all required visualizations for understanding
EV charging patterns, vehicle characteristics, and temporal trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for high-quality research plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class EVChargingVisualizer:
    """
    Comprehensive visualization suite for EV charging research.
    
    Provides all required visualizations including time-series analysis,
    correlation studies, distribution analysis, and geospatial patterns.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the visualizer with processed EV charging data.
        
        Args:
            data: Processed DataFrame with charging sessions and vehicle data
        """
        self.data = data.copy()
        self.figures = {}
        
        # Ensure datetime column
        if 'Start Time' in self.data.columns:
            self.data['Start Time'] = pd.to_datetime(self.data['Start Time'])
    
    def generate_all_visualizations(self, save_path: Optional[str] = None) -> Dict:
        """
        Generate all required visualizations for the research.
        
        Args:
            save_path: Optional path to save figures
            
        Returns:
            Dictionary containing all generated figures
        """
        print("Generating comprehensive EDA visualizations...")
        
        # 1. Time-series visualizations
        self.figures['temporal_overview'] = self.create_temporal_overview()
        self.figures['charging_heatmap'] = self.create_charging_heatmap()
        self.figures['seasonal_decomposition'] = self.create_seasonal_analysis()
        
        # 2. Vehicle and charging characteristics
        self.figures['vehicle_distributions'] = self.create_vehicle_distributions()
        self.figures['charging_power_analysis'] = self.create_charging_power_analysis()
        self.figures['battery_capacity_analysis'] = self.create_battery_analysis()
        
        # 3. Correlation and relationship analysis
        self.figures['correlation_matrix'] = self.create_correlation_analysis()
        self.figures['charging_efficiency'] = self.create_efficiency_analysis()
        
        # 4. Demand forecasting visualizations
        self.figures['demand_patterns'] = self.create_demand_patterns()
        self.figures['load_curve_analysis'] = self.create_load_curves()
        
        # 5. Manufacturer and category analysis
        self.figures['manufacturer_analysis'] = self.create_manufacturer_analysis()
        self.figures['category_comparison'] = self.create_category_comparison()
        
        # 6. Statistical distribution analysis
        self.figures['statistical_distributions'] = self.create_statistical_distributions()
        
        # 7. Charging behavior patterns
        self.figures['charging_patterns'] = self.create_charging_behavior_patterns()
        
        # 8. Research-specific visualizations
        self.figures['federated_data_distribution'] = self.create_federated_distribution()
        
        if save_path:
            self.save_all_figures(save_path)
        
        print(f"Generated {len(self.figures)} visualization categories")
        return self.figures
    
    def create_temporal_overview(self) -> Dict:
        """Create comprehensive temporal analysis visualizations."""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Daily Charging Sessions', 'Hourly Energy Demand',
                'Weekly Patterns', 'Monthly Trends',
                'Session Duration Distribution', 'Energy Consumption Over Time'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        if 'Start Time' in self.data.columns:
            # Daily sessions
            daily_sessions = self.data.groupby(self.data['Start Time'].dt.date).size()
            fig.add_trace(
                go.Scatter(x=daily_sessions.index, y=daily_sessions.values,
                          mode='lines', name='Daily Sessions'),
                row=1, col=1
            )
            
            # Hourly energy demand
            if 'Meter Total(Wh)' in self.data.columns:
                hourly_energy = self.data.groupby(self.data['Start Time'].dt.hour)['Meter Total(Wh)'].sum()
                fig.add_trace(
                    go.Bar(x=hourly_energy.index, y=hourly_energy.values,
                          name='Hourly Energy (Wh)'),
                    row=1, col=2
                )
            
            # Weekly patterns
            weekly_sessions = self.data.groupby(self.data['Start Time'].dt.dayofweek).size()
            weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            fig.add_trace(
                go.Bar(x=weekdays, y=weekly_sessions.values,
                      name='Weekly Sessions'),
                row=2, col=1
            )
            
            # Monthly trends
            monthly_sessions = self.data.groupby(self.data['Start Time'].dt.month).size()
            fig.add_trace(
                go.Scatter(x=monthly_sessions.index, y=monthly_sessions.values,
                          mode='lines+markers', name='Monthly Sessions'),
                row=2, col=2
            )
        
        # Session duration distribution
        if 'Total Duration (s)' in self.data.columns:
            duration_hours = self.data['Total Duration (s)'] / 3600
            fig.add_trace(
                go.Histogram(x=duration_hours, nbinsx=30,
                           name='Duration Distribution'),
                row=3, col=1
            )
        
        # Energy consumption over time
        if 'Start Time' in self.data.columns and 'Meter Total(Wh)' in self.data.columns:
            energy_trend = self.data.groupby(self.data['Start Time'].dt.date)['Meter Total(Wh)'].sum()
            fig.add_trace(
                go.Scatter(x=energy_trend.index, y=energy_trend.values,
                          mode='lines', name='Daily Energy Consumption'),
                row=3, col=2
            )
        
        fig.update_layout(height=1200, title_text="Temporal Analysis Overview")
        return {'plotly_figure': fig}
    
    def create_charging_heatmap(self) -> Dict:
        """Create hour-of-day vs day-of-week charging heatmap."""
        if 'Start Time' not in self.data.columns:
            return {'error': 'No temporal data available'}
        
        # Create hour-day matrix
        self.data['Hour'] = self.data['Start Time'].dt.hour
        self.data['DayOfWeek'] = self.data['Start Time'].dt.dayofweek
        
        heatmap_data = self.data.groupby(['DayOfWeek', 'Hour']).size().unstack(fill_value=0)
        
        # Check if heatmap_data is empty
        if heatmap_data.empty or heatmap_data.sum().sum() == 0:
            return {'error': 'No charging session data available for heatmap generation'}
        
        fig, ax = plt.subplots(figsize=(16, 8))
        
        # Create heatmap
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='d',
                   cmap='YlOrRd',
                   xticklabels=range(24),
                   yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                   ax=ax)
        
        ax.set_title('Charging Session Heatmap: Hour of Day vs Day of Week', fontsize=16)
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Day of Week', fontsize=12)
        
        plt.tight_layout()
        return {'matplotlib_figure': fig}
    
    def create_seasonal_analysis(self) -> Dict:
        """Create seasonal decomposition and analysis."""
        if 'Start Time' not in self.data.columns:
            return {'error': 'No temporal data available'}
        
        # Daily aggregation for seasonal analysis
        daily_data = self.data.groupby(self.data['Start Time'].dt.date).agg({
            'Session_ID': 'count',
            'Meter Total(Wh)': 'sum'
        }).reset_index()
        
        daily_data.columns = ['Date', 'Session_Count', 'Total_Energy']
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Daily Session Count', 'Daily Energy Consumption', 'Monthly Aggregation'],
            vertical_spacing=0.1
        )
        
        # Daily sessions
        fig.add_trace(
            go.Scatter(x=daily_data['Date'], y=daily_data['Session_Count'],
                      mode='lines', name='Daily Sessions'),
            row=1, col=1
        )
        
        # Daily energy
        fig.add_trace(
            go.Scatter(x=daily_data['Date'], y=daily_data['Total_Energy'],
                      mode='lines', name='Daily Energy (Wh)'),
            row=2, col=1
        )
        
        # Monthly aggregation
        monthly_data = daily_data.groupby(daily_data['Date'].dt.to_period('M')).agg({
            'Session_Count': 'sum',
            'Total_Energy': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Bar(x=monthly_data['Date'].astype(str), y=monthly_data['Session_Count'],
                  name='Monthly Sessions'),
            row=3, col=1
        )
        
        fig.update_layout(height=900, title_text="Seasonal Analysis")
        return {'plotly_figure': fig}
    
    def create_vehicle_distributions(self) -> Dict:
        """Create vehicle characteristic distribution plots."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Vehicle Categories', 'Manufacturers', 'Model Years', 'Fuel Types'],
            specs=[[{"type": "pie"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Vehicle categories
        if 'Category' in self.data.columns:
            category_counts = self.data['Category'].value_counts().head(10)
            fig.add_trace(
                go.Pie(labels=category_counts.index, values=category_counts.values,
                      name="Categories"),
                row=1, col=1
            )
        
        # Manufacturers
        if 'Manufacturer' in self.data.columns:
            manufacturer_counts = self.data['Manufacturer'].value_counts().head(10)
            fig.add_trace(
                go.Pie(labels=manufacturer_counts.index, values=manufacturer_counts.values,
                      name="Manufacturers"),
                row=1, col=2
            )
        
        # Model years
        if 'Model Year' in self.data.columns:
            year_counts = self.data['Model Year'].value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=year_counts.index, y=year_counts.values,
                      name="Model Years"),
                row=2, col=1
            )
        
        # Fuel types
        if 'Fuel' in self.data.columns:
            fuel_counts = self.data['Fuel'].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=fuel_counts.index, y=fuel_counts.values,
                      name="Fuel Types"),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Vehicle Distribution Analysis")
        return {'plotly_figure': fig}
    
    def create_charging_power_analysis(self) -> Dict:
        """Analyze charging power characteristics."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Charging power distribution
        if 'Charging_Power_kW' in self.data.columns:
            axes[0, 0].hist(self.data['Charging_Power_kW'].dropna(), bins=30, alpha=0.7)
            axes[0, 0].set_title('Charging Power Distribution')
            axes[0, 0].set_xlabel('Charging Power (kW)')
            axes[0, 0].set_ylabel('Frequency')
        
        # Charging rate vs duration
        if 'Charging_Rate_kW' in self.data.columns and 'Total Duration (s)' in self.data.columns:
            duration_hours = self.data['Total Duration (s)'] / 3600
            axes[0, 1].scatter(self.data['Charging_Rate_kW'], duration_hours, alpha=0.6)
            axes[0, 1].set_title('Charging Rate vs Duration')
            axes[0, 1].set_xlabel('Charging Rate (kW)')
            axes[0, 1].set_ylabel('Duration (hours)')
        
        # Energy consumption distribution
        if 'Meter Total(Wh)' in self.data.columns:
            energy_kwh = self.data['Meter Total(Wh)'] / 1000
            axes[1, 0].hist(energy_kwh.dropna(), bins=30, alpha=0.7)
            axes[1, 0].set_title('Energy Consumption Distribution')
            axes[1, 0].set_xlabel('Energy Consumed (kWh)')
            axes[1, 0].set_ylabel('Frequency')
        
        # Charging efficiency
        if 'Charging_Efficiency' in self.data.columns:
            axes[1, 1].hist(self.data['Charging_Efficiency'].dropna(), bins=30, alpha=0.7)
            axes[1, 1].set_title('Charging Efficiency Distribution')
            axes[1, 1].set_xlabel('Charging Efficiency')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        return {'matplotlib_figure': fig}
    
    def create_battery_analysis(self) -> Dict:
        """Analyze battery capacity and utilization."""
        if 'Battery Capacity kWh' not in self.data.columns:
            return {'error': 'No battery capacity data available'}
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Battery Capacity Distribution', 'Capacity vs Energy Consumed',
                           'SOC Analysis', 'Battery Utilization'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Battery capacity distribution
        capacity_data = self.data['Battery Capacity kWh'].dropna()
        fig.add_trace(
            go.Histogram(x=capacity_data, nbinsx=30, name='Battery Capacity'),
            row=1, col=1
        )
        
        # Capacity vs energy consumed
        if 'Meter Total(Wh)' in self.data.columns:
            energy_kwh = self.data['Meter Total(Wh)'] / 1000
            fig.add_trace(
                go.Scatter(x=self.data['Battery Capacity kWh'], y=energy_kwh,
                          mode='markers', name='Capacity vs Energy'),
                row=1, col=2
            )
        
        # SOC analysis
        if 'Initial_SOC' in self.data.columns and 'Final_SOC' in self.data.columns:
            fig.add_trace(
                go.Histogram(x=self.data['Initial_SOC'], nbinsx=20,
                           name='Initial SOC', opacity=0.7),
                row=2, col=1
            )
            fig.add_trace(
                go.Histogram(x=self.data['Final_SOC'], nbinsx=20,
                           name='Final SOC', opacity=0.7),
                row=2, col=1
            )
        
        # Battery utilization
        if 'Initial_SOC' in self.data.columns and 'Final_SOC' in self.data.columns:
            utilization = self.data['Final_SOC'] - self.data['Initial_SOC']
            fig.add_trace(
                go.Histogram(x=utilization, nbinsx=20,
                           name='SOC Change'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Battery Analysis")
        return {'plotly_figure': fig}
    
    def create_correlation_analysis(self) -> Dict:
        """Create correlation matrix for numeric variables."""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        # Filter out ID columns and select meaningful variables
        meaningful_cols = [col for col in numeric_cols if not any(x in col.lower() 
                          for x in ['id', 'year']) and self.data[col].nunique() > 1]
        
        if len(meaningful_cols) < 2:
            return {'error': 'Insufficient numeric data for correlation analysis'}
        
        correlation_data = self.data[meaningful_cols].corr()
        
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Create correlation heatmap
        mask = np.triu(np.ones_like(correlation_data, dtype=bool))
        sns.heatmap(correlation_data, 
                   mask=mask,
                   annot=True, 
                   fmt='.2f',
                   cmap='coolwarm',
                   center=0,
                   square=True,
                   ax=ax)
        
        ax.set_title('Correlation Matrix - Numeric Variables', fontsize=16)
        plt.tight_layout()
        
        return {'matplotlib_figure': fig}
    
    def create_efficiency_analysis(self) -> Dict:
        """Analyze charging efficiency patterns."""
        if 'Charging_Efficiency' not in self.data.columns:
            return {'error': 'No charging efficiency data available'}
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Efficiency by Hour', 'Efficiency by Manufacturer',
                           'Efficiency vs Power', 'Efficiency by Vehicle Category']
        )
        
        # Efficiency by hour
        if 'Hour' in self.data.columns:
            hourly_eff = self.data.groupby('Hour')['Charging_Efficiency'].mean()
            fig.add_trace(
                go.Bar(x=hourly_eff.index, y=hourly_eff.values,
                      name='Hourly Efficiency'),
                row=1, col=1
            )
        
        # Efficiency by manufacturer
        if 'Manufacturer' in self.data.columns:
            mfr_eff = self.data.groupby('Manufacturer')['Charging_Efficiency'].mean().head(10)
            fig.add_trace(
                go.Bar(x=mfr_eff.index, y=mfr_eff.values,
                      name='Manufacturer Efficiency'),
                row=1, col=2
            )
        
        # Efficiency vs charging power
        if 'Charging_Power_kW' in self.data.columns:
            fig.add_trace(
                go.Scatter(x=self.data['Charging_Power_kW'],
                          y=self.data['Charging_Efficiency'],
                          mode='markers', name='Power vs Efficiency'),
                row=2, col=1
            )
        
        # Efficiency by category
        if 'Category' in self.data.columns:
            cat_eff = self.data.groupby('Category')['Charging_Efficiency'].mean().head(10)
            fig.add_trace(
                go.Bar(x=cat_eff.index, y=cat_eff.values,
                      name='Category Efficiency'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Charging Efficiency Analysis")
        return {'plotly_figure': fig}
    
    def create_demand_patterns(self) -> Dict:
        """Create demand forecasting visualizations."""
        if 'Hour' not in self.data.columns:
            return {'error': 'No temporal data for demand analysis'}
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Hourly Demand Profile', 'Daily Demand Variation',
                           'Weekend vs Weekday', 'Peak Load Analysis']
        )
        
        # Hourly demand profile
        if 'Meter Total(Wh)' in self.data.columns:
            hourly_demand = self.data.groupby('Hour')['Meter Total(Wh)'].sum()
            fig.add_trace(
                go.Scatter(x=hourly_demand.index, y=hourly_demand.values,
                          mode='lines+markers', name='Hourly Demand'),
                row=1, col=1
            )
        
        # Daily demand variation
        if 'Start Time' in self.data.columns and 'Meter Total(Wh)' in self.data.columns:
            daily_demand = self.data.groupby(self.data['Start Time'].dt.date)['Meter Total(Wh)'].sum()
            fig.add_trace(
                go.Scatter(x=daily_demand.index, y=daily_demand.values,
                          mode='lines', name='Daily Demand'),
                row=1, col=2
            )
        
        # Weekend vs weekday
        if 'IsWeekend' in self.data.columns and 'Meter Total(Wh)' in self.data.columns:
            weekend_demand = self.data.groupby(['IsWeekend', 'Hour'])['Meter Total(Wh)'].mean().unstack()
            
            # Check if both weekday and weekend data exist
            if False in weekend_demand.index:
                fig.add_trace(
                    go.Scatter(x=weekend_demand.columns, y=weekend_demand.loc[False],
                              mode='lines', name='Weekday'),
                    row=2, col=1
                )
            if True in weekend_demand.index:
                fig.add_trace(
                    go.Scatter(x=weekend_demand.columns, y=weekend_demand.loc[True],
                              mode='lines', name='Weekend'),
                    row=2, col=1
                )
        
        # Peak load analysis
        if 'Meter Total(Wh)' in self.data.columns:
            peak_hours = self.data.groupby('Hour')['Meter Total(Wh)'].sum().nlargest(6)
            fig.add_trace(
                go.Bar(x=peak_hours.index, y=peak_hours.values,
                      name='Peak Hours'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Demand Pattern Analysis")
        return {'plotly_figure': fig}
    
    def create_load_curves(self) -> Dict:
        """Create load curve analysis for grid impact assessment."""
        if 'Hour' not in self.data.columns or 'Meter Total(Wh)' not in self.data.columns:
            return {'error': 'Insufficient data for load curve analysis'}
        
        # Create typical day load curves
        typical_day = self.data.groupby('Hour').agg({
            'Meter Total(Wh)': ['mean', 'std', 'min', 'max'],
            'Session_ID': 'count'
        }).round(2)
        
        typical_day.columns = ['Mean_Energy', 'Std_Energy', 'Min_Energy', 'Max_Energy', 'Session_Count']
        
        fig = go.Figure()
        
        # Add mean load curve
        fig.add_trace(go.Scatter(
            x=typical_day.index,
            y=typical_day['Mean_Energy'],
            mode='lines+markers',
            name='Average Load',
            line=dict(width=3)
        ))
        
        # Add confidence bands
        fig.add_trace(go.Scatter(
            x=typical_day.index,
            y=typical_day['Mean_Energy'] + typical_day['Std_Energy'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=typical_day.index,
            y=typical_day['Mean_Energy'] - typical_day['Std_Energy'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(0,100,80,0.2)',
            showlegend=False
        ))
        
        # Add min/max envelope
        fig.add_trace(go.Scatter(
            x=typical_day.index,
            y=typical_day['Max_Energy'],
            mode='lines',
            name='Maximum Load',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title='Typical Daily Load Curves',
            xaxis_title='Hour of Day',
            yaxis_title='Energy Demand (Wh)',
            height=600
        )
        
        return {'plotly_figure': fig}
    
    def create_manufacturer_analysis(self) -> Dict:
        """Analyze charging patterns by manufacturer."""
        if 'Manufacturer' not in self.data.columns:
            return {'error': 'No manufacturer data available'}
        
        # Top manufacturers by session count
        top_manufacturers = self.data['Manufacturer'].value_counts().head(10)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Top Manufacturers by Sessions', 'Average Energy by Manufacturer',
                           'Charging Efficiency by Manufacturer', 'Battery Capacity Distribution']
        )
        
        # Session count by manufacturer
        fig.add_trace(
            go.Bar(x=top_manufacturers.values, y=top_manufacturers.index,
                  orientation='h', name='Session Count'),
            row=1, col=1
        )
        
        # Average energy by manufacturer
        if 'Meter Total(Wh)' in self.data.columns:
            avg_energy = self.data.groupby('Manufacturer')['Meter Total(Wh)'].mean().head(10)
            fig.add_trace(
                go.Bar(x=avg_energy.index, y=avg_energy.values,
                      name='Average Energy'),
                row=1, col=2
            )
        
        # Charging efficiency by manufacturer
        if 'Charging_Efficiency' in self.data.columns:
            mfr_efficiency = self.data.groupby('Manufacturer')['Charging_Efficiency'].mean().head(10)
            fig.add_trace(
                go.Bar(x=mfr_efficiency.index, y=mfr_efficiency.values,
                      name='Efficiency'),
                row=2, col=1
            )
        
        # Battery capacity by manufacturer
        if 'Battery Capacity kWh' in self.data.columns:
            mfr_battery = self.data.groupby('Manufacturer')['Battery Capacity kWh'].mean().head(10)
            fig.add_trace(
                go.Bar(x=mfr_battery.index, y=mfr_battery.values,
                      name='Battery Capacity'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Manufacturer Analysis")
        return {'plotly_figure': fig}
    
    def create_category_comparison(self) -> Dict:
        """Compare charging patterns across vehicle categories."""
        if 'Category' not in self.data.columns:
            return {'error': 'No category data available'}
        
        # Create comprehensive category comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Energy consumption by category
        if 'Meter Total(Wh)' in self.data.columns:
            category_energy = self.data.groupby('Category')['Meter Total(Wh)'].mean().head(10)
            category_energy.plot(kind='bar', ax=axes[0, 0])
            axes[0, 0].set_title('Average Energy Consumption by Category')
            axes[0, 0].set_ylabel('Energy (Wh)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Session duration by category
        if 'Total Duration (s)' in self.data.columns:
            duration_hours = self.data['Total Duration (s)'] / 3600
            category_duration = self.data.groupby('Category')[duration_hours.name].mean().head(10)
            category_duration.plot(kind='bar', ax=axes[0, 1])
            axes[0, 1].set_title('Average Session Duration by Category')
            axes[0, 1].set_ylabel('Duration (hours)')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Battery capacity by category
        if 'Battery Capacity kWh' in self.data.columns:
            category_battery = self.data.groupby('Category')['Battery Capacity kWh'].mean().head(10)
            category_battery.plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Average Battery Capacity by Category')
            axes[1, 0].set_ylabel('Battery Capacity (kWh)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Session count by category
        category_sessions = self.data['Category'].value_counts().head(10)
        category_sessions.plot(kind='bar', ax=axes[1, 1])
        axes[1, 1].set_title('Session Count by Category')
        axes[1, 1].set_ylabel('Number of Sessions')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return {'matplotlib_figure': fig}
    
    def create_statistical_distributions(self) -> Dict:
        """Create statistical distribution analysis."""
        numeric_cols = ['Meter Total(Wh)', 'Total Duration (s)', 'Battery Capacity kWh', 'Charging_Rate_kW']
        available_cols = [col for col in numeric_cols if col in self.data.columns]
        
        if not available_cols:
            return {'error': 'No numeric columns available for distribution analysis'}
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[f'{col} Distribution' for col in available_cols[:4]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, col in enumerate(available_cols[:4]):
            data_clean = self.data[col].dropna()
            
            # Create histogram
            fig.add_trace(
                go.Histogram(x=data_clean, nbinsx=30, name=f'{col} Histogram'),
                row=positions[i][0], col=positions[i][1]
            )
        
        fig.update_layout(height=800, title_text="Statistical Distribution Analysis")
        return {'plotly_figure': fig}
    
    def create_charging_behavior_patterns(self) -> Dict:
        """Analyze charging behavior patterns for research insights."""
        if 'Initial_SOC' not in self.data.columns or 'Final_SOC' not in self.data.columns:
            return {'error': 'No SOC data available for behavior analysis'}
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['SOC Start vs End', 'Charging Session Types',
                           'Energy vs Duration', 'Efficiency Distribution'],
            specs=[[{"type": "xy"}, {"type": "pie"}],
                   [{"type": "xy"}, {"type": "xy"}]]
        )
        
        # SOC start vs end
        fig.add_trace(
            go.Scatter(x=self.data['Initial_SOC'], y=self.data['Final_SOC'],
                      mode='markers', name='SOC Pattern'),
            row=1, col=1
        )
        
        # Charging session types (based on duration)
        if 'Total Duration (s)' in self.data.columns:
            duration_hours = self.data['Total Duration (s)'] / 3600
            session_types = pd.cut(duration_hours, 
                                 bins=[0, 1, 4, 8, float('inf')],
                                 labels=['Quick (<1h)', 'Normal (1-4h)', 
                                        'Long (4-8h)', 'Extended (>8h)'])
            
            type_counts = session_types.value_counts()
            fig.add_trace(
                go.Pie(labels=type_counts.index, values=type_counts.values,
                      name="Session Types"),
                row=1, col=2
            )
        
        # Energy vs duration
        if 'Meter Total(Wh)' in self.data.columns and 'Total Duration (s)' in self.data.columns:
            energy_kwh = self.data['Meter Total(Wh)'] / 1000
            duration_hours = self.data['Total Duration (s)'] / 3600
            
            fig.add_trace(
                go.Scatter(x=energy_kwh, y=duration_hours,
                          mode='markers', name='Energy vs Duration'),
                row=2, col=1
            )
        
        # Efficiency distribution
        if 'Charging_Efficiency' in self.data.columns:
            fig.add_trace(
                go.Histogram(x=self.data['Charging_Efficiency'], nbinsx=30,
                           name='Efficiency Distribution'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Charging Behavior Patterns")
        return {'plotly_figure': fig}
    
    def create_federated_distribution(self) -> Dict:
        """Visualize data distribution for federated learning setup."""
        if 'Manufacturer' not in self.data.columns:
            return {'error': 'No manufacturer data for federated analysis'}
        
        # Simulate federated client distribution
        manufacturer_counts = self.data['Manufacturer'].value_counts()
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Data Distribution Across Simulated Clients',
                           'Non-IID Data Characteristics']
        )
        
        # Client data distribution
        top_manufacturers = manufacturer_counts.head(10)
        fig.add_trace(
            go.Bar(x=top_manufacturers.index, y=top_manufacturers.values,
                  name='Samples per Client'),
            row=1, col=1
        )
        
        # Data heterogeneity analysis
        if 'Category' in self.data.columns:
            heterogeneity = self.data.groupby('Manufacturer')['Category'].nunique().head(10)
            fig.add_trace(
                go.Bar(x=heterogeneity.index, y=heterogeneity.values,
                      name='Category Diversity'),
                row=2, col=1
            )
        
        fig.update_layout(height=800, title_text="Federated Learning Data Distribution")
        return {'plotly_figure': fig}
    
    def save_all_figures(self, save_path: str):
        """Save all generated figures to specified path."""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        for name, figure_dict in self.figures.items():
            if 'plotly_figure' in figure_dict:
                figure_dict['plotly_figure'].write_html(f"{save_path}/{name}_plotly.html")
                figure_dict['plotly_figure'].write_image(f"{save_path}/{name}_plotly.png", 
                                                       width=1200, height=800)
            
            if 'matplotlib_figure' in figure_dict:
                figure_dict['matplotlib_figure'].savefig(f"{save_path}/{name}_matplotlib.png", 
                                                        dpi=300, bbox_inches='tight')
        
        print(f"All figures saved to {save_path}")
    
    def generate_summary_statistics(self) -> Dict:
        """Generate comprehensive summary statistics for the dataset."""
        summary = {
            'dataset_overview': {
                'total_records': len(self.data),
                'unique_vehicles': self.data.get('Vehicle ID', pd.Series()).nunique(),
                'unique_sessions': self.data.get('Session_ID', pd.Series()).nunique(),
                'date_range': None
            },
            'charging_statistics': {},
            'vehicle_statistics': {},
            'temporal_statistics': {}
        }
        
        # Date range
        if 'Start Time' in self.data.columns:
            summary['dataset_overview']['date_range'] = {
                'start': self.data['Start Time'].min(),
                'end': self.data['Start Time'].max(),
                'span_days': (self.data['Start Time'].max() - self.data['Start Time'].min()).days
            }
        
        # Charging statistics
        numeric_cols = ['Meter Total(Wh)', 'Total Duration (s)', 'Charging_Rate_kW', 'Charging_Efficiency']
        for col in numeric_cols:
            if col in self.data.columns:
                summary['charging_statistics'][col] = {
                    'mean': self.data[col].mean(),
                    'median': self.data[col].median(),
                    'std': self.data[col].std(),
                    'min': self.data[col].min(),
                    'max': self.data[col].max()
                }
        
        # Vehicle statistics
        categorical_cols = ['Manufacturer', 'Category', 'Fuel']
        for col in categorical_cols:
            if col in self.data.columns:
                summary['vehicle_statistics'][col] = {
                    'unique_count': self.data[col].nunique(),
                    'top_values': self.data[col].value_counts().head(5).to_dict()
                }
        
        # Temporal statistics
        if 'Hour' in self.data.columns:
            summary['temporal_statistics'] = {
                'peak_hour': self.data['Hour'].mode().iloc[0] if not self.data['Hour'].mode().empty else None,
                'sessions_by_hour': self.data['Hour'].value_counts().to_dict(),
                'weekend_sessions': self.data.get('IsWeekend', pd.Series()).sum() if 'IsWeekend' in self.data.columns else None
            }
        
        return summary