#!/usr/bin/env python3
"""
Visualization Utilities for Experiments

This module provides utility functions for creating visualizations across experiments.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class ExperimentVisualizer:
    """Utility class for creating experiment visualizations"""
    
    def __init__(self, style="seaborn-v0_8", figsize=(12, 8)):
        """Initialize visualizer with style settings"""
        self.style = style
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        
        # Set matplotlib style
        plt.style.use(style)
        sns.set_palette("husl")
    
    def plot_federated_convergence(self, federated_results: Dict, save_path: Optional[str] = None):
        """Plot federated learning convergence metrics"""
        if 'metrics' not in federated_results:
            return None
        
        metrics = federated_results['metrics']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Global Loss', 'Global Accuracy', 'Communication Cost', 'Client Participation'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        rounds = list(range(1, len(metrics.get('global_loss_history', [])) + 1))
        
        # Global Loss
        if 'global_loss_history' in metrics:
            fig.add_trace(
                go.Scatter(x=rounds, y=metrics['global_loss_history'], 
                          name='Global Loss', line=dict(color='red')),
                row=1, col=1
            )
        
        # Global Accuracy
        if 'global_accuracy_history' in metrics:
            fig.add_trace(
                go.Scatter(x=rounds, y=metrics['global_accuracy_history'], 
                          name='Global Accuracy', line=dict(color='blue')),
                row=1, col=2
            )
        
        # Communication Cost
        if 'communication_history' in metrics:
            fig.add_trace(
                go.Scatter(x=rounds, y=metrics['communication_history'], 
                          name='Communication Cost (MB)', line=dict(color='green')),
                row=2, col=1
            )
        
        # Client Participation
        if 'participation_history' in metrics:
            fig.add_trace(
                go.Scatter(x=rounds, y=metrics['participation_history'], 
                          name='Active Clients', line=dict(color='orange')),
                row=2, col=2
            )
        
        fig.update_layout(
            title='Federated Learning Convergence Analysis',
            height=600,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_optimization_comparison(self, optimization_results: Dict, save_path: Optional[str] = None):
        """Plot optimization algorithm comparison"""
        if not optimization_results:
            return None
        
        # Extract data for plotting
        algorithms = list(optimization_results.keys())
        costs = [res.total_cost for res in optimization_results.values() if res]
        peak_loads = [res.peak_load for res in optimization_results.values() if res]
        satisfactions = [res.user_satisfaction for res in optimization_results.values() if res]
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Total Cost', 'Peak Load', 'User Satisfaction'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Total Cost
        fig.add_trace(
            go.Bar(x=algorithms, y=costs, name='Total Cost ($)', 
                   marker_color='lightblue'),
            row=1, col=1
        )
        
        # Peak Load
        fig.add_trace(
            go.Bar(x=algorithms, y=peak_loads, name='Peak Load (kW)', 
                   marker_color='lightcoral'),
            row=1, col=2
        )
        
        # User Satisfaction
        fig.add_trace(
            go.Bar(x=algorithms, y=satisfactions, name='User Satisfaction', 
                   marker_color='lightgreen'),
            row=1, col=3
        )
        
        fig.update_layout(
            title='Optimization Algorithm Comparison',
            height=400,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_security_metrics(self, security_results: Dict, save_path: Optional[str] = None):
        """Plot security evaluation metrics"""
        if 'summary_report' not in security_results:
            return None
        
        summary = security_results['summary_report']
        avg_metrics = summary.get('average_metrics', {})
        
        if not avg_metrics:
            return None
        
        # Create radar chart for security metrics
        metrics = ['Detection Rate', 'Robustness', 'Byzantine Tolerance', 'Privacy Protection']
        values = [
            avg_metrics.get('detection_rate', 0),
            avg_metrics.get('robustness_score', 0),
            avg_metrics.get('byzantine_tolerance', 0),
            avg_metrics.get('privacy_protection', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Security Metrics'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Security Evaluation Radar Chart",
            height=500
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def plot_baseline_model_comparison(self, baseline_results: Dict, save_path: Optional[str] = None):
        """Plot baseline model performance comparison"""
        if 'models' not in baseline_results:
            return None
        
        models_data = []
        
        # Extract ML model data
        ml_models = baseline_results['models'].get('machine_learning', {})
        for model_name, model_info in ml_models.items():
            if 'error' not in model_info:
                models_data.append({
                    'Model': model_name,
                    'Type': 'Machine Learning',
                    'Train RMSE': model_info.get('train_rmse', 0),
                    'Val RMSE': model_info.get('val_rmse', 0)
                })
        
        # Extract DL model data
        dl_models = baseline_results['models'].get('deep_learning', {})
        lstm_info = dl_models.get('lstm', {})
        if 'error' not in lstm_info:
            models_data.append({
                'Model': 'LSTM',
                'Type': 'Deep Learning',
                'Train RMSE': lstm_info.get('train_rmse', 0),
                'Val RMSE': lstm_info.get('val_rmse', 0)
            })
        
        if not models_data:
            return None
        
        df = pd.DataFrame(models_data)
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Training RMSE', 'Validation RMSE']
        )
        
        # Training RMSE
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Train RMSE'], 
                   name='Train RMSE', marker_color='skyblue'),
            row=1, col=1
        )
        
        # Validation RMSE
        fig.add_trace(
            go.Bar(x=df['Model'], y=df['Val RMSE'], 
                   name='Val RMSE', marker_color='lightcoral'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Baseline Model Performance Comparison',
            height=400,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def create_experiment_dashboard(self, all_results: Dict, save_path: Optional[str] = None):
        """Create comprehensive dashboard with all experiment results"""
        
        # Create dashboard with multiple subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Federated Learning Convergence', 'Baseline Model Performance',
                'Optimization Comparison', 'Security Metrics',
                'Communication Overhead', 'Privacy vs Accuracy'
            ],
            specs=[[{"colspan": 1}, {"colspan": 1}],
                   [{"colspan": 1}, {"colspan": 1}],
                   [{"colspan": 1}, {"colspan": 1}]]
        )
        
        # Add federated learning data if available
        if 'federated_results' in all_results:
            fed_results = all_results['federated_results']
            if 'metrics' in fed_results and 'global_loss_history' in fed_results['metrics']:
                rounds = list(range(1, len(fed_results['metrics']['global_loss_history']) + 1))
                fig.add_trace(
                    go.Scatter(x=rounds, y=fed_results['metrics']['global_loss_history'],
                              name='Global Loss', line=dict(color='red')),
                    row=1, col=1
                )
        
        # Add baseline model data if available
        if 'baseline_results' in all_results:
            baseline_results = all_results['baseline_results']
            if 'models' in baseline_results:
                ml_models = baseline_results['models'].get('machine_learning', {})
                model_names = []
                val_rmses = []
                
                for model_name, model_info in ml_models.items():
                    if 'error' not in model_info:
                        model_names.append(model_name)
                        val_rmses.append(model_info.get('val_rmse', 0))
                
                if model_names:
                    fig.add_trace(
                        go.Bar(x=model_names, y=val_rmses, name='Val RMSE'),
                        row=1, col=2
                    )
        
        # Add optimization data if available
        if 'optimization_results' in all_results:
            opt_results = all_results['optimization_results']
            if opt_results:
                algorithms = list(opt_results.keys())
                costs = [res.total_cost for res in opt_results.values() if res]
                
                if algorithms and costs:
                    fig.add_trace(
                        go.Bar(x=algorithms, y=costs, name='Total Cost ($)'),
                        row=2, col=1
                    )
        
        # Add security metrics if available
        if 'security_results' in all_results:
            sec_results = all_results['security_results']
            if 'summary_report' in sec_results:
                avg_metrics = sec_results['summary_report'].get('average_metrics', {})
                if avg_metrics:
                    metrics = ['Detection', 'Robustness', 'Byzantine', 'Privacy']
                    values = [
                        avg_metrics.get('detection_rate', 0),
                        avg_metrics.get('robustness_score', 0),
                        avg_metrics.get('byzantine_tolerance', 0),
                        avg_metrics.get('privacy_protection', 0)
                    ]
                    
                    fig.add_trace(
                        go.Bar(x=metrics, y=values, name='Security Metrics'),
                        row=2, col=2
                    )
        
        fig.update_layout(
            title='EV Charging Optimization Research Dashboard',
            height=900,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig
    
    def save_all_plots(self, all_results: Dict, output_dir: str):
        """Save all plots to files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        # Federated learning plot
        if 'federated_results' in all_results:
            fed_path = os.path.join(output_dir, "federated_convergence.html")
            self.plot_federated_convergence(all_results['federated_results'], fed_path)
            saved_files.append(fed_path)
        
        # Baseline models plot
        if 'baseline_results' in all_results:
            baseline_path = os.path.join(output_dir, "baseline_comparison.html")
            self.plot_baseline_model_comparison(all_results['baseline_results'], baseline_path)
            saved_files.append(baseline_path)
        
        # Optimization plot
        if 'optimization_results' in all_results:
            opt_path = os.path.join(output_dir, "optimization_comparison.html")
            self.plot_optimization_comparison(all_results['optimization_results'], opt_path)
            saved_files.append(opt_path)
        
        # Security plot
        if 'security_results' in all_results:
            sec_path = os.path.join(output_dir, "security_metrics.html")
            self.plot_security_metrics(all_results['security_results'], sec_path)
            saved_files.append(sec_path)
        
        # Dashboard
        dashboard_path = os.path.join(output_dir, "experiment_dashboard.html")
        self.create_experiment_dashboard(all_results, dashboard_path)
        saved_files.append(dashboard_path)
        
        return saved_files


def create_quick_visualization(data: Dict, plot_type: str, title: str = "Quick Plot"):
    """Create a quick visualization for any data"""
    
    if plot_type == "line":
        fig = go.Figure()
        for key, values in data.items():
            fig.add_trace(go.Scatter(y=values, name=key, mode='lines+markers'))
        fig.update_layout(title=title)
        return fig
    
    elif plot_type == "bar":
        fig = go.Figure()
        for key, values in data.items():
            fig.add_trace(go.Bar(x=list(range(len(values))), y=values, name=key))
        fig.update_layout(title=title)
        return fig
    
    elif plot_type == "scatter":
        fig = go.Figure()
        for key, values in data.items():
            fig.add_trace(go.Scatter(y=values, name=key, mode='markers'))
        fig.update_layout(title=title)
        return fig
    
    else:
        print(f"Unknown plot type: {plot_type}")
        return None


if __name__ == "__main__":
    # Example usage
    visualizer = ExperimentVisualizer()
    
    # Example data
    sample_data = {
        'loss': [0.5, 0.4, 0.3, 0.25, 0.2],
        'accuracy': [0.7, 0.75, 0.8, 0.82, 0.85]
    }
    
    fig = create_quick_visualization(sample_data, "line", "Sample Training Progress")
    if fig:
        fig.show()