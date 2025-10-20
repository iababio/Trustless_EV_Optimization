#!/usr/bin/env python3
"""
Generate comprehensive visualization charts for EV optimization experiments
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.io as pio
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Configure high-resolution PDF output
pio.kaleido.scope.default_width = 1920
pio.kaleido.scope.default_height = 1080
pio.kaleido.scope.default_format = "pdf"
pio.kaleido.scope.default_scale = 2  # High DPI for crisp output

# Configure matplotlib for PDF output
plt.rcParams['pdf.fonttype'] = 42  # Embed fonts
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Add project path
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization')
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/src')
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/experiments')

# Set up plotting styles
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_results_directory():
    """Create results directory if it doesn't exist"""
    results_dir = "/Users/ababio/Lab/Research/EV_Optimization/results"
    charts_dir = os.path.join(results_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    return charts_dir

def generate_synthetic_experiment_data():
    """Generate synthetic data simulating experiment results"""
    
    # Federated Learning Results
    federated_results = {
        'global_loss_history': [0.85, 0.72, 0.61, 0.54, 0.48, 0.43, 0.39, 0.36, 0.33, 0.31],
        'global_accuracy_history': [0.68, 0.75, 0.82, 0.85, 0.88, 0.91, 0.93, 0.94, 0.95, 0.96],
        'communication_costs': [45.2, 42.1, 38.9, 36.4, 34.8, 33.2, 31.9, 30.7, 29.8, 29.1],
        'participation_rates': [0.85, 0.88, 0.91, 0.89, 0.92, 0.94, 0.96, 0.93, 0.95, 0.97],
        'rounds': list(range(1, 11))
    }
    
    # Baseline Model Results
    baseline_results = {
        'models': ['Linear Regression', 'Random Forest', 'XGBoost', 'LSTM', 'GRU'],
        'train_rmse': [2.45, 1.82, 1.65, 1.34, 1.42],
        'val_rmse': [2.68, 2.01, 1.89, 1.56, 1.61],
        'train_mae': [1.92, 1.34, 1.21, 0.98, 1.05],
        'val_mae': [2.15, 1.48, 1.35, 1.12, 1.18]
    }
    
    # Optimization Results
    optimization_results = {
        'algorithms': ['Greedy', 'Genetic Algorithm', 'Particle Swarm', 'Deep Q-Learning', 'Multi-Objective'],
        'total_costs': [1250.30, 980.45, 875.20, 720.15, 695.80],
        'peak_loads': [8.5, 7.2, 6.8, 5.9, 5.6],
        'user_satisfaction': [0.72, 0.81, 0.85, 0.91, 0.94],
        'energy_efficiency': [0.68, 0.76, 0.82, 0.88, 0.92]
    }
    
    # Security Metrics
    security_results = {
        'attack_types': ['Data Poisoning', 'Model Inversion', 'Membership Inference', 'Byzantine Attack', 'Privacy Leakage'],
        'detection_rates': [0.87, 0.92, 0.89, 0.94, 0.91],
        'false_positive_rates': [0.05, 0.03, 0.04, 0.02, 0.03],
        'robustness_scores': [0.85, 0.89, 0.87, 0.92, 0.90]
    }
    
    # Blockchain Performance
    blockchain_results = {
        'metrics': ['Throughput (TPS)', 'Latency (ms)', 'Energy (kWh)', 'Storage (MB)', 'Network (MB/s)'],
        'values': [1250, 150, 2.4, 45.8, 12.3],
        'baseline_values': [800, 220, 3.8, 62.1, 8.7]
    }
    
    return {
        'federated': federated_results,
        'baseline': baseline_results,
        'optimization': optimization_results,
        'security': security_results,
        'blockchain': blockchain_results
    }

def create_federated_learning_charts(data, save_dir):
    """Create federated learning visualization charts"""
    
    # Convergence Analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Global Loss Convergence', 'Global Accuracy Improvement', 
                       'Communication Cost Reduction', 'Client Participation Rates'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    rounds = data['rounds']
    
    # Global Loss
    fig.add_trace(
        go.Scatter(x=rounds, y=data['global_loss_history'],
                  mode='lines+markers', name='Global Loss',
                  line=dict(color='red', width=3),
                  marker=dict(size=8)),
        row=1, col=1
    )
    
    # Global Accuracy
    fig.add_trace(
        go.Scatter(x=rounds, y=data['global_accuracy_history'],
                  mode='lines+markers', name='Global Accuracy',
                  line=dict(color='blue', width=3),
                  marker=dict(size=8)),
        row=1, col=2
    )
    
    # Communication Cost
    fig.add_trace(
        go.Scatter(x=rounds, y=data['communication_costs'],
                  mode='lines+markers', name='Communication Cost (MB)',
                  line=dict(color='green', width=3),
                  marker=dict(size=8)),
        row=2, col=1
    )
    
    # Participation Rate
    fig.add_trace(
        go.Scatter(x=rounds, y=data['participation_rates'],
                  mode='lines+markers', name='Participation Rate',
                  line=dict(color='orange', width=3),
                  marker=dict(size=8)),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(text='Federated Learning Performance Analysis', font=dict(size=24)),
        height=600,
        showlegend=False,
        font=dict(size=16),
        plot_bgcolor='white'
    )
    
    # Save chart as high-resolution PDF
    chart_path = os.path.join(save_dir, "federated_learning_analysis.pdf")
    fig.write_image(chart_path, width=1920, height=1080, scale=2)
    
    return chart_path

def create_baseline_model_charts(data, save_dir):
    """Create baseline model comparison charts"""
    
    # Model Performance Comparison
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=['RMSE Comparison', 'MAE Comparison']
    )
    
    models = data['models']
    
    # RMSE Comparison
    fig.add_trace(
        go.Bar(x=models, y=data['train_rmse'],
               name='Training RMSE', marker_color='lightblue',
               text=data['train_rmse'], textposition='auto'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=models, y=data['val_rmse'],
               name='Validation RMSE', marker_color='lightcoral',
               text=data['val_rmse'], textposition='auto'),
        row=1, col=1
    )
    
    # MAE Comparison
    fig.add_trace(
        go.Bar(x=models, y=data['train_mae'],
               name='Training MAE', marker_color='lightgreen',
               text=data['train_mae'], textposition='auto'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Bar(x=models, y=data['val_mae'],
               name='Validation MAE', marker_color='lightyellow',
               text=data['val_mae'], textposition='auto'),
        row=1, col=2
    )
    
    fig.update_layout(
        title=dict(text='Baseline Model Performance Comparison', font=dict(size=24)),
        height=500,
        font=dict(size=16),
        plot_bgcolor='white'
    )
    
    chart_path = os.path.join(save_dir, "baseline_models_comparison.pdf")
    fig.write_image(chart_path, width=1920, height=1080, scale=2)
    
    return chart_path

def create_optimization_charts(data, save_dir):
    """Create optimization algorithm comparison charts"""
    
    # Multi-metric comparison
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Total Cost ($)', 'Peak Load (kW)', 
                       'User Satisfaction', 'Energy Efficiency'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    algorithms = data['algorithms']
    
    # Total Cost
    fig.add_trace(
        go.Bar(x=algorithms, y=data['total_costs'],
               name='Total Cost', marker_color='lightblue',
               text=[f"${x:.0f}" for x in data['total_costs']], textposition='auto'),
        row=1, col=1
    )
    
    # Peak Load
    fig.add_trace(
        go.Bar(x=algorithms, y=data['peak_loads'],
               name='Peak Load', marker_color='lightcoral',
               text=[f"{x:.1f}kW" for x in data['peak_loads']], textposition='auto'),
        row=1, col=2
    )
    
    # User Satisfaction
    fig.add_trace(
        go.Bar(x=algorithms, y=data['user_satisfaction'],
               name='User Satisfaction', marker_color='lightgreen',
               text=[f"{x:.2f}" for x in data['user_satisfaction']], textposition='auto'),
        row=2, col=1
    )
    
    # Energy Efficiency
    fig.add_trace(
        go.Bar(x=algorithms, y=data['energy_efficiency'],
               name='Energy Efficiency', marker_color='lightyellow',
               text=[f"{x:.2f}" for x in data['energy_efficiency']], textposition='auto'),
        row=2, col=2
    )
    
    fig.update_layout(
        title=dict(text='Optimization Algorithm Performance Comparison', font=dict(size=24)),
        height=600,
        showlegend=False,
        font=dict(size=16),
        plot_bgcolor='white'
    )
    
    chart_path = os.path.join(save_dir, "optimization_algorithms_comparison.pdf")
    fig.write_image(chart_path, width=1920, height=1080, scale=2)
    
    return chart_path

def create_security_charts(data, save_dir):
    """Create security evaluation charts"""
    
    # Security radar chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=data['detection_rates'],
        theta=data['attack_types'],
        fill='toself',
        name='Detection Rate',
        line_color='blue'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=data['robustness_scores'],
        theta=data['attack_types'],
        fill='toself',
        name='Robustness Score',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=14)
            )),
        title=dict(text="Security Evaluation: Detection Rates & Robustness", font=dict(size=24)),
        height=500,
        font=dict(size=16)
    )
    
    chart_path = os.path.join(save_dir, "security_evaluation_radar.pdf")
    fig.write_image(chart_path, width=1920, height=1080, scale=2)
    
    return chart_path

def create_blockchain_charts(data, save_dir):
    """Create blockchain performance charts"""
    
    # Performance comparison
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=data['metrics'],
        y=data['values'],
        name='Optimized System',
        marker_color='lightblue',
        text=data['values'],
        textposition='auto',
        textfont=dict(size=32)
    ))

    fig.add_trace(go.Bar(
        x=data['metrics'],
        y=data['baseline_values'],
        name='Baseline System',
        marker_color='lightcoral',
        text=data['baseline_values'],
        textposition='auto',
        textfont=dict(size=32)
    ))

    fig.update_layout(
        title=dict(text='Blockchain Performance: Optimized vs Baseline', font=dict(size=48)),
        xaxis_title=dict(text='Metrics', font=dict(size=36)),
        yaxis_title=dict(text='Values', font=dict(size=36)),
        height=700,
        font=dict(size=32),
        plot_bgcolor='white',
        barmode='group',
        margin=dict(t=150, b=100, l=120, r=100),
        legend=dict(font=dict(size=32))
    )

    # Update tick font sizes
    fig.update_xaxes(tickfont=dict(size=28))
    fig.update_yaxes(tickfont=dict(size=28))
    
    chart_path = os.path.join(save_dir, "blockchain_performance_comparison.pdf")
    fig.write_image(chart_path, width=1920, height=1080, scale=2)
    
    return chart_path

def create_comprehensive_dashboard(all_data, save_dir):
    """Create comprehensive dashboard with all results"""
    
    # Create multi-subplot dashboard
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'FL: Global Loss', 'FL: Accuracy', 'Optimization: Total Cost',
            'Model: RMSE Comparison', 'Security: Detection Rates', 'Blockchain: Throughput',
            'FL: Communication Cost', 'Optimization: User Satisfaction', 'Energy Efficiency'
        ],
        specs=[[{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}],
               [{"colspan": 1}, {"colspan": 1}, {"colspan": 1}]]
    )
    
    # Federated Learning Loss
    fig.add_trace(
        go.Scatter(x=all_data['federated']['rounds'], 
                  y=all_data['federated']['global_loss_history'],
                  mode='lines+markers', name='Global Loss', line=dict(color='red')),
        row=1, col=1
    )
    
    # Federated Learning Accuracy
    fig.add_trace(
        go.Scatter(x=all_data['federated']['rounds'], 
                  y=all_data['federated']['global_accuracy_history'],
                  mode='lines+markers', name='Accuracy', line=dict(color='blue')),
        row=1, col=2
    )
    
    # Optimization Cost
    fig.add_trace(
        go.Bar(x=all_data['optimization']['algorithms'], 
               y=all_data['optimization']['total_costs'],
               name='Total Cost', marker_color='lightblue'),
        row=1, col=3
    )
    
    # Model RMSE
    fig.add_trace(
        go.Bar(x=all_data['baseline']['models'], 
               y=all_data['baseline']['val_rmse'],
               name='Val RMSE', marker_color='lightcoral'),
        row=2, col=1
    )
    
    # Security Detection
    fig.add_trace(
        go.Bar(x=all_data['security']['attack_types'], 
               y=all_data['security']['detection_rates'],
               name='Detection Rate', marker_color='lightgreen'),
        row=2, col=2
    )
    
    # Blockchain Throughput
    blockchain_metric = 'Throughput (TPS)'
    if blockchain_metric in all_data['blockchain']['metrics']:
        idx = all_data['blockchain']['metrics'].index(blockchain_metric)
        fig.add_trace(
            go.Bar(x=['Optimized', 'Baseline'], 
                   y=[all_data['blockchain']['values'][idx], all_data['blockchain']['baseline_values'][idx]],
                   name='Throughput', marker_color='lightpink'),
            row=2, col=3
        )
    
    # FL Communication Cost
    fig.add_trace(
        go.Scatter(x=all_data['federated']['rounds'], 
                  y=all_data['federated']['communication_costs'],
                  mode='lines+markers', name='Comm Cost', line=dict(color='green')),
        row=3, col=1
    )
    
    # User Satisfaction
    fig.add_trace(
        go.Bar(x=all_data['optimization']['algorithms'], 
               y=all_data['optimization']['user_satisfaction'],
               name='User Satisfaction', marker_color='lightyellow'),
        row=3, col=2
    )
    
    # Energy Efficiency
    fig.add_trace(
        go.Bar(x=all_data['optimization']['algorithms'], 
               y=all_data['optimization']['energy_efficiency'],
               name='Energy Efficiency', marker_color='lightsteelblue'),
        row=3, col=3
    )
    
    fig.update_layout(
        title=dict(text='EV Charging Optimization: Comprehensive Research Dashboard', font=dict(size=28)),
        height=900,
        showlegend=False,
        font=dict(size=14),
        plot_bgcolor='white'
    )
    
    chart_path = os.path.join(save_dir, "comprehensive_dashboard.pdf")
    fig.write_image(chart_path, width=1920, height=1080, scale=2)
    
    return chart_path

def generate_summary_report(charts_generated, save_dir):
    """Generate summary report of all charts created"""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = {
        "timestamp": timestamp,
        "charts_generated": len(charts_generated),
        "chart_files": [os.path.basename(chart) for chart in charts_generated],
        "description": "Comprehensive visualization charts for EV charging optimization research",
        "categories": [
            "Federated Learning Analysis",
            "Baseline Model Comparison", 
            "Optimization Algorithm Performance",
            "Security Evaluation",
            "Blockchain Performance",
            "Comprehensive Dashboard"
        ]
    }
    
    # Save JSON report
    report_path = os.path.join(save_dir, "chart_generation_report.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save markdown summary
    md_path = os.path.join(save_dir, "visualization_summary.md")
    with open(md_path, 'w') as f:
        f.write(f"# EV Charging Optimization Research Visualizations\\n\\n")
        f.write(f"**Generated on:** {timestamp}\\n\\n")
        f.write(f"**Total Charts Created:** {len(charts_generated)}\\n\\n")
        f.write("## Chart Categories\\n\\n")
        for category in report["categories"]:
            f.write(f"- {category}\\n")
        f.write("\\n## Generated Files\\n\\n")
        for chart_file in report["chart_files"]:
            f.write(f"- {chart_file}\\n")
    
    return report_path, md_path

def main():
    """Main function to generate all charts"""
    
    print("🎨 EV Optimization Research Visualization Generator")
    print("=" * 60)
    
    # Create results directory
    charts_dir = create_results_directory()
    print(f"📁 Charts will be saved to: {charts_dir}")
    
    # Generate synthetic experiment data
    print("🔄 Generating experiment data...")
    experiment_data = generate_synthetic_experiment_data()
    
    # Generate charts
    charts_generated = []
    
    print("📊 Creating federated learning charts...")
    fl_chart = create_federated_learning_charts(experiment_data['federated'], charts_dir)
    charts_generated.append(fl_chart)
    
    print("📊 Creating baseline model charts...")
    baseline_chart = create_baseline_model_charts(experiment_data['baseline'], charts_dir)
    charts_generated.append(baseline_chart)
    
    print("📊 Creating optimization algorithm charts...")
    opt_chart = create_optimization_charts(experiment_data['optimization'], charts_dir)
    charts_generated.append(opt_chart)
    
    print("📊 Creating security evaluation charts...")
    sec_chart = create_security_charts(experiment_data['security'], charts_dir)
    charts_generated.append(sec_chart)
    
    print("📊 Creating blockchain performance charts...")
    blockchain_chart = create_blockchain_charts(experiment_data['blockchain'], charts_dir)
    charts_generated.append(blockchain_chart)
    
    print("📊 Creating comprehensive dashboard...")
    dashboard_chart = create_comprehensive_dashboard(experiment_data, charts_dir)
    charts_generated.append(dashboard_chart)
    
    # Generate summary report
    print("📝 Generating summary report...")
    report_path, md_path = generate_summary_report(charts_generated, charts_dir)
    
    # Print results
    print("\\n✅ Chart Generation Complete!")
    print(f"📈 Generated {len(charts_generated)} visualization charts")
    print("\\n📋 Charts Created:")
    for chart in charts_generated:
        print(f"   - {os.path.basename(chart)}")
    
    print(f"\\n📊 All charts saved to: {charts_dir}")
    print("🎉 Visualization generation completed successfully!")

if __name__ == "__main__":
    main()