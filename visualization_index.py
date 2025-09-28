#!/usr/bin/env python3
"""
Comprehensive Visualization Index Generator
Creates an HTML index of all generated research visualizations
"""

import os
import json
from datetime import datetime

def create_visualization_index():
    """Create a comprehensive HTML index of all visualizations"""
    
    results_dir = "/Users/ababio/Lab/Research/EV_Optimization/results"
    
    # Define all visualization categories and their files
    categories = {
        "Basic Experiment Charts": {
            "path": "charts",
            "files": [
                ("federated_learning_analysis.html", "Federated Learning Performance Analysis"),
                ("baseline_models_comparison.html", "Baseline Model Performance Comparison"),
                ("optimization_algorithms_comparison.html", "Optimization Algorithm Performance"),
                ("security_evaluation_radar.html", "Security Evaluation Radar Chart"),
                ("blockchain_performance_comparison.html", "Blockchain Performance Analysis"),
                ("comprehensive_dashboard.html", "Comprehensive Research Dashboard")
            ]
        },
        "Exploratory Data Analysis (EDA)": {
            "path": "research_analytics", 
            "files": [
                ("eda_time_series_overview.html", "Time-Series Multi-Panel Overview"),
                ("usage_heatmaps.html", "Hour×Weekday Usage Heatmaps"),
                ("seasonal_decomposition.html", "STL Seasonal Decomposition"),
                ("autocorrelation_analysis.html", "ACF/PACF Analysis"),
                ("correlation_analysis.html", "Feature Correlation Matrix"),
                ("session_distributions.html", "Session-Level Distributions"),
                ("geospatial_analysis.html", "Geospatial Station Analysis")
            ]
        },
        "Federated Learning Analytics": {
            "path": "research_analytics",
            "files": [
                ("federated_learning_analytics.html", "FL Convergence & Communication Analysis")
            ]
        },
        "Forecasting & Optimization Metrics": {
            "path": "research_analytics",
            "files": [
                ("forecasting_evaluation.html", "Forecasting Model Evaluation (MAE, RMSE, MAPE, CRPS)"),
                ("optimization_metrics.html", "Optimization & Operational Metrics")
            ]
        },
        "Explainability & Feature Attribution": {
            "path": "explainability_charts",
            "files": [
                ("feature_importance_comparison.html", "Feature Importance Analysis"),
                ("partial_dependence_ice.html", "Partial Dependence & ICE Plots"),
                ("shap_summary_plot.html", "SHAP Values Summary"),
                ("shap_dependence_plots.html", "SHAP Dependence Analysis"),
                ("model_performance_radar.html", "Model Performance Comparison"),
                ("prediction_intervals.html", "Prediction Intervals & Uncertainty"),
                ("federated_client_contributions.html", "Federated Client Contribution Analysis")
            ]
        }
    }
    
    # Generate HTML index
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>EV Charging Optimization Research - Visualization Index</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                padding: 30px;
            }}
            h1 {{
                text-align: center;
                color: #2c3e50;
                font-size: 2.5em;
                margin-bottom: 10px;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }}
            .subtitle {{
                text-align: center;
                color: #7f8c8d;
                font-size: 1.2em;
                margin-bottom: 30px;
            }}
            .stats {{
                display: flex;
                justify-content: space-around;
                margin: 30px 0;
                padding: 20px;
                background: linear-gradient(45deg, #3498db, #2ecc71);
                border-radius: 10px;
                color: white;
            }}
            .stat-item {{
                text-align: center;
            }}
            .stat-number {{
                font-size: 2em;
                font-weight: bold;
                display: block;
            }}
            .category {{
                margin: 30px 0;
                border: 2px solid #ecf0f1;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }}
            .category-header {{
                background: linear-gradient(45deg, #34495e, #2c3e50);
                color: white;
                padding: 15px 20px;
                font-size: 1.3em;
                font-weight: bold;
            }}
            .category-content {{
                padding: 20px;
                background: #f8f9fa;
            }}
            .visualization-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
            }}
            .viz-card {{
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                border-left: 4px solid #3498db;
            }}
            .viz-card:hover {{
                transform: translateY(-5px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }}
            .viz-link {{
                text-decoration: none;
                color: #2c3e50;
                font-weight: bold;
                font-size: 1.1em;
                display: block;
                margin-bottom: 8px;
            }}
            .viz-link:hover {{
                color: #3498db;
            }}
            .viz-description {{
                color: #7f8c8d;
                font-size: 0.95em;
            }}
            .footer {{
                text-align: center;
                margin-top: 40px;
                padding: 20px;
                background: #ecf0f1;
                border-radius: 10px;
                color: #7f8c8d;
            }}
            .research-areas {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 20px 0;
                justify-content: center;
            }}
            .research-tag {{
                background: linear-gradient(45deg, #e74c3c, #f39c12);
                color: white;
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                font-weight: bold;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 EV Charging Optimization Research</h1>
            <p class="subtitle">Comprehensive Visualization Dashboard</p>
            
            <div class="research-areas">
                <span class="research-tag">Federated Learning</span>
                <span class="research-tag">Time Series Analysis</span>
                <span class="research-tag">Optimization</span>
                <span class="research-tag">Security</span>
                <span class="research-tag">Blockchain</span>
                <span class="research-tag">Explainability</span>
            </div>
            
            <div class="stats">
                <div class="stat-item">
                    <span class="stat-number">{sum(len(cat['files']) for cat in categories.values())}</span>
                    <span>Total Visualizations</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">{len(categories)}</span>
                    <span>Research Categories</span>
                </div>
                <div class="stat-item">
                    <span class="stat-number">3</span>
                    <span>Directories</span>
                </div>
            </div>
    """
    
    # Add each category
    for category_name, category_info in categories.items():
        html_content += f"""
            <div class="category">
                <div class="category-header">
                    📊 {category_name} ({len(category_info['files'])} visualizations)
                </div>
                <div class="category-content">
                    <div class="visualization-grid">
        """
        
        for filename, description in category_info['files']:
            file_path = f"{category_info['path']}/{filename}"
            html_content += f"""
                        <div class="viz-card">
                            <a href="{file_path}" class="viz-link" target="_blank">
                                📈 {filename.replace('.html', '').replace('_', ' ').title()}
                            </a>
                            <div class="viz-description">{description}</div>
                        </div>
            """
        
        html_content += """
                    </div>
                </div>
            </div>
        """
    
    # Add footer
    html_content += f"""
            <div class="footer">
                <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p><strong>Research Focus:</strong> Trustless Edge-Based Real-Time ML for EV Charging Optimization</p>
                <p>📁 All visualizations are interactive HTML files powered by Plotly</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save index file
    index_path = os.path.join(results_dir, "visualization_index.html")
    with open(index_path, 'w') as f:
        f.write(html_content)
    
    return index_path

def generate_summary_report():
    """Generate final summary report"""
    
    results_dir = "/Users/ababio/Lab/Research/EV_Optimization/results"
    
    # Count all visualizations
    total_charts = 0
    directories = ['charts', 'research_analytics', 'explainability_charts']
    
    for directory in directories:
        dir_path = os.path.join(results_dir, directory)
        if os.path.exists(dir_path):
            html_files = [f for f in os.listdir(dir_path) if f.endswith('.html')]
            total_charts += len(html_files)
    
    summary = {
        "research_project": "EV Charging Optimization - Trustless Edge-Based Real-Time ML",
        "generation_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_visualizations": total_charts,
        "categories_implemented": {
            "Exploratory Data Analysis": [
                "Time-series overview with rolling means",
                "Hour×weekday usage heatmaps", 
                "STL seasonal decomposition",
                "ACF/PACF autocorrelation analysis",
                "Feature correlation matrix",
                "Session-level distributions",
                "Geospatial station analysis"
            ],
            "Federated Learning Analytics": [
                "Global loss & accuracy convergence",
                "Client participation tracking",
                "Communication overhead analysis", 
                "Weight divergence monitoring",
                "Shapley contribution analysis",
                "Client contribution heatmaps"
            ],
            "Forecasting Model Evaluation": [
                "MAE, RMSE, MAPE comparisons",
                "CRPS score analysis",
                "Coverage intervals (80%, 95%)",
                "Prediction interval visualization"
            ],
            "Optimization & Operational Metrics": [
                "Peak load reduction analysis",
                "Energy cost savings comparison",
                "User satisfaction scoring",
                "Constraint violation tracking"
            ],
            "Explainability & Feature Attribution": [
                "Feature importance comparison",
                "Partial dependence plots (PDP)",
                "Individual conditional expectation (ICE)",
                "SHAP value analysis",
                "SHAP dependence plots",
                "Model performance radar charts"
            ],
            "Security & Blockchain": [
                "Security evaluation radar charts",
                "Attack detection analysis",
                "Blockchain performance metrics",
                "Robustness scoring"
            ]
        },
        "research_insights": {
            "temporal_patterns": "Clear daily and weekly seasonality patterns identified in charging behavior",
            "optimization_performance": "Multi-objective optimization achieves best balance of cost, satisfaction, and efficiency",
            "federated_learning": "Global model convergence with efficient communication and client contribution tracking",
            "explainability": "Battery level, hour of day, and electricity price are most influential features",
            "forecasting_accuracy": "Transformer models show best prediction performance with reliable uncertainty estimates"
        },
        "file_structure": {
            "charts/": "Basic experiment visualizations (6 files)",
            "research_analytics/": "Comprehensive EDA and research analytics (10 files)", 
            "explainability_charts/": "Explainability and feature attribution (7 files)",
            "visualization_index.html": "Master index of all visualizations"
        }
    }
    
    # Save comprehensive summary
    summary_path = os.path.join(results_dir, "research_visualization_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_path, summary

def main():
    """Generate visualization index and summary"""
    
    print("📊 Generating Comprehensive Visualization Index...")
    
    # Create HTML index
    index_path = create_visualization_index()
    print(f"✅ Created visualization index: {index_path}")
    
    # Generate summary report
    summary_path, summary = generate_summary_report()
    print(f"✅ Created summary report: {summary_path}")
    
    print(f"\\n🎉 Visualization Index Generation Complete!")
    print(f"📈 Total Visualizations: {summary['total_visualizations']}")
    print(f"📂 Categories: {len(summary['categories_implemented'])}")
    print(f"🌐 Access via: visualization_index.html")

if __name__ == "__main__":
    main()