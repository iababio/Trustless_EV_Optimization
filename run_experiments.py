#!/usr/bin/env python3
"""
Main Experiment Runner for EV Charging Optimization Research

This script provides a command-line interface to run all experiments.
"""

import sys
import os
import argparse
import subprocess
import time

# Add paths
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/experiments')
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/src')

# Try to import utilities, fall back to basic functionality if imports fail
try:
    from experiments.utils.experiment_runner import ExperimentRunner
    from experiments.utils.visualization_utils import ExperimentVisualizer
    from experiments.utils.report_generator import ExperimentReportGenerator
    UTILS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Warning: Could not import utility modules: {e}")
    print("🔄 Falling back to basic experiment execution...")
    UTILS_AVAILABLE = False


def run_experiment_directly(experiment_number, output_dir):
    """Direct execution fallback when utilities aren't available"""
    
    experiment_files = {
        1: "01_data_analysis_experiment.py",
        2: "02_baseline_models_experiment.py", 
        3: "03_federated_learning_experiment.py",
        4: "04_blockchain_validation_experiment.py",
        5: "05_optimization_experiment.py",
        6: "06_security_evaluation_experiment.py"
    }
    
    experiment_names = {
        1: "Data Analysis and EDA",
        2: "Baseline Models",
        3: "Federated Learning", 
        4: "Blockchain Validation",
        5: "Optimization",
        6: "Security Evaluation"
    }
    
    if experiment_number not in experiment_files:
        print(f"❌ Invalid experiment number: {experiment_number}")
        return False
    
    exp_file = experiment_files[experiment_number]
    exp_name = experiment_names[experiment_number]
    exp_path = f"/Users/ababio/Lab/Research/EV_Optimization/experiments/{exp_file}"
    
    print(f"\n🚀 Running Experiment {experiment_number}: {exp_name}")
    print("=" * 60)
    
    if not os.path.exists(exp_path):
        print(f"❌ Experiment file not found: {exp_path}")
        return False
    
    start_time = time.time()
    try:
        result = subprocess.run([sys.executable, exp_path], 
                              cwd="/Users/ababio/Lab/Research/EV_Optimization",
                              check=True)
        execution_time = time.time() - start_time
        print(f"\n✅ Experiment {experiment_number} completed in {execution_time:.2f} seconds")
        return True
        
    except subprocess.CalledProcessError as e:
        execution_time = time.time() - start_time
        print(f"\n❌ Experiment {experiment_number} failed after {execution_time:.2f} seconds")
        print(f"Error: {e}")
        return False


def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(
        description='EV Charging Optimization Research Experiment Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_experiments.py --all                    # Run all experiments
  python run_experiments.py --experiment 1           # Run data analysis only
  python run_experiments.py --subset 1 2 3           # Run first three experiments
  python run_experiments.py --complete               # Run complete research demo
  python run_experiments.py --generate-report        # Generate report from existing results
        """
    )
    
    # Experiment selection options
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--all', '-a', action='store_true',
                      help='Run all individual experiments (1-6)')
    group.add_argument('--experiment', '-e', type=int, choices=range(1, 7),
                      help='Run specific experiment (1-6)')
    group.add_argument('--subset', '-s', nargs='+', type=int, choices=range(1, 7),
                      help='Run subset of experiments')
    group.add_argument('--complete', '-c', action='store_true',
                      help='Run complete research demonstration')
    
    # Output and reporting options
    parser.add_argument('--output', '-o', type=str,
                       default="/Users/ababio/Lab/Research/EV_Optimization/results",
                       help='Output directory for results')
    parser.add_argument('--generate-report', '-r', action='store_true',
                       help='Generate comprehensive report')
    parser.add_argument('--generate-plots', '-p', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Print header
    print("🔬 EV Charging Optimization Research Suite")
    print("=" * 60)
    print("Trustless Edge-Based Real-Time ML for EV Charging Optimization")
    print("=" * 60)
    
    # Initialize components (if available)
    if UTILS_AVAILABLE:
        runner = ExperimentRunner(output_dir=args.output)
        visualizer = ExperimentVisualizer()
        reporter = ExperimentReportGenerator(output_dir=args.output)
    else:
        runner = visualizer = reporter = None
    
    results = None
    
    # Run experiments based on arguments
    if args.complete:
        print("\n🚀 Running Complete Research Demonstration")
        print("-" * 50)
        
        # Import and run the complete demo
        from experiments.complete_research_demonstration import main as run_complete_demo
        try:
            run_complete_demo()
            print("\n✅ Complete research demonstration finished!")
        except Exception as e:
            print(f"\n❌ Complete demo failed: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    elif args.all:
        print("\n🚀 Running All Individual Experiments")
        print("-" * 50)
        if UTILS_AVAILABLE and runner:
            results = runner.run_all_experiments()
        else:
            # Fallback: run each experiment directly
            for exp_num in range(1, 7):
                run_experiment_directly(exp_num, args.output)
    
    elif args.experiment:
        print(f"\n🚀 Running Experiment {args.experiment}")
        print("-" * 50)
        if UTILS_AVAILABLE and runner:
            result = runner.run_single_experiment(args.experiment)
            results = {args.experiment: result} if result else {}
        else:
            # Fallback: run experiment directly
            run_experiment_directly(args.experiment, args.output)
    
    elif args.subset:
        print(f"\n🚀 Running Experiments: {args.subset}")
        print("-" * 50)
        if UTILS_AVAILABLE and runner:
            results = runner.run_experiments_subset(args.subset)
        else:
            # Fallback: run each experiment directly
            for exp_num in args.subset:
                run_experiment_directly(exp_num, args.output)
    
    # Generate visualizations if requested
    if args.generate_plots and results and UTILS_AVAILABLE and visualizer:
        print("\n📊 Generating Visualizations...")
        print("-" * 30)
        
        try:
            # Convert results to expected format for visualizer
            viz_results = {}
            
            for exp_num, result in results.items():
                if result and hasattr(result, '__dict__'):
                    # Map experiment results to visualization format
                    if exp_num == 2:  # Baseline models
                        viz_results['baseline_results'] = result
                    elif exp_num == 3:  # Federated learning
                        viz_results['federated_results'] = result
                    elif exp_num == 5:  # Optimization
                        viz_results['optimization_results'] = result
                    elif exp_num == 6:  # Security
                        viz_results['security_results'] = result
            
            if viz_results:
                saved_plots = visualizer.save_all_plots(viz_results, args.output)
                print(f"✅ Generated {len(saved_plots)} visualization files:")
                for plot_file in saved_plots:
                    print(f"   - {os.path.basename(plot_file)}")
            else:
                print("⚠️ No suitable results found for visualization")
                
        except Exception as e:
            print(f"❌ Visualization generation failed: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Generate report if requested
    if args.generate_report and UTILS_AVAILABLE and reporter:
        print("\n📝 Generating Comprehensive Report...")
        print("-" * 30)
        
        try:
            # Use results if available, otherwise create sample report
            report_data = results if results else {
                'timestamp': 'Generated from experiment runner',
                'note': 'Report generated without running experiments'
            }
            
            markdown_path = reporter.generate_markdown_report(report_data)
            json_path = reporter.generate_json_summary(report_data)
            csv_path = reporter.generate_csv_metrics(report_data)
            
            print("✅ Generated report files:")
            print(f"   - Markdown: {os.path.basename(markdown_path)}")
            print(f"   - JSON: {os.path.basename(json_path)}")
            print(f"   - CSV: {os.path.basename(csv_path)}")
            
        except Exception as e:
            print(f"❌ Report generation failed: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
    
    # Show experiment status
    if results and UTILS_AVAILABLE and runner:
        print("\n📊 Final Experiment Status")
        print("-" * 30)
        runner.get_experiment_status()
    
    # Show available options if no action taken
    if not any([args.all, args.experiment, args.subset, args.complete, 
               args.generate_report, args.generate_plots]):
        print("\n📋 Available Experiments:")
        print("  1. Data Analysis and EDA")
        print("  2. Baseline Model Development")
        print("  3. Federated Learning Simulation")
        print("  4. Blockchain Validation")
        print("  5. Multi-Objective Optimization")
        print("  6. Security Evaluation")
        print("\n📋 Available Actions:")
        print("  --all                 Run all experiments")
        print("  --complete            Run complete research demo")
        print("  --generate-report     Generate comprehensive report")
        print("  --generate-plots      Generate visualization plots")
        print("\nUse --help for more options.")
    
    print(f"\n📁 Results saved to: {args.output}")
    print("🎉 Research suite execution completed!")


if __name__ == "__main__":
    main()