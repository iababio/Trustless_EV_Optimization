#!/usr/bin/env python3
"""
Experiment Runner Utility

This utility provides functions to run all experiments in sequence or individually.
"""

import sys
import os
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/src')
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/experiments')

import time
import warnings
warnings.filterwarnings('ignore')

# Import experiment modules
try:
    sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/experiments')
    import importlib.util
    
    # Import experiment modules dynamically
    def import_experiment(exp_file):
        spec = importlib.util.spec_from_file_location("experiment", exp_file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    exp1 = import_experiment('/Users/ababio/Lab/Research/EV_Optimization/experiments/01_data_analysis_experiment.py')
    exp2 = import_experiment('/Users/ababio/Lab/Research/EV_Optimization/experiments/02_baseline_models_experiment.py')
    exp3 = import_experiment('/Users/ababio/Lab/Research/EV_Optimization/experiments/03_federated_learning_experiment.py')
    exp4 = import_experiment('/Users/ababio/Lab/Research/EV_Optimization/experiments/04_blockchain_validation_experiment.py')
    exp5 = import_experiment('/Users/ababio/Lab/Research/EV_Optimization/experiments/05_optimization_experiment.py')
    exp6 = import_experiment('/Users/ababio/Lab/Research/EV_Optimization/experiments/06_security_evaluation_experiment.py')
    
except ImportError as e:
    print(f"Warning: Could not import experiment modules: {e}")
    print("Falling back to direct execution method...")
    exp1 = exp2 = exp3 = exp4 = exp5 = exp6 = None


class ExperimentRunner:
    """Utility class for running experiments"""
    
    def __init__(self, output_dir="/Users/ababio/Lab/Research/EV_Optimization/results"):
        self.output_dir = output_dir
        self.results = {}
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
    
    def run_single_experiment(self, experiment_number):
        """Run a single experiment by number"""
        
        # Define experiment names and file paths
        experiment_info = {
            1: ("Data Analysis and EDA", '/Users/ababio/Lab/Research/EV_Optimization/experiments/01_data_analysis_experiment.py'),
            2: ("Baseline Models", '/Users/ababio/Lab/Research/EV_Optimization/experiments/02_baseline_models_experiment.py'),
            3: ("Federated Learning", '/Users/ababio/Lab/Research/EV_Optimization/experiments/03_federated_learning_experiment.py'),
            4: ("Blockchain Validation", '/Users/ababio/Lab/Research/EV_Optimization/experiments/04_blockchain_validation_experiment.py'),
            5: ("Optimization", '/Users/ababio/Lab/Research/EV_Optimization/experiments/05_optimization_experiment.py'),
            6: ("Security Evaluation", '/Users/ababio/Lab/Research/EV_Optimization/experiments/06_security_evaluation_experiment.py')
        }
        
        if experiment_number not in experiment_info:
            print(f"❌ Invalid experiment number: {experiment_number}")
            return None
        
        name, exp_file = experiment_info[experiment_number]
        print(f"\n🚀 Running Experiment {experiment_number}: {name}")
        print("=" * 60)
        
        start_time = time.time()
        try:
            # Try to use imported module first, fallback to direct execution
            experiment_modules = {
                1: exp1, 2: exp2, 3: exp3, 4: exp4, 5: exp5, 6: exp6
            }
            
            exp_module = experiment_modules.get(experiment_number)
            
            if exp_module and hasattr(exp_module, 'main'):
                result = exp_module.main()
            else:
                # Fallback: execute the file directly
                print(f"📝 Executing {exp_file} directly...")
                import subprocess
                result = subprocess.run([sys.executable, exp_file], 
                                      capture_output=True, text=True, cwd=os.path.dirname(exp_file))
                if result.returncode != 0:
                    raise Exception(f"Script execution failed: {result.stderr}")
                print(result.stdout)
                result = "executed_successfully"
            
            execution_time = time.time() - start_time
            
            self.results[experiment_number] = {
                'name': name,
                'result': result,
                'execution_time': execution_time,
                'status': 'success'
            }
            
            print(f"\n✅ Experiment {experiment_number} completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            self.results[experiment_number] = {
                'name': name,
                'result': None,
                'execution_time': execution_time,
                'status': 'failed',
                'error': str(e)
            }
            
            print(f"\n❌ Experiment {experiment_number} failed after {execution_time:.2f} seconds")
            print(f"Error: {str(e)}")
            return None
    
    def run_all_experiments(self):
        """Run all experiments in sequence"""
        print("🔬 Running All Experiments")
        print("=" * 80)
        
        total_start_time = time.time()
        
        for i in range(1, 7):
            self.run_single_experiment(i)
        
        total_time = time.time() - total_start_time
        
        # Print summary
        print("\n📊 Experiment Summary")
        print("=" * 80)
        
        successful = 0
        failed = 0
        
        for exp_num, result in self.results.items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"{status_icon} Experiment {exp_num}: {result['name']} "
                  f"({result['execution_time']:.2f}s)")
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
                if 'error' in result:
                    print(f"   Error: {result['error']}")
        
        print(f"\n📈 Results: {successful} successful, {failed} failed")
        print(f"⏱️ Total execution time: {total_time:.2f} seconds")
        
        return self.results
    
    def run_experiments_subset(self, experiment_numbers):
        """Run a subset of experiments"""
        print(f"🔬 Running Experiments: {experiment_numbers}")
        print("=" * 60)
        
        results = {}
        for exp_num in experiment_numbers:
            result = self.run_single_experiment(exp_num)
            results[exp_num] = result
        
        return results
    
    def get_experiment_status(self):
        """Get status of all experiments"""
        if not self.results:
            print("No experiments have been run yet.")
            return
        
        print("📊 Experiment Status")
        print("-" * 40)
        
        for exp_num, result in self.results.items():
            status = result['status'].upper()
            time_str = f"{result['execution_time']:.2f}s"
            print(f"Experiment {exp_num}: {status} ({time_str})")


def main():
    """Main function for running experiments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run EV Charging Optimization Experiments')
    parser.add_argument('--experiment', '-e', type=int, choices=range(1, 7),
                       help='Run specific experiment (1-6)')
    parser.add_argument('--all', '-a', action='store_true',
                       help='Run all experiments')
    parser.add_argument('--subset', '-s', nargs='+', type=int, choices=range(1, 7),
                       help='Run subset of experiments')
    parser.add_argument('--output', '-o', type=str,
                       default="/Users/ababio/Lab/Research/EV_Optimization/results",
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output)
    
    if args.all:
        runner.run_all_experiments()
    elif args.experiment:
        runner.run_single_experiment(args.experiment)
    elif args.subset:
        runner.run_experiments_subset(args.subset)
    else:
        print("🔬 EV Charging Optimization Experiment Runner")
        print("=" * 50)
        print("Available experiments:")
        print("  1. Data Analysis and EDA")
        print("  2. Baseline Models")
        print("  3. Federated Learning")
        print("  4. Blockchain Validation")
        print("  5. Optimization")
        print("  6. Security Evaluation")
        print("\nUsage:")
        print("  python experiment_runner.py --all")
        print("  python experiment_runner.py --experiment 1")
        print("  python experiment_runner.py --subset 1 2 3")


if __name__ == "__main__":
    main()