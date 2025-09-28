#!/usr/bin/env python3
"""
Experiment 5: Multi-Objective Charging Optimization

This script implements and evaluates multi-objective optimization algorithms for EV charging scheduling.
"""

import sys
import os
sys.path.append('/Users/ababio/Lab/Research/EV_Optimization/src')

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from optimization.algorithms.charging_optimizer import (
    ChargingOptimizationSuite, OptimizationObjective
)


def main():
    """Run optimization experiment"""
    print("⚡ Experiment 5: Multi-Objective Charging Optimization")
    print("=" * 50)

    # Initialize optimization suite
    optimization_suite = ChargingOptimizationSuite()

    # Create synthetic charging sessions for optimization
    print("🚗 Creating synthetic charging sessions...")
    charging_sessions = optimization_suite.create_synthetic_sessions(n_sessions=15, random_seed=42)

    print(f"📊 Generated {len(charging_sessions)} charging sessions:")
    total_energy = 0
    for i, session in enumerate(charging_sessions):
        energy = session.energy_required()
        time_available = session.max_charging_time()
        total_energy += energy
        print(f"   - Session {i+1}: {energy:.1f} kWh needed, {time_available:.1f}h available")
        if i >= 4:  # Show first 5 sessions
            print("   ... and more")
            break
    
    print(f"\n📈 Total energy demand: {total_energy:.1f} kWh")

    # Create realistic grid constraints
    grid_constraints = optimization_suite.create_realistic_grid_constraints()
    print(f"\n🏭 Grid constraints configured:")
    print(f"   - Max total load: {grid_constraints.max_total_load} kW")
    print(f"   - Peak load penalty: ${grid_constraints.peak_load_penalty}/kW")
    print(f"   - Time-of-use rates: {len(grid_constraints.time_of_use_rates)} hourly rates")
    
    # Show sample TOU rates
    print("   - Sample TOU rates (first 6 hours):")
    for hour, rate in list(grid_constraints.time_of_use_rates.items())[:6]:
        print(f"     Hour {hour}: ${rate:.3f}/kWh")

    # Define optimization objectives
    objectives = [
        OptimizationObjective.MINIMIZE_COST,
        OptimizationObjective.MINIMIZE_PEAK_LOAD,
        OptimizationObjective.MAXIMIZE_USER_SATISFACTION
    ]

    print(f"\n🎯 Optimization objectives: {len(objectives)} objectives defined")
    for i, obj in enumerate(objectives, 1):
        print(f"   {i}. {obj.value}")

    # Run optimization comparison study
    print("\n🚀 Running Optimization Comparison Study")
    print("=" * 50)

    # Run comparison across all algorithms
    print("🔄 Testing multiple optimization algorithms...")
    optimization_results = optimization_suite.run_comparison_study(
        sessions=charging_sessions,
        constraints=grid_constraints,
        objectives=objectives
    )

    # Generate comparison report
    if optimization_results:
        print(f"\n📊 Tested {len(optimization_results)} optimization algorithms")
        
        comparison_df = optimization_suite.generate_comparison_report()
        print("\n📋 Optimization Algorithm Comparison:")
        print(comparison_df.to_string(index=False))
        
        # Analyze results
        print("\n📈 Performance Analysis:")
        if not comparison_df.empty:
            best_cost_idx = comparison_df['Total Cost ($)'].idxmin()
            best_peak_idx = comparison_df['Peak Load (kW)'].idxmin()
            best_satisfaction_idx = comparison_df['User Satisfaction'].idxmax()
            
            print(f"   - Lowest Cost: {comparison_df.loc[best_cost_idx, 'Algorithm']} "
                  f"(${comparison_df.loc[best_cost_idx, 'Total Cost ($)']:.2f})")
            print(f"   - Lowest Peak Load: {comparison_df.loc[best_peak_idx, 'Algorithm']} "
                  f"({comparison_df.loc[best_peak_idx, 'Peak Load (kW)']:.1f} kW)")
            print(f"   - Highest User Satisfaction: {comparison_df.loc[best_satisfaction_idx, 'Algorithm']} "
                  f"({comparison_df.loc[best_satisfaction_idx, 'User Satisfaction']:.3f})")
        
        # Pareto efficiency analysis
        pareto_analysis = optimization_suite.get_pareto_analysis()
        if pareto_analysis:
            print(f"\n🏆 Pareto Analysis:")
            print(f"   - Pareto optimal solutions: {pareto_analysis.get('pareto_optimal', 0)}")
            print(f"   - Trade-off solutions: {pareto_analysis.get('trade_off_solutions', 0)}")
            
            # Show trade-offs
            if 'trade_offs' in pareto_analysis:
                print("\n⚖️ Key Trade-offs Identified:")
                for trade_off in pareto_analysis['trade_offs'][:3]:
                    print(f"   - {trade_off}")

    # Test algorithm sensitivity
    print("\n🔬 Algorithm Sensitivity Analysis:")
    
    # Test with different session counts
    session_counts = [5, 10, 20]
    for n_sessions in session_counts:
        test_sessions = optimization_suite.create_synthetic_sessions(n_sessions=n_sessions, random_seed=42)
        test_results = optimization_suite.run_comparison_study(
            sessions=test_sessions,
            constraints=grid_constraints,
            objectives=objectives
        )
        
        if test_results:
            avg_cost = np.mean([res.total_cost for res in test_results.values() if res])
            print(f"   - {n_sessions} sessions: Average cost ${avg_cost:.2f}")

    print("\n✅ Optimization experiment completed!")
    
    return optimization_results, comparison_df


if __name__ == "__main__":
    main()