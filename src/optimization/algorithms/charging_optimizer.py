"""
Multi-Objective Charging Optimization Algorithms for EV Charging Research

This module implements various optimization algorithms for EV charging scheduling
considering multiple objectives: cost minimization, peak load reduction, 
user satisfaction, and grid stability.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import scipy.optimize as opt
from scipy.optimize import differential_evolution, minimize
import pulp
import warnings
warnings.filterwarnings('ignore')


class OptimizationObjective(Enum):
    """Types of optimization objectives."""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_PEAK_LOAD = "minimize_peak_load"
    MAXIMIZE_USER_SATISFACTION = "maximize_user_satisfaction"
    MAXIMIZE_GRID_STABILITY = "maximize_grid_stability"
    MINIMIZE_WAIT_TIME = "minimize_wait_time"


@dataclass
class ChargingSession:
    """Represents a charging session request."""
    session_id: str
    vehicle_id: str
    arrival_time: float  # Hours from start of optimization period
    departure_time: float
    initial_soc: float  # State of charge (0-1)
    target_soc: float
    battery_capacity: float  # kWh
    max_charging_rate: float  # kW
    charging_efficiency: float = 0.9
    price_sensitivity: float = 1.0  # User's price sensitivity
    urgency: float = 1.0  # How urgent the charging is
    
    def energy_required(self) -> float:
        """Calculate energy required for this session."""
        return (self.target_soc - self.initial_soc) * self.battery_capacity
    
    def max_charging_time(self) -> float:
        """Calculate maximum available charging time."""
        return self.departure_time - self.arrival_time
    
    def min_charging_time(self) -> float:
        """Calculate minimum time needed to reach target SOC."""
        energy_needed = self.energy_required()
        return energy_needed / (self.max_charging_rate * self.charging_efficiency)


@dataclass
class GridConstraints:
    """Grid operational constraints."""
    max_total_load: float  # kW
    peak_load_penalty: float  # Cost per kW over threshold
    time_of_use_rates: Dict[int, float]  # Hour -> price per kWh
    renewable_availability: Dict[int, float]  # Hour -> renewable fraction
    voltage_constraints: Tuple[float, float] = (0.95, 1.05)  # Min, max voltage
    
    def get_electricity_price(self, hour: int) -> float:
        """Get electricity price for given hour."""
        return self.time_of_use_rates.get(hour % 24, 0.1)
    
    def get_renewable_fraction(self, hour: int) -> float:
        """Get renewable energy fraction for given hour."""
        return self.renewable_availability.get(hour % 24, 0.3)


@dataclass
class OptimizationResult:
    """Results from charging optimization."""
    charging_schedule: Dict[str, List[float]]  # session_id -> hourly charging rates
    total_cost: float
    peak_load: float
    user_satisfaction: float
    grid_stability: float
    objective_values: Dict[str, float]
    execution_time: float
    convergence_info: Dict[str, Any] = field(default_factory=dict)


class BaseChargingOptimizer:
    """Base class for charging optimization algorithms."""
    
    def __init__(self, time_horizon: int = 24, time_resolution: float = 1.0):
        """
        Initialize optimizer.
        
        Args:
            time_horizon: Optimization time horizon in hours
            time_resolution: Time resolution in hours (e.g., 0.25 for 15-min intervals)
        """
        self.time_horizon = time_horizon
        self.time_resolution = time_resolution
        self.time_steps = int(time_horizon / time_resolution)
        
    def optimize(self, sessions: List[ChargingSession], 
                constraints: GridConstraints,
                objectives: List[OptimizationObjective],
                objective_weights: Optional[Dict[OptimizationObjective, float]] = None) -> OptimizationResult:
        """
        Optimize charging schedule.
        
        Args:
            sessions: List of charging sessions to schedule
            constraints: Grid constraints
            objectives: List of optimization objectives
            objective_weights: Weights for multi-objective optimization
            
        Returns:
            Optimization result
        """
        raise NotImplementedError("Subclasses must implement optimize method")
    
    def calculate_cost(self, schedule: Dict[str, List[float]], 
                      sessions: List[ChargingSession],
                      constraints: GridConstraints) -> float:
        """Calculate total charging cost."""
        total_cost = 0.0
        
        for hour in range(self.time_steps):
            hourly_load = sum(
                schedule[session.session_id][hour] 
                for session in sessions 
                if session.session_id in schedule
            )
            
            # Energy cost
            electricity_price = constraints.get_electricity_price(hour)
            energy_cost = hourly_load * self.time_resolution * electricity_price
            
            # Peak load penalty
            if hourly_load > constraints.max_total_load:
                peak_penalty = (hourly_load - constraints.max_total_load) * constraints.peak_load_penalty
                energy_cost += peak_penalty
            
            total_cost += energy_cost
        
        return total_cost
    
    def calculate_peak_load(self, schedule: Dict[str, List[float]]) -> float:
        """Calculate peak load."""
        hourly_loads = []
        for hour in range(self.time_steps):
            hourly_load = sum(
                schedule[session_id][hour] 
                for session_id, hourly_schedule in schedule.items()
            )
            hourly_loads.append(hourly_load)
        
        return max(hourly_loads)
    
    def calculate_user_satisfaction(self, schedule: Dict[str, List[float]],
                                  sessions: List[ChargingSession]) -> float:
        """Calculate average user satisfaction."""
        satisfactions = []
        
        for session in sessions:
            if session.session_id not in schedule:
                satisfactions.append(0.0)
                continue
            
            session_schedule = schedule[session.session_id]
            
            # Calculate energy delivered
            energy_delivered = sum(
                session_schedule[hour] * self.time_resolution * session.charging_efficiency
                for hour in range(self.time_steps)
            )
            
            # Energy satisfaction
            energy_needed = session.energy_required()
            energy_satisfaction = min(1.0, energy_delivered / energy_needed) if energy_needed > 0 else 1.0
            
            # Time satisfaction (prefer earlier completion)
            total_charging_time = sum(1 for rate in session_schedule if rate > 0) * self.time_resolution
            available_time = session.max_charging_time()
            time_satisfaction = 1.0 - (total_charging_time / available_time) if available_time > 0 else 1.0
            
            # Combined satisfaction
            satisfaction = 0.7 * energy_satisfaction + 0.3 * time_satisfaction
            satisfactions.append(satisfaction)
        
        return np.mean(satisfactions) if satisfactions else 0.0
    
    def calculate_grid_stability(self, schedule: Dict[str, List[float]],
                               constraints: GridConstraints) -> float:
        """Calculate grid stability metric."""
        hourly_loads = []
        renewable_alignment = []
        
        for hour in range(self.time_steps):
            hourly_load = sum(
                schedule[session_id][hour] 
                for session_id, hourly_schedule in schedule.items()
            )
            hourly_loads.append(hourly_load)
            
            # Renewable energy alignment
            renewable_fraction = constraints.get_renewable_fraction(hour)
            renewable_alignment.append(hourly_load * renewable_fraction)
        
        # Load variance (lower is better)
        load_variance = np.var(hourly_loads)
        max_load = max(hourly_loads) if hourly_loads else 1.0
        normalized_variance = load_variance / (max_load ** 2) if max_load > 0 else 0.0
        
        # Renewable alignment (higher is better)
        total_renewable = sum(renewable_alignment)
        total_load = sum(hourly_loads)
        renewable_utilization = total_renewable / total_load if total_load > 0 else 0.0
        
        # Combined stability score
        stability = 0.6 * (1 - normalized_variance) + 0.4 * renewable_utilization
        return max(0.0, min(1.0, stability))


class GreedyChargingOptimizer(BaseChargingOptimizer):
    """
    Simple greedy charging optimizer.
    
    Schedules charging during cheapest available hours for each session.
    """
    
    def optimize(self, sessions: List[ChargingSession], 
                constraints: GridConstraints,
                objectives: List[OptimizationObjective],
                objective_weights: Optional[Dict[OptimizationObjective, float]] = None) -> OptimizationResult:
        """Optimize using greedy algorithm."""
        import time
        start_time = time.time()
        
        schedule = {}
        
        # Sort sessions by urgency
        sorted_sessions = sorted(sessions, key=lambda s: s.urgency, reverse=True)
        
        for session in sorted_sessions:
            session_schedule = [0.0] * self.time_steps
            
            # Available time slots
            start_slot = int(session.arrival_time / self.time_resolution)
            end_slot = int(session.departure_time / self.time_resolution)
            
            # Calculate energy needed
            energy_needed = session.energy_required()
            
            # Get electricity prices for available slots
            slot_prices = []
            for slot in range(start_slot, min(end_slot, self.time_steps)):
                hour = int(slot * self.time_resolution)
                price = constraints.get_electricity_price(hour)
                slot_prices.append((slot, price))
            
            # Sort by price (greedy approach)
            slot_prices.sort(key=lambda x: x[1])
            
            # Schedule charging in cheapest slots
            remaining_energy = energy_needed
            for slot, _ in slot_prices:
                if remaining_energy <= 0:
                    break
                
                # Maximum charging in this slot
                max_charging = min(
                    session.max_charging_rate,
                    remaining_energy / (self.time_resolution * session.charging_efficiency)
                )
                
                session_schedule[slot] = max_charging
                remaining_energy -= max_charging * self.time_resolution * session.charging_efficiency
            
            schedule[session.session_id] = session_schedule
        
        # Calculate metrics
        total_cost = self.calculate_cost(schedule, sessions, constraints)
        peak_load = self.calculate_peak_load(schedule)
        user_satisfaction = self.calculate_user_satisfaction(schedule, sessions)
        grid_stability = self.calculate_grid_stability(schedule, constraints)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            charging_schedule=schedule,
            total_cost=total_cost,
            peak_load=peak_load,
            user_satisfaction=user_satisfaction,
            grid_stability=grid_stability,
            objective_values={
                'cost': total_cost,
                'peak_load': peak_load,
                'user_satisfaction': user_satisfaction,
                'grid_stability': grid_stability
            },
            execution_time=execution_time
        )


class LinearProgrammingOptimizer(BaseChargingOptimizer):
    """
    Linear programming optimizer for charging scheduling.
    
    Formulates the charging optimization as a linear program.
    """
    
    def optimize(self, sessions: List[ChargingSession], 
                constraints: GridConstraints,
                objectives: List[OptimizationObjective],
                objective_weights: Optional[Dict[OptimizationObjective, float]] = None) -> OptimizationResult:
        """Optimize using linear programming."""
        import time
        start_time = time.time()
        
        if objective_weights is None:
            objective_weights = {obj: 1.0 for obj in objectives}
        
        # Create LP problem
        prob = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMinimize)
        
        # Decision variables: charging rate for each session at each time slot
        charging_vars = {}
        for session in sessions:
            for hour in range(self.time_steps):
                var_name = f"charge_{session.session_id}_{hour}"
                charging_vars[(session.session_id, hour)] = pulp.LpVariable(
                    var_name, 
                    lowBound=0, 
                    upBound=session.max_charging_rate
                )
        
        # Auxiliary variables for peak load
        peak_load_var = pulp.LpVariable("peak_load", lowBound=0)
        
        # Objective function
        objective = 0
        
        # Cost objective
        if OptimizationObjective.MINIMIZE_COST in objectives:
            weight = objective_weights.get(OptimizationObjective.MINIMIZE_COST, 1.0)
            for hour in range(self.time_steps):
                price = constraints.get_electricity_price(hour)
                hourly_charging = pulp.lpSum([
                    charging_vars[(session.session_id, hour)] 
                    for session in sessions
                ])
                objective += weight * price * hourly_charging * self.time_resolution
        
        # Peak load objective
        if OptimizationObjective.MINIMIZE_PEAK_LOAD in objectives:
            weight = objective_weights.get(OptimizationObjective.MINIMIZE_PEAK_LOAD, 1.0)
            objective += weight * peak_load_var
        
        prob += objective
        
        # Constraints
        
        # Energy requirements
        for session in sessions:
            energy_needed = session.energy_required()
            total_energy = pulp.lpSum([
                charging_vars[(session.session_id, hour)] * self.time_resolution * session.charging_efficiency
                for hour in range(self.time_steps)
            ])
            prob += total_energy >= energy_needed
        
        # Time availability constraints
        for session in sessions:
            start_slot = int(session.arrival_time / self.time_resolution)
            end_slot = int(session.departure_time / self.time_resolution)
            
            for hour in range(self.time_steps):
                if hour < start_slot or hour >= end_slot:
                    prob += charging_vars[(session.session_id, hour)] == 0
        
        # Grid capacity constraints
        for hour in range(self.time_steps):
            hourly_load = pulp.lpSum([
                charging_vars[(session.session_id, hour)] 
                for session in sessions
            ])
            prob += hourly_load <= constraints.max_total_load
            
            # Peak load constraints
            prob += peak_load_var >= hourly_load
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        schedule = {}
        for session in sessions:
            session_schedule = []
            for hour in range(self.time_steps):
                charging_rate = charging_vars[(session.session_id, hour)].varValue or 0.0
                session_schedule.append(charging_rate)
            schedule[session.session_id] = session_schedule
        
        # Calculate metrics
        total_cost = self.calculate_cost(schedule, sessions, constraints)
        peak_load = self.calculate_peak_load(schedule)
        user_satisfaction = self.calculate_user_satisfaction(schedule, sessions)
        grid_stability = self.calculate_grid_stability(schedule, constraints)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            charging_schedule=schedule,
            total_cost=total_cost,
            peak_load=peak_load,
            user_satisfaction=user_satisfaction,
            grid_stability=grid_stability,
            objective_values={
                'cost': total_cost,
                'peak_load': peak_load,
                'user_satisfaction': user_satisfaction,
                'grid_stability': grid_stability
            },
            execution_time=execution_time,
            convergence_info={'status': pulp.LpStatus[prob.status]}
        )


class GeneticAlgorithmOptimizer(BaseChargingOptimizer):
    """
    Genetic algorithm optimizer for multi-objective charging optimization.
    
    Uses differential evolution for continuous optimization.
    """
    
    def __init__(self, time_horizon: int = 24, time_resolution: float = 1.0,
                 population_size: int = 50, max_generations: int = 100):
        """Initialize GA optimizer."""
        super().__init__(time_horizon, time_resolution)
        self.population_size = population_size
        self.max_generations = max_generations
    
    def optimize(self, sessions: List[ChargingSession], 
                constraints: GridConstraints,
                objectives: List[OptimizationObjective],
                objective_weights: Optional[Dict[OptimizationObjective, float]] = None) -> OptimizationResult:
        """Optimize using genetic algorithm."""
        import time
        start_time = time.time()
        
        if objective_weights is None:
            objective_weights = {obj: 1.0 for obj in objectives}
        
        # Prepare bounds for decision variables
        bounds = []
        session_indices = {}
        var_index = 0
        
        for i, session in enumerate(sessions):
            session_indices[session.session_id] = []
            for hour in range(self.time_steps):
                # Check if this hour is within the session's availability
                hour_time = hour * self.time_resolution
                if session.arrival_time <= hour_time < session.departure_time:
                    bounds.append((0, session.max_charging_rate))
                else:
                    bounds.append((0, 0))  # Cannot charge outside availability
                
                session_indices[session.session_id].append(var_index)
                var_index += 1
        
        def objective_function(x):
            """Multi-objective function to minimize."""
            # Convert decision variables to schedule format
            schedule = {}
            for session in sessions:
                session_schedule = []
                for hour in range(self.time_steps):
                    var_idx = session_indices[session.session_id][hour]
                    session_schedule.append(x[var_idx])
                schedule[session.session_id] = session_schedule
            
            # Calculate objectives
            total_objective = 0.0
            
            if OptimizationObjective.MINIMIZE_COST in objectives:
                cost = self.calculate_cost(schedule, sessions, constraints)
                weight = objective_weights.get(OptimizationObjective.MINIMIZE_COST, 1.0)
                total_objective += weight * cost / 1000.0  # Normalize
            
            if OptimizationObjective.MINIMIZE_PEAK_LOAD in objectives:
                peak_load = self.calculate_peak_load(schedule)
                weight = objective_weights.get(OptimizationObjective.MINIMIZE_PEAK_LOAD, 1.0)
                total_objective += weight * peak_load / 100.0  # Normalize
            
            if OptimizationObjective.MAXIMIZE_USER_SATISFACTION in objectives:
                satisfaction = self.calculate_user_satisfaction(schedule, sessions)
                weight = objective_weights.get(OptimizationObjective.MAXIMIZE_USER_SATISFACTION, 1.0)
                total_objective += weight * (1.0 - satisfaction)  # Minimize negative satisfaction
            
            if OptimizationObjective.MAXIMIZE_GRID_STABILITY in objectives:
                stability = self.calculate_grid_stability(schedule, constraints)
                weight = objective_weights.get(OptimizationObjective.MAXIMIZE_GRID_STABILITY, 1.0)
                total_objective += weight * (1.0 - stability)  # Minimize negative stability
            
            # Add constraint penalties
            penalty = 0.0
            
            # Energy requirement constraints
            for session in sessions:
                session_schedule = schedule[session.session_id]
                energy_delivered = sum(
                    session_schedule[hour] * self.time_resolution * session.charging_efficiency
                    for hour in range(self.time_steps)
                )
                energy_needed = session.energy_required()
                
                if energy_delivered < energy_needed * 0.95:  # Allow 5% tolerance
                    penalty += 1000.0 * (energy_needed - energy_delivered)
            
            # Grid capacity constraints
            for hour in range(self.time_steps):
                hourly_load = sum(schedule[session.session_id][hour] for session in sessions)
                if hourly_load > constraints.max_total_load:
                    penalty += 1000.0 * (hourly_load - constraints.max_total_load)
            
            return total_objective + penalty
        
        # Run differential evolution
        result = differential_evolution(
            objective_function,
            bounds,
            popsize=self.population_size // len(bounds) + 1,
            maxiter=self.max_generations,
            seed=42,
            disp=False
        )
        
        # Extract final schedule
        x_optimal = result.x
        schedule = {}
        for session in sessions:
            session_schedule = []
            for hour in range(self.time_steps):
                var_idx = session_indices[session.session_id][hour]
                session_schedule.append(x_optimal[var_idx])
            schedule[session.session_id] = session_schedule
        
        # Calculate final metrics
        total_cost = self.calculate_cost(schedule, sessions, constraints)
        peak_load = self.calculate_peak_load(schedule)
        user_satisfaction = self.calculate_user_satisfaction(schedule, sessions)
        grid_stability = self.calculate_grid_stability(schedule, constraints)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            charging_schedule=schedule,
            total_cost=total_cost,
            peak_load=peak_load,
            user_satisfaction=user_satisfaction,
            grid_stability=grid_stability,
            objective_values={
                'cost': total_cost,
                'peak_load': peak_load,
                'user_satisfaction': user_satisfaction,
                'grid_stability': grid_stability
            },
            execution_time=execution_time,
            convergence_info={
                'success': result.success,
                'nfev': result.nfev,
                'nit': result.nit,
                'fun': result.fun
            }
        )


class ReinforcementLearningOptimizer(BaseChargingOptimizer):
    """
    Reinforcement learning optimizer (simplified version for research).
    
    Uses a simple Q-learning approach for demonstration purposes.
    """
    
    def __init__(self, time_horizon: int = 24, time_resolution: float = 1.0,
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.1):
        """Initialize RL optimizer."""
        super().__init__(time_horizon, time_resolution)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.q_table = {}
    
    def optimize(self, sessions: List[ChargingSession], 
                constraints: GridConstraints,
                objectives: List[OptimizationObjective],
                objective_weights: Optional[Dict[OptimizationObjective, float]] = None) -> OptimizationResult:
        """Optimize using reinforcement learning (simplified)."""
        import time
        start_time = time.time()
        
        # For simplicity, use a greedy approach with some randomization
        # In a full implementation, this would be a proper RL agent
        
        schedule = {}
        
        for session in sessions:
            session_schedule = [0.0] * self.time_steps
            
            # State: available time slots and energy needed
            start_slot = int(session.arrival_time / self.time_resolution)
            end_slot = int(session.departure_time / self.time_resolution)
            energy_needed = session.energy_required()
            
            # Simple policy: distribute charging evenly with some randomness
            available_slots = list(range(start_slot, min(end_slot, self.time_steps)))
            
            if available_slots and energy_needed > 0:
                # Add randomness to explore different strategies
                np.random.shuffle(available_slots)
                
                remaining_energy = energy_needed
                for slot in available_slots:
                    if remaining_energy <= 0:
                        break
                    
                    # Random charging rate (exploration)
                    if np.random.random() < self.exploration_rate:
                        charging_rate = np.random.uniform(0, session.max_charging_rate)
                    else:
                        # Greedy charging rate
                        charging_rate = min(
                            session.max_charging_rate,
                            remaining_energy / (self.time_resolution * session.charging_efficiency)
                        )
                    
                    session_schedule[slot] = charging_rate
                    remaining_energy -= charging_rate * self.time_resolution * session.charging_efficiency
            
            schedule[session.session_id] = session_schedule
        
        # Calculate metrics
        total_cost = self.calculate_cost(schedule, sessions, constraints)
        peak_load = self.calculate_peak_load(schedule)
        user_satisfaction = self.calculate_user_satisfaction(schedule, sessions)
        grid_stability = self.calculate_grid_stability(schedule, constraints)
        
        execution_time = time.time() - start_time
        
        return OptimizationResult(
            charging_schedule=schedule,
            total_cost=total_cost,
            peak_load=peak_load,
            user_satisfaction=user_satisfaction,
            grid_stability=grid_stability,
            objective_values={
                'cost': total_cost,
                'peak_load': peak_load,
                'user_satisfaction': user_satisfaction,
                'grid_stability': grid_stability
            },
            execution_time=execution_time,
            convergence_info={'method': 'simplified_rl'}
        )


class ChargingOptimizationSuite:
    """
    Comprehensive suite for comparing different charging optimization algorithms.
    """
    
    def __init__(self):
        """Initialize optimization suite."""
        self.optimizers = {
            'greedy': GreedyChargingOptimizer(),
            'linear_programming': LinearProgrammingOptimizer(),
            'genetic_algorithm': GeneticAlgorithmOptimizer(),
            'reinforcement_learning': ReinforcementLearningOptimizer()
        }
        
        self.results = {}
    
    def create_synthetic_sessions(self, n_sessions: int = 20, 
                                random_seed: int = 42) -> List[ChargingSession]:
        """Create synthetic charging sessions for testing."""
        np.random.seed(random_seed)
        
        sessions = []
        for i in range(n_sessions):
            # Random arrival and departure times
            arrival = np.random.uniform(0, 18)  # Arrive between 0-18 hours
            duration = np.random.uniform(2, 8)   # Stay 2-8 hours
            departure = min(24, arrival + duration)
            
            # Random vehicle characteristics
            battery_capacity = np.random.uniform(30, 100)  # 30-100 kWh
            initial_soc = np.random.uniform(0.1, 0.6)      # 10-60% initial
            target_soc = np.random.uniform(0.7, 1.0)       # 70-100% target
            max_charging_rate = np.random.choice([7.4, 11, 22, 50])  # Common rates
            
            session = ChargingSession(
                session_id=f"session_{i}",
                vehicle_id=f"vehicle_{i}",
                arrival_time=arrival,
                departure_time=departure,
                initial_soc=initial_soc,
                target_soc=target_soc,
                battery_capacity=battery_capacity,
                max_charging_rate=max_charging_rate,
                price_sensitivity=np.random.uniform(0.5, 1.5),
                urgency=np.random.uniform(0.5, 1.5)
            )
            
            sessions.append(session)
        
        return sessions
    
    def create_realistic_grid_constraints(self) -> GridConstraints:
        """Create realistic grid constraints."""
        # Time-of-use pricing (simplified)
        tou_rates = {}
        for hour in range(24):
            if 6 <= hour <= 9 or 17 <= hour <= 21:  # Peak hours
                tou_rates[hour] = 0.25  # $0.25/kWh
            elif 10 <= hour <= 16:  # Mid-peak
                tou_rates[hour] = 0.15  # $0.15/kWh
            else:  # Off-peak
                tou_rates[hour] = 0.08  # $0.08/kWh
        
        # Renewable availability (simplified solar pattern)
        renewable_availability = {}
        for hour in range(24):
            if 6 <= hour <= 18:  # Daylight hours
                # Parabolic pattern peaking at noon
                normalized_hour = (hour - 12) / 6  # -1 to 1
                renewable_fraction = max(0, 0.8 * (1 - normalized_hour**2))
            else:
                renewable_fraction = 0.1  # Some wind at night
            
            renewable_availability[hour] = renewable_fraction
        
        return GridConstraints(
            max_total_load=500.0,  # 500 kW maximum
            peak_load_penalty=10.0,  # $10/kW penalty
            time_of_use_rates=tou_rates,
            renewable_availability=renewable_availability
        )
    
    def run_comparison_study(self, sessions: Optional[List[ChargingSession]] = None,
                           constraints: Optional[GridConstraints] = None,
                           objectives: Optional[List[OptimizationObjective]] = None) -> Dict[str, OptimizationResult]:
        """Run comparison study across all optimization algorithms."""
        
        if sessions is None:
            sessions = self.create_synthetic_sessions()
        
        if constraints is None:
            constraints = self.create_realistic_grid_constraints()
        
        if objectives is None:
            objectives = [
                OptimizationObjective.MINIMIZE_COST,
                OptimizationObjective.MINIMIZE_PEAK_LOAD,
                OptimizationObjective.MAXIMIZE_USER_SATISFACTION
            ]
        
        results = {}
        
        for name, optimizer in self.optimizers.items():
            print(f"Running {name} optimizer...")
            try:
                result = optimizer.optimize(sessions, constraints, objectives)
                results[name] = result
                print(f"  - Cost: ${result.total_cost:.2f}")
                print(f"  - Peak Load: {result.peak_load:.1f} kW")
                print(f"  - User Satisfaction: {result.user_satisfaction:.3f}")
                print(f"  - Execution Time: {result.execution_time:.3f}s")
            except Exception as e:
                print(f"  - Failed: {e}")
                results[name] = None
        
        self.results = results
        return results
    
    def generate_comparison_report(self) -> pd.DataFrame:
        """Generate comparison report of optimization results."""
        if not self.results:
            raise ValueError("No results available. Run comparison study first.")
        
        report_data = []
        
        for name, result in self.results.items():
            if result is not None:
                report_data.append({
                    'Algorithm': name,
                    'Total Cost ($)': result.total_cost,
                    'Peak Load (kW)': result.peak_load,
                    'User Satisfaction': result.user_satisfaction,
                    'Grid Stability': result.grid_stability,
                    'Execution Time (s)': result.execution_time,
                    'Convergence': result.convergence_info.get('success', 'N/A')
                })
        
        return pd.DataFrame(report_data)
    
    def get_pareto_analysis(self) -> Dict[str, Any]:
        """Analyze Pareto efficiency of solutions."""
        if not self.results:
            raise ValueError("No results available.")
        
        # Extract objectives for Pareto analysis
        solutions = []
        names = []
        
        for name, result in self.results.items():
            if result is not None:
                # Use negative values for maximization objectives
                solutions.append([
                    result.total_cost,
                    result.peak_load,
                    -result.user_satisfaction,  # Negative for minimization
                    -result.grid_stability      # Negative for minimization
                ])
                names.append(name)
        
        if not solutions:
            return {}
        
        solutions = np.array(solutions)
        
        # Find Pareto front
        pareto_indices = []
        for i in range(len(solutions)):
            is_pareto = True
            for j in range(len(solutions)):
                if i != j:
                    # Check if solution j dominates solution i
                    if all(solutions[j] <= solutions[i]) and any(solutions[j] < solutions[i]):
                        is_pareto = False
                        break
            if is_pareto:
                pareto_indices.append(i)
        
        pareto_solutions = [names[i] for i in pareto_indices]
        
        return {
            'pareto_optimal': pareto_solutions,
            'all_solutions': list(zip(names, solutions.tolist())),
            'pareto_indices': pareto_indices
        }