"""
Data Loading and Preprocessing Pipeline for EV Charging Research

This module handles loading, cleaning, and preprocessing the EV charging dataset
for federated learning and optimization research.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class EVChargingDataLoader:
    """
    Comprehensive data loader for EV charging optimization research.
    
    Handles missing values, feature engineering, and temporal splits
    suitable for federated learning experiments.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the CSV file containing vehicle charging data
        """
        self.data_path = data_path
        self.raw_data = None
        self.processed_data = None
        self.feature_columns = []
        self.target_columns = []
        
    def load_raw_data(self) -> pd.DataFrame:
        """Load raw data from CSV file."""
        try:
            self.raw_data = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.raw_data)} records from {self.data_path}")
            logger.info(f"Dataset shape: {self.raw_data.shape}")
            return self.raw_data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def analyze_missing_values(self) -> Dict[str, float]:
        """Analyze missing values in the dataset."""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
            
        missing_analysis = {}
        for col in self.raw_data.columns:
            missing_count = self.raw_data[col].isnull().sum()
            missing_percent = (missing_count / len(self.raw_data)) * 100
            missing_analysis[col] = missing_percent
            
        # Sort by missing percentage
        missing_analysis = dict(sorted(missing_analysis.items(), 
                                     key=lambda x: x[1], reverse=True))
        
        logger.info("Missing value analysis completed")
        return missing_analysis
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean and preprocess the raw data.
        
        Returns:
            Cleaned DataFrame ready for feature engineering
        """
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_raw_data() first.")
            
        df = self.raw_data.copy()
        
        # Convert numeric columns with mixed types
        numeric_columns = [
            'Alternative Fuel Economy Combined',
            'Conventional Fuel Economy Combined',
            'Electric-Only Range',
            'Total Range',
            'Charging Rate Level 2 (kW)',
            'Charging Rate DC Fast (kW)',
            'Charging Speed Level 2 (miles added per hour of charging)',
            'Battery Capacity kWh'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle missing charging data by creating synthetic sessions
        df = self._create_charging_sessions(df)
        
        # Clean categorical variables
        df = self._clean_categorical_variables(df)
        
        # Remove outliers
        df = self._remove_outliers(df)
        
        logger.info(f"Data cleaning completed. Shape: {df.shape}")
        return df
    
    def _create_charging_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic charging sessions for research purposes.
        
        Since the original dataset lacks temporal charging data,
        we create realistic charging sessions based on vehicle characteristics.
        """
        # Create charging sessions for vehicles with charging capabilities
        charging_vehicles = df[df['Fuel'].str.contains('Electric', na=False)].copy()
        
        if len(charging_vehicles) == 0:
            logger.warning("No electric vehicles found for charging session creation")
            return df
        
        # Generate synthetic charging data
        np.random.seed(42)  # For reproducibility
        
        sessions_per_vehicle = np.random.poisson(3, len(charging_vehicles))  # 3 sessions on average
        
        charging_sessions = []
        session_id = 0
        
        for idx, (_, vehicle) in enumerate(charging_vehicles.iterrows()):
            n_sessions = sessions_per_vehicle[idx]
            
            for session in range(n_sessions):
                # Generate realistic charging session
                battery_capacity = vehicle.get('Battery Capacity kWh', 50)  # Default 50 kWh
                if pd.isna(battery_capacity):
                    battery_capacity = 50
                
                # Random start time within the last year
                base_date = datetime(2023, 1, 1)
                random_days = np.random.randint(0, 365)
                random_hours = np.random.randint(0, 24)
                start_time = base_date + timedelta(days=random_days, hours=random_hours)
                
                # Charging session characteristics
                initial_soc = np.random.uniform(0.1, 0.8)  # Start between 10-80% SOC
                final_soc = np.random.uniform(initial_soc + 0.1, 1.0)  # End between initial+10% and 100%
                
                energy_consumed = battery_capacity * (final_soc - initial_soc)
                
                # Charging power based on charger type
                if np.random.random() < 0.7:  # 70% Level 2 charging
                    charging_power = np.random.uniform(3.3, 11)  # Level 2: 3.3-11 kW
                    charger_type = "Level_2"
                else:  # 30% DC Fast charging
                    charging_power = np.random.uniform(50, 150)  # DC Fast: 50-150 kW
                    charger_type = "DC_Fast"
                
                # Duration calculation (with realistic charging curves)
                avg_charging_rate = charging_power * 0.8  # 80% efficiency
                duration_hours = energy_consumed / avg_charging_rate
                duration_seconds = duration_hours * 3600
                
                session_data = vehicle.copy()
                session_data['Session_ID'] = session_id
                session_data['Start Time'] = start_time
                session_data['Meter Start (Wh)'] = int(energy_consumed * 1000 * initial_soc)
                session_data['Meter End(Wh)'] = int(energy_consumed * 1000 * final_soc)
                session_data['Meter Total(Wh)'] = int(energy_consumed * 1000)
                session_data['Total Duration (s)'] = int(duration_seconds)
                session_data['Charger_name'] = f"{charger_type}_Station_{np.random.randint(1, 100)}"
                session_data['Charging_Power_kW'] = charging_power
                session_data['Initial_SOC'] = initial_soc
                session_data['Final_SOC'] = final_soc
                
                charging_sessions.append(session_data)
                session_id += 1
        
        if charging_sessions:
            charging_df = pd.DataFrame(charging_sessions)
            logger.info(f"Created {len(charging_sessions)} synthetic charging sessions")
            return charging_df
        else:
            return df
    
    def _clean_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize categorical variables."""
        # Standardize fuel types
        if 'Fuel' in df.columns:
            df['Fuel'] = df['Fuel'].fillna('Unknown')
            df['Is_Electric'] = df['Fuel'].str.contains('Electric', na=False)
            df['Is_Hybrid'] = df['Fuel'].str.contains('Hybrid', na=False)
        
        # Clean manufacturer names
        if 'Manufacturer' in df.columns:
            df['Manufacturer'] = df['Manufacturer'].fillna('Unknown')
        
        # Clean vehicle categories
        if 'Category' in df.columns:
            df['Category'] = df['Category'].fillna('Unknown')
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove statistical outliers from key numeric columns only."""
        # Only remove outliers from key charging metrics to avoid removing all data
        outlier_columns = [
            'Meter Total(Wh)', 'Total Duration (s)', 'Charging_Power_kW', 
            'Battery Capacity kWh', 'Charging_Rate_kW'
        ]
        
        for col in outlier_columns:
            if col in df.columns and df[col].notna().sum() > 0:
                Q1 = df[col].quantile(0.05)  # Use more conservative percentiles
                Q3 = df[col].quantile(0.95)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR  # More lenient bounds
                upper_bound = Q3 + 3 * IQR
                
                initial_count = len(df)
                df = df[(df[col] >= lower_bound) | (df[col] <= upper_bound) | df[col].isna()]
                removed_count = initial_count - len(df)
                
                if removed_count > 0:
                    logger.info(f"Removed {removed_count} outliers from {col}")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for machine learning models.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        feature_df = df.copy()
        
        # Temporal features
        if 'Start Time' in feature_df.columns:
            feature_df['Start Time'] = pd.to_datetime(feature_df['Start Time'])
            feature_df['Hour'] = feature_df['Start Time'].dt.hour
            feature_df['DayOfWeek'] = feature_df['Start Time'].dt.dayofweek
            feature_df['Month'] = feature_df['Start Time'].dt.month
            feature_df['IsWeekend'] = feature_df['DayOfWeek'].isin([5, 6])
            feature_df['Season'] = feature_df['Month'].map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        # Vehicle efficiency features
        if 'Battery Capacity kWh' in feature_df.columns:
            feature_df['Battery_Capacity_Normalized'] = (
                feature_df['Battery Capacity kWh'] / 
                feature_df['Battery Capacity kWh'].max()
            )
        
        # Charging efficiency features
        if 'Meter Total(Wh)' in feature_df.columns and 'Total Duration (s)' in feature_df.columns:
            feature_df['Charging_Rate_kW'] = (
                feature_df['Meter Total(Wh)'] / 1000
            ) / (feature_df['Total Duration (s)'] / 3600)
            
            feature_df['Charging_Efficiency'] = feature_df['Charging_Rate_kW'] / (
                feature_df.get('Charging_Power_kW', feature_df['Charging_Rate_kW'])
            )
        
        # Demand prediction targets
        if 'Hour' in feature_df.columns:
            # Create hourly demand aggregation
            hourly_demand = feature_df.groupby(['Hour']).agg({
                'Meter Total(Wh)': 'sum',
                'Session_ID': 'count'
            }).reset_index()
            
            hourly_demand.columns = ['Hour', 'Total_Energy_Demand_Wh', 'Session_Count']
            feature_df = feature_df.merge(hourly_demand, on='Hour', how='left')
        
        # One-hot encode categorical variables
        categorical_cols = ['Manufacturer', 'Category', 'Fuel', 'Season']
        for col in categorical_cols:
            if col in feature_df.columns:
                dummies = pd.get_dummies(feature_df[col], prefix=col)
                feature_df = pd.concat([feature_df, dummies], axis=1)
        
        logger.info(f"Feature engineering completed. Shape: {feature_df.shape}")
        self.processed_data = feature_df
        return feature_df
    
    def create_federated_splits(self, df: pd.DataFrame, n_clients: int = 10) -> Dict[int, pd.DataFrame]:
        """
        Create data splits for federated learning simulation.
        
        Args:
            df: Processed DataFrame
            n_clients: Number of federated clients to simulate
            
        Returns:
            Dictionary mapping client_id to their data subset
        """
        if 'Manufacturer' not in df.columns:
            raise ValueError("Manufacturer column required for realistic federated splits")
        
        # Create realistic non-IID splits based on manufacturers
        manufacturers = df['Manufacturer'].value_counts()
        
        client_data = {}
        client_id = 0
        
        # Primary manufacturers get their own clients
        for manufacturer in manufacturers.head(n_clients // 2).index:
            manufacturer_data = df[df['Manufacturer'] == manufacturer].copy()
            if len(manufacturer_data) > 10:  # Minimum data per client
                client_data[client_id] = manufacturer_data
                client_id += 1
        
        # Remaining data distributed among remaining clients
        remaining_data = df[~df['Manufacturer'].isin(client_data.keys())]
        if len(remaining_data) > 0:
            chunk_size = len(remaining_data) // (n_clients - len(client_data))
            
            for i in range(n_clients - len(client_data)):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < (n_clients - len(client_data) - 1) else len(remaining_data)
                
                if start_idx < len(remaining_data):
                    client_data[client_id] = remaining_data.iloc[start_idx:end_idx].copy()
                    client_id += 1
        
        # Log client data distribution
        for client_id, data in client_data.items():
            logger.info(f"Client {client_id}: {len(data)} samples, "
                       f"Manufacturers: {data['Manufacturer'].nunique()}")
        
        return client_data
    
    def get_feature_target_split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split processed data into features and targets for ML models.
        
        Args:
            df: Processed DataFrame with engineered features
            
        Returns:
            Tuple of (features_df, targets_df)
        """
        # Define target columns for different prediction tasks
        target_columns = [
            'Meter Total(Wh)',           # Energy demand prediction
            'Total Duration (s)',        # Charging duration prediction
            'Charging_Rate_kW',          # Charging rate prediction
            'Total_Energy_Demand_Wh',    # Hourly demand prediction
            'Session_Count'              # Session count prediction
        ]
        
        # Define feature columns (exclude targets and non-feature columns)
        exclude_columns = target_columns + [
            'Vehicle ID', 'Session_ID', 'Start Time', 'Meter Start (Wh)', 
            'Meter End(Wh)', 'Manufacturer URL', 'Notes', 'Charger_name'
        ]
        
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Handle missing target values
        targets_df = df[target_columns].copy()
        features_df = df[feature_columns].copy()
        
        # Fill missing features with appropriate defaults
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        categorical_features = features_df.select_dtypes(include=['object', 'bool']).columns
        
        features_df[numeric_features] = features_df[numeric_features].fillna(0)
        features_df[categorical_features] = features_df[categorical_features].fillna('Unknown')
        
        logger.info(f"Feature matrix shape: {features_df.shape}")
        logger.info(f"Target matrix shape: {targets_df.shape}")
        
        return features_df, targets_df
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """Save processed data to file."""
        try:
            df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise