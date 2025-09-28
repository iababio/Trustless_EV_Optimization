"""
Baseline Forecasting Models for EV Charging Demand Prediction

This module implements various baseline models for comparison with
federated learning approaches in the research.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 24):
        """
        Initialize the dataset.
        
        Args:
            X: Feature matrix
            y: Target values
            sequence_length: Length of input sequences
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        return (
            self.X[idx:idx + self.sequence_length],
            self.y[idx + self.sequence_length - 1]
        )


class LightweightLSTM(nn.Module):
    """Lightweight LSTM for EV charging demand forecasting."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LightweightLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        """Forward pass."""
        # Handle 2D input by adding sequence dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension (batch_size, seq_len=1, features)
            
        batch_size = x.size(0)
        
        # Initialize hidden state with correct device
        device = x.device
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        # LSTM forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # Take output from the last time step
        if out.size(1) == 1:
            out = out.squeeze(1)  # Remove sequence dimension if seq_len=1
        else:
            out = out[:, -1, :]  # Take last time step
            
        out = self.dropout(out)
        out = self.fc(out)
        
        return out


class BaselineModelSuite:
    """
    Comprehensive suite of baseline models for EV charging demand forecasting.
    
    Includes traditional time series models, machine learning models,
    and deep learning models for comparison with federated approaches.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the baseline model suite.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        torch.manual_seed(random_state)
        
        self.models = {}
        self.scalers = {}
        self.results = {}
        
        # Model configurations
        self.lstm_config = {
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'sequence_length': 24
        }
        
        self.xgb_config = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        
    def prepare_time_series_data(self, data: pd.DataFrame, 
                               target_col: str = 'Meter Total(Wh)',
                               feature_cols: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare time series data for modeling.
        
        Args:
            data: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature columns
            
        Returns:
            Tuple of (features, target)
        """
        if feature_cols is None:
            # Auto-select numeric features
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [col for col in numeric_cols 
                          if col != target_col and 'ID' not in col.upper()]
        
        # Ensure temporal ordering if datetime column exists
        if 'Start Time' in data.columns:
            data = data.sort_values('Start Time')
        
        # Handle missing values
        features = data[feature_cols].fillna(0)
        target = data[target_col].fillna(0)
        
        return features, target
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features for time series forecasting.
        
        Args:
            data: Input DataFrame with datetime column
            
        Returns:
            DataFrame with temporal features
        """
        feature_data = data.copy()
        
        if 'Start Time' in data.columns:
            dt_col = pd.to_datetime(data['Start Time'])
            
            # Basic temporal features
            feature_data['hour'] = dt_col.dt.hour
            feature_data['day_of_week'] = dt_col.dt.dayofweek
            feature_data['month'] = dt_col.dt.month
            feature_data['day_of_year'] = dt_col.dt.dayofyear
            
            # Cyclical features
            feature_data['hour_sin'] = np.sin(2 * np.pi * feature_data['hour'] / 24)
            feature_data['hour_cos'] = np.cos(2 * np.pi * feature_data['hour'] / 24)
            feature_data['dow_sin'] = np.sin(2 * np.pi * feature_data['day_of_week'] / 7)
            feature_data['dow_cos'] = np.cos(2 * np.pi * feature_data['day_of_week'] / 7)
            feature_data['month_sin'] = np.sin(2 * np.pi * feature_data['month'] / 12)
            feature_data['month_cos'] = np.cos(2 * np.pi * feature_data['month'] / 12)
            
            # Lag features
            if 'Meter Total(Wh)' in data.columns:
                for lag in [1, 24, 168]:  # 1 hour, 1 day, 1 week
                    feature_data[f'lag_{lag}'] = data['Meter Total(Wh)'].shift(lag)
                
                # Rolling statistics
                for window in [3, 24, 168]:
                    feature_data[f'rolling_mean_{window}'] = (
                        data['Meter Total(Wh)'].rolling(window=window).mean()
                    )
                    feature_data[f'rolling_std_{window}'] = (
                        data['Meter Total(Wh)'].rolling(window=window).std()
                    )
        
        return feature_data.fillna(0)
    
    def train_naive_models(self, data: pd.DataFrame, 
                          target_col: str = 'Meter Total(Wh)') -> Dict[str, Any]:
        """
        Train naive baseline models.
        
        Args:
            data: Training data
            target_col: Target column name
            
        Returns:
            Dictionary of trained naive models
        """
        naive_models = {}
        
        target_series = data[target_col].fillna(0)
        
        # Persistence model (naive forecast)
        naive_models['persistence'] = {
            'type': 'naive',
            'last_value': target_series.iloc[-1],
            'predictions': lambda steps: [target_series.iloc[-1]] * steps
        }
        
        # Seasonal naive (24-hour seasonality)
        if len(target_series) >= 24:
            seasonal_pattern = target_series.iloc[-24:].values
            naive_models['seasonal_naive'] = {
                'type': 'seasonal_naive',
                'pattern': seasonal_pattern,
                'predictions': lambda steps: list(seasonal_pattern) * (steps // 24 + 1)[:steps]
            }
        
        # Moving average
        for window in [3, 12, 24]:
            if len(target_series) >= window:
                ma_value = target_series.iloc[-window:].mean()
                naive_models[f'moving_avg_{window}'] = {
                    'type': 'moving_average',
                    'window': window,
                    'value': ma_value,
                    'predictions': lambda steps, val=ma_value: [val] * steps
                }
        
        return naive_models
    
    def train_arima_model(self, data: pd.DataFrame, 
                         target_col: str = 'Meter Total(Wh)',
                         order: Tuple[int, int, int] = (2, 1, 2)) -> Dict[str, Any]:
        """
        Train ARIMA model for time series forecasting.
        
        Args:
            data: Training data
            target_col: Target column name
            order: ARIMA order (p, d, q)
            
        Returns:
            Dictionary containing trained ARIMA model
        """
        try:
            target_series = data[target_col].fillna(method='ffill').fillna(0)
            
            # Remove zero variance periods
            if target_series.var() == 0:
                target_series = target_series + np.random.normal(0, 0.001, len(target_series))
            
            # Fit ARIMA model
            model = ARIMA(target_series, order=order)
            fitted_model = model.fit()
            
            return {
                'model': fitted_model,
                'type': 'arima',
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': fitted_model.resid
            }
            
        except Exception as e:
            print(f"ARIMA model training failed: {e}")
            return {'error': str(e)}
    
    def train_machine_learning_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                                    X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train traditional machine learning models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary of trained ML models
        """
        ml_models = {}
        
        # Prepare data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        self.scalers['ml_scaler'] = scaler
        
        # Linear Regression
        try:
            lr_model = LinearRegression()
            lr_model.fit(X_train_scaled, y_train)
            
            train_pred = lr_model.predict(X_train_scaled)
            val_pred = lr_model.predict(X_val_scaled)
            
            ml_models['linear_regression'] = {
                'model': lr_model,
                'type': 'linear_regression',
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'val_mae': mean_absolute_error(y_val, val_pred)
            }
        except Exception as e:
            ml_models['linear_regression'] = {'error': str(e)}
        
        # Random Forest
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                n_jobs=-1
            )
            rf_model.fit(X_train, y_train)  # RF doesn't need scaling
            
            train_pred = rf_model.predict(X_train)
            val_pred = rf_model.predict(X_val)
            
            ml_models['random_forest'] = {
                'model': rf_model,
                'type': 'random_forest',
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'feature_importance': rf_model.feature_importances_
            }
        except Exception as e:
            ml_models['random_forest'] = {'error': str(e)}
        
        # XGBoost
        try:
            xgb_model = xgb.XGBRegressor(**self.xgb_config)
            xgb_model.fit(X_train, y_train, 
                         eval_set=[(X_val, y_val)], 
                         early_stopping_rounds=10, 
                         verbose=False)
            
            train_pred = xgb_model.predict(X_train)
            val_pred = xgb_model.predict(X_val)
            
            ml_models['xgboost'] = {
                'model': xgb_model,
                'type': 'xgboost',
                'train_rmse': np.sqrt(mean_squared_error(y_train, train_pred)),
                'val_rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
                'train_mae': mean_absolute_error(y_train, train_pred),
                'val_mae': mean_absolute_error(y_val, val_pred),
                'feature_importance': xgb_model.feature_importances_
            }
        except Exception as e:
            ml_models['xgboost'] = {'error': str(e)}
        
        return ml_models
    
    def train_lstm_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train LSTM model for time series forecasting.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Dictionary containing trained LSTM model
        """
        try:
            # Prepare data
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_val_scaled = scaler_X.transform(X_val)
            
            y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
            y_val_scaled = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()
            
            self.scalers['lstm_scaler_X'] = scaler_X
            self.scalers['lstm_scaler_y'] = scaler_y
            
            # Create datasets
            train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled, 
                                            self.lstm_config['sequence_length'])
            val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, 
                                          self.lstm_config['sequence_length'])
            
            train_loader = DataLoader(train_dataset, 
                                    batch_size=self.lstm_config['batch_size'], 
                                    shuffle=True)
            val_loader = DataLoader(val_dataset, 
                                  batch_size=self.lstm_config['batch_size'], 
                                  shuffle=False)
            
            # Initialize model
            model = LightweightLSTM(
                input_size=X_train.shape[1],
                hidden_size=self.lstm_config['hidden_size'],
                num_layers=self.lstm_config['num_layers'],
                dropout=self.lstm_config['dropout']
            )
            
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), 
                                 lr=self.lstm_config['learning_rate'])
            
            # Training loop
            train_losses = []
            val_losses = []
            
            model.train()
            for epoch in range(self.lstm_config['epochs']):
                epoch_train_loss = 0
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_train_loss += loss.item()
                
                # Validation
                model.eval()
                epoch_val_loss = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs.squeeze(), batch_y)
                        epoch_val_loss += loss.item()
                
                train_losses.append(epoch_train_loss / len(train_loader))
                val_losses.append(epoch_val_loss / len(val_loader))
                
                # Early stopping
                if epoch > 10 and val_losses[-1] > val_losses[-5]:
                    break
                
                model.train()
            
            # Generate predictions
            model.eval()
            train_predictions = []
            val_predictions = []
            
            with torch.no_grad():
                for batch_X, _ in train_loader:
                    outputs = model(batch_X)
                    train_predictions.extend(outputs.squeeze().numpy())
                
                for batch_X, _ in val_loader:
                    outputs = model(batch_X)
                    val_predictions.extend(outputs.squeeze().numpy())
            
            # Inverse transform predictions
            train_pred_orig = scaler_y.inverse_transform(
                np.array(train_predictions).reshape(-1, 1)
            ).flatten()
            val_pred_orig = scaler_y.inverse_transform(
                np.array(val_predictions).reshape(-1, 1)
            ).flatten()
            
            # Calculate metrics (adjust for sequence length)
            seq_len = self.lstm_config['sequence_length']
            y_train_adj = y_train.iloc[seq_len-1:seq_len-1+len(train_pred_orig)]
            y_val_adj = y_val.iloc[seq_len-1:seq_len-1+len(val_pred_orig)]
            
            return {
                'model': model,
                'type': 'lstm',
                'train_rmse': np.sqrt(mean_squared_error(y_train_adj, train_pred_orig)),
                'val_rmse': np.sqrt(mean_squared_error(y_val_adj, val_pred_orig)),
                'train_mae': mean_absolute_error(y_train_adj, train_pred_orig),
                'val_mae': mean_absolute_error(y_val_adj, val_pred_orig),
                'train_losses': train_losses,
                'val_losses': val_losses,
                'config': self.lstm_config
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def train_all_baselines(self, data: pd.DataFrame, 
                          target_col: str = 'Meter Total(Wh)',
                          train_split: float = 0.7,
                          val_split: float = 0.2) -> Dict[str, Any]:
        """
        Train all baseline models for comprehensive comparison.
        
        Args:
            data: Input dataset
            target_col: Target column name
            train_split: Training data proportion
            val_split: Validation data proportion
            
        Returns:
            Dictionary containing all trained models and results
        """
        print("Training comprehensive baseline model suite...")
        
        # Prepare data with temporal features
        enhanced_data = self.create_temporal_features(data)
        
        # Prepare features and target
        features, target = self.prepare_time_series_data(enhanced_data, target_col)
        
        # Time-based splits
        n_samples = len(enhanced_data)
        train_end = int(n_samples * train_split)
        val_end = int(n_samples * (train_split + val_split))
        
        X_train = features.iloc[:train_end]
        y_train = target.iloc[:train_end]
        X_val = features.iloc[train_end:val_end]
        y_val = target.iloc[train_end:val_end]
        X_test = features.iloc[val_end:]
        y_test = target.iloc[val_end:]
        
        results = {
            'data_splits': {
                'train_size': len(X_train),
                'val_size': len(X_val),
                'test_size': len(X_test)
            },
            'models': {}
        }
        
        # Train naive models
        print("Training naive models...")
        naive_models = self.train_naive_models(enhanced_data.iloc[:train_end], target_col)
        results['models']['naive'] = naive_models
        
        # Train ARIMA model
        print("Training ARIMA model...")
        arima_model = self.train_arima_model(enhanced_data.iloc[:train_end], target_col)
        results['models']['arima'] = arima_model
        
        # Train ML models
        print("Training machine learning models...")
        ml_models = self.train_machine_learning_models(X_train, y_train, X_val, y_val)
        results['models']['machine_learning'] = ml_models
        
        # Train LSTM model
        print("Training LSTM model...")
        lstm_model = self.train_lstm_model(X_train, y_train, X_val, y_val)
        results['models']['deep_learning'] = {'lstm': lstm_model}
        
        # Store test data for evaluation
        results['test_data'] = {
            'X_test': X_test,
            'y_test': y_test
        }
        
        self.results = results
        print("Baseline model training completed!")
        
        return results
    
    def evaluate_models(self, results: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate all trained models on test data."""
        if results is None:
            results = self.results
            
        if not results:
            raise ValueError("No trained models found. Run train_all_baselines first.")
        
        evaluation_results = {}
        
        # Extract test data
        X_test = results['test_data']['X_test']
        y_test = results['test_data']['y_test']
        
        # Evaluate ML models
        for model_name, model_info in results['models']['machine_learning'].items():
            if 'error' not in model_info:
                model = model_info['model']
                
                if model_name == 'linear_regression':
                    X_test_scaled = self.scalers['ml_scaler'].transform(X_test)
                    predictions = model.predict(X_test_scaled)
                else:
                    predictions = model.predict(X_test)
                
                evaluation_results[model_name] = {
                    'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                    'mae': mean_absolute_error(y_test, predictions),
                    'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
                }
        
        return evaluation_results
    
    def get_model_comparison(self) -> pd.DataFrame:
        """Generate model comparison table."""
        if not self.results:
            raise ValueError("No trained models found.")
        
        comparison_data = []
        
        # ML models
        for model_name, model_info in self.results['models']['machine_learning'].items():
            if 'error' not in model_info:
                comparison_data.append({
                    'Model': model_name,
                    'Type': 'Machine Learning',
                    'Train RMSE': model_info.get('train_rmse', np.nan),
                    'Val RMSE': model_info.get('val_rmse', np.nan),
                    'Train MAE': model_info.get('train_mae', np.nan),
                    'Val MAE': model_info.get('val_mae', np.nan)
                })
        
        # LSTM model
        lstm_info = self.results['models']['deep_learning']['lstm']
        if 'error' not in lstm_info:
            comparison_data.append({
                'Model': 'LSTM',
                'Type': 'Deep Learning',
                'Train RMSE': lstm_info.get('train_rmse', np.nan),
                'Val RMSE': lstm_info.get('val_rmse', np.nan),
                'Train MAE': lstm_info.get('train_mae', np.nan),
                'Val MAE': lstm_info.get('val_mae', np.nan)
            })
        
        return pd.DataFrame(comparison_data)