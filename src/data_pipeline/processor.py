"""Advanced data processing pipeline for EV charging data with comprehensive metrics."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import structlog
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..metrics.collector import MetricsCollector, MetricType

logger = structlog.get_logger(__name__)


@dataclass
class ChargingEvent:
    """Standardized charging event data structure."""
    
    station_id: str
    timestamp: datetime
    meter_start_wh: float
    meter_end_wh: float
    meter_total_wh: float
    duration_seconds: int
    charger_name: Optional[str] = None
    session_id: Optional[str] = None


class DataProcessor(ABC):
    """Abstract base class for data processors."""
    
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        """Process the input data."""
        pass


class EVChargingDataProcessor:
    """Comprehensive EV charging data processor with metrics tracking."""
    
    def __init__(self, metrics_collector: Optional[MetricsCollector] = None):
        self.metrics_collector = metrics_collector
        self.processors: List[DataProcessor] = []
        self.validation_rules = self._setup_validation_rules()
        self.feature_pipeline = True  # Indicate feature pipeline is available
        
        logger.info("EVChargingDataProcessor initialized")
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Define data validation rules."""
        return {
            "meter_total_wh": {"min": 0, "max": 100000},  # Reasonable energy range
            "duration_seconds": {"min": 0, "max": 86400 * 7},  # Max 1 week
            "required_columns": [
                "timestamp", "meter_total_wh"  # Simplified required columns
            ],
            "original_required_columns": [
                "Start Time", "Meter Start (Wh)", "Meter End(Wh)", 
                "Meter Total(Wh)", "Total Duration (s)"
            ],
        }
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load EV charging data from file with metrics tracking."""
        start_time = datetime.now()
        
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.csv':
                data = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.json', '.jsonl']:
                data = pd.read_json(file_path, lines=file_path.suffix == '.jsonl')
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # Record loading metrics
            if self.metrics_collector:
                processing_time = (datetime.now() - start_time).total_seconds()
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    processing_time,
                    "data_loader",
                    metadata={"operation": "load_data", "file_size": file_path.stat().st_size}
                )
            
            logger.info(
                "Data loaded successfully",
                file_path=str(file_path),
                rows=len(data),
                columns=len(data.columns)
            )
            
            return data
            
        except Exception as e:
            logger.error("Data loading failed", file_path=file_path, error=str(e))
            raise
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Validate data quality with comprehensive metrics."""
        start_time = datetime.now()
        validation_report = {
            "original_rows": len(data),
            "validation_errors": [],
            "data_quality_score": 1.0,
        }
        
        # Check required columns (try simplified first, then original)
        missing_columns = []
        required_cols = self.validation_rules["required_columns"]
        
        # Check if we have the simplified required columns
        for col in required_cols:
            if col not in data.columns:
                missing_columns.append(col)
        
        # If simplified columns are missing, check for original format
        if missing_columns:
            missing_original = []
            for col in self.validation_rules["original_required_columns"]:
                if col not in data.columns:
                    missing_original.append(col)
            
            # Only error if both formats are missing
            if missing_original and len(missing_original) == len(self.validation_rules["original_required_columns"]):
                error_msg = f"Missing required columns. Expected either: {required_cols} or {self.validation_rules['original_required_columns']}"
                validation_report["validation_errors"].append(error_msg)
                validation_report["data_quality_score"] *= 0.5
                logger.error("Critical data validation error", error=error_msg)
            else:
                # Clear missing columns if original format is present
                missing_columns = []
        
        # Clean and standardize data
        clean_data = data.copy()
        
        try:
            # Standardize column names
            clean_data = self._standardize_columns(clean_data)
            
            # Convert data types
            clean_data = self._convert_data_types(clean_data)
            
            # Remove invalid records
            clean_data = self._remove_invalid_records(clean_data, validation_report)
            
            # Handle missing values
            clean_data = self._handle_missing_values(clean_data, validation_report)
            
            validation_report["cleaned_rows"] = len(clean_data)
            validation_report["rows_removed"] = validation_report["original_rows"] - len(clean_data)
            validation_report["data_quality_score"] *= (len(clean_data) / validation_report["original_rows"])
            
        except Exception as e:
            logger.error("Data validation failed", error=str(e))
            validation_report["validation_errors"].append(f"Validation error: {str(e)}")
        
        # Record validation metrics
        if self.metrics_collector:
            processing_time = (datetime.now() - start_time).total_seconds()
            self.metrics_collector.record_metric(
                MetricType.INFERENCE_TIME,
                processing_time,
                "data_validator",
                metadata={"operation": "validate_data"}
            )
            
            # Record data quality metrics
            self.metrics_collector.record_metric(
                MetricType.ACCURACY,  # Using accuracy as data quality proxy
                validation_report["data_quality_score"],
                "data_validator"
            )
        
        logger.info(
            "Data validation completed",
            original_rows=validation_report["original_rows"],
            cleaned_rows=validation_report["cleaned_rows"],
            quality_score=validation_report["data_quality_score"]
        )
        
        return clean_data, validation_report
    
    def _standardize_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names."""
        column_mapping = {
            "Start Time": "timestamp",
            "Meter Start (Wh)": "meter_start_wh",
            "Meter End(Wh)": "meter_end_wh",
            "Meter Total(Wh)": "meter_total_wh",
            "Total Duration (s)": "duration_seconds",
            "Charger_name": "charger_name",
        }
        
        return data.rename(columns=column_mapping)
    
    def _convert_data_types(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert data to appropriate types."""
        if "timestamp" in data.columns:
            # Try multiple timestamp formats
            try:
                # First try the original format
                data["timestamp"] = pd.to_datetime(data["timestamp"], format="%d.%m.%Y %H:%M", errors="raise")
            except:
                try:
                    # Try ISO format (common in test data)
                    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
                except:
                    # Fall back to infer format
                    data["timestamp"] = pd.to_datetime(data["timestamp"], infer_datetime_format=True, errors="coerce")
        
        # Convert numeric columns
        numeric_columns = ["meter_start_wh", "meter_end_wh", "meter_total_wh", "duration_seconds", "session_duration_min"]
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        
        return data
    
    def _remove_invalid_records(self, data: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Remove invalid records based on validation rules."""
        initial_count = len(data)
        
        # Remove rows with invalid energy values
        if "meter_total_wh" in data.columns:
            valid_energy = (
                (data["meter_total_wh"] >= self.validation_rules["meter_total_wh"]["min"]) &
                (data["meter_total_wh"] <= self.validation_rules["meter_total_wh"]["max"])
            )
            data = data[valid_energy]
        
        # Remove rows with invalid duration (check both possible duration columns)
        if "duration_seconds" in data.columns:
            valid_duration = (
                (data["duration_seconds"] >= self.validation_rules["duration_seconds"]["min"]) &
                (data["duration_seconds"] <= self.validation_rules["duration_seconds"]["max"])
            )
            data = data[valid_duration]
        elif "session_duration_min" in data.columns:
            # Convert minutes to seconds for validation
            duration_seconds = data["session_duration_min"] * 60
            valid_duration = (
                (duration_seconds >= self.validation_rules["duration_seconds"]["min"]) &
                (duration_seconds <= self.validation_rules["duration_seconds"]["max"])
            )
            data = data[valid_duration]
        
        removed_count = initial_count - len(data)
        if removed_count > 0:
            report["validation_errors"].append(f"Removed {removed_count} invalid records")
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values appropriately."""
        # Fill missing charger names
        if "charger_name" in data.columns:
            data["charger_name"] = data["charger_name"].fillna("unknown")
        
        # Remove rows with missing critical values
        critical_columns = ["timestamp", "meter_total_wh"]
        for col in critical_columns:
            if col in data.columns:
                initial_count = len(data)
                data = data.dropna(subset=[col])
                removed_count = initial_count - len(data)
                if removed_count > 0:
                    report["validation_errors"].append(f"Removed {removed_count} rows with missing {col}")
        
        return data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for ML models."""
        start_time = datetime.now()
        
        try:
            features_data = data.copy()
            
            # Temporal features
            features_data = self._create_temporal_features(features_data)
            
            # Statistical features
            features_data = self._create_statistical_features(features_data)
            
            # Domain-specific features
            features_data = self._create_domain_features(features_data)
            
            # Aggregate features
            features_data = self._create_aggregate_features(features_data)
            
            # Record feature engineering metrics
            if self.metrics_collector:
                processing_time = (datetime.now() - start_time).total_seconds()
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    processing_time,
                    "feature_engineer",
                    metadata={"operation": "create_features", "features_count": len(features_data.columns)}
                )
            
            logger.info(
                "Feature engineering completed",
                original_features=len(data.columns),
                total_features=len(features_data.columns)
            )
            
            return features_data
            
        except Exception as e:
            logger.error("Feature engineering failed", error=str(e))
            raise
    
    def _create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        if "timestamp" not in data.columns:
            return data
        
        data["hour"] = data["timestamp"].dt.hour
        data["day_of_week"] = data["timestamp"].dt.dayofweek
        data["month"] = data["timestamp"].dt.month
        data["quarter"] = data["timestamp"].dt.quarter
        data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
        
        # Peak hours (typical high usage)
        data["is_peak_hour"] = data["hour"].isin([8, 9, 10, 17, 18, 19]).astype(int)
        
        # Time since epoch (for trend analysis)
        data["timestamp_numeric"] = data["timestamp"].astype(np.int64) // 10**9
        
        # Cyclical encoding for temporal features
        data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
        data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
        data["day_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
        data["day_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
        
        return data
    
    def _create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features."""
        # Energy z-score and percentile
        if "meter_total_wh" in data.columns:
            energy_mean = data["meter_total_wh"].mean()
            energy_std = data["meter_total_wh"].std()
            if energy_std > 0:
                data["energy_z_score"] = (data["meter_total_wh"] - energy_mean) / energy_std
            else:
                data["energy_z_score"] = 0
            data["energy_percentile"] = data["meter_total_wh"].rank(pct=True)
            data["energy_log"] = np.log1p(data["meter_total_wh"])  # log1p handles zeros better
        
        # Duration z-score and percentile
        if "session_duration_min" in data.columns:
            duration_mean = data["session_duration_min"].mean()
            duration_std = data["session_duration_min"].std()
            if duration_std > 0:
                data["duration_z_score"] = (data["session_duration_min"] - duration_mean) / duration_std
            else:
                data["duration_z_score"] = 0
            data["duration_percentile"] = data["session_duration_min"].rank(pct=True)
            data["duration_log"] = np.log1p(data["session_duration_min"])
        
        # Energy efficiency
        if all(col in data.columns for col in ["meter_total_wh", "duration_seconds"]):
            data["energy_per_second"] = data["meter_total_wh"] / (data["duration_seconds"] + 1e-6)
            data["energy_per_minute"] = data["energy_per_second"] * 60
        elif all(col in data.columns for col in ["meter_total_wh", "session_duration_min"]):
            # Use session_duration_min if duration_seconds is not available
            data["energy_per_minute"] = data["meter_total_wh"] / (data["session_duration_min"] + 1e-6)
        
        # Charging rate
        if "energy_per_minute" in data.columns and data["energy_per_minute"].sum() > 0:
            data["charging_rate_category"] = pd.cut(
                data["energy_per_minute"],
                bins=[-np.inf, 10, 50, 100, np.inf],
                labels=["very_slow", "slow", "normal", "fast"]
            )
        
        return data
    
    def _create_domain_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create EV charging domain-specific features."""
        # Session type based on energy consumed
        if "meter_total_wh" in data.columns:
            data["session_type"] = pd.cut(
                data["meter_total_wh"],
                bins=[-np.inf, 100, 1000, 10000, np.inf],
                labels=["minimal", "short", "regular", "long"]
            )
        
        # Charger utilization (simplified)
        if "charger_name" in data.columns:
            charger_counts = data["charger_name"].value_counts()
            data["charger_popularity"] = data["charger_name"].map(charger_counts)
        
        return data
    
    def _create_aggregate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create aggregated features."""
        if len(data) < 2:
            return data
        
        # Rolling statistics (if we have enough data)
        if "meter_total_wh" in data.columns and len(data) > 10:
            data = data.sort_values("timestamp", na_position="last")
            
            # 5-point rolling averages
            data["energy_rolling_mean"] = data["meter_total_wh"].rolling(window=5, min_periods=1).mean()
            data["energy_rolling_std"] = data["meter_total_wh"].rolling(window=5, min_periods=1).std()
            
            # Lag features
            data["energy_lag_1"] = data["meter_total_wh"].shift(1)
            data["energy_diff"] = data["meter_total_wh"] - data["energy_lag_1"]
        
        return data
    
    def create_time_series_features(
        self,
        data: pd.DataFrame,
        target_column: str = "meter_total_wh",
        window_size: int = 24
    ) -> pd.DataFrame:
        """Create time series features for forecasting models."""
        if len(data) < window_size:
            logger.warning(f"Insufficient data for time series features (need {window_size}, got {len(data)})")
            return data
        
        ts_data = data.copy().sort_values("timestamp")
        
        # Create lag features
        for lag in range(1, min(window_size, len(data))):
            ts_data[f"{target_column}_lag_{lag}"] = ts_data[target_column].shift(lag)
        
        # Moving averages
        for window in [3, 6, 12, 24]:
            if len(ts_data) > window:
                ts_data[f"{target_column}_ma_{window}"] = (
                    ts_data[target_column].rolling(window=window, min_periods=1).mean()
                )
        
        # Exponential smoothing
        ts_data[f"{target_column}_ewm"] = ts_data[target_column].ewm(span=12).mean()
        
        return ts_data.dropna()
    
    def get_data_insights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data insights for research."""
        insights = {
            "dataset_summary": {
                "total_records": len(data),
                "date_range": None,
                "total_energy_consumed": 0,
                "avg_session_duration": 0,
                "unique_chargers": 0,
            },
            "summary_stats": {},
            "patterns": {},
            "quality_metrics": {},
            "correlations": {},
            "peak_usage_hours": [],
            "station_utilization": {},
        }
        
        try:
            # Basic statistics
            if "timestamp" in data.columns and not data["timestamp"].isna().all():
                insights["dataset_summary"]["date_range"] = {
                    "start": data["timestamp"].min().isoformat() if pd.notna(data["timestamp"].min()) else None,
                    "end": data["timestamp"].max().isoformat() if pd.notna(data["timestamp"].max()) else None,
                }
            
            if "meter_total_wh" in data.columns:
                insights["dataset_summary"]["total_energy_consumed"] = float(data["meter_total_wh"].sum())
                insights["dataset_summary"]["avg_energy_per_session"] = float(data["meter_total_wh"].mean())
            
            if "duration_seconds" in data.columns:
                insights["dataset_summary"]["avg_session_duration"] = float(data["duration_seconds"].mean())
            
            if "charger_name" in data.columns:
                insights["dataset_summary"]["unique_chargers"] = int(data["charger_name"].nunique())
            
            # Summary statistics
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                insights["summary_stats"] = data[numeric_columns].describe().to_dict()
            
            # Pattern analysis
            if "hour_of_day" in data.columns:
                hourly_usage = data.groupby("hour_of_day")["meter_total_wh"].mean() if "meter_total_wh" in data.columns else data.groupby("hour_of_day").size()
                insights["patterns"]["peak_hours"] = hourly_usage.nlargest(3).index.tolist()
                insights["peak_usage_hours"] = hourly_usage.nlargest(3).index.tolist()
            elif "hour" in data.columns:
                hourly_usage = data.groupby("hour")["meter_total_wh"].mean() if "meter_total_wh" in data.columns else data.groupby("hour").size()
                insights["patterns"]["peak_hours"] = hourly_usage.nlargest(3).index.tolist()
                insights["peak_usage_hours"] = hourly_usage.nlargest(3).index.tolist()
            
            # Station utilization
            if "station_id" in data.columns:
                station_counts = data["station_id"].value_counts().to_dict()
                insights["station_utilization"] = station_counts
            
            # Data quality metrics
            insights["quality_metrics"] = {
                "missing_values": data.isnull().sum().to_dict(),
                "duplicate_records": int(data.duplicated().sum()),
                "data_completeness": float(1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) if len(data) > 0 and len(data.columns) > 0 else 1.0,
            }
            
            # Correlation analysis
            if len(numeric_columns) > 1:
                correlations = data[numeric_columns].corr()
                insights["correlations"] = correlations.to_dict()
            
        except Exception as e:
            logger.error("Data insights generation failed", error=str(e))
            insights["error"] = str(e)
        
        return insights
    
    def process_pipeline(self, file_path: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute complete data processing pipeline."""
        pipeline_start = datetime.now()
        
        try:
            # Step 1: Load data
            raw_data = self.load_data(file_path)
            
            # Step 2: Validate and clean
            clean_data, validation_report = self.validate_data(raw_data)
            
            # Step 3: Feature engineering
            features_data = self.create_features(clean_data)
            
            # Step 4: Generate insights
            insights = self.get_data_insights(features_data)
            
            # Compile pipeline report
            processing_time = (datetime.now() - pipeline_start).total_seconds()
            pipeline_report = {
                "original_shape": raw_data.shape,
                "processed_shape": features_data.shape,
                "validation_results": validation_report,
                "processing_time": processing_time,
                "processing_time_seconds": processing_time,  # Keep for backward compatibility
                "insights": insights,
                "pipeline_status": "success",
            }
            
            # Record overall pipeline metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    pipeline_report["processing_time_seconds"],
                    "data_pipeline",
                    metadata={"operation": "full_pipeline"}
                )
            
            logger.info(
                "Data processing pipeline completed successfully",
                processing_time=pipeline_report["processing_time_seconds"],
                final_records=len(features_data)
            )
            
            return features_data, pipeline_report
            
        except Exception as e:
            logger.error("Data processing pipeline failed", error=str(e))
            pipeline_report = {
                "processing_time_seconds": (datetime.now() - pipeline_start).total_seconds(),
                "pipeline_status": "failed",
                "error": str(e),
            }
            return pd.DataFrame(), pipeline_report
    
    def process_pipeline_from_dataframe(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute complete data processing pipeline from DataFrame."""
        pipeline_start = datetime.now()
        
        try:
            # Step 1: Validate and clean
            clean_data, validation_report = self.validate_data(data)
            
            # Step 2: Feature engineering
            features_data = self.create_features(clean_data)
            
            # Step 3: Generate insights
            insights = self.get_data_insights(features_data)
            
            # Compile pipeline report
            pipeline_report = {
                "processing_time": (datetime.now() - pipeline_start).total_seconds(),
                "original_shape": data.shape,
                "processed_shape": features_data.shape,
                "validation_results": validation_report,
                "insights": insights,
                "pipeline_status": "success",
            }
            
            # Record overall pipeline metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    pipeline_report["processing_time"],
                    "data_pipeline",
                    metadata={"operation": "dataframe_pipeline"}
                )
            
            logger.info(
                "DataFrame processing pipeline completed successfully",
                processing_time=pipeline_report["processing_time"],
                final_records=len(features_data)
            )
            
            return features_data, pipeline_report
            
        except Exception as e:
            logger.error("DataFrame processing pipeline failed", error=str(e))
            pipeline_report = {
                "processing_time": (datetime.now() - pipeline_start).total_seconds(),
                "original_shape": data.shape,
                "processed_shape": (0, 0),
                "pipeline_status": "failed",
                "error": str(e),
            }
            # Return original data on error to avoid returning None
            return data, pipeline_report
    
    def validate_data_structure(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data structure and return issues."""
        issues = []
        
        # Check required columns
        for col in self.validation_rules["required_columns"]:
            if col not in data.columns:
                issues.append(f"Missing required column: {col}")
        
        # Check for completely empty DataFrame
        if data.empty:
            issues.append("DataFrame is empty")
            
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def validate_data_quality(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate data quality and return issues."""
        issues = []
        
        # Check for negative energy values
        if "meter_total_wh" in data.columns:
            negative_energy = data["meter_total_wh"] < 0
            if negative_energy.any():
                issues.append(f"Found {negative_energy.sum()} records with negative energy values")
        
        # Check for negative duration values
        if "session_duration_min" in data.columns:
            negative_duration = data["session_duration_min"] < 0
            if negative_duration.any():
                issues.append(f"Found {negative_duration.sum()} records with negative duration values")
        
        # Check for excessive missing values
        missing_pct = data.isnull().mean()
        high_missing = missing_pct > 0.5
        if high_missing.any():
            high_missing_cols = high_missing[high_missing].index.tolist()
            issues.append(f"Columns with >50% missing values: {high_missing_cols}")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def create_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from timestamp data."""
        features_df = data.copy()
        
        if "timestamp" in data.columns:
            features_df = self._create_temporal_features(features_df)
        
        return features_df
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create statistical features from numerical data."""
        features_df = data.copy()
        features_df = self._create_statistical_features(features_df)
        return features_df
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data by removing invalid values and handling missing data."""
        clean_data = data.copy()
        
        # Remove infinite values
        numeric_columns = clean_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            clean_data = clean_data[~np.isinf(clean_data[col])]
        
        # Remove negative values where they shouldn't exist
        if "meter_total_wh" in clean_data.columns:
            clean_data = clean_data[clean_data["meter_total_wh"] >= 0]
            
        if "session_duration_min" in clean_data.columns:
            clean_data = clean_data[clean_data["session_duration_min"] >= 0]
        
        # Handle missing values
        clean_data = self.handle_missing_data(clean_data)
        
        return clean_data
    
    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data with appropriate strategies."""
        handled_data = data.copy()
        
        # Fill numeric columns with median
        numeric_columns = handled_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if handled_data[col].isnull().any():
                median_val = handled_data[col].median()
                handled_data[col] = handled_data[col].fillna(median_val)
        
        # Fill categorical columns with mode
        categorical_columns = handled_data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if handled_data[col].isnull().any():
                mode_val = handled_data[col].mode().iloc[0] if not handled_data[col].mode().empty else 'unknown'
                handled_data[col] = handled_data[col].fillna(mode_val)
        
        return handled_data
    
    def calculate_feature_importance(self, data: pd.DataFrame, target_column: str) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:
            # Simple correlation-based importance
            numeric_data = data.select_dtypes(include=[np.number])
            
            if target_column not in numeric_data.columns:
                return {}
                
            correlations = numeric_data.corr()[target_column].abs().drop(target_column)
            
            # Remove NaN values
            correlations = correlations.dropna()
            
            if correlations.empty or correlations.max() == 0:
                return {}
            
            # Normalize to 0-1 range
            max_corr = correlations.max()
            importance_scores = (correlations / max_corr).to_dict()
            
            # Ensure all values are between 0 and 1 and not NaN
            importance_scores = {k: v for k, v in importance_scores.items() 
                               if not np.isnan(v) and 0 <= v <= 1}
            
            return importance_scores
            
        except Exception as e:
            logger.warning(f"Feature importance calculation failed: {e}")
            return {}