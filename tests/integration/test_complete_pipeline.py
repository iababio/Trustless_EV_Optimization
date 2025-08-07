"""Integration tests for complete EV optimization pipeline."""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.data_pipeline.processor import EVChargingDataProcessor
from src.metrics.collector import MetricsCollector


class TestCompletePipeline:
    """Integration tests for the complete EV charging optimization pipeline."""

    @pytest.fixture
    def complete_test_data(self):
        """Create comprehensive test data for full pipeline testing."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate realistic EV charging data with seasonal patterns
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='30min')
        
        data = {
            'timestamp': dates,
            'station_id': np.random.choice(['station_001', 'station_002', 'station_003', 'station_004'], n_samples),
            'meter_total_wh': np.random.exponential(4000, n_samples),
            'session_duration_min': np.random.gamma(2, 25, n_samples),
            'connector_type': np.random.choice(['Type1', 'Type2', 'CCS', 'CHAdeMO'], n_samples),
            'user_id': np.random.choice([f'user_{i:03d}' for i in range(50)], n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Add realistic patterns
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['month'] = df['timestamp'].dt.month
        
        # Add temperature data (seasonal pattern)
        df['temperature'] = 15 + 10 * np.sin((df['timestamp'].dt.dayofyear - 80) * 2 * np.pi / 365) + np.random.normal(0, 3, n_samples)
        
        # Add peak hour effects
        peak_hours = [8, 9, 17, 18, 19]  # Morning and evening peaks
        df['is_peak_hour'] = df['hour_of_day'].isin(peak_hours)
        df.loc[df['is_peak_hour'], 'meter_total_wh'] *= np.random.uniform(1.2, 1.8, sum(df['is_peak_hour']))
        
        # Add weekend effects
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df.loc[df['is_weekend'], 'meter_total_wh'] *= np.random.uniform(0.7, 1.1, sum(df['is_weekend']))
        
        return df

    @pytest.fixture
    def temp_data_file_comprehensive(self, complete_test_data):
        """Create temporary CSV file with comprehensive test data."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            complete_test_data.to_csv(f.name, index=False)
            yield f.name
        Path(f.name).unlink()

    def test_end_to_end_data_processing(self, temp_data_file_comprehensive, metrics_collector):
        """Test complete data processing pipeline end-to-end."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        # Process complete pipeline
        processed_data, report = processor.process_pipeline(temp_data_file_comprehensive)
        
        # Validate processed data
        assert isinstance(processed_data, pd.DataFrame)
        assert len(processed_data) > 0
        assert processed_data['meter_total_wh'].notna().all()
        
        # Validate processing report
        assert 'original_shape' in report
        assert 'processed_shape' in report
        assert 'validation_results' in report
        assert 'feature_engineering' in report
        assert 'processing_time' in report
        
        # Check feature engineering
        expected_features = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'energy_z_score', 'duration_z_score'
        ]
        for feature in expected_features:
            assert feature in processed_data.columns
        
        # Validate metrics were recorded
        metrics = metrics_collector.export_metrics(format_type="dict")
        assert metrics['total_metrics'] > 0

    def test_ml_pipeline_integration(self, complete_test_data, metrics_collector):
        """Test ML model training pipeline integration."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        # Process data
        processed_data = processor.process_pipeline_from_dataframe(complete_test_data)
        
        # Prepare ML data
        feature_columns = [col for col in processed_data.columns 
                          if col != 'meter_total_wh' and processed_data[col].dtype in ['float64', 'int64']]
        X = processed_data[feature_columns].fillna(0).values
        y = processed_data['meter_total_wh'].values
        
        # Remove invalid targets
        valid_mask = ~np.isnan(y) & (y >= 0)
        X, y = X[valid_mask], y[valid_mask]
        
        assert len(X) > 50  # Ensure sufficient data
        assert X.shape[1] > 5  # Ensure multiple features
        
        # Test data splitting
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        assert len(X_train) > len(X_test)
        assert len(X_train) > 0 and len(X_test) > 0

    @pytest.mark.skipif(True, reason="Requires XGBoost")
    def test_xgboost_integration(self, complete_test_data, metrics_collector):
        """Test XGBoost model integration in pipeline."""
        from src.ml_models.xgboost_model import LightweightXGBoost
        
        processor = EVChargingDataProcessor(metrics_collector)
        processed_data = processor.process_pipeline_from_dataframe(complete_test_data)
        
        # Prepare data for XGBoost
        feature_columns = [col for col in processed_data.columns 
                          if col != 'meter_total_wh' and processed_data[col].dtype in ['float64', 'int64']]
        X = processed_data[feature_columns].fillna(0).values
        y = processed_data['meter_total_wh'].values
        
        valid_mask = ~np.isnan(y) & (y >= 0)
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) > 100:  # Only test if sufficient data
            # Train XGBoost model
            model = LightweightXGBoost(
                model_name="integration_test",
                metrics_collector=metrics_collector,
                n_estimators=20
            )
            
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            training_summary = model.train(X_train, y_train, X_val, y_val)
            
            assert 'training_time' in training_summary
            assert model.model is not None
            
            # Test prediction
            predictions = model.predict(X_val)
            assert len(predictions) == len(y_val)
            assert not np.isnan(predictions).any()

    def test_metrics_collection_integration(self, complete_test_data, metrics_collector):
        """Test comprehensive metrics collection throughout pipeline."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        # Start system monitoring
        metrics_collector.start_system_monitoring(interval_seconds=0.1)
        
        try:
            # Process data (should record processing metrics)
            processed_data = processor.process_pipeline_from_dataframe(complete_test_data)
            
            # Simulate ML training metrics
            metrics_collector.record_ml_metrics(
                accuracy=0.85,
                loss=0.25,
                component="test_model",
                additional_metrics={'r2_score': 0.78}
            )
            
            # Simulate federated learning metrics
            for round_id in range(1, 4):
                metrics_collector.record_federated_metrics(
                    round_id=round_id,
                    num_clients=5,
                    round_duration=45.0 + round_id * 5,
                    convergence_metric=0.7 + round_id * 0.05,
                    component="fl_server"
                )
            
            # Simulate blockchain metrics
            metrics_collector.record_blockchain_metrics(
                transaction_time=2.3,
                gas_used=145000,
                validation_success=True,
                component="model_validator"
            )
            
            # Wait a bit for system monitoring
            import time
            time.sleep(0.3)
            
        finally:
            metrics_collector.stop_system_monitoring()
        
        # Export and validate metrics
        metrics = metrics_collector.export_metrics(format_type="dict")
        
        assert metrics['total_metrics'] > 0
        
        # Check different metric types are present
        metric_types = set(m.get('metric_type') for m in metrics['metrics'])
        expected_types = {'ml_performance', 'federated_learning', 'blockchain', 'system_performance'}
        
        # Should have at least some of the expected types
        assert len(metric_types.intersection(expected_types)) >= 2

    def test_data_insights_generation(self, complete_test_data, metrics_collector):
        """Test comprehensive data insights generation."""
        processor = EVChargingDataProcessor(metrics_collector)
        processed_data = processor.process_pipeline_from_dataframe(complete_test_data)
        
        # Generate insights
        insights = processor.get_data_insights(processed_data)
        
        # Validate insights structure
        assert 'summary_stats' in insights
        assert 'correlations' in insights
        assert 'peak_usage_hours' in insights
        assert 'station_utilization' in insights
        assert 'temporal_patterns' in insights
        
        # Validate summary statistics
        stats = insights['summary_stats']
        assert 'meter_total_wh' in stats
        assert 'mean' in stats['meter_total_wh']
        assert 'std' in stats['meter_total_wh']
        
        # Validate correlations
        correlations = insights['correlations']
        assert isinstance(correlations, dict)
        
        # Validate peak hours detection
        peak_hours = insights['peak_usage_hours']
        assert isinstance(peak_hours, list)
        assert all(0 <= hour <= 23 for hour in peak_hours)

    def test_feature_importance_analysis(self, complete_test_data, metrics_collector):
        """Test feature importance analysis in pipeline."""
        processor = EVChargingDataProcessor(metrics_collector)
        processed_data = processor.process_pipeline_from_dataframe(complete_test_data)
        
        # Calculate feature importance
        importance_scores = processor.calculate_feature_importance(
            processed_data, 
            target_column='meter_total_wh'
        )
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) > 0
        
        # Check importance scores are valid
        for feature, score in importance_scores.items():
            assert isinstance(score, (int, float))
            assert 0 <= score <= 1
            assert feature in processed_data.columns

    def test_data_quality_validation(self, metrics_collector):
        """Test data quality validation in pipeline."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        # Create data with quality issues
        problematic_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='1H'),
            'station_id': ['station_001'] * 100,
            'meter_total_wh': np.concatenate([
                np.random.exponential(3000, 80),  # Normal data
                [-1000, -500],  # Negative values
                [np.nan] * 10,  # Missing values
                [np.inf, -np.inf]  # Invalid values
            ])[:100],
            'session_duration_min': np.concatenate([
                np.random.gamma(2, 20, 85),  # Normal data
                [-30, -10],  # Negative durations
                [np.nan] * 8,  # Missing values
                [1000000]  # Unrealistic value
            ])[:100]
        })
        
        # Process with quality issues
        is_valid, issues = processor.validate_data_quality(problematic_data)
        
        # Should detect quality issues
        assert not is_valid
        assert len(issues) > 0
        assert any('negative' in issue.lower() or 'missing' in issue.lower() for issue in issues)

    def test_pipeline_error_handling(self, metrics_collector):
        """Test error handling throughout the pipeline."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        # Test with completely invalid data
        invalid_data = pd.DataFrame({
            'wrong_column': [1, 2, 3],
            'another_wrong_column': ['a', 'b', 'c']
        })
        
        # Should handle gracefully
        try:
            result = processor.process_pipeline_from_dataframe(invalid_data)
            # Should return something (possibly empty or cleaned data)
            assert result is not None
        except Exception as e:
            # If exception is raised, it should be informative
            assert len(str(e)) > 0

    def test_pipeline_performance_metrics(self, complete_test_data, metrics_collector):
        """Test performance metrics collection during pipeline execution."""
        import time
        
        processor = EVChargingDataProcessor(metrics_collector)
        
        start_time = time.time()
        processed_data = processor.process_pipeline_from_dataframe(complete_test_data)
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time
        assert processing_time < 30  # 30 seconds max
        
        # Should process reasonable amount of data
        assert len(processed_data) > 0
        assert len(processed_data) <= len(complete_test_data)  # Should not increase data

    def test_reproducible_processing(self, complete_test_data, metrics_collector):
        """Test that processing is reproducible."""
        processor1 = EVChargingDataProcessor(metrics_collector)
        processor2 = EVChargingDataProcessor(metrics_collector)
        
        # Process same data twice
        result1 = processor1.process_pipeline_from_dataframe(complete_test_data.copy())
        result2 = processor2.process_pipeline_from_dataframe(complete_test_data.copy())
        
        # Results should be identical (or very similar for stochastic operations)
        assert len(result1) == len(result2)
        assert list(result1.columns) == list(result2.columns)
        
        # Check that deterministic features are identical
        deterministic_columns = ['hour_of_day', 'day_of_week', 'month']
        for col in deterministic_columns:
            if col in result1.columns and col in result2.columns:
                pd.testing.assert_series_equal(result1[col], result2[col], check_names=False)

    def test_pipeline_scalability(self, metrics_collector):
        """Test pipeline performance with varying data sizes."""
        processor = EVChargingDataProcessor(metrics_collector)
        processing_times = {}
        
        # Test with different data sizes
        data_sizes = [100, 500, 1000]
        
        for size in data_sizes:
            # Generate data of specific size
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='30min'),
                'station_id': ['station_001'] * size,
                'meter_total_wh': np.random.exponential(4000, size),
                'session_duration_min': np.random.gamma(2, 25, size),
                'hour_of_day': np.random.randint(0, 24, size),
                'day_of_week': np.random.randint(0, 7, size),
            })
            
            start_time = time.time()
            processed = processor.process_pipeline_from_dataframe(test_data)
            processing_times[size] = time.time() - start_time
            
            # Validate processing completed
            assert len(processed) > 0
        
        # Processing time should scale reasonably
        # (not necessarily linear due to overhead, but should not be exponential)
        for size in data_sizes:
            assert processing_times[size] < size * 0.01  # Less than 0.01 seconds per sample