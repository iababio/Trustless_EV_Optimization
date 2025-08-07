"""Unit tests for the EV charging data processor."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from src.data_pipeline.processor import EVChargingDataProcessor
from src.metrics.collector import MetricsCollector, MetricType


class TestEVChargingDataProcessor:
    """Test suite for EVChargingDataProcessor."""

    def test_initialization(self, metrics_collector):
        """Test processor initialization."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        assert processor.metrics_collector == metrics_collector
        assert hasattr(processor, 'feature_pipeline')
        assert hasattr(processor, 'validation_rules')

    def test_load_data_valid_file(self, temp_data_file, data_processor):
        """Test loading data from valid CSV file."""
        data = data_processor.load_data(temp_data_file)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'timestamp' in data.columns
        assert 'meter_total_wh' in data.columns

    def test_load_data_invalid_file(self, data_processor):
        """Test loading data from non-existent file."""
        with pytest.raises(FileNotFoundError):
            data_processor.load_data("non_existent_file.csv")

    def test_validate_data_structure(self, sample_ev_data, data_processor):
        """Test data structure validation."""
        # Valid data should pass
        is_valid, issues = data_processor.validate_data_structure(sample_ev_data)
        assert is_valid
        assert len(issues) == 0

        # Missing required columns should fail
        invalid_data = sample_ev_data.drop('meter_total_wh', axis=1)
        is_valid, issues = data_processor.validate_data_structure(invalid_data)
        assert not is_valid
        assert any('meter_total_wh' in issue for issue in issues)

    def test_validate_data_quality(self, sample_ev_data, data_processor):
        """Test data quality validation."""
        # Clean data should pass
        is_valid, issues = data_processor.validate_data_quality(sample_ev_data)
        assert is_valid

        # Data with issues should be detected
        corrupted_data = sample_ev_data.copy()
        corrupted_data.loc[0, 'meter_total_wh'] = -1000  # Negative energy
        corrupted_data.loc[1, 'session_duration_min'] = -50  # Negative duration
        
        is_valid, issues = data_processor.validate_data_quality(corrupted_data)
        assert not is_valid
        assert len(issues) > 0

    def test_create_temporal_features(self, sample_ev_data, data_processor):
        """Test temporal feature engineering."""
        features_df = data_processor.create_temporal_features(sample_ev_data)
        
        expected_features = [
            'hour_of_day', 'day_of_week', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns

        # Test cyclical encoding
        assert features_df['hour_sin'].between(-1, 1).all()
        assert features_df['hour_cos'].between(-1, 1).all()

    def test_create_statistical_features(self, sample_ev_data, data_processor):
        """Test statistical feature engineering."""
        features_df = data_processor.create_statistical_features(sample_ev_data)
        
        expected_features = [
            'energy_z_score', 'duration_z_score', 'energy_percentile',
            'duration_percentile', 'energy_log', 'duration_log'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns

    def test_process_pipeline_complete(self, temp_data_file, data_processor):
        """Test complete processing pipeline."""
        processed_data, report = data_processor.process_pipeline(temp_data_file)
        
        assert isinstance(processed_data, pd.DataFrame)
        assert isinstance(report, dict)
        assert len(processed_data) > 0
        
        # Check report structure
        expected_keys = ['original_shape', 'processed_shape', 'validation_results', 'processing_time']
        for key in expected_keys:
            assert key in report

    def test_get_data_insights(self, sample_ev_data, data_processor):
        """Test data insights generation."""
        insights = data_processor.get_data_insights(sample_ev_data)
        
        assert isinstance(insights, dict)
        assert 'summary_stats' in insights
        assert 'correlations' in insights
        assert 'peak_usage_hours' in insights
        assert 'station_utilization' in insights

    def test_feature_importance_calculation(self, sample_ev_data, data_processor):
        """Test feature importance calculation."""
        # Add some features first
        enhanced_data = data_processor.create_temporal_features(sample_ev_data)
        enhanced_data = data_processor.create_statistical_features(enhanced_data)
        
        importance_scores = data_processor.calculate_feature_importance(
            enhanced_data, target_column='meter_total_wh'
        )
        
        assert isinstance(importance_scores, dict)
        assert len(importance_scores) > 0
        
        # Scores should be between 0 and 1
        for score in importance_scores.values():
            assert 0 <= score <= 1

    def test_data_cleaning(self, data_processor):
        """Test data cleaning functionality."""
        # Create data with various issues
        dirty_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='1H'),
            'meter_total_wh': [1000, -500, np.nan, 2000, 0, 5000, 10000, np.inf, 3000, 1500],
            'session_duration_min': [30, 45, 60, np.nan, -10, 120, 90, 75, np.inf, 40],
            'station_id': ['A', 'B', None, 'A', 'C', 'B', 'A', 'B', 'C', 'A']
        })
        
        cleaned_data = data_processor.clean_data(dirty_data)
        
        # Should remove/fix problematic values
        assert not cleaned_data['meter_total_wh'].isna().any()
        assert not cleaned_data['session_duration_min'].isna().any()
        assert (cleaned_data['meter_total_wh'] >= 0).all()
        assert (cleaned_data['session_duration_min'] >= 0).all()
        assert not np.isinf(cleaned_data['meter_total_wh']).any()

    def test_metrics_recording(self, data_processor, sample_ev_data):
        """Test that metrics are properly recorded during processing."""
        # Process data and check metrics were recorded
        data_processor.process_pipeline_from_dataframe(sample_ev_data)
        
        # Check that metrics were collected
        metrics = data_processor.metrics_collector.export_metrics(format_type="dict")
        assert metrics['total_metrics'] > 0

    def test_handle_missing_data(self, data_processor):
        """Test handling of missing data."""
        # Create data with missing values
        incomplete_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=5, freq='1H'),
            'meter_total_wh': [1000, np.nan, 2000, np.nan, 1500],
            'session_duration_min': [30, 45, np.nan, 90, 40],
            'station_id': ['A', 'B', 'A', 'B', 'A']
        })
        
        handled_data = data_processor.handle_missing_data(incomplete_data)
        
        # Should not have any NaN values after handling
        assert not handled_data.isna().any().any()

    @patch('src.data_pipeline.processor.logger')
    def test_error_handling(self, mock_logger, data_processor):
        """Test error handling in data processing."""
        # Test with completely invalid data
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})
        
        # Should handle gracefully and log errors
        result = data_processor.process_pipeline_from_dataframe(invalid_data)
        
        assert result is not None  # Should return something even on error
        mock_logger.error.assert_called()  # Should log the error