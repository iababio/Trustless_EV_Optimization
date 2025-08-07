"""Unit tests for XGBoost models."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

try:
    from src.ml_models.xgboost_model import LightweightXGBoost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
class TestLightweightXGBoost:
    """Test suite for LightweightXGBoost."""

    @pytest.fixture
    def sample_model(self, metrics_collector):
        """Create a sample XGBoost model for testing."""
        return LightweightXGBoost(
            model_name="test_xgb",
            metrics_collector=metrics_collector,
            max_depth=3,
            n_estimators=10,
            learning_rate=0.1
        )

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 8
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 10 + 50  # Target values around 50
        
        return X, y

    def test_model_initialization(self, sample_model):
        """Test XGBoost model initialization."""
        assert sample_model.model_name == "test_xgb"
        assert sample_model.max_depth == 3
        assert sample_model.n_estimators == 10
        assert sample_model.learning_rate == 0.1
        assert sample_model.model is None  # Not trained yet

    def test_model_training(self, sample_model, sample_data):
        """Test model training."""
        X, y = sample_data
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        training_summary = sample_model.train(X_train, y_train, X_val, y_val)
        
        assert sample_model.model is not None
        assert 'training_time' in training_summary
        assert 'final_train_score' in training_summary
        assert 'final_val_score' in training_summary

    def test_model_prediction(self, sample_model, sample_data):
        """Test model prediction."""
        X, y = sample_data
        
        # Train first
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        # Make predictions
        predictions = sample_model.predict(X_val)
        
        assert len(predictions) == len(X_val)
        assert isinstance(predictions, np.ndarray)
        assert not np.isnan(predictions).any()

    def test_model_evaluation(self, sample_model, sample_data):
        """Test model evaluation."""
        X, y = sample_data
        
        # Train first
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate
        metrics = sample_model.evaluate(X_val, y_val)
        
        assert 'r2_score' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert isinstance(metrics['r2_score'], (int, float))

    def test_feature_importance(self, sample_model, sample_data):
        """Test feature importance extraction."""
        X, y = sample_data
        
        # Train first
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        # Get feature importance
        importance = sample_model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == X.shape[1]  # One importance per feature
        assert all(isinstance(v, (int, float)) for v in importance.values())

    def test_model_summary(self, sample_model, sample_data):
        """Test model summary generation."""
        X, y = sample_data
        
        # Train first
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        summary = sample_model.get_model_summary()
        
        assert 'model_name' in summary
        assert 'architecture' in summary
        assert 'parameters' in summary
        assert 'feature_importance' in summary

    def test_edge_optimization(self, sample_model, sample_data):
        """Test edge deployment optimization."""
        X, y = sample_data
        
        # Train first
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        # Apply optimization
        optimization_report = sample_model.optimize_for_edge(pruning_threshold=0.01)
        
        assert 'pruning_applied' in optimization_report
        assert 'original_features' in optimization_report
        assert 'selected_features' in optimization_report

    def test_inference_with_metrics(self, sample_model, sample_data):
        """Test inference with timing metrics."""
        X, y = sample_data
        
        # Train first
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        # Make predictions with metrics
        predictions, metrics = sample_model.predict(X_val, return_metrics=True)
        
        assert len(predictions) == len(X_val)
        assert 'inference_time' in metrics
        assert 'throughput' in metrics
        assert metrics['inference_time'] > 0

    def test_cross_validation(self, sample_model, sample_data):
        """Test cross-validation functionality."""
        X, y = sample_data
        
        cv_scores = sample_model.cross_validate(X, y, cv_folds=3)
        
        assert 'cv_scores' in cv_scores
        assert 'mean_score' in cv_scores
        assert 'std_score' in cv_scores
        assert len(cv_scores['cv_scores']) == 3

    def test_hyperparameter_optimization(self, sample_model, sample_data):
        """Test hyperparameter optimization."""
        X, y = sample_data
        
        # Define parameter grid
        param_grid = {
            'max_depth': [2, 3],
            'n_estimators': [5, 10],
            'learning_rate': [0.1, 0.2]
        }
        
        best_params = sample_model.optimize_hyperparameters(X, y, param_grid, cv_folds=2)
        
        assert 'best_params' in best_params
        assert 'best_score' in best_params
        assert isinstance(best_params['best_params'], dict)

    def test_model_serialization(self, sample_model, sample_data, temp_dir):
        """Test model save and load functionality."""
        X, y = sample_data
        
        # Train first
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        # Save model
        model_path = temp_dir / "test_model.pkl"
        sample_model.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_model = LightweightXGBoost(model_name="loaded_model")
        loaded_model.load_model(str(model_path))
        
        # Compare predictions
        original_pred = sample_model.predict(X_val)
        loaded_pred = loaded_model.predict(X_val)
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)

    def test_early_stopping(self, metrics_collector):
        """Test early stopping functionality."""
        model = LightweightXGBoost(
            model_name="early_stop_test",
            metrics_collector=metrics_collector,
            n_estimators=100,
            early_stopping_rounds=5
        )
        
        # Create simple data that should converge quickly
        X = np.random.randn(100, 5)
        y = X.sum(axis=1) + np.random.randn(100) * 0.1  # Simple linear relationship
        
        split_idx = 80
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        training_summary = model.train(X_train, y_train, X_val, y_val)
        
        # Should have stopped early (less than 100 estimators)
        assert 'actual_estimators' in training_summary
        assert training_summary['actual_estimators'] < 100

    def test_feature_selection(self, sample_model, sample_data):
        """Test automatic feature selection."""
        X, y = sample_data
        
        # Add some noise features
        noise_features = np.random.randn(X.shape[0], 5)
        X_with_noise = np.concatenate([X, noise_features], axis=1)
        
        # Train model
        split_idx = int(len(X_with_noise) * 0.8)
        X_train, X_val = X_with_noise[:split_idx], X_with_noise[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        # Apply feature selection
        selected_features = sample_model.select_features(importance_threshold=0.01)
        
        assert 'selected_indices' in selected_features
        assert 'importance_scores' in selected_features
        assert len(selected_features['selected_indices']) <= X_with_noise.shape[1]

    def test_model_interpretability(self, sample_model, sample_data):
        """Test model interpretability features."""
        X, y = sample_data
        
        # Train first
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        sample_model.train(X_train, y_train, X_val, y_val)
        
        # Get interpretability report
        interp_report = sample_model.get_interpretability_report(X_val[:5])
        
        assert 'feature_importance' in interp_report
        assert 'sample_explanations' in interp_report
        assert len(interp_report['sample_explanations']) == 5

    def test_error_handling(self, sample_model):
        """Test error handling for invalid inputs."""
        # Test prediction without training
        with pytest.raises(ValueError):
            sample_model.predict(np.random.randn(10, 5))
        
        # Test training with mismatched data
        X = np.random.randn(10, 5)
        y = np.random.randn(15)  # Wrong size
        
        with pytest.raises(ValueError):
            sample_model.train(X, y)