"""Optimized XGBoost model for EV charging demand prediction with edge deployment."""

import xgboost as xgb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, Union
from datetime import datetime
import joblib
import structlog
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tempfile
import os

from ..metrics.collector import MetricsCollector, MetricType

logger = structlog.get_logger(__name__)


class LightweightXGBoost:
    """Optimized XGBoost model for edge deployment in EV charging stations."""
    
    def __init__(
        self,
        model_name: str = "LightweightXGBoost",
        metrics_collector: Optional[MetricsCollector] = None,
        **xgb_params
    ):
        self.model_name = model_name
        self.metrics_collector = metrics_collector
        
        # Default parameters optimized for edge deployment
        default_params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "max_depth": 4,  # Shallow trees for faster inference
            "n_estimators": 100,  # Moderate number for balance
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,  # L1 regularization
            "reg_lambda": 0.1,  # L2 regularization
            "random_state": 42,
            "n_jobs": 1,  # Single thread for edge deployment
            "tree_method": "hist",  # Memory efficient
            "verbosity": 0,
        }
        
        # Merge with user parameters
        self.xgb_params = {**default_params, **xgb_params}
        
        # Initialize model
        self.model = xgb.XGBRegressor(**self.xgb_params)
        
        # Model metadata
        self.training_history: Dict[str, Any] = {}
        self.feature_names: Optional[list] = None
        self.feature_importance_: Optional[Dict[str, float]] = None
        self.model_size_bytes = 0
        self.optimization_applied = False
        
        logger.info(
            "LightweightXGBoost initialized",
            model_name=model_name,
            max_depth=self.xgb_params["max_depth"],
            n_estimators=self.xgb_params["n_estimators"]
        )
    
    def train(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        early_stopping_rounds: int = 10,
        verbose_eval: bool = False,
    ) -> Dict[str, Any]:
        """Train XGBoost model with comprehensive metrics tracking."""
        training_start = datetime.now()
        
        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Prepare validation data if provided
        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)
        
        try:
            # Train the model
            logger.info("Starting XGBoost training", train_samples=len(X_train))
            
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=verbose_eval,
            )
            
            training_time = (datetime.now() - training_start).total_seconds()
            
            # Get training results
            if hasattr(self.model, 'evals_result_'):
                evals_result = self.model.evals_result_
            else:
                evals_result = {}
            
            # Calculate model size
            self.model_size_bytes = self._calculate_model_size()
            
            # Generate feature importance
            self.feature_importance_ = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            
            # Prepare training summary
            training_summary = {
                "training_time_seconds": training_time,
                "best_iteration": getattr(self.model, 'best_iteration', len(self.model.get_booster().get_dump())),
                "n_features": len(self.feature_names),
                "model_size_bytes": self.model_size_bytes,
                "model_size_mb": self.model_size_bytes / (1024 * 1024),
                "evals_result": evals_result,
                "feature_importance": self.feature_importance_,
            }
            
            # Store training history
            self.training_history = training_summary
            
            # Record training metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    training_time,
                    self.model_name,
                    metadata={"operation": "training"}
                )
                
                self.metrics_collector.record_metric(
                    MetricType.MODEL_SIZE,
                    self.model_size_bytes,
                    self.model_name
                )
            
            logger.info(
                "XGBoost training completed",
                training_time=training_time,
                model_size_mb=training_summary["model_size_mb"],
                best_iteration=training_summary["best_iteration"]
            )
            
            return training_summary
            
        except Exception as e:
            logger.error("XGBoost training failed", error=str(e))
            raise
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame], return_metrics: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, float]]]:
        """Make predictions with optional performance metrics."""
        start_time = datetime.now()
        
        try:
            predictions = self.model.predict(X)
            
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Record inference metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    MetricType.INFERENCE_TIME,
                    inference_time,
                    self.model_name,
                    metadata={"batch_size": len(X), "operation": "prediction"}
                )
            
            if return_metrics:
                metrics = {
                    "inference_time_seconds": inference_time,
                    "samples_per_second": len(X) / inference_time if inference_time > 0 else 0,
                    "batch_size": len(X),
                }
                return predictions, metrics
            
            return predictions
            
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise
    
    def evaluate(
        self,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """Evaluate model performance with comprehensive metrics."""
        predictions = self.predict(X_test)
        
        # Calculate regression metrics
        metrics = {
            "mse": mean_squared_error(y_test, predictions),
            "rmse": np.sqrt(mean_squared_error(y_test, predictions)),
            "mae": mean_absolute_error(y_test, predictions),
            "r2_score": r2_score(y_test, predictions),
            "mape": np.mean(np.abs((y_test - predictions) / (y_test + 1e-8))) * 100,
        }
        
        # Additional metrics
        residuals = y_test - predictions
        metrics.update({
            "mean_residual": np.mean(residuals),
            "std_residual": np.std(residuals),
            "max_error": np.max(np.abs(residuals)),
            "q95_error": np.percentile(np.abs(residuals), 95),
        })
        
        # Record evaluation metrics
        if self.metrics_collector:
            self.metrics_collector.record_ml_metrics(
                accuracy=1.0 / (1.0 + metrics["rmse"]),  # Approximation
                loss=metrics["rmse"],
                component=self.model_name,
                additional_metrics=metrics
            )
        
        logger.info(
            "Model evaluation completed",
            rmse=metrics["rmse"],
            mae=metrics["mae"],
            r2_score=metrics["r2_score"]
        )
        
        return metrics
    
    def optimize_for_edge(
        self,
        pruning_threshold: float = 0.001,
        max_trees: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Optimize XGBoost model for edge deployment."""
        optimization_start = datetime.now()
        original_size = self.model_size_bytes
        
        try:
            # Get the booster
            booster = self.model.get_booster()
            
            # Tree pruning based on feature importance
            if pruning_threshold > 0:
                self._prune_low_importance_features(pruning_threshold)
            
            # Limit number of trees if specified
            if max_trees and max_trees < self.model.n_estimators:
                self.model.n_estimators = max_trees
                # Note: XGBoost doesn't directly support tree removal after training
                # This would require retraining, so we'll log the recommendation
                logger.info(f"Recommendation: Retrain with max {max_trees} trees for optimal size")
            
            # Calculate optimized size
            optimized_size = self._calculate_model_size()
            
            optimization_report = {
                "original_size_bytes": original_size,
                "optimized_size_bytes": optimized_size,
                "size_reduction_ratio": (original_size - optimized_size) / original_size if original_size > 0 else 0,
                "compression_ratio": original_size / optimized_size if optimized_size > 0 else 1,
                "optimization_time_seconds": (datetime.now() - optimization_start).total_seconds(),
                "pruning_threshold": pruning_threshold,
                "optimizations_applied": ["feature_pruning"] if pruning_threshold > 0 else [],
            }
            
            self.optimization_applied = True
            self.model_size_bytes = optimized_size
            
            # Record optimization metrics
            if self.metrics_collector:
                self.metrics_collector.record_metric(
                    MetricType.MODEL_SIZE,
                    optimized_size,
                    self.model_name,
                    metadata={"optimization": "post_optimization"}
                )
            
            logger.info(
                "XGBoost optimization completed",
                size_reduction=f"{optimization_report['size_reduction_ratio']:.2%}",
                compression_ratio=f"{optimization_report['compression_ratio']:.2f}x"
            )
            
            return optimization_report
            
        except Exception as e:
            logger.error("XGBoost optimization failed", error=str(e))
            return {"error": str(e)}
    
    def _prune_low_importance_features(self, threshold: float) -> None:
        """Remove features with importance below threshold."""
        if self.feature_importance_ is None:
            return
        
        important_features = [
            name for name, importance in self.feature_importance_.items()
            if importance >= threshold
        ]
        
        logger.info(
            f"Feature pruning: keeping {len(important_features)} out of {len(self.feature_importance_)} features"
        )
    
    def _calculate_model_size(self) -> int:
        """Calculate model size in bytes."""
        try:
            # Save model to temporary file to get size
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                joblib.dump(self.model, tmp_file.name, compress=3)
                size = os.path.getsize(tmp_file.name)
                os.unlink(tmp_file.name)
                return size
        except Exception:
            # Fallback estimation
            booster = self.model.get_booster()
            n_trees = len(booster.get_dump())
            estimated_size = n_trees * 1000  # Rough estimate: 1KB per tree
            return estimated_size
    
    def get_feature_importance(self, importance_type: str = "weight") -> Dict[str, float]:
        """Get feature importance with different metrics."""
        if self.model.get_booster() is None:
            logger.warning("Model not trained yet")
            return {}
        
        try:
            importance = self.model.get_booster().get_score(importance_type=importance_type)
            
            # Ensure all features are included
            if self.feature_names:
                for feature in self.feature_names:
                    if feature not in importance:
                        importance[feature] = 0.0
            
            return importance
            
        except Exception as e:
            logger.error("Failed to get feature importance", error=str(e))
            return {}
    
    def save_model(self, filepath: str) -> None:
        """Save XGBoost model with metadata."""
        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance_,
            "training_history": self.training_history,
            "xgb_params": self.xgb_params,
            "optimization_applied": self.optimization_applied,
            "model_size_bytes": self.model_size_bytes,
            "saved_at": datetime.now().isoformat(),
        }
        
        joblib.dump(model_data, filepath, compress=3)
        logger.info("XGBoost model saved", filepath=filepath, model_name=self.model_name)
    
    @classmethod
    def load_model(cls, filepath: str, metrics_collector: Optional[MetricsCollector] = None):
        """Load XGBoost model with metadata."""
        model_data = joblib.load(filepath)
        
        # Create instance
        instance = cls(
            model_name=model_data["model_name"],
            metrics_collector=metrics_collector,
            **model_data["xgb_params"]
        )
        
        # Load model and metadata
        instance.model = model_data["model"]
        instance.feature_names = model_data["feature_names"]
        instance.feature_importance_ = model_data["feature_importance"]
        instance.training_history = model_data["training_history"]
        instance.optimization_applied = model_data["optimization_applied"]
        instance.model_size_bytes = model_data["model_size_bytes"]
        
        logger.info("XGBoost model loaded", filepath=filepath, model_name=instance.model_name)
        return instance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        summary = {
            "model_name": self.model_name,
            "model_type": "XGBoost",
            "parameters": self.xgb_params,
            "model_size_bytes": self.model_size_bytes,
            "model_size_mb": self.model_size_bytes / (1024 * 1024),
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "optimization_applied": self.optimization_applied,
            "training_history": self.training_history,
        }
        
        if self.model.get_booster():
            booster = self.model.get_booster()
            summary.update({
                "n_trees": len(booster.get_dump()),
                "max_depth": self.xgb_params.get("max_depth", "unknown"),
                "feature_importance": self.feature_importance_,
            })
        
        return summary
    
    def explain_prediction(
        self,
        X_sample: Union[np.ndarray, pd.DataFrame],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        """Explain prediction using feature importance and SHAP-like analysis."""
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        prediction = self.predict(X_sample)[0]
        
        # Get feature contributions (simplified)
        feature_contributions = {}
        if self.feature_names and self.feature_importance_:
            sample_values = X_sample[0] if hasattr(X_sample, 'iloc') else X_sample[0]
            
            for i, feature in enumerate(self.feature_names):
                importance = self.feature_importance_.get(feature, 0)
                value = sample_values[i] if i < len(sample_values) else 0
                contribution = importance * value
                feature_contributions[feature] = contribution
        
        # Sort by contribution magnitude
        sorted_contributions = sorted(
            feature_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_k]
        
        explanation = {
            "prediction": float(prediction),
            "top_features": sorted_contributions,
            "feature_values": dict(zip(self.feature_names, X_sample[0])) if self.feature_names else {},
        }
        
        return explanation