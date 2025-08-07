"""Performance benchmarks for ML models."""

import pytest
import time
import numpy as np
import pandas as pd
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

try:
    import torch
    from src.ml_models.lstm import LightweightLSTM
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from src.ml_models.xgboost_model import LightweightXGBoost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.data_pipeline.processor import EVChargingDataProcessor
from src.metrics.collector import MetricsCollector


class TestModelPerformance:
    """Performance benchmarks for ML models."""

    @pytest.fixture
    def performance_data(self):
        """Generate large dataset for performance testing."""
        np.random.seed(42)
        n_samples = 10000
        n_features = 20
        
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='5min'),
            'meter_total_wh': np.random.exponential(4000, n_samples),
            'session_duration_min': np.random.gamma(2, 25, n_samples),
        }
        
        # Add engineered features
        for i in range(n_features):
            data[f'feature_{i}'] = np.random.randn(n_samples)
        
        df = pd.DataFrame(data)
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        return df

    @pytest.fixture
    def metrics_collector(self):
        """Create metrics collector for performance tests."""
        return MetricsCollector("performance_test", enable_prometheus=False)

    def test_data_processing_performance(self, performance_data, metrics_collector):
        """Benchmark data processing performance."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        # Measure processing time
        start_time = time.time()
        processed_data = processor.process_pipeline_from_dataframe(performance_data)
        processing_time = time.time() - start_time
        
        # Performance assertions
        assert processing_time < 10.0  # Should process 10k samples in < 10 seconds
        assert len(processed_data) > 0
        
        # Calculate throughput
        throughput = len(performance_data) / processing_time
        assert throughput > 1000  # > 1000 samples/second
        
        print(f"Data processing: {processing_time:.2f}s, {throughput:.0f} samples/sec")

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_lstm_inference_performance(self, metrics_collector):
        """Benchmark LSTM inference performance."""
        model = LightweightLSTM(
            input_size=20,
            hidden_size=32,
            num_layers=2,
            metrics_collector=metrics_collector
        )
        
        # Generate test data
        batch_sizes = [1, 8, 32, 64]
        seq_len = 10
        
        for batch_size in batch_sizes:
            test_input = torch.randn(batch_size, seq_len, 20)
            
            # Warm up
            with torch.no_grad():
                _ = model.forward(test_input)
            
            # Benchmark inference
            num_runs = 100
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(num_runs):
                    _ = model.forward(test_input)
            
            total_time = time.time() - start_time
            avg_time = total_time / num_runs
            throughput = batch_size / avg_time
            
            # Performance assertions
            assert avg_time < 0.1  # < 100ms per batch
            assert throughput > 10  # > 10 samples/second
            
            print(f"LSTM batch_size={batch_size}: {avg_time*1000:.2f}ms, {throughput:.0f} samples/sec")

    @pytest.mark.skipif(not HAS_XGBOOST, reason="XGBoost not available")
    def test_xgboost_training_performance(self, performance_data, metrics_collector):
        """Benchmark XGBoost training performance."""
        # Prepare data
        feature_columns = [col for col in performance_data.columns 
                          if col != 'meter_total_wh' and performance_data[col].dtype in ['float64', 'int64']]
        X = performance_data[feature_columns].values
        y = performance_data['meter_total_wh'].values
        
        # Remove invalid values
        valid_mask = ~np.isnan(y)
        X, y = X[valid_mask], y[valid_mask]
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Benchmark training
        model = LightweightXGBoost(
            model_name="performance_test",
            metrics_collector=metrics_collector,
            n_estimators=100
        )
        
        start_time = time.time()
        training_summary = model.train(X_train, y_train, X_val, y_val)
        training_time = time.time() - start_time
        
        # Performance assertions
        assert training_time < 30.0  # < 30 seconds for 8k samples
        assert 'training_time' in training_summary
        
        # Test inference performance
        start_time = time.time()
        predictions = model.predict(X_val)
        inference_time = time.time() - start_time
        
        inference_throughput = len(X_val) / inference_time
        assert inference_throughput > 1000  # > 1000 predictions/second
        
        print(f"XGBoost training: {training_time:.2f}s, inference: {inference_throughput:.0f} samples/sec")

    def test_memory_usage_monitoring(self, performance_data, metrics_collector):
        """Monitor memory usage during processing."""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        processor = EVChargingDataProcessor(metrics_collector)
        
        # Process data while monitoring memory
        processed_data = processor.process_pipeline_from_dataframe(performance_data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory usage should be reasonable
        assert memory_increase < 500  # < 500 MB increase
        assert len(processed_data) > 0
        
        print(f"Memory usage: {initial_memory:.1f}MB -> {peak_memory:.1f}MB (+{memory_increase:.1f}MB)")

    def test_concurrent_processing_performance(self, metrics_collector):
        """Test performance with concurrent processing."""
        def process_batch(batch_id):
            """Process a single batch of data."""
            np.random.seed(batch_id)  # Different seed for each batch
            batch_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=1000, freq='5min'),
                'meter_total_wh': np.random.exponential(4000, 1000),
                'session_duration_min': np.random.gamma(2, 25, 1000),
                'hour_of_day': np.random.randint(0, 24, 1000),
                'day_of_week': np.random.randint(0, 7, 1000),
                'feature_1': np.random.randn(1000),
                'feature_2': np.random.randn(1000),
            })
            
            processor = EVChargingDataProcessor(metrics_collector)
            return processor.process_pipeline_from_dataframe(batch_data)
        
        # Test with different numbers of threads
        thread_counts = [1, 2, 4]
        
        for num_threads in thread_counts:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_batch, i) for i in range(num_threads * 2)]
                results = [future.result() for future in futures]
            
            total_time = time.time() - start_time
            
            # Validate results
            assert len(results) == num_threads * 2
            for result in results:
                assert len(result) > 0
            
            print(f"Concurrent processing ({num_threads} threads): {total_time:.2f}s")

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
    def test_model_optimization_performance(self, metrics_collector):
        """Benchmark model optimization techniques."""
        model = LightweightLSTM(
            input_size=10,
            hidden_size=64,
            num_layers=3,
            metrics_collector=metrics_collector
        )
        
        # Measure original model size
        original_params = model.count_parameters()['total_parameters']
        
        # Generate test data
        test_input = torch.randn(16, 8, 10)
        
        # Benchmark original model
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model.forward(test_input)
        original_time = time.time() - start_time
        
        # Apply optimization
        optimization_start = time.time()
        optimization_report = model.optimize_for_edge(pruning_ratio=0.5, quantize=False)
        optimization_time = time.time() - optimization_start
        
        # Benchmark optimized model
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model.forward(test_input)
        optimized_time = time.time() - start_time
        
        # Performance assertions
        assert optimization_time < 10.0  # Optimization should be fast
        assert 'optimized_size' in optimization_report
        
        # Optimized model should be smaller (if pruning was applied)
        if optimization_report.get('pruning_applied', False):
            optimized_params = model.count_parameters()['total_parameters']
            assert optimized_params < original_params
        
        print(f"Model optimization: {optimization_time:.2f}s")
        print(f"Inference time: {original_time:.3f}s -> {optimized_time:.3f}s")

    def test_batch_processing_scalability(self, metrics_collector):
        """Test scalability with different batch sizes."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        batch_sizes = [100, 500, 1000, 5000]
        processing_times = {}
        
        for batch_size in batch_sizes:
            # Generate data of specific size
            test_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=batch_size, freq='5min'),
                'meter_total_wh': np.random.exponential(4000, batch_size),
                'session_duration_min': np.random.gamma(2, 25, batch_size),
                'hour_of_day': np.random.randint(0, 24, batch_size),
                'day_of_week': np.random.randint(0, 7, batch_size),
            })
            
            # Benchmark processing
            start_time = time.time()
            processed = processor.process_pipeline_from_dataframe(test_data)
            processing_time = time.time() - start_time
            
            processing_times[batch_size] = processing_time
            throughput = batch_size / processing_time
            
            # Validate results
            assert len(processed) > 0
            assert throughput > 100  # > 100 samples/second
            
            print(f"Batch size {batch_size}: {processing_time:.2f}s, {throughput:.0f} samples/sec")
        
        # Check scalability (should be roughly linear)
        time_per_sample = {size: time/size for size, time in processing_times.items()}
        
        # Time per sample should be relatively consistent
        times = list(time_per_sample.values())
        assert max(times) / min(times) < 3  # Within 3x of each other

    def test_memory_efficiency(self, metrics_collector):
        """Test memory efficiency with large datasets."""
        process = psutil.Process()
        
        # Test with increasingly large datasets
        dataset_sizes = [1000, 5000, 10000]
        
        for size in dataset_sizes:
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate large dataset
            large_data = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=size, freq='1min'),
                'meter_total_wh': np.random.exponential(4000, size),
                'session_duration_min': np.random.gamma(2, 25, size),
            })
            
            # Add many features
            for i in range(50):
                large_data[f'feature_{i}'] = np.random.randn(size)
            
            processor = EVChargingDataProcessor(metrics_collector)
            processed = processor.process_pipeline_from_dataframe(large_data)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_per_sample = (peak_memory - initial_memory) * 1024 / size  # KB per sample
            
            # Clean up
            del large_data, processed
            
            # Memory usage should scale reasonably
            assert memory_per_sample < 100  # < 100 KB per sample
            
            print(f"Dataset size {size}: {memory_per_sample:.2f} KB/sample")

    @pytest.mark.slow
    def test_stress_testing(self, metrics_collector):
        """Stress test the system with extreme workloads."""
        processor = EVChargingDataProcessor(metrics_collector)
        
        # Create very large dataset
        stress_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=50000, freq='30s'),
            'meter_total_wh': np.random.exponential(4000, 50000),
            'session_duration_min': np.random.gamma(2, 25, 50000),
            'hour_of_day': np.random.randint(0, 24, 50000),
            'day_of_week': np.random.randint(0, 7, 50000),
        })
        
        # Process under stress
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            processed = processor.process_pipeline_from_dataframe(stress_data)
            stress_time = time.time() - start_time
            peak_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Should complete within reasonable bounds
            assert stress_time < 60  # < 1 minute for 50k samples
            assert peak_memory - start_memory < 1000  # < 1GB memory increase
            assert len(processed) > 0
            
            print(f"Stress test: {stress_time:.2f}s, {peak_memory-start_memory:.1f}MB memory")
            
        except Exception as e:
            # If it fails, should fail gracefully
            assert "memory" in str(e).lower() or "timeout" in str(e).lower()