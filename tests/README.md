# 🧪 Test Suite for EV Charging Optimization System

## Overview

Comprehensive test suite for the trustless edge-based real-time ML system for EV charging optimization. The test suite covers unit tests, integration tests, performance benchmarks, and security validation.

## 📁 Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest fixtures and configuration
├── test_runner.py              # Comprehensive test runner
├── README.md                   # This file
│
├── unit/                       # Unit tests for individual components
│   ├── test_data_processor.py  # Data processing pipeline tests
│   ├── test_metrics_collector.py  # Metrics collection tests
│   ├── test_lstm_model.py      # LSTM model tests
│   ├── test_xgboost_model.py   # XGBoost model tests
│   └── test_blockchain_validator.py  # Blockchain validator tests
│
├── integration/                # Integration tests for complete workflows
│   ├── test_federated_learning.py  # End-to-end FL testing
│   └── test_complete_pipeline.py   # Full system integration
│
├── performance/                # Performance and benchmark tests
│   └── test_model_performance.py   # ML model performance tests
│
├── security/                   # Security and privacy tests
│   └── test_privacy_protection.py  # Privacy mechanism tests
│
└── utils/                      # Test utilities and helpers
    ├── __init__.py
    └── test_helpers.py          # Helper functions and mock objects
```

## 🚀 Quick Start

### Run All Tests
```bash
# Using the test runner
python tests/test_runner.py --all --verbose

# Using Makefile (recommended)
make test

# Using pytest directly
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
make test-unit
python tests/test_runner.py --unit

# Integration tests
make test-integration  
python tests/test_runner.py --integration

# Performance tests
make test-performance
python tests/test_runner.py --performance

# Security tests
make test-security
python tests/test_runner.py --security

# Quick smoke test
make test-smoke
python tests/test_runner.py --smoke
```

### Run with Coverage
```bash
make test-coverage
python tests/test_runner.py --all --coverage
```

## 🧪 Test Categories

### Unit Tests (`tests/unit/`)
Test individual components in isolation:
- **Data Processing**: CSV loading, feature engineering, validation
- **Metrics Collection**: ML metrics, FL metrics, system metrics
- **ML Models**: LSTM training, XGBoost optimization, edge deployment
- **Blockchain**: Smart contract validation, mock validators

### Integration Tests (`tests/integration/`)  
Test complete workflows:
- **Federated Learning**: End-to-end FL simulation with multiple clients
- **Complete Pipeline**: Data processing → ML training → FL → Blockchain validation

### Performance Tests (`tests/performance/`)
Benchmark system performance:
- **Data Processing**: Throughput and latency benchmarks
- **ML Inference**: Model prediction speed and memory usage
- **Scalability**: Performance with varying data sizes
- **Concurrent Processing**: Multi-threaded performance

### Security Tests (`tests/security/`)
Validate privacy and security mechanisms:
- **Differential Privacy**: Noise addition and budget tracking
- **Data Anonymization**: PII removal and k-anonymity
- **Input Validation**: SQL injection and XSS protection
- **Model Security**: Poisoning attack detection

## 📋 Test Configuration

### Pytest Configuration (`pytest.ini`)
```ini
[tool:pytest]
testpaths = tests
markers =
    unit: Unit tests for individual components
    integration: Integration tests for complete workflows  
    performance: Performance and benchmark tests
    security: Security and privacy tests
    slow: Slow tests (>30 seconds)
    gpu: Tests requiring GPU
    blockchain: Tests requiring blockchain connectivity
```

### Running Specific Markers
```bash
# Run only fast tests
pytest -m "not slow"

# Run only GPU tests (if GPU available)  
pytest -m gpu

# Run blockchain tests
pytest -m blockchain
```

## 🛠️ Test Dependencies

### Core Testing Dependencies
- `pytest>=7.4.0` - Testing framework
- `pytest-cov>=4.1.0` - Coverage reporting  
- `pytest-mock>=3.11.0` - Mocking utilities
- `pytest-asyncio>=0.21.0` - Async testing support

### Optional Dependencies (Auto-detected)
- `torch>=2.0.0` - For LSTM model tests
- `xgboost>=1.7.0` - For XGBoost model tests
- `web3>=6.0.0` - For blockchain tests
- `flwr>=1.5.0` - For federated learning tests

### Install Test Dependencies
```bash
# Core testing dependencies
pip install -r requirements-test.txt

# Optional ML dependencies  
pip install torch xgboost flwr web3
```

## 📊 Test Reports

### Coverage Reports
```bash
# Generate HTML coverage report
make test-coverage

# View coverage report
open test_results/coverage_html/index.html
```

### Performance Reports
```bash
# Run performance benchmarks
make test-performance

# Results saved to test_results/
```

### Comprehensive Test Report
```bash  
# Generate detailed test report
python tests/test_runner.py --all --report test_results/full_report.md
```

## 🔧 Writing Tests

### Test File Structure
```python
"""Test module docstring describing what is being tested."""

import pytest
from unittest.mock import Mock, patch

# Import the module being tested
from src.component.module import ComponentClass


class TestComponentClass:
    """Test suite for ComponentClass."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return {"key": "value"}
    
    def test_initialization(self, sample_data):
        """Test component initialization."""
        component = ComponentClass(sample_data)
        assert component is not None
    
    def test_main_functionality(self, sample_data):
        """Test main component functionality."""
        component = ComponentClass(sample_data)
        result = component.process()
        assert result is not None
        assert isinstance(result, expected_type)
```

### Common Test Patterns

#### Testing with Mock Dependencies
```python
@patch('src.module.external_dependency')
def test_with_mock(self, mock_dependency):
    """Test component with mocked external dependency."""
    mock_dependency.return_value = "mocked_result"
    
    component = ComponentClass()
    result = component.use_dependency()
    
    assert result == "mocked_result"
    mock_dependency.assert_called_once()
```

#### Testing Async Functions
```python
@pytest.mark.asyncio
async def test_async_function(self):
    """Test async function."""
    result = await async_function()
    assert result is not None
```

#### Testing Performance
```python
def test_performance_benchmark(self, benchmark):
    """Test performance with pytest-benchmark."""
    result = benchmark(expensive_function, arg1, arg2)
    assert result is not None
```

## 🚨 Common Issues & Solutions

### Import Errors
```bash
# Add project root to Python path
export PYTHONPATH="$PWD:$PYTHONPATH"
pytest tests/
```

### Optional Dependencies
```python
# Skip tests when dependencies unavailable
@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
def test_pytorch_model(self):
    # Test code here
    pass
```

### Slow Tests
```python
# Mark slow tests
@pytest.mark.slow
def test_long_running_operation(self):
    # Long-running test code
    pass

# Skip slow tests in CI
pytest -m "not slow"
```

## 📈 Test Metrics

### Current Test Coverage
- **Unit Tests**: 85%+ coverage target
- **Integration Tests**: Key workflows covered
- **Performance Tests**: All models benchmarked
- **Security Tests**: Privacy mechanisms validated

### Test Execution Time
- **Smoke Tests**: < 10 seconds
- **Unit Tests**: < 60 seconds  
- **Integration Tests**: < 300 seconds
- **Full Suite**: < 600 seconds

## 🔄 Continuous Integration

### GitHub Actions Integration
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-test.txt
      - name: Run tests
        run: make ci-test
```

### Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## 💡 Best Practices

1. **Test Naming**: Use descriptive test names that explain what is being tested
2. **Test Isolation**: Each test should be independent and not rely on other tests
3. **Fixtures**: Use pytest fixtures for reusable test data and setup
4. **Mocking**: Mock external dependencies to ensure tests run in isolation
5. **Performance**: Mark slow tests and skip them in regular runs
6. **Coverage**: Aim for high test coverage but focus on quality over quantity
7. **Documentation**: Document complex test scenarios and edge cases

## 📚 Additional Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)
- [Test-Driven Development](https://testdriven.io/)
- [Mock Objects in Python](https://realpython.com/python-mock-library/)

---

**🎯 Goal**: Ensure the EV charging optimization system is robust, reliable, and ready for production deployment through comprehensive testing!