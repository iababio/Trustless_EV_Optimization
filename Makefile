# Makefile for EV Charging Optimization System

.PHONY: help install test test-unit test-integration test-performance test-security lint format clean demo blockchain docs

# Default target
help:
	@echo "🚗⚡ EV Charging Optimization - Available Commands"
	@echo "=================================================="
	@echo ""
	@echo "📦 Installation & Setup:"
	@echo "  install          Install dependencies and setup environment"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-optional Install optional dependencies (PyTorch, XGBoost, etc.)"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-performance Run performance benchmarks"
	@echo "  test-security    Run security tests"
	@echo "  test-smoke       Run quick smoke tests"
	@echo "  test-coverage    Run tests with coverage report"
	@echo ""
	@echo "🔧 Code Quality:"
	@echo "  lint             Run code linting (ruff, mypy)"
	@echo "  format           Format code (black, isort)"
	@echo "  security-scan    Run security scan (bandit)"
	@echo "  type-check       Run type checking (mypy)"
	@echo ""
	@echo "🚀 Demo & Blockchain:"
	@echo "  demo             Run the quick demo"
	@echo "  demo-full        Run the complete demo with all features"
	@echo "  blockchain-start Start local blockchain server"
	@echo "  blockchain-deploy Deploy smart contracts"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  docs             Generate documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  clean            Clean temporary files and caches"
	@echo "  clean-all        Deep clean including dependencies"

# Installation targets
install:
	@echo "📦 Installing core dependencies..."
	pip install -r requirements.txt

install-dev: install
	@echo "🛠️ Installing development dependencies..."
	pip install pytest pytest-cov pytest-xdist pytest-mock
	pip install black isort ruff mypy bandit
	pip install pre-commit

install-optional:
	@echo "⚡ Installing optional ML dependencies..."
	pip install torch torchvision || echo "⚠️ PyTorch installation failed"
	pip install xgboost || echo "⚠️ XGBoost installation failed"  
	pip install flwr || echo "⚠️ Flower installation failed"
	pip install web3 || echo "⚠️ Web3.py installation failed"
	pip install structlog || echo "⚠️ Structlog installation failed"
	pip install prometheus-client || echo "⚠️ Prometheus client installation failed"

# Testing targets
test:
	@echo "🧪 Running all tests..."
	python tests/test_runner.py --all --verbose

test-unit:
	@echo "🔬 Running unit tests..."
	python tests/test_runner.py --unit --verbose

test-integration:
	@echo "🔗 Running integration tests..."
	python tests/test_runner.py --integration --verbose

test-performance:
	@echo "⚡ Running performance tests..."
	python tests/test_runner.py --performance --verbose

test-security:
	@echo "🔒 Running security tests..."
	python tests/test_runner.py --security --verbose

test-smoke:
	@echo "💨 Running smoke tests..."
	python test_simple.py

test-coverage:
	@echo "📊 Running tests with coverage..."
	python tests/test_runner.py --all --coverage --verbose --report test_results/coverage_report.md

# Code quality targets
lint:
	@echo "🔍 Running code linting..."
	ruff check src/ tests/ --fix
	ruff format src/ tests/

format:
	@echo "✨ Formatting code..."
	black src/ tests/
	isort src/ tests/

security-scan:
	@echo "🛡️ Running security scan..."
	bandit -r src/ -f json -o test_results/security_report.json || true
	bandit -r src/

type-check:
	@echo "🔍 Running type checking..."
	mypy src/ --ignore-missing-imports --no-strict-optional

# Demo targets
demo:
	@echo "🚀 Running quick demo..."
	python demo.py

demo-full:
	@echo "🚀 Running complete demo with all features..."
	python examples/complete_demo.py

# Blockchain targets
blockchain-start:
	@echo "🔗 Starting local blockchain server..."
	./start_blockchain.sh

blockchain-deploy:
	@echo "📜 Deploying smart contracts..."
	./deploy_contracts.sh

# Documentation targets
docs:
	@echo "📚 Generating documentation..."
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs/ docs/_build/html; \
	else \
		echo "⚠️ Sphinx not installed. Install with: pip install sphinx sphinx-rtd-theme"; \
	fi

docs-serve: docs
	@echo "🌐 Serving documentation at http://localhost:8000"
	@cd docs/_build/html && python -m http.server 8000

# Cleanup targets
clean:
	@echo "🧹 Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -rf test_results/*.xml test_results/*.json

clean-all: clean
	@echo "🧹 Deep cleaning..."
	rm -rf venv/
	rm -rf .venv/
	rm -rf node_modules/
	rm -rf test_results/
	rm -rf docs/_build/

# CI/CD targets
ci-install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-xdist

ci-test:
	python tests/test_runner.py --all --coverage --report test_results/ci_report.md

ci-quality:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

# Development workflow
dev-setup: install-dev install-optional
	@echo "🛠️ Development environment setup complete!"
	@echo "💡 Next steps:"
	@echo "   1. Run 'make test-smoke' to verify installation"
	@echo "   2. Run 'make demo' to see the system in action"
	@echo "   3. Run 'make blockchain-start' to start blockchain server"

# Quick validation
validate: test-smoke lint type-check
	@echo "✅ System validation complete!"

# Show system status
status:
	@echo "📊 System Status"
	@echo "==============="
	@echo "Python: $(shell python --version)"
	@echo "Pip packages:"
	@pip list | grep -E "(torch|xgboost|flwr|web3|pytest)" || echo "  No optional packages found"
	@echo ""
	@echo "🧪 Running quick health check..."
	@python health_check.py