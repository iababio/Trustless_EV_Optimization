"""Comprehensive test runner for the EV optimization system."""

# Fix protobuf compatibility issue with Web3.py
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', 'python')

import pytest
import sys
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestRunner:
    """Main test runner with different test categories and reporting."""
    
    def __init__(self):
        self.project_root = project_root
        self.test_dir = self.project_root / "tests"
        self.results_dir = self.project_root / "test_results"
        self.results_dir.mkdir(exist_ok=True)
    
    def run_unit_tests(self, verbose: bool = False, coverage: bool = False) -> Dict:
        """Run unit tests."""
        print("🧪 Running Unit Tests...")
        
        cmd = [
            "python", "-m", "pytest", 
            str(self.test_dir / "unit"),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html:test_results/coverage_html",
                "--cov-report=term",
                "--cov-report=json:test_results/coverage.json"
            ])
        
        return self._run_pytest(cmd, "unit_tests")
    
    def run_integration_tests(self, verbose: bool = False) -> Dict:
        """Run integration tests."""
        print("🔗 Running Integration Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "integration"),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        return self._run_pytest(cmd, "integration_tests")
    
    def run_performance_tests(self, verbose: bool = False) -> Dict:
        """Run performance tests."""
        print("⚡ Running Performance Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "performance"),
            "-v" if verbose else "-q",
            "--tb=short",
            "-m", "not slow"  # Skip slow tests by default
        ]
        
        return self._run_pytest(cmd, "performance_tests")
    
    def run_security_tests(self, verbose: bool = False) -> Dict:
        """Run security tests."""
        print("🔒 Running Security Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir / "security"),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        return self._run_pytest(cmd, "security_tests")
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = False, 
                     include_slow: bool = False) -> Dict:
        """Run all test categories."""
        print("🚀 Running Complete Test Suite...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-v" if verbose else "-q",
            "--tb=short"
        ]
        
        if not include_slow:
            cmd.extend(["-m", "not slow"])
        
        if coverage:
            cmd.extend([
                "--cov=src",
                "--cov-report=html:test_results/coverage_html",
                "--cov-report=term",
                "--cov-report=xml:test_results/coverage.xml",
                "--cov-report=json:test_results/coverage.json"
            ])
        
        return self._run_pytest(cmd, "all_tests")
    
    def run_smoke_tests(self) -> Dict:
        """Run quick smoke tests to verify basic functionality."""
        print("💨 Running Smoke Tests...")
        
        cmd = [
            "python", "-m", "pytest",
            str(self.test_dir),
            "-k", "test_initialization or test_basic",
            "-q", "--tb=line"
        ]
        
        return self._run_pytest(cmd, "smoke_tests")
    
    def _run_pytest(self, cmd: List[str], test_type: str) -> Dict:
        """Run pytest with given command and return results."""
        import subprocess
        
        start_time = time.time()
        
        try:
            # Run pytest and capture output
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            
            duration = time.time() - start_time
            
            # Parse pytest output for summary
            output_lines = result.stdout.split('\n')
            summary_line = ""
            for line in output_lines:
                if "passed" in line or "failed" in line or "error" in line:
                    if any(keyword in line for keyword in ["==", "FAILED", "ERROR"]):
                        summary_line = line.strip()
                        break
            
            return {
                "test_type": test_type,
                "return_code": result.returncode,
                "duration": duration,
                "summary": summary_line,
                "output": result.stdout,
                "errors": result.stderr,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test_type": test_type,
                "return_code": -1,
                "duration": time.time() - start_time,
                "summary": "Tests timed out after 10 minutes",
                "output": "",
                "errors": "Timeout",
                "success": False
            }
        except Exception as e:
            return {
                "test_type": test_type,
                "return_code": -1,
                "duration": time.time() - start_time,
                "summary": f"Test execution failed: {str(e)}",
                "output": "",
                "errors": str(e),
                "success": False
            }
    
    def generate_test_report(self, results: List[Dict], output_file: Optional[str] = None) -> str:
        """Generate a comprehensive test report."""
        if output_file is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_file = self.results_dir / f"test_report_{timestamp}.md"
        
        report = []
        report.append("# EV Charging Optimization - Test Report")
        report.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['success'])
        failed_tests = total_tests - passed_tests
        total_duration = sum(r['duration'] for r in results)
        
        report.append("## 📊 Summary")
        report.append(f"- **Total Test Categories:** {total_tests}")
        report.append(f"- **Passed:** {passed_tests} ✅")
        report.append(f"- **Failed:** {failed_tests} ❌")
        report.append(f"- **Total Duration:** {total_duration:.2f} seconds")
        report.append(f"- **Success Rate:** {(passed_tests/total_tests*100):.1f}%")
        report.append("")
        
        # Detailed results
        report.append("## 📝 Detailed Results")
        
        for result in results:
            status_emoji = "✅" if result['success'] else "❌"
            report.append(f"### {status_emoji} {result['test_type'].replace('_', ' ').title()}")
            report.append(f"- **Duration:** {result['duration']:.2f}s")
            report.append(f"- **Return Code:** {result['return_code']}")
            
            if result['summary']:
                report.append(f"- **Summary:** {result['summary']}")
            
            if not result['success'] and result['errors']:
                report.append("- **Errors:**")
                report.append("```")
                report.append(result['errors'])
                report.append("```")
            
            report.append("")
        
        # System information
        report.append("## 💻 System Information")
        report.append(f"- **Python Version:** {sys.version}")
        report.append(f"- **Platform:** {sys.platform}")
        report.append(f"- **Working Directory:** {os.getcwd()}")
        
        # Check for optional dependencies
        report.append("")
        report.append("## 📦 Dependencies Status")
        
        dependencies = [
            ("torch", "PyTorch for LSTM models"),
            ("xgboost", "XGBoost for gradient boosting"),
            ("flwr", "Flower for federated learning"),
            ("web3", "Web3.py for blockchain"),
            ("structlog", "Structured logging"),
            ("prometheus_client", "Prometheus metrics")
        ]
        
        for dep, description in dependencies:
            try:
                __import__(dep)
                report.append(f"- ✅ **{dep}:** Available - {description}")
            except ImportError:
                report.append(f"- ❌ **{dep}:** Not available - {description}")
        
        report_content = '\n'.join(report)
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report_content)
        
        print(f"📄 Test report saved to: {output_file}")
        return report_content
    
    def print_summary(self, results: List[Dict]):
        """Print a summary of test results to console."""
        print("\n" + "="*60)
        print("🎯 TEST EXECUTION SUMMARY")
        print("="*60)
        
        for result in results:
            status = "PASS" if result['success'] else "FAIL"
            status_emoji = "✅" if result['success'] else "❌"
            test_name = result['test_type'].replace('_', ' ').title()
            
            print(f"{status_emoji} {test_name:<25} | {status:<4} | {result['duration']:.2f}s")
            
            if result['summary']:
                print(f"    └─ {result['summary']}")
        
        print("="*60)
        
        # Overall status
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r['success'])
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        overall_status = "✅ ALL TESTS PASSED" if passed_tests == total_tests else "❌ SOME TESTS FAILED"
        print(f"{overall_status} ({passed_tests}/{total_tests} - {success_rate:.1f}%)")
        print("="*60)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="EV Optimization Test Runner")
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--integration", action="store_true", help="Run integration tests") 
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--security", action="store_true", help="Run security tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--smoke", action="store_true", help="Run smoke tests")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--include-slow", action="store_true", help="Include slow tests")
    parser.add_argument("--report", "-r", help="Output file for test report")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    results = []
    
    # Determine which tests to run
    if args.smoke:
        results.append(runner.run_smoke_tests())
    elif args.all:
        results.append(runner.run_all_tests(
            verbose=args.verbose, 
            coverage=args.coverage,
            include_slow=args.include_slow
        ))
    else:
        if args.unit:
            results.append(runner.run_unit_tests(verbose=args.verbose, coverage=args.coverage))
        if args.integration:
            results.append(runner.run_integration_tests(verbose=args.verbose))
        if args.performance:
            results.append(runner.run_performance_tests(verbose=args.verbose))
        if args.security:
            results.append(runner.run_security_tests(verbose=args.verbose))
    
    # If no specific tests requested, run smoke tests
    if not results:
        print("No specific tests requested. Running smoke tests...")
        results.append(runner.run_smoke_tests())
    
    # Print summary and generate report
    runner.print_summary(results)
    
    if args.report or len(results) > 1:
        runner.generate_test_report(results, args.report)
    
    # Exit with error code if any tests failed
    if any(not r['success'] for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()