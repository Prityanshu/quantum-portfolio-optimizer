"""
Test Advanced Quantum Portfolio Optimizer
=========================================
This script tests the advanced quantum portfolio optimizer integration.
"""

import requests
import json
import time

def test_advanced_quantum_optimizer():
    """Test the advanced quantum optimizer endpoint"""
    
    print("=" * 60)
    print("🧪 TESTING ADVANCED QUANTUM PORTFOLIO OPTIMIZER")
    print("=" * 60)
    
    # Test 1: Health check
    print("\n1. Testing health check...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Health check passed")
            print(f"   Version: {health_data.get('version', 'unknown')}")
            print(f"   Advanced Quantum Optimizer: {health_data.get('services', {}).get('advanced_quantum_optimizer', False)}")
            print(f"   Quantum Backend: {health_data.get('services', {}).get('quantum_backend', False)}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return
    
    # Test 2: Advanced quantum optimizer test
    print("\n2. Testing advanced quantum optimizer...")
    try:
        response = requests.post("http://localhost:8000/test-advanced-quantum", timeout=120)
        if response.status_code == 200:
            test_data = response.json()
            print(f"✅ Advanced quantum test passed")
            print(f"   Test symbols: {test_data.get('test_symbols', [])}")
            print(f"   Quantum portfolio: {test_data.get('quantum_portfolio', {})}")
            print(f"   Quantum Sharpe ratio: {test_data.get('quantum_sharpe', 0):.3f}")
            print(f"   Classical Sharpe ratio: {test_data.get('classical_sharpe', 0):.3f}")
            print(f"   Execution time: {test_data.get('execution_time', 0):.2f}s")
            print(f"   Backend used: {test_data.get('backend_used', 'unknown')}")
            print(f"   Quantum available: {test_data.get('quantum_available', False)}")
        else:
            print(f"❌ Advanced quantum test failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Advanced quantum test error: {e}")
    
    # Test 3: Portfolio optimization with advanced method
    print("\n3. Testing portfolio optimization with advanced QAOA...")
    try:
        optimization_data = {
            "investment_amount": 10000,
            "stock_symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
            "optimization_method": "qaoa",
            "risk_tolerance": "moderate",
            "quantum_backend": "aer_simulator",
            "qaoa_layers": 2,
            "max_iterations": 50,
            "include_visualization": False
        }
        
        response = requests.post(
            "http://localhost:8000/optimize-portfolio",
            json=optimization_data,
            timeout=180
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Portfolio optimization passed")
            print(f"   Method: {result.get('optimization_method', 'unknown')}")
            print(f"   Sharpe ratio: {result.get('sharpe_ratio', 0):.3f}")
            print(f"   Expected return: {result.get('expected_return', 0):.3f}")
            print(f"   Risk: {result.get('risk_std_dev', 0):.3f}")
            print(f"   Allocations: {result.get('allocations', {})}")
            
            # Check if quantum metrics are present
            quantum_metrics = result.get('quantum_metrics')
            if quantum_metrics:
                print(f"   Quantum backend: {quantum_metrics.get('backend_used', 'unknown')}")
                print(f"   QAOA layers: {quantum_metrics.get('qaoa_layers', 0)}")
                print(f"   Execution time: {quantum_metrics.get('execution_time', 0):.2f}s")
            
            # Check performance comparison
            comparison = result.get('performance_comparison', {})
            if comparison:
                print(f"   Performance comparison:")
                for method, metrics in comparison.items():
                    print(f"     {method}: Sharpe={metrics.get('sharpe_ratio', 0):.3f}")
        else:
            print(f"❌ Portfolio optimization failed: {response.status_code}")
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Portfolio optimization error: {e}")
    
    print("\n" + "=" * 60)
    print("🏁 TESTING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_advanced_quantum_optimizer() 