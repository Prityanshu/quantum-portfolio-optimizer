import requests
import json
import time
import sys

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint"""
    print("--- Health Check ---")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: SUCCESS")
            print(f"   Status: {data['status']}")
            print(f"   Cache Size: {data['cache_size']}")
            return True
        else:
            print(f"❌ Health Check: FAIL - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check: FAIL - {str(e)}")
        return False

def test_server_status():
    """Test the server status endpoint"""
    print("\n--- Server Status ---")
    try:
        response = requests.get(f"{BASE_URL}/servers/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server Status: SUCCESS")
            print(f"   Server ID: {data['server_id']}")
            print(f"   Status: {data['status']}")
            return True
        else:
            print(f"❌ Server Status: FAIL - Status {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server Status: FAIL - {str(e)}")
        return False

def test_stock_validation():
    """Test stock symbol validation"""
    print("\n--- Stock Validation ---")
    try:
        test_stocks = ["AAPL", "GOOGL", "MSFT", "INVALID_SYMBOL"]
        payload = {"symbols": test_stocks}
        
        response = requests.post(f"{BASE_URL}/stocks/validate", 
                               json=payload, 
                               headers={"Content-Type": "application/json"}, 
                               timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stock Validation: SUCCESS")
            for stock in data['stocks']:
                status = "✅" if stock['valid'] else "❌"
                print(f"   {status} {stock['symbol']}: ${stock['current_price']:.2f}")
            return True
        else:
            print(f"❌ Stock Validation: FAIL - Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Stock Validation: FAIL - {str(e)}")
        return False

def test_portfolio_optimization():
    """Test portfolio optimization"""
    print("\n--- Portfolio Optimization ---")
    try:
        # Test with valid stocks
        test_request = {
            "investment_amount": 10000,
            "stock_symbols": ["AAPL", "GOOGL", "MSFT", "TSLA"],
            "optimization_method": "quantum",
            "risk_tolerance": "moderate"
        }
        
        print(f"Testing with: {test_request['stock_symbols']}")
        print("This may take 30-60 seconds to fetch stock data...")
        
        response = requests.post(f"{BASE_URL}/portfolio/optimize", 
                               json=test_request, 
                               headers={"Content-Type": "application/json"}, 
                               timeout=120)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Portfolio Optimization: SUCCESS")
            print(f"   Expected Return: {data['expected_return']:.2f}%")
            print(f"   Risk (Std Dev): {data['risk_std_dev']:.2f}%")
            print(f"   Sharpe Ratio: {data['sharpe_ratio']:.3f}")
            print(f"   Method: {data['optimization_method']}")
            print(f"   Risk Tolerance: {data['risk_tolerance']}")
            print("   Allocations:")
            for symbol, allocation in data['allocations'].items():
                investment = data['investment_allocations'][symbol]
                print(f"     {symbol}: {allocation:.1f}% (${investment:.2f})")
            return True
        else:
            print(f"❌ Portfolio Optimization: FAIL - Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Portfolio Optimization: FAIL - {str(e)}")
        return False

def test_mpt_optimization():
    """Test Modern Portfolio Theory optimization"""
    print("\n--- MPT Optimization ---")
    try:
        test_request = {
            "investment_amount": 5000,
            "stock_symbols": ["AAPL", "MSFT", "AMZN"],
            "optimization_method": "mpt",  # Using MPT instead of quantum
            "risk_tolerance": "conservative"
        }
        
        print(f"Testing MPT with: {test_request['stock_symbols']}")
        
        response = requests.post(f"{BASE_URL}/portfolio/optimize", 
                               json=test_request, 
                               headers={"Content-Type": "application/json"}, 
                               timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ MPT Optimization: SUCCESS")
            print(f"   Expected Return: {data['expected_return']:.2f}%")
            print(f"   Risk (Std Dev): {data['risk_std_dev']:.2f}%")
            print(f"   Sharpe Ratio: {data['sharpe_ratio']:.3f}")
            print(f"   Method: {data['optimization_method']}")
            print("   Allocations:")
            for symbol, allocation in data['allocations'].items():
                investment = data['investment_allocations'][symbol]
                print(f"     {symbol}: {allocation:.1f}% (${investment:.2f})")
            return True
        else:
            print(f"❌ MPT Optimization: FAIL - Status {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ MPT Optimization: FAIL - {str(e)}")
        return False

def main():
    print("=" * 60)
    print("QUANTUM PORTFOLIO MANAGEMENT API TESTS")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("Waiting for server to start...")
    time.sleep(3)
    
    # Run tests
    tests = [
        ("Health Check", test_health_check),
        ("Server Status", test_server_status),
        ("Stock Validation", test_stock_validation),
        ("Portfolio Optimization", test_portfolio_optimization),
        ("MPT Optimization", test_mpt_optimization)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for result in results.values() if result)
    total_count = len(results)
    
    print(f"\nOverall: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("🎉 All tests passed! Your Quantum Portfolio Management API is working perfectly!")
    elif passed_count > 0:
        print("⚠️  Some tests passed. Check the failures above.")
    else:
        print("❌ All tests failed. Make sure the server is running.")
        print("\nTo start the server, run: startup.bat")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()