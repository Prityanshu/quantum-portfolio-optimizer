import requests
import json
import time

def test_20_stocks_validation():
    """Test stock validation with 20 diverse stocks"""
    
    # 20 stocks from different sectors
    stocks_20 = [
        # Technology
        "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "META", "NFLX",
        # Financial
        "JPM", "BAC", "WFC", "GS",
        # Healthcare
        "JNJ", "PFE", "UNH", "ABBV", "TMO",
        # Energy
        "XOM", "CVX", "COP", "SLB"
    ]
    
    print("=" * 60)
    print("TESTING STOCK VALIDATION WITH 20 STOCKS")
    print("=" * 60)
    print(f"Testing {len(stocks_20)} stocks: {', '.join(stocks_20)}")
    print()
    
    # API endpoint
    url = "http://localhost:8000/validate-stocks"
    
    # Request payload
    payload = {
        "symbols": stocks_20
    }
    
    try:
        print("Sending request to API...")
        start_time = time.time()
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Time: {response_time:.2f} seconds")
        print()
        
        if response.status_code == 200:
            data = response.json()
            
            print("VALIDATION RESULTS:")
            print("-" * 40)
            
            if "stocks" in data:
                stocks_data = data["stocks"]
                valid_count = 0
                invalid_count = 0
                
                for stock in stocks_data:
                    symbol = stock.get("symbol", "Unknown")
                    valid = stock.get("valid", False)
                    price = stock.get("current_price", 0)
                    name = stock.get("name", "Unknown")
                    error = stock.get("error", "")
                    
                    if valid:
                        valid_count += 1
                        print(f"✅ {symbol}: ${price:.2f} - {name}")
                    else:
                        invalid_count += 1
                        print(f"❌ {symbol}: {error}")
                
                print()
                print("SUMMARY:")
                print(f"✅ Valid stocks: {valid_count}")
                print(f"❌ Invalid stocks: {invalid_count}")
                print(f"📊 Success rate: {(valid_count/len(stocks_20)*100):.1f}%")
                
            else:
                print("Unexpected response format:")
                print(json.dumps(data, indent=2))
                
        else:
            print(f"Error: {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Response:", response.text)
    except Exception as e:
        print(f"Unexpected error: {e}")

def test_portfolio_optimization_with_20_stocks():
    """Test portfolio optimization with the 20 stocks"""
    
    print("\n" + "=" * 60)
    print("TESTING PORTFOLIO OPTIMIZATION WITH 20 STOCKS")
    print("=" * 60)
    
    # Use a subset of stocks for optimization (to avoid too many parameters)
    optimization_stocks = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "NVDA", "JPM", "JNJ", "XOM"]
    
    print(f"Testing optimization with {len(optimization_stocks)} stocks: {', '.join(optimization_stocks)}")
    print()
    
    url = "http://localhost:8000/optimize-portfolio"
    
    payload = {
        "investment_amount": 50000,
        "stock_symbols": optimization_stocks,
        "optimization_method": "qaoa",
        "risk_tolerance": "moderate",
        "quantum_backend": "aer_simulator",
        "qaoa_layers": 3,
        "max_iterations": 100,
        "include_visualization": True
    }
    
    try:
        print("Sending optimization request...")
        start_time = time.time()
        
        response = requests.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120  # Longer timeout for optimization
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Time: {response_time:.2f} seconds")
        print()
        
        if response.status_code == 200:
            data = response.json()
            
            print("OPTIMIZATION RESULTS:")
            print("-" * 40)
            print(f"Sharpe Ratio: {data.get('sharpe_ratio', 'N/A'):.3f}")
            print(f"Expected Return: {data.get('expected_return', 'N/A'):.2f}%")
            print(f"Risk (Std Dev): {data.get('risk_std_dev', 'N/A'):.2f}%")
            print(f"Optimization Method: {data.get('optimization_method', 'N/A')}")
            print(f"Risk Tolerance: {data.get('risk_tolerance', 'N/A')}")
            print()
            
            print("PORTFOLIO ALLOCATIONS:")
            print("-" * 40)
            allocations = data.get('allocations', {})
            for symbol, allocation in allocations.items():
                print(f"{symbol}: {allocation:.2f}%")
            
            print()
            print("INVESTMENT AMOUNTS:")
            print("-" * 40)
            investment_amounts = data.get('investment_allocations', {})
            for symbol, amount in investment_amounts.items():
                print(f"{symbol}: ${amount:,.2f}")
            
            if data.get('quantum_metrics'):
                print()
                print("QUANTUM METRICS:")
                print("-" * 40)
                qm = data['quantum_metrics']
                print(f"Backend Used: {qm.get('backend_used', 'N/A')}")
                print(f"QAOA Layers: {qm.get('qaoa_layers', 'N/A')}")
                print(f"Execution Time: {qm.get('execution_time', 'N/A'):.2f}s")
                print(f"Convergence: {qm.get('convergence_achieved', 'N/A')}")
                
        else:
            print(f"Error: {response.status_code}")
            print("Response:", response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("Response:", response.text)
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    # Test stock validation
    test_20_stocks_validation()
    
    # Test portfolio optimization
    test_portfolio_optimization_with_20_stocks()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETE!")
    print("=" * 60) 