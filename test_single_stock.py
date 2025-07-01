"""
Test Single Stock Validation
===========================
This script tests validation of a single stock to see which data source is used.
"""

import requests
import json
import time

def test_single_stock():
    """Test validation of a single stock"""
    
    # Test data
    test_data = {
        "symbols": ["AAPL"]
    }
    
    print("=" * 50)
    print("TESTING SINGLE STOCK VALIDATION")
    print("=" * 50)
    print(f"Testing stock: {test_data['symbols'][0]}")
    
    # Send request
    start_time = time.time()
    try:
        response = requests.post(
            "http://localhost:8000/validate-stocks",
            json=test_data,
            timeout=30
        )
        end_time = time.time()
        
        print(f"Response Status: {response.status_code}")
        print(f"Response Time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            
            print("\nVALIDATION RESULT:")
            print("-" * 40)
            
            for stock in data.get('stocks', []):
                print(f"Symbol: {stock['symbol']}")
                print(f"Name: {stock['name']}")
                print(f"Current Price: ${stock['current_price']:.2f}")
                print(f"Valid: {stock['valid']}")
                if stock.get('volume'):
                    print(f"Volume: {stock['volume']:,}")
                if stock.get('market_cap'):
                    print(f"Market Cap: ${stock['market_cap']:,}")
                if stock.get('pe_ratio'):
                    print(f"P/E Ratio: {stock['pe_ratio']:.2f}")
                if stock.get('error'):
                    print(f"Error: {stock['error']}")
                print()
                
        else:
            print(f"Error: {response.text}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_stock() 