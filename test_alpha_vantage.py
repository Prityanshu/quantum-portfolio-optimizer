"""
Test Alpha Vantage API Key
==========================
This script tests if your Alpha Vantage API key is working correctly.
"""

import os
import requests
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded")
except ImportError:
    print("⚠️  python-dotenv not installed")
except Exception as e:
    print(f"⚠️  Error loading .env: {e}")

def test_alpha_vantage_api():
    """Test Alpha Vantage API with your key"""
    
    # Get API key from environment
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    if not api_key:
        print("❌ ALPHA_VANTAGE_API_KEY not found in environment variables")
        print("Please create a .env file with: ALPHA_VANTAGE_API_KEY=YT1O6NSSF1OHA9OA")
        return False
    
    print(f"✅ Found API key: {api_key[:8]}...{api_key[-4:]}")
    
    # Test with a simple API call
    url = "https://www.alphavantage.co/query"
    params = {
        'function': 'GLOBAL_QUOTE',
        'symbol': 'AAPL',
        'apikey': api_key
    }
    
    try:
        print("🔄 Testing Alpha Vantage API...")
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if 'Global Quote' in data and data['Global Quote']:
                quote = data['Global Quote']
                print("✅ API call successful!")
                print(f"   Symbol: {quote.get('01. symbol', 'N/A')}")
                print(f"   Price: ${quote.get('05. price', 'N/A')}")
                print(f"   Change: {quote.get('09. change', 'N/A')}")
                print(f"   Volume: {quote.get('06. volume', 'N/A')}")
                return True
            else:
                print("❌ No quote data returned")
                print(f"Response: {data}")
                return False
        else:
            print(f"❌ API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False

def test_environment_setup():
    """Test environment variable setup"""
    print("=" * 50)
    print("ENVIRONMENT VARIABLE TEST")
    print("=" * 50)
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
        with open('.env', 'r') as f:
            content = f.read()
            if 'ALPHA_VANTAGE_API_KEY' in content:
                print("✅ ALPHA_VANTAGE_API_KEY found in .env file")
            else:
                print("❌ ALPHA_VANTAGE_API_KEY not found in .env file")
    else:
        print("❌ .env file not found")
        print("Creating .env file with your API key...")
        
        # Create .env file
        env_content = """# Stock Data API Keys
ALPHA_VANTAGE_API_KEY=YT1O6NSSF1OHA9OA

# Other API keys (optional)
IEX_CLOUD_API_KEY=your_iex_cloud_key_here
POLYGON_API_KEY=your_polygon_key_here
FINNHUB_API_KEY=your_finnhub_key_here
"""
        
        try:
            with open('.env', 'w') as f:
                f.write(env_content)
            print("✅ Created .env file")
            
            # Reload environment variables
            from dotenv import load_dotenv
            load_dotenv()
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
    
    # Check environment variable
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if api_key:
        print(f"✅ Environment variable loaded: {api_key[:8]}...{api_key[-4:]}")
    else:
        print("❌ Environment variable not loaded")

if __name__ == "__main__":
    print("Alpha Vantage API Key Test")
    print("=" * 50)
    
    # Test environment setup
    test_environment_setup()
    
    print("\n" + "=" * 50)
    print("API FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Test API functionality
    if test_alpha_vantage_api():
        print("\n🎉 SUCCESS! Your Alpha Vantage API key is working correctly!")
        print("You can now use it in your quantum portfolio optimization app.")
    else:
        print("\n❌ FAILED! Please check your API key and try again.") 