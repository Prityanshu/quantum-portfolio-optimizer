"""
Data Source Configuration for Quantum Portfolio Optimization
===========================================================

This file contains configuration settings for different stock data providers.
Set your API keys as environment variables or in a .env file.

ENVIRONMENT VARIABLES TO SET:
- ALPHA_VANTAGE_API_KEY
- IEX_CLOUD_API_KEY  
- POLYGON_API_KEY
- FINNHUB_API_KEY

RECOMMENDED SETUP FOR PRODUCTION:
1. Alpha Vantage (Primary) - Free tier with 500 requests/day
2. IEX Cloud (Secondary) - Free tier with 500K messages/month
3. Polygon.io (Real-time) - Free tier with 5 calls/minute
4. Enhanced Yahoo Finance (Fallback) - Free but rate limited
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Environment variables loaded from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Error loading .env file: {e}")

@dataclass
class DataSourceInfo:
    """Information about a data source"""
    name: str
    website: str
    free_tier: str
    paid_tier: str
    features: List[str]
    setup_instructions: str
    api_key_url: str

# Data Source Information
DATA_SOURCES = {
    "alpha_vantage": DataSourceInfo(
        name="Alpha Vantage",
        website="https://www.alphavantage.co/",
        free_tier="500 requests/day",
        paid_tier="$49.99/month for 1200 requests/minute",
        features=[
            "Real-time quotes",
            "Historical data",
            "Technical indicators",
            "Fundamental data",
            "Forex & Crypto"
        ],
        setup_instructions="""
        1. Go to https://www.alphavantage.co/support/#api-key
        2. Sign up for a free account
        3. Get your API key
        4. Set environment variable: ALPHA_VANTAGE_API_KEY=your_key_here
        """,
        api_key_url="https://www.alphavantage.co/support/#api-key"
    ),
    
    "iex_cloud": DataSourceInfo(
        name="IEX Cloud",
        website="https://iexcloud.io/",
        free_tier="500,000 messages/month",
        paid_tier="$9/month for 1M messages",
        features=[
            "Real-time quotes",
            "Historical data",
            "Fundamental data",
            "News sentiment",
            "Options data"
        ],
        setup_instructions="""
        1. Go to https://iexcloud.io/cloud-login#/register
        2. Sign up for a free account
        3. Get your API key from the dashboard
        4. Set environment variable: IEX_CLOUD_API_KEY=your_key_here
        """,
        api_key_url="https://iexcloud.io/cloud-login#/register"
    ),
    
    "polygon": DataSourceInfo(
        name="Polygon.io",
        website="https://polygon.io/",
        free_tier="5 API calls/minute",
        paid_tier="$29/month for 5 calls/second",
        features=[
            "Real-time trades & quotes",
            "Historical data",
            "Options data",
            "Forex data",
            "Crypto data"
        ],
        setup_instructions="""
        1. Go to https://polygon.io/
        2. Sign up for a free account
        3. Get your API key from the dashboard
        4. Set environment variable: POLYGON_API_KEY=your_key_here
        """,
        api_key_url="https://polygon.io/"
    ),
    
    "finnhub": DataSourceInfo(
        name="Finnhub",
        website="https://finnhub.io/",
        free_tier="60 API calls/minute",
        paid_tier="$7.99/month for 1000 calls/minute",
        features=[
            "Real-time quotes",
            "Historical data",
            "News sentiment",
            "Fundamental data",
            "Insider transactions"
        ],
        setup_instructions="""
        1. Go to https://finnhub.io/
        2. Sign up for a free account
        3. Get your API key from the dashboard
        4. Set environment variable: FINNHUB_API_KEY=your_key_here
        """,
        api_key_url="https://finnhub.io/"
    )
}

def print_setup_instructions():
    """Print setup instructions for all data sources"""
    print("=" * 80)
    print("STOCK DATA SOURCE SETUP INSTRUCTIONS")
    print("=" * 80)
    print()
    
    for key, source in DATA_SOURCES.items():
        print(f"📊 {source.name}")
        print("-" * 50)
        print(f"Website: {source.website}")
        print(f"Free Tier: {source.free_tier}")
        print(f"Paid Tier: {source.paid_tier}")
        print(f"Features: {', '.join(source.features)}")
        print(f"API Key URL: {source.api_key_url}")
        print()
        print("Setup Instructions:")
        print(source.setup_instructions)
        print()
        print("=" * 80)
        print()

def check_api_keys() -> Dict[str, bool]:
    """Check which API keys are configured"""
    keys = {
        "alpha_vantage": bool(os.getenv('ALPHA_VANTAGE_API_KEY')),
        "iex_cloud": bool(os.getenv('IEX_CLOUD_API_KEY')),
        "polygon": bool(os.getenv('POLYGON_API_KEY')),
        "finnhub": bool(os.getenv('FINNHUB_API_KEY'))
    }
    
    print("API Key Status:")
    print("-" * 30)
    for source, has_key in keys.items():
        status = "✅ Configured" if has_key else "❌ Not configured"
        print(f"{DATA_SOURCES[source].name}: {status}")
    
    return keys

def get_recommended_setup() -> List[str]:
    """Get recommended setup order based on free tier limits"""
    return [
        "alpha_vantage",  # 500 requests/day - good for testing
        "iex_cloud",      # 500K messages/month - excellent for production
        "polygon",        # 5 calls/minute - good for real-time
        "finnhub"         # 60 calls/minute - good backup
    ]

def create_env_template():
    """Create a template .env file"""
    template = """# Stock Data API Keys
# Get your free API keys from the following websites:

# Alpha Vantage (500 requests/day free)
# https://www.alphavantage.co/support/#api-key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here

# IEX Cloud (500K messages/month free)
# https://iexcloud.io/cloud-login#/register
IEX_CLOUD_API_KEY=your_iex_cloud_key_here

# Polygon.io (5 calls/minute free)
# https://polygon.io/
POLYGON_API_KEY=your_polygon_key_here

# Finnhub (60 calls/minute free)
# https://finnhub.io/
FINNHUB_API_KEY=your_finnhub_key_here

# Optional: Load this file in your application
# from dotenv import load_dotenv
# load_dotenv()
"""
    
    with open('.env.template', 'w') as f:
        f.write(template)
    
    print("Created .env.template file")
    print("Copy this file to .env and add your API keys")

if __name__ == "__main__":
    print_setup_instructions()
    check_api_keys()
    create_env_template()
    
    print("\n🎯 RECOMMENDED SETUP ORDER:")
    print("-" * 30)
    for i, source in enumerate(get_recommended_setup(), 1):
        print(f"{i}. {DATA_SOURCES[source].name}")
    
    print("\n💡 TIP: Start with Alpha Vantage for testing, then add IEX Cloud for production use.") 