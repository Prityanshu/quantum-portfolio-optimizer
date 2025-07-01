"""
Alternative Stock Data Sources for Quantum Portfolio Optimization
===============================================================

This module provides multiple reliable alternatives to Yahoo Finance for fetching
real-time and historical stock data. Each source has different pricing, rate limits,
and features.

CHOOSING THE RIGHT DATA SOURCE:
1. FREE TIERS: Alpha Vantage, IEX Cloud, Polygon.io
2. PAID TIERS: Bloomberg, Refinitiv, FactSet
3. REAL-TIME: Polygon.io, IEX Cloud, Finnhub
4. HISTORICAL: Alpha Vantage, Quandl, Yahoo Finance (with better handling)
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class DataSourceConfig:
    """Configuration for different data sources"""
    name: str
    api_key: str
    base_url: str
    rate_limit: int  # requests per minute
    free_tier_limit: int
    pricing_tier: str  # free, basic, premium, enterprise
    features: List[str]  # real-time, historical, fundamentals, etc.

class StockDataProvider(ABC):
    """Abstract base class for stock data providers"""
    
    @abstractmethod
    async def get_stock_price(self, symbol: str) -> float:
        """Get current stock price"""
        pass
    
    @abstractmethod
    async def get_historical_data(self, symbol: str, period: str) -> pd.DataFrame:
        """Get historical price data"""
        pass
    
    @abstractmethod
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and is tradeable"""
        pass

# ============================================================================
# 1. ALPHA VANTAGE (Recommended for Free Tier)
# ============================================================================

class AlphaVantageProvider(StockDataProvider):
    """
    Alpha Vantage - Excellent free tier with 500 requests/day
    - Real-time and historical data
    - Technical indicators
    - Fundamental data
    - Free tier: 500 requests/day
    - Paid: $49.99/month for 1200 requests/minute
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_stock_price(self, symbol: str) -> float:
        """Get real-time stock price"""
        session = await self._get_session()
        
        params = {
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
            'apikey': self.api_key
        }
        
        try:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Global Quote' in data and data['Global Quote']:
                        return float(data['Global Quote']['05. price'])
                return 0.0
        except Exception as e:
            logger.error(f"Alpha Vantage error for {symbol}: {e}")
            return 0.0
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical daily data"""
        session = await self._get_session()
        
        # Map period to Alpha Vantage function
        function_map = {
            "1d": "TIME_SERIES_INTRADAY",
            "1w": "TIME_SERIES_DAILY",
            "1m": "TIME_SERIES_DAILY",
            "3m": "TIME_SERIES_DAILY",
            "6m": "TIME_SERIES_DAILY",
            "1y": "TIME_SERIES_DAILY",
            "2y": "TIME_SERIES_DAILY",
            "5y": "TIME_SERIES_DAILY"
        }
        
        function = function_map.get(period, "TIME_SERIES_DAILY")
        
        params = {
            'function': function,
            'symbol': symbol,
            'apikey': self.api_key,
            'outputsize': 'full' if period in ['2y', '5y'] else 'compact'
        }
        
        try:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    if 'Time Series (Daily)' in data:
                        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
                        df.index = pd.to_datetime(df.index)
                        df = df.astype(float)
                        df.columns = ['open', 'high', 'low', 'close', 'volume']
                        
                        # Filter by period
                        if period != "1d":
                            end_date = datetime.now()
                            if period == "1w":
                                start_date = end_date - timedelta(days=7)
                            elif period == "1m":
                                start_date = end_date - timedelta(days=30)
                            elif period == "3m":
                                start_date = end_date - timedelta(days=90)
                            elif period == "6m":
                                start_date = end_date - timedelta(days=180)
                            elif period == "1y":
                                start_date = end_date - timedelta(days=365)
                            elif period == "2y":
                                start_date = end_date - timedelta(days=730)
                            elif period == "5y":
                                start_date = end_date - timedelta(days=1825)
                            
                            df = df[df.index >= start_date]
                        
                        return df.sort_index()
                    
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Alpha Vantage historical error for {symbol}: {e}")
            return pd.DataFrame()
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists"""
        price = await self.get_stock_price(symbol)
        return price > 0

# ============================================================================
# 2. IEX CLOUD (Excellent for Real-time Data)
# ============================================================================

class IEXCloudProvider(StockDataProvider):
    """
    IEX Cloud - Great for real-time data and fundamentals
    - Real-time quotes
    - Historical data
    - Fundamental data
    - Free tier: 500,000 messages/month
    - Paid: $9/month for 1M messages
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://cloud.iexapis.com/stable"
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_stock_price(self, symbol: str) -> float:
        """Get real-time stock price"""
        session = await self._get_session()
        
        url = f"{self.base_url}/stock/{symbol}/quote"
        params = {'token': self.api_key}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('latestPrice', 0))
                return 0.0
        except Exception as e:
            logger.error(f"IEX Cloud error for {symbol}: {e}")
            return 0.0
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data"""
        session = await self._get_session()
        
        # Map period to IEX range
        range_map = {
            "1d": "1d",
            "1w": "5d",
            "1m": "1m",
            "3m": "3m",
            "6m": "6m",
            "1y": "1y",
            "2y": "2y",
            "5y": "5y"
        }
        
        range_param = range_map.get(period, "1y")
        url = f"{self.base_url}/stock/{symbol}/chart/{range_param}"
        params = {'token': self.api_key}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    df = pd.DataFrame(data)
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    return df
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"IEX Cloud historical error for {symbol}: {e}")
            return pd.DataFrame()
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists"""
        price = await self.get_stock_price(symbol)
        return price > 0

# ============================================================================
# 3. POLYGON.IO (Best for Real-time and High Frequency)
# ============================================================================

class PolygonProvider(StockDataProvider):
    """
    Polygon.io - Excellent for real-time and high-frequency data
    - Real-time trades and quotes
    - Historical data
    - Options data
    - Free tier: 5 API calls/minute
    - Paid: $29/month for 5 calls/second
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_stock_price(self, symbol: str) -> float:
        """Get real-time stock price"""
        session = await self._get_session()
        
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{symbol}"
        params = {'apikey': self.api_key}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data and data['results']:
                        return float(data['results']['last']['p'])
                return 0.0
        except Exception as e:
            logger.error(f"Polygon error for {symbol}: {e}")
            return 0.0
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data"""
        session = await self._get_session()
        
        # Calculate date range
        end_date = datetime.now()
        if period == "1d":
            start_date = end_date - timedelta(days=1)
        elif period == "1w":
            start_date = end_date - timedelta(days=7)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "2y":
            start_date = end_date - timedelta(days=730)
        elif period == "5y":
            start_date = end_date - timedelta(days=1825)
        else:
            start_date = end_date - timedelta(days=365)
        
        url = f"{self.base_url}/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
        params = {'apikey': self.api_key, 'adjusted': 'true'}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'results' in data:
                        df = pd.DataFrame(data['results'])
                        df['t'] = pd.to_datetime(df['t'], unit='ms')
                        df.set_index('t', inplace=True)
                        df.columns = ['open', 'high', 'low', 'close', 'volume', 'vwap', 'transactions']
                        return df
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Polygon historical error for {symbol}: {e}")
            return pd.DataFrame()
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists"""
        price = await self.get_stock_price(symbol)
        return price > 0

# ============================================================================
# 4. FINNHUB (Good for Real-time and News)
# ============================================================================

class FinnhubProvider(StockDataProvider):
    """
    Finnhub - Good for real-time data and news sentiment
    - Real-time quotes
    - News sentiment
    - Fundamental data
    - Free tier: 60 API calls/minute
    - Paid: $7.99/month for 1000 calls/minute
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_stock_price(self, symbol: str) -> float:
        """Get real-time stock price"""
        session = await self._get_session()
        
        url = f"{self.base_url}/quote"
        params = {'symbol': symbol, 'token': self.api_key}
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    return float(data.get('c', 0))  # Current price
                return 0.0
        except Exception as e:
            logger.error(f"Finnhub error for {symbol}: {e}")
            return 0.0
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data"""
        session = await self._get_session()
        
        # Calculate date range
        end_timestamp = int(datetime.now().timestamp())
        if period == "1d":
            start_timestamp = int((datetime.now() - timedelta(days=1)).timestamp())
        elif period == "1w":
            start_timestamp = int((datetime.now() - timedelta(days=7)).timestamp())
        elif period == "1m":
            start_timestamp = int((datetime.now() - timedelta(days=30)).timestamp())
        elif period == "3m":
            start_timestamp = int((datetime.now() - timedelta(days=90)).timestamp())
        elif period == "6m":
            start_timestamp = int((datetime.now() - timedelta(days=180)).timestamp())
        elif period == "1y":
            start_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
        elif period == "2y":
            start_timestamp = int((datetime.now() - timedelta(days=730)).timestamp())
        elif period == "5y":
            start_timestamp = int((datetime.now() - timedelta(days=1825)).timestamp())
        else:
            start_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
        
        url = f"{self.base_url}/stock/candle"
        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': start_timestamp,
            'to': end_timestamp,
            'token': self.api_key
        }
        
        try:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    if data['s'] == 'ok':
                        df = pd.DataFrame({
                            'timestamp': data['t'],
                            'open': data['o'],
                            'high': data['h'],
                            'low': data['l'],
                            'close': data['c'],
                            'volume': data['v']
                        })
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                        df.set_index('timestamp', inplace=True)
                        return df
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Finnhub historical error for {symbol}: {e}")
            return pd.DataFrame()
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists"""
        price = await self.get_stock_price(symbol)
        return price > 0

# ============================================================================
# 5. ENHANCED YAHOO FINANCE (With Better Error Handling)
# ============================================================================

class EnhancedYahooFinanceProvider(StockDataProvider):
    """
    Enhanced Yahoo Finance with better error handling and retry logic
    - Free but with rate limiting
    - Good historical data
    - Multiple retry attempts
    - Fallback mechanisms
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.session = None
    
    async def _get_session(self):
        if self.session is None:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def get_stock_price(self, symbol: str) -> float:
        """Get current stock price with retry logic"""
        session = await self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                # Use Yahoo Finance API endpoint
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                params = {
                    'range': '1d',
                    'interval': '1m',
                    'includePrePost': 'false'
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                            result = data['chart']['result'][0]
                            if 'meta' in result and 'regularMarketPrice' in result['meta']:
                                return float(result['meta']['regularMarketPrice'])
                    
                    if response.status == 429:  # Rate limited
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                        
            except Exception as e:
                logger.warning(f"Yahoo Finance attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
            
        return 0.0
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data with retry logic"""
        session = await self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
                params = {
                    'range': period,
                    'interval': '1d',
                    'includePrePost': 'false'
                }
                
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                            result = data['chart']['result'][0]
                            if 'timestamp' in result and 'indicators' in result:
                                timestamps = result['timestamp']
                                quotes = result['indicators']['quote'][0]
                                
                                df = pd.DataFrame({
                                    'timestamp': timestamps,
                                    'open': quotes.get('open', []),
                                    'high': quotes.get('high', []),
                                    'low': quotes.get('low', []),
                                    'close': quotes.get('close', []),
                                    'volume': quotes.get('volume', [])
                                })
                                
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                                df.set_index('timestamp', inplace=True)
                                df = df.dropna()
                                return df
                    
                    if response.status == 429:  # Rate limited
                        await asyncio.sleep(self.retry_delay * (attempt + 1))
                        continue
                        
            except Exception as e:
                logger.warning(f"Yahoo Finance historical attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
                    continue
            
        return pd.DataFrame()
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol exists"""
        price = await self.get_stock_price(symbol)
        return price > 0

# ============================================================================
# 6. DATA PROVIDER MANAGER (Main Interface)
# ============================================================================

class DataProviderManager:
    """
    Manages multiple data providers with fallback logic
    """
    
    def __init__(self):
        self.providers = []
        self.current_provider_index = 0
        
    def add_provider(self, provider: StockDataProvider, priority: int = 0):
        """Add a data provider with priority"""
        self.providers.append((priority, provider))
        self.providers.sort(key=lambda x: x[0], reverse=True)  # Sort by priority
    
    async def get_stock_price(self, symbol: str) -> float:
        """Get stock price from available providers"""
        for priority, provider in self.providers:
            try:
                price = await provider.get_stock_price(symbol)
                if price > 0:
                    logger.info(f"Got price for {symbol} from {provider.__class__.__name__}: ${price}")
                    return price
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed for {symbol}: {e}")
                continue
        
        logger.error(f"All providers failed for {symbol}")
        return 0.0
    
    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical data from available providers"""
        for priority, provider in self.providers:
            try:
                df = await provider.get_historical_data(symbol, period)
                if not df.empty:
                    logger.info(f"Got historical data for {symbol} from {provider.__class__.__name__}: {len(df)} rows")
                    return df
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed for {symbol}: {e}")
                continue
        
        logger.error(f"All providers failed for {symbol}")
        return pd.DataFrame()
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate symbol from available providers"""
        for priority, provider in self.providers:
            try:
                if await provider.validate_symbol(symbol):
                    logger.info(f"Validated {symbol} with {provider.__class__.__name__}")
                    return True
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} validation failed for {symbol}: {e}")
                continue
        
        return False

# ============================================================================
# 7. SETUP AND CONFIGURATION
# ============================================================================

def setup_data_providers() -> DataProviderManager:
    """
    Setup data providers with API keys from environment variables
    """
    manager = DataProviderManager()
    
    # 1. Alpha Vantage (Free tier - 500 requests/day)
    alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    if alpha_vantage_key:
        manager.add_provider(AlphaVantageProvider(alpha_vantage_key), priority=5)
        logger.info("Added Alpha Vantage provider")
    
    # 2. IEX Cloud (Free tier - 500K messages/month)
    iex_key = os.getenv('IEX_CLOUD_API_KEY')
    if iex_key:
        manager.add_provider(IEXCloudProvider(iex_key), priority=4)
        logger.info("Added IEX Cloud provider")
    
    # 3. Polygon.io (Free tier - 5 calls/minute)
    polygon_key = os.getenv('POLYGON_API_KEY')
    if polygon_key:
        manager.add_provider(PolygonProvider(polygon_key), priority=3)
        logger.info("Added Polygon.io provider")
    
    # 4. Finnhub (Free tier - 60 calls/minute)
    finnhub_key = os.getenv('FINNHUB_API_KEY')
    if finnhub_key:
        manager.add_provider(FinnhubProvider(finnhub_key), priority=2)
        logger.info("Added Finnhub provider")
    
    # 5. Enhanced Yahoo Finance (Fallback)
    manager.add_provider(EnhancedYahooFinanceProvider(), priority=1)
    logger.info("Added Enhanced Yahoo Finance provider")
    
    return manager

# ============================================================================
# 8. USAGE EXAMPLE
# ============================================================================

async def example_usage():
    """Example of how to use the data provider manager"""
    
    # Setup providers
    manager = setup_data_providers()
    
    # Test symbols
    symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
    
    print("Testing Data Providers:")
    print("=" * 50)
    
    for symbol in symbols:
        print(f"\nTesting {symbol}:")
        
        # Get current price
        price = await manager.get_stock_price(symbol)
        print(f"  Current Price: ${price:.2f}")
        
        # Get historical data
        hist_data = await manager.get_historical_data(symbol, "1m")
        if not hist_data.empty:
            print(f"  Historical Data: {len(hist_data)} days")
            print(f"  Latest Close: ${hist_data['close'].iloc[-1]:.2f}")
        
        # Validate symbol
        is_valid = await manager.validate_symbol(symbol)
        print(f"  Valid: {is_valid}")

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage()) 