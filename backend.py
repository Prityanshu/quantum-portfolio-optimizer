# requirements.txt
"""
fastapi==0.104.1
uvicorn==0.24.0
redis==5.0.1
celery==5.3.4
yfinance==0.2.18
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
pydantic==2.4.2
python-multipart==0.0.6
aioredis==2.0.1
httpx==0.25.0
python-dotenv==1.0.0
"""

# config.py
import os
from typing import List, Dict
from pydantic import BaseSettings

class Settings(BaseSettings):
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # Celery Configuration
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # Server Pool Configuration
    SERVER_POOL: List[Dict] = [
        {"id": "server_1", "url": "http://localhost:8001", "api_key": "quantum_key_1", "active": True},
        {"id": "server_2", "url": "http://localhost:8002", "api_key": "quantum_key_2", "active": True},
        {"id": "server_3", "url": "http://localhost:8003", "api_key": "quantum_key_3", "active": True},
        {"id": "server_4", "url": "http://localhost:8004", "api_key": "quantum_key_4", "active": True},
        {"id": "server_5", "url": "http://localhost:8005", "api_key": "quantum_key_5", "active": True},
    ]
    
    # Rate Limiting
    MAX_REQUESTS_PER_MINUTE: int = 20
    
    # Cache Settings
    CACHE_TTL_SECONDS: int = 300  # 5 minutes
    
    class Config:
        env_file = ".env"

settings = Settings()

# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

class StockInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol (e.g., AAPL)")
    
class PortfolioRequest(BaseModel):
    investment_amount: float = Field(..., ge=100, description="Investment amount in USD")
    stock_symbols: List[str] = Field(..., min_items=2, max_items=50, description="List of stock symbols")
    optimization_method: str = Field(default="quantum", description="quantum or mpt")
    risk_tolerance: Optional[str] = Field(default="moderate", description="conservative, moderate, aggressive")

class PortfolioResponse(BaseModel):
    request_id: str
    status: str
    sharpe_ratio: Optional[float] = None
    expected_return: Optional[float] = None
    risk_std_dev: Optional[float] = None
    allocations: Optional[Dict[str, float]] = None
    investment_amounts: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None
    server_id: Optional[str] = None
    created_at: datetime
    error_message: Optional[str] = None

class ServerStatus(BaseModel):
    server_id: str
    url: str
    active: bool
    current_load: int
    last_ping: Optional[datetime] = None
    response_time_ms: Optional[float] = None

# database.py
import redis
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import asyncio
import aioredis

class RedisManager:
    def __init__(self):
        self.redis_client = None
    
    async def connect(self):
        self.redis_client = await aioredis.from_url(
            f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}"
        )
    
    async def disconnect(self):
        if self.redis_client:
            await self.redis_client.close()
    
    async def set_server_load(self, server_id: str, load: int):
        await self.redis_client.hset("server_loads", server_id, load)
    
    async def get_server_load(self, server_id: str) -> int:
        load = await self.redis_client.hget("server_loads", server_id)
        return int(load) if load else 0
    
    async def get_all_server_loads(self) -> Dict[str, int]:
        loads = await self.redis_client.hgetall("server_loads")
        return {k.decode(): int(v) for k, v in loads.items()}
    
    async def cache_portfolio_result(self, request_id: str, result: Dict, ttl: int = 300):
        await self.redis_client.setex(f"portfolio:{request_id}", ttl, json.dumps(result))
    
    async def get_cached_portfolio_result(self, request_id: str) -> Optional[Dict]:
        result = await self.redis_client.get(f"portfolio:{request_id}")
        return json.loads(result) if result else None
    
    async def rate_limit_check(self, client_ip: str) -> bool:
        key = f"rate_limit:{client_ip}"
        current = await self.redis_client.get(key)
        
        if current is None:
            await self.redis_client.setex(key, 60, 1)
            return True
        
        if int(current) >= settings.MAX_REQUESTS_PER_MINUTE:
            return False
        
        await self.redis_client.incr(key)
        return True

redis_manager = RedisManager()

# load_balancer.py
import asyncio
import httpx
from typing import Optional, Dict
import time
import random

class LoadBalancer:
    def __init__(self):
        self.servers = {server["id"]: server for server in settings.SERVER_POOL}
        self.server_loads = {}
    
    async def get_least_loaded_server(self) -> Optional[Dict]:
        """Get the server with the least load"""
        active_servers = [s for s in self.servers.values() if s["active"]]
        
        if not active_servers:
            return None
        
        # Get current loads from Redis
        loads = await redis_manager.get_all_server_loads()
        
        # Find server with minimum load
        min_load = float('inf')
        selected_server = None
        
        for server in active_servers:
            server_id = server["id"]
            current_load = loads.get(server_id, 0)
            
            if current_load < min_load:
                min_load = current_load
                selected_server = server
        
        return selected_server
    
    async def increment_server_load(self, server_id: str):
        """Increment server load"""
        current_load = await redis_manager.get_server_load(server_id)
        await redis_manager.set_server_load(server_id, current_load + 1)
    
    async def decrement_server_load(self, server_id: str):
        """Decrement server load"""
        current_load = await redis_manager.get_server_load(server_id)
        new_load = max(0, current_load - 1)
        await redis_manager.set_server_load(server_id, new_load)
    
    async def health_check(self, server: Dict) -> bool:
        """Check if server is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{server['url']}/health")
                return response.status_code == 200
        except:
            return False
    
    async def periodic_health_check(self):
        """Periodically check server health"""
        while True:
            for server in self.servers.values():
                is_healthy = await self.health_check(server)
                server["active"] = is_healthy
            
            await asyncio.sleep(30)  # Check every 30 seconds

load_balancer = LoadBalancer()

# quantum_optimizer.py
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple
import asyncio
from datetime import datetime, timedelta

class QuantumPortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    async def get_stock_data(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """Fetch stock data from yfinance"""
        try:
            # Download data for all symbols
            data = yf.download(symbols, period=period, progress=False)
            
            # Handle single stock case
            if len(symbols) == 1:
                return data['Adj Close'].to_frame(symbols[0])
            
            # Return adjusted close prices
            return data['Adj Close'].dropna()
        
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns"""
        return prices.pct_change().dropna()
    
    def calculate_portfolio_metrics(self, weights: np.array, returns: pd.DataFrame) -> Tuple[float, float, float]:
        """Calculate portfolio return, risk, and Sharpe ratio"""
        # Annual return
        portfolio_return = np.sum(returns.mean() * weights) * 252
        
        # Annual volatility
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
        
        return portfolio_return, portfolio_std, sharpe_ratio
    
    def quantum_optimization_objective(self, weights: np.array, returns: pd.DataFrame, risk_tolerance: str = "moderate") -> float:
        """Quantum-inspired optimization objective function"""
        portfolio_return, portfolio_std, sharpe_ratio = self.calculate_portfolio_metrics(weights, returns)
        
        # Risk tolerance multipliers
        risk_multipliers = {
            "conservative": 0.5,
            "moderate": 1.0,
            "aggressive": 2.0
        }
        
        risk_mult = risk_multipliers.get(risk_tolerance, 1.0)
        
        # Quantum-inspired objective: maximize Sharpe ratio with entropy regularization
        entropy = -np.sum(weights * np.log(weights + 1e-10))  # Add small value to avoid log(0)
        
        # Objective: maximize Sharpe ratio + entropy bonus - risk penalty
        objective = -(sharpe_ratio + 0.1 * entropy - risk_mult * portfolio_std)
        
        return objective
    
    def mpt_optimization_objective(self, weights: np.array, returns: pd.DataFrame) -> float:
        """Traditional MPT optimization (maximize Sharpe ratio)"""
        _, _, sharpe_ratio = self.calculate_portfolio_metrics(weights, returns)
        return -sharpe_ratio  # Minimize negative Sharpe ratio
    
    async def optimize_portfolio(self, symbols: List[str], investment_amount: float, 
                               method: str = "quantum", risk_tolerance: str = "moderate") -> Dict:
        """Optimize portfolio allocation"""
        try:
            # Get stock data
            prices = await self.get_stock_data(symbols)
            returns = self.calculate_returns(prices)
            
            n_assets = len(symbols)
            
            # Initial guess: equal weights
            initial_weights = np.array([1/n_assets] * n_assets)
            
            # Constraints: weights sum to 1
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            
            # Bounds: weights between 0 and 1
            bounds = tuple((0, 1) for _ in range(n_assets))
            
            # Choose optimization method
            if method.lower() == "quantum":
                objective_func = lambda w: self.quantum_optimization_objective(w, returns, risk_tolerance)
            else:
                objective_func = lambda w: self.mpt_optimization_objective(w, returns)
            
            # Optimize
            result = minimize(
                objective_func,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if not result.success:
                raise Exception("Optimization failed")
            
            # Calculate final metrics
            optimal_weights = result.x
            portfolio_return, portfolio_std, sharpe_ratio = self.calculate_portfolio_metrics(optimal_weights, returns)
            
            # Create allocation dictionary
            allocations = {symbol: float(weight * 100) for symbol, weight in zip(symbols, optimal_weights)}
            
            # Calculate investment amounts
            investment_amounts = {symbol: float(weight * investment_amount) for symbol, weight in zip(symbols, optimal_weights)}
            
            return {
                "sharpe_ratio": float(sharpe_ratio),
                "expected_return": float(portfolio_return * 100),  # Convert to percentage
                "risk_std_dev": float(portfolio_std * 100),  # Convert to percentage
                "allocations": allocations,
                "investment_amounts": investment_amounts,
                "method": method,
                "symbols": symbols
            }
            
        except Exception as e:
            raise Exception(f"Portfolio optimization failed: {str(e)}")

# celery_app.py
from celery import Celery
import asyncio
from datetime import datetime

celery_app = Celery(
    "quantum_portfolio",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    result_expires=3600,  # Results expire after 1 hour
)

@celery_app.task(bind=True)
def optimize_portfolio_task(self, symbols: List[str], investment_amount: float, 
                          method: str = "quantum", risk_tolerance: str = "moderate"):
    """Celery task for portfolio optimization"""
    try:
        optimizer = QuantumPortfolioOptimizer()
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            optimizer.optimize_portfolio(symbols, investment_amount, method, risk_tolerance)
        )
        
        loop.close()
        
        return {
            "status": "completed",
            "result": result,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.utcnow().isoformat()
        }

# main.py (FastAPI Application)
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uuid
import time
from datetime import datetime
import asyncio

app = FastAPI(
    title="Quantum Portfolio Management API",
    description="Scalable quantum-inspired portfolio optimization service",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    await redis_manager.connect()
    # Start periodic health checks
    asyncio.create_task(load_balancer.periodic_health_check())

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up connections on shutdown"""
    await redis_manager.disconnect()

async def get_client_ip(request: Request) -> str:
    """Get client IP address"""
    return request.client.host

async def rate_limit_dependency(request: Request):
    """Rate limiting dependency"""
    client_ip = await get_client_ip(request)
    
    if not await redis_manager.rate_limit_check(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Quantum Portfolio Management API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@app.get("/servers/status")
async def get_server_status():
    """Get status of all servers"""
    loads = await redis_manager.get_all_server_loads()
    
    server_statuses = []
    for server in settings.SERVER_POOL:
        server_id = server["id"]
        status = ServerStatus(
            server_id=server_id,
            url=server["url"],
            active=server["active"],
            current_load=loads.get(server_id, 0)
        )
        server_statuses.append(status)
    
    return {"servers": server_statuses}

@app.post("/portfolio/optimize", response_model=PortfolioResponse)
async def optimize_portfolio(
    request: PortfolioRequest,
    background_tasks: BackgroundTasks,
    client_request: Request,
    _: None = Depends(rate_limit_dependency)
):
    """Optimize portfolio allocation"""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    try:
        # Select least loaded server
        selected_server = await load_balancer.get_least_loaded_server()
        
        if not selected_server:
            raise HTTPException(status_code=503, detail="No servers available")
        
        # Increment server load
        await load_balancer.increment_server_load(selected_server["id"])
        
        try:
            # Submit optimization task
            task = optimize_portfolio_task.delay(
                request.stock_symbols,
                request.investment_amount,
                request.optimization_method,
                request.risk_tolerance
            )
            
            # Wait for task completion (with timeout)
            result = task.get(timeout=60)  # 60 second timeout
            
            if result["status"] == "failed":
                raise Exception(result["error"])
            
            optimization_result = result["result"]
            processing_time = time.time() - start_time
            
            # Create response
            response = PortfolioResponse(
                request_id=request_id,
                status="completed",
                sharpe_ratio=optimization_result["sharpe_ratio"],
                expected_return=optimization_result["expected_return"],
                risk_std_dev=optimization_result["risk_std_dev"],
                allocations=optimization_result["allocations"],
                investment_amounts=optimization_result["investment_amounts"],
                processing_time=processing_time,
                server_id=selected_server["id"],
                created_at=datetime.utcnow()
            )
            
            # Cache result
            background_tasks.add_task(
                redis_manager.cache_portfolio_result,
                request_id,
                response.dict()
            )
            
            return response
            
        finally:
            # Decrement server load
            background_tasks.add_task(
                load_balancer.decrement_server_load,
                selected_server["id"]
            )
            
    except Exception as e:
        return PortfolioResponse(
            request_id=request_id,
            status="failed",
            created_at=datetime.utcnow(),
            error_message=str(e),
            processing_time=time.time() - start_time
        )

@app.get("/portfolio/{request_id}", response_model=PortfolioResponse)
async def get_portfolio_result(request_id: str):
    """Get cached portfolio optimization result"""
    result = await redis_manager.get_cached_portfolio_result(request_id)
    
    if not result:
        raise HTTPException(status_code=404, detail="Portfolio result not found")
    
    return PortfolioResponse(**result)

@app.post("/stocks/validate")
async def validate_stocks(stocks: List[StockInput]):
    """Validate stock symbols"""
    try:
        symbols = [stock.symbol.upper() for stock in stocks]
        
        # Try to fetch basic info for validation
        optimizer = QuantumPortfolioOptimizer()
        test_data = await optimizer.get_stock_data(symbols, period="5d")
        
        valid_symbols = list(test_data.columns)
        invalid_symbols = [s for s in symbols if s not in valid_symbols]
        
        return {
            "valid_symbols": valid_symbols,
            "invalid_symbols": invalid_symbols,
            "total_valid": len(valid_symbols),
            "total_invalid": len(invalid_symbols)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Stock validation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT)