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