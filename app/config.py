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