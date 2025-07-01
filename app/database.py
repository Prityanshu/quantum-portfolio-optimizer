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