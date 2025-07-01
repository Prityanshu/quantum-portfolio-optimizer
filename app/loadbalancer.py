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