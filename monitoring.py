import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemMonitor:
    def __init__(self, servers: List[Dict]):
        self.servers = servers
        self.metrics = {
            "server_health": {},
            "response_times": {},
            "error_rates": {},
            "load_distribution": {}
        }
    
    async def check_server_health(self, server: Dict) -> Dict:
        """Check individual server health"""
        try:
            start_time = time.time()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{server['url']}/health", timeout=5) as response:
                    response_time = (time.time() - start_time) * 1000  # ms
                    
                    return {
                        "server_id": server["id"],
                        "healthy": response.status == 200,
                        "response_time_ms": response_time,
                        "status_code": response.status,
                        "timestamp": datetime.utcnow().isoformat()
                    }
        except Exception as e:
            return {
                "server_id": server["id"],
                "healthy": False,
                "response_time_ms": None,
                "status_code": None,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def get_server_loads(self) -> Dict:
        """Get current server loads"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/servers/status") as response:
                    if response.status == 200:
                        data = await response.json()
                        return {s["server_id"]: s["current_load"] for s in data["servers"]}
                    return {}
        except Exception as e:
            logger.error(f"Failed to get server loads: {e}")
            return {}
    
    async def monitor_system(self):
        """Main monitoring loop"""
        while True:
            try:
                # Check server health
                health_tasks = [self.check_server_health(server) for server in self.servers]
                health_results = await asyncio.gather(*health_tasks)
                
                # Get server loads
                loads = await self.get_server_loads()
                
                # Update metrics
                timestamp = datetime.utcnow()
                
                for result in health_results:
                    server_id = result["server_id"]
                    
                    # Health metrics
                    if server_id not in self.metrics["server_health"]:
                        self.metrics["server_health"][server_id] = []
                    
                    self.metrics["server_health"][server_id].append(result)
                    
                    # Keep only last 100 entries
                    self.metrics["server_health"][server_id] = \
                        self.metrics["server_health"][server_id][-100:]
                    
                    # Response time metrics
                    if result["response_time_ms"]:
                        if server_id not in self.metrics["response_times"]:
                            self.metrics["response_times"][server_id] = []
                        
                        self.metrics["response_times"][server_id].append({
                            "timestamp": timestamp.isoformat(),
                            "response_time_ms": result["response_time_ms"]
                        })
                        
                        self.metrics["response_times"][server_id] = \
                            self.metrics["response_times"][server_id][-100:]
                
                # Load distribution
                self.metrics["load_distribution"] = {
                    "timestamp": timestamp.isoformat(),
                    "loads": loads
                }
                
                # Log summary
                healthy_servers = sum(1 for r in health_results if r["healthy"])
                total_load = sum(loads.values())
                avg_response_time = sum(
                    r["response_time_ms"] for r in health_results 
                    if r["response_time_ms"]
                ) / len([r for r in health_results if r["response_time_ms"]])
                
                logger.info(f"System Status - Healthy: {healthy_servers}/{len(self.servers)}, "
                           f"Total Load: {total_load}, Avg Response: {avg_response_time:.2f}ms")
                
                # Save metrics to file
                with open("metrics.json", "w") as f:
                    json.dump(self.metrics, f, indent=2)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            await asyncio.sleep(30)  # Monitor every 30 seconds
    
    def generate_report(self) -> str:
        """Generate monitoring report"""
        report = []
        report.append("=" * 60)
        report.append("QUANTUM PORTFOLIO MANAGEMENT - SYSTEM REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.utcnow().isoformat()}")
        report.append("")
        
        # Server Health Summary
        report.append("SERVER HEALTH SUMMARY:")
        report.append("-" * 40)
        
        for server_id, health_data in self.metrics["server_health"].items():
            if health_data:
                recent_health = health_data[-10:]  # Last 10 checks
                uptime_pct = (sum(1 for h in recent_health if h["healthy"]) / len(recent_health)) * 100
                
                avg_response = None
                response_times = [h["response_time_ms"] for h in recent_health if h["response_time_ms"]]
                if response_times:
                    avg_response = sum(response_times) / len(response_times)
                
                report.append(f"{server_id}:")
                report.append(f"  Uptime: {uptime_pct:.1f}%")
                if avg_response:
                    report.append(f"  Avg Response: {avg_response:.2f}ms")
                report.append("")
        
        # Load Distribution
        if "loads" in self.metrics["load_distribution"]:
            report.append("CURRENT LOAD DISTRIBUTION:")
            report.append("-" * 40)
            loads = self.metrics["load_distribution"]["loads"]
            total_load = sum(loads.values())
            
            for server_id, load in loads.items():
                pct = (load / total_load * 100) if total_load > 0 else 0
                report.append(f"{server_id}: {load} requests ({pct:.1f}%)")
            
            report.append(f"Total Active Requests: {total_load}")
        
        return "\n".join(report)

async def main():
    """Main monitoring function"""
    servers = [
        {"id": "server_1", "url": "http://localhost:8001"},
        {"id": "server_2", "url": "http://localhost:8002"},
        {"id": "server_3", "url": "http://localhost:8003"},
        {"id": "server_4", "url": "http://localhost:8004"},
        {"id": "server_5", "url": "http://localhost:8005"},
    ]
    
    monitor = SystemMonitor(servers)
    
    # Start monitoring
    await monitor.monitor_system()

if __name__ == "__main__":
    asyncio.run(main())