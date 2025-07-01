import asyncio
import aiohttp
import time
import random
from concurrent.futures import ThreadPoolExecutor
import statistics

class PerformanceTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.results = []
    
    async def single_optimization_request(self, session, request_data):
        """Single optimization request"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/portfolio/optimize",
                json=request_data,
                timeout=120
            ) as response:
                response_time = time.time() - start_time
                data = await response.json()
                
                return {
                    "success": response.status == 200,
                    "response_time": response_time,
                    "status_code": response.status,
                    "server_id": data.get("server_id") if response.status == 200 else None,
                    "error": data.get("error_message") if response.status != 200 else None
                }
        except Exception as e:
            return {
                "success": False,
                "response_time": time.time() - start_time,
                "status_code": None,
                "server_id": None,
                "error": str(e)
            }
    
    async def load_test(self, concurrent_users=10, requests_per_user=5):
        """Perform load testing"""
        print(f"Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        # Sample stock combinations
        stock_combinations = [
            ["AAPL", "GOOGL", "MSFT", "TSLA"],
            ["AMZN", "NFLX", "NVDA", "AMD"],
            ["JPM", "BAC", "WFC", "GS"],
            ["JNJ", "PFE", "UNH", "ABBV"],
            ["XOM", "CVX", "COP", "SLB"]
        ]
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for user in range(concurrent_users):
                for request in range(requests_per_user):
                    request_data = {
                        "investment_amount": random.randint(1000, 100000),
                        "stock_symbols": random.choice(stock_combinations),
                        "optimization_method": random.choice(["quantum", "mpt"]),
                        "risk_tolerance": random.choice(["conservative", "moderate", "aggressive"])
                    }
                    
                    task = self.single_optimization_request(session, request_data)
                    tasks.append(task)
            
            # Execute all requests
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # Process results
            successful_results = [r for r in results if isinstance(r, dict) and r["success"]]
            failed_results = [r for r in results if isinstance(r, dict) and not r["success"]]
            
            print(f"\nLoad Test Results:")
            print(f"Total Time: {total_time:.2f}s")
            print(f"Total Requests: {len(results)}")
            print(f"Successful: {len(successful_results)}")
            print(f"Failed: {len(failed_results)}")
            print(f"Success Rate: {len(successful_results)/len(results)*100:.1f}%")
            
            if successful_results:
                response_times = [r["response_time"] for r in successful_results]
                print(f"Response Times:")
                print(f"  Min: {min(response_times):.2f}s")
                print(f"  Max: {max(response_times):.2f}s")
                print(f"  Mean: {statistics.mean(response_times):.2f}s")
                print(f"  Median: {statistics.median(response_times):.2f}s")
                
                # Server distribution
                server_counts = {}
                for result in successful_results:
                    server_id = result["server_id"]
                    if server_id:
                        server_counts[server_id] = server_counts.get(server_id, 0) + 1
                
                print(f"Server Distribution:")
                for server_id, count in server_counts.items():
                    print(f"  {server_id}: {count} requests")
            
            if failed_results:
                print(f"Common Errors:")
                error_counts = {}
                for result in failed_results:
                    error = result.get("error", "Unknown")
                    error_counts[error] = error_counts.get(error, 0) + 1
                
                for error, count in error_counts.items():
                    print(f"  {error}: {count} times")

async def main():
    """Main performance testing function"""
    tester = PerformanceTester()
    
    # Test different load scenarios
    scenarios = [
        (5, 2),   # Light load: 5 users, 2 requests each
        (10, 3),  # Medium load: 10 users, 3 requests each
        (15, 2),  # Heavy load: 15 users, 2 requests each
    ]
    
    for users, requests in scenarios:
        print(f"\n{'='*60}")
        print(f"SCENARIO: {users} concurrent users, {requests} requests each")
        print(f"{'='*60}")
        
        await tester.load_test(users, requests)
        
        # Wait between scenarios
        print("\nWaiting 10 seconds before next scenario...")
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())