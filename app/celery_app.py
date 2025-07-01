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