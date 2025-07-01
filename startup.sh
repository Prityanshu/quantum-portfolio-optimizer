@echo off
echo ===================================================
echo QUANTUM PORTFOLIO MANAGEMENT - WINDOWS STARTUP
echo ===================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and add it to your PATH
    pause
    exit /b 1
)

REM Check if Redis is running (optional - will use in-memory fallback)
echo Checking Redis connection...
python -c "import redis; r=redis.Redis(); r.ping()" >nul 2>&1
if errorlevel 1 (
    echo WARNING: Redis not available - using in-memory caching
) else (
    echo Redis connection successful
)

REM Install required packages
echo Installing required packages...
pip install fastapi uvicorn yfinance numpy pandas qiskit qiskit-optimization scikit-learn redis celery requests

REM Create necessary directories
if not exist "logs" mkdir logs
if not exist "cache" mkdir cache

REM Start the main server
echo ===================================================
echo Starting Quantum Portfolio Management Server...
echo Server will be available at: http://localhost:8000
echo API Documentation: http://localhost:8000/docs
echo ===================================================
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload