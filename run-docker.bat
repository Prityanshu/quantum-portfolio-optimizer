@echo off
echo ===================================================
echo QUANTUM PORTFOLIO MANAGEMENT - DOCKER STARTUP
echo ===================================================

echo Checking Docker...
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running
    echo Please install Docker Desktop and make sure it's running
    pause
    exit /b 1
)

echo Docker found! Building and starting the application...
echo This may take a few minutes for the first build...

REM Build and start the services
docker-compose up --build -d

echo.
echo ===================================================
echo Quantum Portfolio Management API is starting...
echo ===================================================
echo Server URL: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo Health Check: http://localhost:8000/health
echo ===================================================
echo.

echo Waiting for service to be ready...
timeout /t 10 /nobreak >nul

echo Testing connection...
curl -f http://localhost:8000/health 2>nul
if errorlevel 1 (
    echo Service is still starting up. Please wait a moment and try:
    echo curl http://localhost:8000/health
) else (
    echo ✅ Service is ready!
)

echo.
echo To stop the service, run: docker-compose down
echo To view logs, run: docker-compose logs -f
echo.
pause