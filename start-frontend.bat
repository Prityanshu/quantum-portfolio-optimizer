@echo off
echo Starting Quantum Portfolio Frontend...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.7+ and try again
    pause
    exit /b 1
)

REM Navigate to frontend directory
cd frontend

REM Start the frontend server with correct host binding
echo Starting frontend server on http://127.0.0.1:3000
echo Press Ctrl+C to stop the server
echo.
python -m http.server 3000 --bind 127.0.0.1

pause 