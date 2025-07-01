#!/bin/bash

echo "Starting Quantum Portfolio Frontend..."
echo

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7+ and try again"
    exit 1
fi

# Navigate to frontend directory
cd frontend

# Start the frontend server with correct host binding
echo "Starting frontend server on http://127.0.0.1:3000"
echo "Press Ctrl+C to stop the server"
echo
python3 -m http.server 3000 --bind 127.0.0.1 