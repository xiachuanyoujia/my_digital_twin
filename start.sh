#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo "  My Digital Twin - Quick Start"
echo "========================================"
echo ""

echo "[1/5] Checking Python..."
python --version

echo ""
echo "[2/5] Installing backend dependencies..."
cd "$ROOT/backend"
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "[3/5] Downloading assets (Swagger + Model)..."
python setup.py

echo ""
echo "[4/5] Starting backend on port 8000..."
python -m app.main &
BACKEND_PID=$!

echo ""
echo "[5/5] Starting frontend on port 5173..."
cd "$ROOT/frontend"
if [ -d "node_modules" ]; then
    echo "  node_modules found, skipping npm install."
else
    npm install
fi
npx vite --host &
FRONTEND_PID=$!

echo ""
echo "========================================"
echo "  All services are starting!"
echo ""
echo "  Backend API:  http://localhost:8000/docs"
echo "  Frontend:     http://localhost:5173"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop all services"

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
