@echo off
title My Digital Twin - Starting...

set "ROOT=%~dp0"

echo ========================================
echo   My Digital Twin - Quick Start
echo ========================================
echo.

echo [1/5] Checking Python...
python --version
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.11+
    pause
    exit /b 1
)
echo   Python OK.

echo.
echo [2/5] Checking backend dependencies...
rem pip will auto-skip packages already installed
pip install -r "%ROOT%backend\requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install backend dependencies.
    pause
    exit /b 1
)
echo   Backend dependencies OK.

echo.
echo [3/5] Downloading assets (Swagger + Model)...
python "%ROOT%backend\setup.py"
if %errorlevel% neq 0 (
    echo [ERROR] Asset download failed. Check network and retry.
    pause
    exit /b 1
)

echo.
echo [4/5] Starting backend on port 8000...
start "DT-Backend" /D "%ROOT%backend" cmd /k "echo Starting FastAPI... && python -m app.main"

echo [5/5] Checking frontend dependencies...
cd /d "%ROOT%frontend"
if exist "node_modules\" (
    echo   node_modules found, skipping npm install.
) else (
    echo   Installing frontend packages...
    call npm install
)
echo   Frontend dependencies OK.

echo   Starting frontend on port 5173...
start "DT-Frontend" /D "%ROOT%frontend" cmd /k "echo Starting Vite... && npx vite --host"

echo.
echo ========================================
echo   All services are starting!
echo.
echo   Backend API:  http://localhost:8000/docs
echo   Frontend:     http://localhost:5173
echo ========================================
echo.
echo You can close this window.
pause >nul
