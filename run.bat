@echo off
setlocal enabledelayedexpansion

echo ========================================
echo AI Agent - Web Interface Launcher
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [OK] Python is installed
python --version

REM Check if .env file exists
if not exist ".env" (
    echo.
    echo [WARNING] .env file not found!
    echo Please create a .env file with the following variables:
    echo   - OPENAI_API_KEY=your-openai-api-key-here
    echo   - DATABASE_URL=sqlite:///./users.db
    echo.
    echo The application may fail to start without these variables.
    echo.
    pause
)

REM Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo.
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo.
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] Failed to upgrade pip, continuing anyway...
)

REM Install/update requirements
echo.
echo [INFO] Installing/updating dependencies...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt file not found!
    pause
    exit /b 1
)

pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    echo Please check requirements.txt and try again
    pause
    exit /b 1
)
echo [OK] Dependencies installed



REM Run the application
echo.
echo ========================================
echo Starting web interface...
echo ========================================
echo.
echo [INFO] Starting FastAPI server in a new window...
start "FastAPI Server" cmd /k "call venv\Scripts\activate.bat && python .\rag_pinecone_fastapi.py"
timeout /t 3 /nobreak >nul
echo [INFO] Starting Streamlit UI...
echo.
streamlit run streamlit_ui.py

REM If we get here, the application has exited
echo.
echo [INFO] Application has stopped
pause