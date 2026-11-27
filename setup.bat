@echo off
title Buildathon setup
cd /d "%~dp0"
echo Buildathon setup - one click install
echo.
where python >nul 2>&1
if errorlevel 1 (
  echo Python is not found. Install Python 3.12 and tick "Add python to PATH", then double-click setup.bat again.
  pause
  exit /b 1
)
where psql >nul 2>&1
if errorlevel 1 (
  echo psql is not found. Install PostgreSQL 17, add "C:\Program Files\PostgreSQL\17\bin" to PATH, then double-click setup.bat again.
  pause
  exit /b 1
)
echo.
echo Make the database - your Postgres password is asked ONCE now.
echo Type the password you set when you installed PostgreSQL.
echo Type it, do not paste it. Nothing shows while you type, that is normal.
echo If it says "role buildathon already exists", that is fine, it will reuse it.
psql -U postgres -v ON_ERROR_STOP=0 -c "CREATE USER buildathon WITH PASSWORD 'buildathon';" -c "CREATE DATABASE buildathon OWNER buildathon;"
echo Database done.
echo.
echo Make the virtual environment and install - this takes a few minutes the first time.
cd anpr-backend
if not exist venv (
  python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
  echo Install failed. Check the error above, then double-click setup.bat again.
  pause
  exit /b 1
)
if not exist .env (
  copy .env.example .env
)
echo.
echo Create the tables.
python -m app.test_db
python -m app.create_tables
if not exist models (
  mkdir models
)
if not exist models\yolov8-plate.pt (
  echo Download the plate model - about 6 MB.
  powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest https://huggingface.co/joker5914/yolov8n-license-plate/resolve/main/best.pt -OutFile models\yolov8-plate.pt"
)
echo.
echo Setup is done.
echo Next: double-click run.bat to start the backend, then open dashboard\dashboard.html
pause
