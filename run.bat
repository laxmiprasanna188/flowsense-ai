@echo off
cd /d "%~dp0"
start "Backend" cmd /k "cd anpr-backend && venv\Scripts\activate && uvicorn app.main:app --reload"
timeout /t 5 /nobreak >nul
start "" "dashboard\dashboard.html"
exit
