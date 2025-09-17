@echo off
echo Starting Crop Recommendation System...
echo =====================================
echo.

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Start the backend server
echo Starting backend API server...
start /B cmd /c "cd backend && python app.py"

REM Wait for server to start
timeout /t 3 /nobreak > nul

REM Open the frontend in browser
echo Opening application in browser...
start http://localhost:8000
start frontend\index.html

echo.
echo =====================================
echo Application is running!
echo Backend API: http://localhost:8000
echo Frontend: Open index.html in your browser
echo.
echo Press Ctrl+C to stop the server
echo =====================================

REM Keep the backend running
cd backend
python app.py
