@echo off
echo ========================================================
echo   Starting IT3091 Machine Learning Model API Server
echo   Group 2026-AI-08K - Ames House Price Prediction
echo ========================================================
echo.

IF EXIST ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) ELSE (
    python app.py
)

pause
