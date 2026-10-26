@echo off
echo ===================================================================
echo Starting AI Urban Problem Hotspot Decision Support System Dashboard
echo ===================================================================
cd /d "%~dp0"
streamlit run dashboard\app.py
pause
