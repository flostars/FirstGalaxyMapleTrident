@echo off
echo 🚀 Starting ExoVision AI Platform...
echo.

echo Installing requirements...
pip install pandas scikit-learn xgboost fastapi uvicorn streamlit plotly joblib

echo.
echo Starting Streamlit UI...
echo Open your browser to: http://localhost:8501
echo.

streamlit run ui.py

pause
