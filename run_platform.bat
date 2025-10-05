@echo off
echo 🚀 Starting ExoVision AI Platform...
echo.

echo Installing requirements...
pip install pandas scikit-learn xgboost fastapi uvicorn streamlit plotly joblib torch transformers

echo.
echo Training quick advanced model...
python quick_train_advanced.py

echo.
echo Starting Streamlit UI...
echo Open your browser to: http://localhost:8501
echo.

streamlit run ui.py

pause
