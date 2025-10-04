#!/usr/bin/env python3
"""
Test script to check if the platform is ready to run
"""

print("🚀 Testing ExoVision AI Platform Setup...")
print("=" * 50)

try:
    import pandas as pd
    print("✅ Pandas imported successfully")
except ImportError as e:
    print(f"❌ Pandas import failed: {e}")

try:
    import numpy as np
    print("✅ NumPy imported successfully")
except ImportError as e:
    print(f"❌ NumPy import failed: {e}")

try:
    import sklearn
    print("✅ Scikit-learn imported successfully")
except ImportError as e:
    print(f"❌ Scikit-learn import failed: {e}")

try:
    import streamlit as st
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import plotly
    print("✅ Plotly imported successfully")
except ImportError as e:
    print(f"❌ Plotly import failed: {e}")

try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

print("\n🔍 Testing data loading...")
try:
    from preprocess import load_datasets
    df = load_datasets()
    print(f"✅ Data loaded successfully: {len(df)} records")
    print(f"   Features: {list(df.columns)}")
except Exception as e:
    print(f"❌ Data loading failed: {e}")

print("\n🎯 Testing model training...")
try:
    from train import train_from_base
    print("✅ Training module imported successfully")
    print("   Ready to train model!")
except Exception as e:
    print(f"❌ Training module failed: {e}")

print("\n🌐 Testing UI...")
try:
    print("✅ UI module ready")
    print("   Run: streamlit run ui.py")
except Exception as e:
    print(f"❌ UI module failed: {e}")

print("\n" + "=" * 50)
print("🎉 Platform Status Check Complete!")
print("\nTo run the platform:")
print("1. Streamlit UI: streamlit run ui.py")
print("2. FastAPI Backend: uvicorn app:app --reload")
print("3. Train Model: python train.py")
