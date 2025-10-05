"""Streamlit UI for interacting with the exoplanet classifier."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


import io
from typing import Any, Dict

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

from exoplanet_app import preprocess, train
import sys
sys.path.append(str(Path(__file__).parent))
from predict_advanced_simple import SimplifiedAdvancedPredictor

st.set_page_config(
    page_title="ExoVision AI - Cosmic Exoplanet Hunter", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🌌"
)

COSMIC_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600;700&display=swap');

:root {
    --primary-cyan: #00f5ff;
    --secondary-purple: #8b5cf6;
    --accent-pink: #f472b6;
    --dark-bg: #0a0a0f;
    --darker-bg: #050508;
    --card-bg: rgba(15, 15, 35, 0.8);
    --border-glow: rgba(0, 245, 255, 0.3);
    --text-glow: rgba(0, 245, 255, 0.8);
    --neon-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
}

* {
    box-sizing: border-box;
}

body {
    background: 
        radial-gradient(circle at 20% 80%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(244, 114, 182, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(0, 245, 255, 0.05) 0%, transparent 50%),
        linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 25%, #16213e 50%, #0f0f23 75%, #050508 100%);
    color: #ffffff;
    font-family: 'Exo 2', 'Segoe UI', sans-serif;
    margin: 0;
    padding: 0;
    min-height: 100vh;
    overflow-x: hidden;
}

.stApp {
    background: transparent !important;
}

/* Animated background particles */
body::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(2px 2px at 20px 30px, var(--primary-cyan), transparent),
        radial-gradient(2px 2px at 40px 70px, var(--secondary-purple), transparent),
        radial-gradient(1px 1px at 90px 40px, var(--accent-pink), transparent),
        radial-gradient(1px 1px at 130px 80px, var(--primary-cyan), transparent),
        radial-gradient(2px 2px at 160px 30px, var(--secondary-purple), transparent);
    background-repeat: repeat;
    background-size: 200px 100px;
    animation: sparkle 20s linear infinite;
    pointer-events: none;
    z-index: -1;
}

@keyframes sparkle {
    0% { transform: translateY(0px); }
    100% { transform: translateY(-100px); }
}

.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1400px;
    background: var(--card-bg);
    border-radius: 20px;
    border: 1px solid var(--border-glow);
    box-shadow: var(--neon-shadow);
    backdrop-filter: blur(10px);
    margin: 1rem;
}

/* Headers with cyberpunk styling */
h1 {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 3rem;
    background: linear-gradient(45deg, var(--primary-cyan), var(--secondary-purple), var(--accent-pink));
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin: 2rem 0;
    animation: gradientShift 3s ease-in-out infinite;
    text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
}

@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

h2 {
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    color: var(--primary-cyan);
    text-shadow: 0 0 15px rgba(0, 245, 255, 0.6);
    border-bottom: 2px solid var(--border-glow);
    padding-bottom: 0.5rem;
    margin-top: 2rem;
}

h3 {
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    color: var(--primary-cyan);
    text-shadow: 0 0 15px rgba(0, 245, 255, 0.8);
    background: linear-gradient(45deg, var(--primary-cyan), var(--secondary-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Cyberpunk buttons */
.stButton > button {
    background: linear-gradient(45deg, var(--primary-cyan), var(--secondary-purple));
    color: white;
    border: 2px solid var(--primary-cyan);
    border-radius: 25px;
    padding: 0.8rem 2rem;
    font-family: 'Orbitron', monospace;
    font-weight: 700;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    box-shadow: 
        0 0 20px rgba(0, 245, 255, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-3px) scale(1.05);
    box-shadow: 
        0 10px 30px rgba(0, 245, 255, 0.4),
        0 0 40px rgba(139, 92, 246, 0.3),
        inset 0 1px 0 rgba(255, 255, 255, 0.3);
    border-color: var(--accent-pink);
}

.stButton > button:active {
    transform: translateY(-1px) scale(1.02);
}

/* Cyberpunk input fields */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: var(--card-bg) !important;
    border: 2px solid var(--border-glow) !important;
    border-radius: 15px !important;
    color: white !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 400 !important;
    padding: 0.8rem 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.3) !important;
}

/* Mission selection dropdown specific styling */
.stSelectbox > div > div {
    background: var(--card-bg) !important;
    border: 2px solid var(--primary-cyan) !important;
    border-radius: 15px !important;
    color: white !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.8rem 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 
        inset 0 2px 10px rgba(0, 0, 0, 0.3),
        0 0 15px rgba(0, 245, 255, 0.2) !important;
}

/* Fix for selected text visibility in dropdown */
.stSelectbox > div > div > div {
    color: white !important;
    font-weight: 600 !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.5) !important;
}

.stSelectbox [data-baseweb="select"] > div > div {
    color: white !important;
    font-weight: 600 !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.5) !important;
}

/* Fix for all text content in selectbox */
.stSelectbox * {
    color: white !important;
}

.stSelectbox input {
    color: white !important;
    background: transparent !important;
}

.stSelectbox span {
    color: white !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.5) !important;
}

.stSelectbox > div > div:hover {
    border-color: var(--accent-pink) !important;
    box-shadow: 
        inset 0 2px 10px rgba(0, 0, 0, 0.3),
        0 0 20px rgba(0, 245, 255, 0.4) !important;
}

.stSelectbox > div > div:focus {
    border-color: var(--primary-cyan) !important;
    box-shadow: 
        inset 0 2px 10px rgba(0, 0, 0, 0.3),
        0 0 25px rgba(0, 245, 255, 0.5) !important;
    outline: none !important;
}

/* Dropdown options styling */
.stSelectbox [data-baseweb="select"] > div {
    background: var(--card-bg) !important;
    border: 2px solid var(--primary-cyan) !important;
    border-radius: 15px !important;
    color: white !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.8rem 1rem !important;
    box-shadow: 
        inset 0 2px 10px rgba(0, 0, 0, 0.3),
        0 0 15px rgba(0, 245, 255, 0.2) !important;
}

.stSelectbox [data-baseweb="select"] > div:hover {
    border-color: var(--accent-pink) !important;
    box-shadow: 
        inset 0 2px 10px rgba(0, 0, 0, 0.3),
        0 0 20px rgba(0, 245, 255, 0.4) !important;
}

/* Dropdown menu options */
.stSelectbox [role="listbox"] {
    background: var(--darker-bg) !important;
    border: 2px solid var(--border-glow) !important;
    border-radius: 15px !important;
    box-shadow: 0 0 25px rgba(0, 245, 255, 0.3) !important;
}

.stSelectbox [role="option"] {
    background: var(--card-bg) !important;
    color: white !important;
    font-family: 'Exo 2', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.8rem 1rem !important;
    border-bottom: 1px solid var(--border-glow) !important;
}

.stSelectbox [role="option"]:hover {
    background: rgba(0, 245, 255, 0.1) !important;
    color: var(--primary-cyan) !important;
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.6) !important;
}

.stSelectbox [role="option"][aria-selected="true"] {
    background: linear-gradient(45deg, var(--primary-cyan), var(--secondary-purple)) !important;
    color: white !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.8) !important;
}

/* Additional fixes for dropdown text visibility */
.stSelectbox [data-baseweb="select"] {
    color: white !important;
}

.stSelectbox [data-baseweb="select"] > div {
    color: white !important;
}

.stSelectbox [data-baseweb="select"] > div > div {
    color: white !important;
    background: var(--card-bg) !important;
}

.stSelectbox [data-baseweb="select"] > div > div > div {
    color: white !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.5) !important;
}

/* Fix for the selected value display */
.stSelectbox [data-baseweb="select"] [data-baseweb="select__value-container"] {
    color: white !important;
}

.stSelectbox [data-baseweb="select"] [data-baseweb="select__single-value"] {
    color: white !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.5) !important;
    font-weight: 600 !important;
}

.stSelectbox [data-baseweb="select"] [data-baseweb="select__placeholder"] {
    color: rgba(255, 255, 255, 0.7) !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.3) !important;
}

/* ULTRA-AGGRESSIVE FIXES FOR MISSION SELECTION TEXT VISIBILITY */

/* Nuclear option - force ALL elements to be white and properly aligned */
.stSelectbox,
.stSelectbox *,
.stSelectbox div,
.stSelectbox span,
.stSelectbox p,
.stSelectbox input,
.stSelectbox button,
.stSelectbox [role="combobox"],
.stSelectbox [data-baseweb="select"],
.stSelectbox [data-baseweb="select__single-value"],
.stSelectbox [data-baseweb="select__placeholder"] {
    color: white !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.9) !important;
    font-weight: 700 !important;
    background: transparent !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Target specific Streamlit classes */
.stSelectbox .st-ay,
.stSelectbox .st-au,
.stSelectbox .st-af,
.stSelectbox .st-ai,
.stSelectbox .st-di,
.stSelectbox .st-dj,
.stSelectbox .st-ag,
.stSelectbox .st-b0,
.stSelectbox .st-ah,
.stSelectbox .st-d1,
.stSelectbox .st-dk,
.stSelectbox .st-dl,
.stSelectbox .st-dm,
.stSelectbox .st-dn,
.stSelectbox .st-do,
.stSelectbox .st-dp,
.stSelectbox .st-dq,
.stSelectbox .st-dr,
.stSelectbox .st-ds,
.stSelectbox .st-dt,
.stSelectbox .st-du,
.stSelectbox .st-dv,
.stSelectbox .st-dw,
.stSelectbox .st-dx,
.stSelectbox .st-dy,
.stSelectbox .st-dz,
.stSelectbox .st-e0,
.stSelectbox .st-e1,
.stSelectbox .st-e2,
.stSelectbox .st-e3,
.stSelectbox .st-e4,
.stSelectbox .st-e5,
.stSelectbox .st-e6,
.stSelectbox .st-e7,
.stSelectbox .st-e8,
.stSelectbox .st-e9 {
    color: white !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.9) !important;
    font-weight: 700 !important;
    background: transparent !important;
    opacity: 1 !important;
    visibility: visible !important;
}

/* Force text content to be visible */
.stSelectbox [value="kepler"],
.stSelectbox [value="k2"], 
.stSelectbox [value="tess"],
.stSelectbox [value="neossat"] {
    color: white !important;
    text-shadow: 0 0 15px rgba(255, 255, 255, 1) !important;
    font-weight: 900 !important;
    background: transparent !important;
    opacity: 1 !important;
    visibility: visible !important;
    display: block !important;
}

/* Override any BaseWeb styles */
.stSelectbox [data-baseweb="select"] * {
    color: white !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.9) !important;
    font-weight: 700 !important;
}

/* Force the selected value to be visible and centered */
.stSelectbox [data-baseweb="select__single-value"] {
    color: white !important;
    text-shadow: 0 0 15px rgba(255, 255, 255, 1) !important;
    font-weight: 900 !important;
    background: transparent !important;
    opacity: 1 !important;
    visibility: visible !important;
    text-align: center !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    height: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Target the dropdown options */
.stSelectbox [data-baseweb="select__option"] {
    color: white !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.9) !important;
    font-weight: 700 !important;
    background: rgba(0, 0, 0, 0.8) !important;
}

/* Force visibility on all text nodes */
.stSelectbox *:not(svg):not(path) {
    color: white !important;
    text-shadow: 0 0 10px rgba(255, 255, 255, 0.9) !important;
    font-weight: 700 !important;
}

.stSelectbox > div > div:focus,
.stNumberInput > div > div > input:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--primary-cyan) !important;
    box-shadow: 
        inset 0 2px 10px rgba(0, 0, 0, 0.3),
        0 0 20px rgba(0, 245, 255, 0.3) !important;
    outline: none !important;
}

/* Cyberpunk tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: var(--darker-bg);
    padding: 10px;
    border-radius: 15px;
    border: 1px solid var(--border-glow);
}

.stTabs [data-baseweb="tab"] {
    background: var(--card-bg);
    border: 2px solid transparent;
    border-radius: 12px;
    color: var(--primary-cyan);
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    font-size: 1rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 1rem 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.stTabs [data-baseweb="tab"]::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(0, 245, 255, 0.1), transparent);
    transition: left 0.5s;
}

.stTabs [data-baseweb="tab"]:hover::before {
    left: 100%;
}

.stTabs [data-baseweb="tab"]:hover {
    border-color: var(--primary-cyan);
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
    transform: translateY(-2px);
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(45deg, var(--primary-cyan), var(--secondary-purple));
    color: white;
    border-color: var(--accent-pink);
    box-shadow: 
        0 0 25px rgba(0, 245, 255, 0.4),
        0 0 40px rgba(139, 92, 246, 0.2);
    transform: translateY(-3px);
}

/* Cyberpunk metrics and cards */
.metric-container {
    background: var(--card-bg);
    border: 2px solid var(--border-glow);
    border-radius: 15px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: var(--neon-shadow);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--primary-cyan), var(--secondary-purple), var(--accent-pink));
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 0.5; }
    50% { opacity: 1; }
}

.metric-container:hover {
    transform: translateY(-5px);
    box-shadow: 
        0 10px 30px rgba(0, 245, 255, 0.3),
        0 0 50px rgba(139, 92, 246, 0.2);
    border-color: var(--accent-pink);
}

/* Status messages with cyberpunk styling */
.stSuccess {
    background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 200, 0, 0.05));
    border: 2px solid #00ff00;
    border-radius: 15px;
    color: #00ff00;
    padding: 1rem;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(0, 255, 0, 0.5);
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);
}

.stError {
    background: linear-gradient(135deg, rgba(255, 0, 0, 0.1), rgba(200, 0, 0, 0.05));
    border: 2px solid #ff0000;
    border-radius: 15px;
    color: #ff0000;
    padding: 1rem;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(255, 0, 0, 0.5);
    box-shadow: 0 0 20px rgba(255, 0, 0, 0.2);
}

.stWarning {
    background: linear-gradient(135deg, rgba(255, 165, 0, 0.1), rgba(255, 140, 0, 0.05));
    border: 2px solid #ffa500;
    border-radius: 15px;
    color: #ffa500;
    padding: 1rem;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(255, 165, 0, 0.5);
    box-shadow: 0 0 20px rgba(255, 165, 0, 0.2);
}

.stInfo {
    background: linear-gradient(135deg, rgba(0, 245, 255, 0.1), rgba(0, 200, 255, 0.05));
    border: 2px solid var(--primary-cyan);
    border-radius: 15px;
    color: var(--primary-cyan);
    padding: 1rem;
    font-family: 'Orbitron', monospace;
    font-weight: 600;
    text-shadow: 0 0 10px rgba(0, 245, 255, 0.5);
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.2);
}

/* Data tables with cyberpunk styling */
.stDataFrame {
    background: var(--card-bg);
    border: 2px solid var(--border-glow);
    border-radius: 15px;
    box-shadow: var(--neon-shadow);
    backdrop-filter: blur(10px);
    overflow: hidden;
}

.stDataFrame table {
    background: transparent !important;
}

.stDataFrame th {
    background: linear-gradient(45deg, var(--primary-cyan), var(--secondary-purple)) !important;
    color: white !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    border: none !important;
    padding: 1rem !important;
}

.stDataFrame td {
    background: rgba(15, 15, 35, 0.5) !important;
    color: white !important;
    border: 1px solid var(--border-glow) !important;
    padding: 0.8rem !important;
    font-family: 'Exo 2', sans-serif !important;
}

.stDataFrame tr:nth-child(even) td {
    background: rgba(25, 25, 55, 0.3) !important;
}

.stDataFrame tr:hover td {
    background: rgba(0, 245, 255, 0.1) !important;
    box-shadow: inset 0 0 10px rgba(0, 245, 255, 0.2);
}

/* Sidebar styling */
.css-1d391kg {
    background: var(--darker-bg) !important;
    border-right: 2px solid var(--border-glow) !important;
}

/* File uploader styling */
.stFileUploader {
    background: var(--card-bg);
    border: 2px dashed var(--border-glow);
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
}

.stFileUploader:hover {
    border-color: var(--primary-cyan);
    background: rgba(0, 245, 255, 0.05);
    box-shadow: 0 0 20px rgba(0, 245, 255, 0.2);
}

/* Progress bars */
.stProgress > div > div > div {
    background: linear-gradient(90deg, var(--primary-cyan), var(--secondary-purple)) !important;
    border-radius: 10px !important;
    box-shadow: 0 0 10px rgba(0, 245, 255, 0.3) !important;
}

/* Spinner styling */
.stSpinner {
    color: var(--primary-cyan) !important;
}

/* Custom scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--darker-bg);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(45deg, var(--primary-cyan), var(--secondary-purple));
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(45deg, var(--accent-pink), var(--primary-cyan));
}

/* Enhanced text readability */
.stSubheader, .stMarkdown h3, .stMarkdown h4 {
    color: var(--primary-cyan) !important;
    text-shadow: 0 0 15px rgba(0, 245, 255, 0.8) !important;
    font-weight: 700 !important;
}

.stMarkdown p, .stMarkdown div {
    color: #ffffff !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.3) !important;
}

/* Input field labels */
.stNumberInput label, .stSelectbox label, .stTextInput label {
    color: var(--primary-cyan) !important;
    font-weight: 600 !important;
    text-shadow: 0 0 10px rgba(0, 245, 255, 0.6) !important;
    font-family: 'Orbitron', monospace !important;
}

/* Mission selection specific styling - Fixed positioning and text wrapping */
.stSelectbox > label {
    color: var(--primary-cyan) !important;
    font-weight: 700 !important;
    text-shadow: 0 0 15px rgba(0, 245, 255, 0.8) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.1rem !important;
    text-align: left !important;
    white-space: nowrap !important;
    overflow: visible !important;
    width: auto !important;
    max-width: none !important;
    position: relative !important;
    left: 0 !important;
    transform: none !important;
    padding: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    display: block !important;
    margin-bottom: 0.5rem !important;
}

/* Section headers */
.stMarkdown h2 {
    color: var(--primary-cyan) !important;
    text-shadow: 0 0 20px rgba(0, 245, 255, 0.8) !important;
    font-weight: 700 !important;
    border-bottom: 2px solid var(--border-glow) !important;
    padding-bottom: 0.5rem !important;
}

/* File uploader text */
.stFileUploader label {
    color: var(--primary-cyan) !important;
    font-weight: 600 !important;
    text-shadow: 0 0 10px rgba(0, 245, 255, 0.6) !important;
}

/* Metric labels */
.stMetric label {
    color: var(--primary-cyan) !important;
    font-weight: 600 !important;
    text-shadow: 0 0 10px rgba(0, 245, 255, 0.6) !important;
}

/* Enhanced visibility for all text elements */
.stMarkdown, .stMarkdown *, .stText, .stText * {
    color: #ffffff !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.3) !important;
}

/* Specific styling for section titles */
.stMarkdown h3, .stMarkdown h4 {
    background: linear-gradient(45deg, var(--primary-cyan), var(--secondary-purple)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-shadow: 0 0 15px rgba(0, 245, 255, 0.8) !important;
}

/* Fix for prediction matrix feature tags */
.stMarkdown code, .stMarkdown pre, .stMarkdown span {
    background: var(--card-bg) !important;
    color: var(--primary-cyan) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.6rem !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.6) !important;
    box-shadow: 0 0 10px rgba(0, 245, 255, 0.2) !important;
    display: inline-block !important;
    margin: 0.2rem !important;
}

/* Fix for inline code and feature names */
.stMarkdown p code, .stMarkdown li code, .stMarkdown div code {
    background: linear-gradient(135deg, var(--card-bg), rgba(0, 245, 255, 0.1)) !important;
    color: var(--primary-cyan) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 6px !important;
    padding: 0.2rem 0.5rem !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.6) !important;
    box-shadow: 0 0 8px rgba(0, 245, 255, 0.2) !important;
    display: inline-block !important;
    margin: 0.1rem !important;
    font-size: 0.9rem !important;
}

/* Fix for all text content in prediction matrix */
.stMarkdown p, .stMarkdown div, .stMarkdown span {
    color: #ffffff !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.3) !important;
    line-height: 1.6 !important;
}

/* Fix for specific prediction matrix text */
.prediction-matrix-text {
    color: #ffffff !important;
    text-shadow: 0 0 5px rgba(255, 255, 255, 0.3) !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.6 !important;
}

/* Fix for feature tags specifically */
.feature-tag {
    background: linear-gradient(135deg, var(--card-bg), rgba(0, 245, 255, 0.1)) !important;
    color: var(--primary-cyan) !important;
    border: 1px solid var(--border-glow) !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.6rem !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 600 !important;
    text-shadow: 0 0 8px rgba(0, 245, 255, 0.6) !important;
    box-shadow: 0 0 10px rgba(0, 245, 255, 0.2) !important;
    display: inline-block !important;
    margin: 0.2rem !important;
    font-size: 0.9rem !important;
}

/* Responsive design */
@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    
    .main .block-container {
        margin: 0.5rem;
        padding: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
    }
}

/* Loading animation */
@keyframes cyberpunkGlow {
    0%, 100% { 
        box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
    }
    50% { 
        box-shadow: 0 0 40px rgba(0, 245, 255, 0.6), 0 0 60px rgba(139, 92, 246, 0.3);
    }
}

.cyberpunk-glow {
    animation: cyberpunkGlow 2s ease-in-out infinite;
}
</style>
"""
st.markdown(COSMIC_CSS, unsafe_allow_html=True)

# Add JavaScript to force text visibility
st.markdown("""
<script>
// Force mission selection text to be visible and centered
function forceTextVisibility() {
    const selectboxes = document.querySelectorAll('.stSelectbox');
    selectboxes.forEach(selectbox => {
        const allElements = selectbox.querySelectorAll('*');
        allElements.forEach(el => {
            if (el.tagName !== 'SVG' && el.tagName !== 'PATH') {
                el.style.color = 'white !important';
                el.style.textShadow = '0 0 10px rgba(255, 255, 255, 0.9) !important';
                el.style.fontWeight = '700 !important';
                el.style.background = 'transparent !important';
                el.style.opacity = '1 !important';
                el.style.visibility = 'visible !important';
                el.style.textAlign = 'center !important';
                el.style.display = 'flex !important';
                el.style.alignItems = 'center !important';
                el.style.justifyContent = 'center !important';
            }
        });
        
        // Specifically target the value display area
        const valueElements = selectbox.querySelectorAll('[data-baseweb="select__single-value"]');
        valueElements.forEach(el => {
            el.style.textAlign = 'center !important';
            el.style.display = 'flex !important';
            el.style.alignItems = 'center !important';
            el.style.justifyContent = 'center !important';
            el.style.width = '100% !important';
            el.style.height = '100% !important';
            el.style.padding = '0 !important';
            el.style.margin = '0 !important';
        });
        
        // Fix label positioning and prevent text wrapping
        const labels = selectbox.querySelectorAll('label');
        labels.forEach(label => {
            label.style.textAlign = 'left !important';
            label.style.whiteSpace = 'nowrap !important';
            label.style.overflow = 'visible !important';
            label.style.width = 'auto !important';
            label.style.maxWidth = 'none !important';
            label.style.position = 'relative !important';
            label.style.left = '0 !important';
            label.style.transform = 'none !important';
            label.style.padding = '0 !important';
            label.style.marginLeft = '0 !important';
            label.style.marginRight = '0 !important';
        });
    });
}

// Run on page load and when selectbox changes
document.addEventListener('DOMContentLoaded', forceTextVisibility);
setInterval(forceTextVisibility, 1000);
</script>
""", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin: 2rem 0;">
    <h1 style="font-family: 'Orbitron', monospace; font-weight: 900; font-size: 4rem; 
        background: linear-gradient(45deg, #00f5ff, #8b5cf6, #f472b6);
        background-size: 200% 200%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientShift 3s ease-in-out infinite;
        text-shadow: 0 0 30px rgba(0, 245, 255, 0.5);
        margin: 0;">
        🚀 EXOVISION AI 🚀
    </h1>
    <p style="font-family: 'Exo 2', sans-serif; font-size: 1.5rem; color: #00f5ff; 
       text-shadow: 0 0 15px rgba(0, 245, 255, 0.6); margin: 1rem 0;">
        COSMIC EXOPLANET HUNTER
    </p>
    <p style="font-family: 'Exo 2', sans-serif; font-size: 1rem; color: #8b5cf6; 
       margin: 0.5rem 0;">
        Advanced AI-Powered Multi-Modal Exoplanet Detection System
    </p>
</div>
""", unsafe_allow_html=True)

px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = [
    "#7FDBFF",
    "#39CCCC",
    "#B10DC9",
    "#FF851B",
    "#FFDC00",
    "#2ECC40",
    "#01FF70",
    "#F012BE",
]
px.defaults.width = None
px.defaults.height = 520


@st.cache_data(show_spinner=False)
def _load_base_dataset() -> pd.DataFrame:
    return preprocess.load_datasets()


@st.cache_data(show_spinner=False)
def _load_star_map_data() -> pd.DataFrame:
    return preprocess.load_star_map_data()


def _show_metrics(metrics: Dict[str, Any], label: str) -> None:
    st.subheader(label)
    cols = st.columns(len(metrics))
    for (name, value), col in zip(metrics.items(), cols):
        col.metric(name.capitalize(), f"{value:.3f}")


base_data = None
try:
    base_data = _load_base_dataset()
except Exception as exc:
    st.warning(f"Unable to load base datasets: {exc}")

star_map_data = None
star_map_error = None
try:
    star_map_data = _load_star_map_data()
except Exception as exc:
    star_map_error = str(exc)

training_tab, prediction_tab, advanced_tab, insights_tab = st.tabs([
    "⚡ NEURAL TRAINING", 
    "🔮 PREDICTION MATRIX", 
    "🤖 QUANTUM AI CORE", 
    "📊 DATA VISUALIZATION"
])

with training_tab:
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.6); border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            ⚡ NEURAL TRAINING
        </h2>
        <p style="font-family: 'Exo 2', sans-serif; color: #8b5cf6; margin: 1rem 0;">
            Machine Learning Model Training • Algorithm Selection • Performance Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    algorithm_label = st.selectbox("Model selection", ("Random Forest (baseline)", "XGBoost"))
    algorithm_key = "random_forest" if "Random Forest" in algorithm_label else "xgboost"
    base_model_file = st.file_uploader("Upload base model (optional)", type=("pkl", "joblib"), key="base_model")
    training_file = st.file_uploader("Upload CSV for retraining (optional)", type="csv")

    if st.button("Train / Retrain", use_container_width=True):
        with st.spinner("Training model..."):
            try:
                base_bytes = base_model_file.getvalue() if base_model_file is not None else None
                if training_file is not None:
                    metrics = train.retrain_model(
                        training_file.getvalue(),
                        algorithm=algorithm_key,
                        base_model=base_bytes,
                    )
                else:
                    df = preprocess.load_datasets()
                    metrics = train.train_model(
                        df,
                        algorithm=algorithm_key,
                        base_model=base_bytes,
                    )
            except Exception as exc:  # pragma: no cover - surfaced in UI
                st.error(f"Training failed: {exc}")
            else:
                _show_metrics(metrics, "Latest Training Metrics")
                st.success("Model training complete.")

    # Show previously stored metrics if available
    try:
        past_metrics = train.get_stored_metrics()
        _show_metrics(past_metrics, "Stored Metrics")
    except FileNotFoundError:
        pass

with prediction_tab:
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.6); border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            🔮 PREDICTION MATRIX
        </h2>
        <p style="font-family: 'Exo 2', sans-serif; color: #8b5cf6; margin: 1rem 0;">
            Advanced Exoplanet Classification • Multi-Feature Analysis • Real-Time Processing
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div class="prediction-matrix-text">
        Upload a CSV containing the eleven numeric features used for training:
        <br><br>
        <span class="feature-tag">pl_orbper</span>
        <span class="feature-tag">pl_orbsmax</span>
        <span class="feature-tag">pl_rade</span>
        <span class="feature-tag">pl_bmasse</span>
        <span class="feature-tag">pl_eqt</span>
        <span class="feature-tag">pl_insol</span>
        <span class="feature-tag">st_teff</span>
        <span class="feature-tag">st_rad</span>
        <span class="feature-tag">st_mass</span>
        <span class="feature-tag">st_met</span>
        <span class="feature-tag">st_logg</span>
        <br><br>
        Extra columns are optional and will be echoed back alongside predictions.
    </div>
    """, unsafe_allow_html=True)
    prediction_file = st.file_uploader("Upload CSV for prediction", type="csv", key="prediction")

    if st.button("Run Prediction", use_container_width=True):
        if prediction_file is None:
            st.warning("Please upload a CSV file first.")
        else:
            try:
                payload = train.get_trained_model()
            except FileNotFoundError:
                st.error("Model not trained yet. Train the model before running predictions.")
            else:
                dataframe = pd.read_csv(io.BytesIO(prediction_file.getvalue()))
                features = preprocess.prepare_features_frame(dataframe)
                predictions = payload["model"].predict(features)
                results = dataframe.copy()
                results["prediction"] = predictions
                st.subheader("Predicted Dispositions")
                st.dataframe(results, use_container_width=True)

with advanced_tab:
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.6); border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            🤖 QUANTUM AI CORE - MULTI-MODAL EXOPLANET DETECTION
        </h2>
        <p style="font-family: 'Exo 2', sans-serif; color: #8b5cf6; margin: 1rem 0;">
            Advanced Neural Network Processing • Cross-Mission Learning • Real-Time Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize advanced predictor
    if 'advanced_predictor' not in st.session_state:
        st.session_state.advanced_predictor = SimplifiedAdvancedPredictor()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 1rem 0;">
            🧪 SINGLE PREDICTION TEST
        </h3>
        """, unsafe_allow_html=True)
        
        # Input parameters
        st.markdown("""
        <div style="color: #00f5ff; font-family: 'Orbitron', monospace; font-weight: 600; 
           text-shadow: 0 0 10px rgba(0, 245, 255, 0.6); margin: 0.5rem 0;">
            🚀 ORBITAL PARAMETERS
        </div>
        """, unsafe_allow_html=True)
        period = st.number_input("Orbital Period (days)", min_value=0.1, max_value=1000.0, value=10.5)
        depth = st.number_input("Transit Depth (ppm)", min_value=1, max_value=100000, value=1000)
        duration = st.number_input("Transit Duration (hours)", min_value=0.1, max_value=100.0, value=3.0)
        
        # Stellar parameters
        st.markdown("""
        <h4 style="font-family: 'Orbitron', monospace; font-weight: 600; color: #8b5cf6; 
           text-shadow: 0 0 10px rgba(139, 92, 246, 0.8); margin: 1rem 0;">
            ⭐ STELLAR PARAMETERS
        </h4>
        """, unsafe_allow_html=True)
        teff = st.number_input("Effective Temperature (K)", min_value=2000, max_value=10000, value=5800)
        logg = st.number_input("Surface Gravity (cgs)", min_value=0.0, max_value=6.0, value=4.4)
        radius = st.number_input("Stellar Radius (Solar radii)", min_value=0.1, max_value=100.0, value=1.0)
        mass = st.number_input("Stellar Mass (Solar masses)", min_value=0.1, max_value=10.0, value=1.0)
        metallicity = st.number_input("Metallicity (dex)", min_value=-2.0, max_value=1.0, value=0.0)
        
        st.markdown("""
        <div style="color: #00f5ff; font-family: 'Orbitron', monospace; font-weight: 600; 
           text-shadow: 0 0 10px rgba(0, 245, 255, 0.6); margin: 0.5rem 0;">
            🛰️ MISSION SELECTION
        </div>
        """, unsafe_allow_html=True)
        mission = st.selectbox("Mission", ["kepler", "k2", "tess"])
        
        if st.button("🔍 Analyze with Advanced AI"):
            with st.spinner("Analyzing with multi-modal AI..."):
                stellar_params = {
                    'pl_orbper': period,
                    'pl_orbsmax': 0.1,
                    'pl_rade': 1.2,
                    'pl_bmasse': 2.1,
                    'pl_eqt': 300.0,
                    'pl_insol': 1.0,
                    'st_teff': teff,
                    'st_rad': radius,
                    'st_mass': mass,
                    'st_met': metallicity,
                    'st_logg': logg
                }
                
                result = st.session_state.advanced_predictor.predict_single(
                    period=period,
                    depth=depth/1e6,  # Convert ppm to fraction
                    duration=duration,
                    stellar_params=stellar_params,
                    mission=mission
                )
                
                # Display results with cyberpunk styling
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 200, 0, 0.05));
                   border: 2px solid #00ff00; border-radius: 15px; padding: 1rem; margin: 1rem 0;
                   font-family: 'Orbitron', monospace; font-weight: 600; color: #00ff00;
                   text-shadow: 0 0 10px rgba(0, 255, 0, 0.5); box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);">
                    🎯 **QUANTUM PREDICTION**: {result['prediction']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: rgba(0, 245, 255, 0.1); border: 2px solid #00f5ff; border-radius: 15px; 
                   padding: 1rem; margin: 1rem 0; color: #00f5ff; font-family: 'Exo 2', sans-serif;">
                    🎯 **NEURAL CONFIDENCE**: {result['confidence']:.1%}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, rgba(255, 165, 0, 0.1), rgba(255, 140, 0, 0.05));
                   border: 2px solid #ffa500; border-radius: 15px; padding: 1rem; margin: 1rem 0;
                   font-family: 'Orbitron', monospace; font-weight: 600; color: #ffa500;
                   text-shadow: 0 0 10px rgba(255, 165, 0, 0.5); box-shadow: 0 0 20px rgba(255, 165, 0, 0.2);">
                    ⚠️ **UNCERTAINTY LEVEL**: {result['uncertainty']:.1%}
                </div>
                """, unsafe_allow_html=True)
                
                # Probabilities
                prob_col1, prob_col2 = st.columns(2)
                with prob_col1:
                    st.metric("No Exoplanet", f"{result['probabilities']['no_exoplanet']:.1%}")
                with prob_col2:
                    st.metric("Exoplanet", f"{result['probabilities']['exoplanet']:.1%}")
                
                # Feature importance
                st.markdown("""
                <h4 style="font-family: 'Orbitron', monospace; font-weight: 600; color: #8b5cf6; 
                   text-shadow: 0 0 10px rgba(139, 92, 246, 0.8); margin: 1rem 0;">
                    🔬 FEATURE IMPORTANCE
                </h4>
                """, unsafe_allow_html=True)
                importance_col1, importance_col2 = st.columns(2)
                with importance_col1:
                    st.metric("Light Curve", f"{result['feature_importance']['light_curve']:.1%}")
                with importance_col2:
                    st.metric("Stellar Features", f"{np.mean(result['feature_importance']['stellar_features']):.1%}")
    
    with col2:
        st.markdown("""
        <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 1rem 0;">
            📊 BATCH PREDICTION
        </h3>
        """, unsafe_allow_html=True)
        
        # Upload CSV for batch prediction
        st.markdown("""
        <div style="color: #00f5ff; font-family: 'Orbitron', monospace; font-weight: 600; 
           text-shadow: 0 0 10px rgba(0, 245, 255, 0.6); margin: 0.5rem 0;">
            📁 DATA UPLOAD INTERFACE
        </div>
        """, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload CSV file for batch prediction", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.write(f"📁 Loaded {len(df)} records")
                
                if st.button("🚀 Run Advanced Batch Prediction"):
                    with st.spinner("Processing with advanced AI..."):
                        results = st.session_state.advanced_predictor.predict_batch(df)
                        
                        st.success(f"✅ Processed {len(results)} records")
                        
                        # Show results
                        st.dataframe(results[['prediction', 'confidence', 'uncertainty', 'mission']], use_container_width=True)
                        
                        # Download results
                        csv = results.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="advanced_predictions.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"❌ Error processing file: {e}")
        
        st.markdown("""
        <div style="background: rgba(15, 15, 35, 0.8); border: 2px solid rgba(0, 245, 255, 0.3); 
           border-radius: 15px; padding: 1.5rem; margin: 1rem 0; box-shadow: 0 0 20px rgba(0, 245, 255, 0.5);
           backdrop-filter: blur(10px); position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; 
               background: linear-gradient(90deg, #00f5ff, #8b5cf6, #f472b6);
               animation: pulse 2s ease-in-out infinite;"></div>
            <h3 style="font-family: 'Orbitron', monospace; color: #00f5ff; margin: 0 0 1rem 0;">
                🎯 QUANTUM AI STATUS
            </h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.advanced_predictor.advanced_available:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 200, 0, 0.05));
               border: 2px solid #00ff00; border-radius: 15px; padding: 1rem; margin: 1rem 0;
               font-family: 'Orbitron', monospace; font-weight: 600; color: #00ff00;
               text-shadow: 0 0 10px rgba(0, 255, 0, 0.5); box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);">
                ✅ QUANTUM AI CORE ONLINE
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(0, 245, 255, 0.1); border: 2px solid #00f5ff; border-radius: 15px; 
               padding: 1rem; margin: 1rem 0; color: #00f5ff; font-family: 'Exo 2', sans-serif;">
                <h4 style="color: #00f5ff; margin: 0 0 1rem 0;">🧠 NEURAL ARCHITECTURE:</h4>
                <ul style="margin: 0; padding-left: 1.5rem;">
                    <li>🔬 Light Curve CNN Processing</li>
                    <li>⭐ Stellar Parameter Encoding</li>
                    <li>🔗 Cross-Modal Attention Fusion</li>
                    <li>🚀 Mission-Specific Adapters</li>
                    <li>📊 Uncertainty Quantification</li>
                    <li>⚡ Real-Time Inference Engine</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(255, 165, 0, 0.1), rgba(255, 140, 0, 0.05));
               border: 2px solid #ffa500; border-radius: 15px; padding: 1rem; margin: 1rem 0;
               font-family: 'Orbitron', monospace; font-weight: 600; color: #ffa500;
               text-shadow: 0 0 10px rgba(255, 165, 0, 0.5); box-shadow: 0 0 20px rgba(255, 165, 0, 0.2);">
                ⚠️ QUANTUM AI CORE - FALLBACK MODE
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="background: rgba(139, 92, 246, 0.1); border: 2px solid #8b5cf6; border-radius: 15px; 
               padding: 1rem; margin: 1rem 0; color: #8b5cf6; font-family: 'Exo 2', sans-serif;">
                <h4 style="color: #8b5cf6; margin: 0 0 1rem 0;">🔄 SIMPLIFIED AI MODE:</h4>
                <ul style="margin: 0; padding-left: 1.5rem;">
                    <li>🔍 Heuristic-Based Light Curve Analysis</li>
                    <li>⭐ Stellar Parameter Scoring</li>
                    <li>🌌 Mission-Aware Predictions</li>
                    <li>📊 Confidence Estimation</li>
                    <li>🛡️ Fallback ML Algorithms</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

with insights_tab:
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.6); border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            📊 DATA VISUALIZATION
        </h2>
        <p style="font-family: 'Exo 2', sans-serif; color: #8b5cf6; margin: 1rem 0;">
            Dataset Analysis • Statistical Insights • Interactive Visualizations
        </p>
    </div>
    """, unsafe_allow_html=True)
    if base_data is not None and not base_data.empty:
        disposition_counts = base_data[preprocess.LABEL_COLUMN].value_counts().reset_index()
        disposition_counts.columns = ["Disposition", "Count"]
        chart = px.bar(
            disposition_counts,
            x="Disposition",
            y="Count",
            title="Distribution of Planet Candidates by Status",
            color="Disposition",
        )
        chart.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(chart, use_container_width=True)

        st.subheader("Orbital Period vs Planet Radius")
        scatter_fig = px.scatter(
            base_data,
            x="pl_orbper",
            y="pl_rade",
            color=preprocess.LABEL_COLUMN,
            title="Verification Dataset View",
            hover_data=["pl_eqt", "pl_insol", "st_teff", "st_rad"],
        )
        scatter_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(scatter_fig, use_container_width=True)

    if star_map_data is not None and not star_map_data.empty:
        st.subheader("Galactic Star Map")
        star_display = star_map_data.copy()
        radius_series = star_display["planet_radius"]
        if radius_series.notna().any():
            median_radius = float(radius_series.median(skipna=True))
        else:
            median_radius = 1.0
        star_display["radius_display"] = radius_series.fillna(median_radius).clip(lower=0.1)
        star_display["discovery_year_label"] = star_display["discovery_year"].apply(
            lambda val: str(int(val)) if pd.notna(val) else "Unknown"
        )
        star_display["identifier"] = star_display["identifier"].fillna("N/A")
        star_fig = px.scatter(
            star_display,
            x="ra",
            y="dec",
            color="discovery_year_label",
            size="radius_display",
            size_max=18,
            hover_name="identifier",
            hover_data={
                "dataset": True,
                preprocess.LABEL_COLUMN: True,
                "planet_radius": True,
                "discovery_year": True,
            },
            labels={
                "ra": "Right Ascension (deg)",
                "dec": "Declination (deg)",
                "discovery_year_label": "Discovery Year",
                "radius_display": "Radius (Earth radii)",
            },
            title="Star Map from Training Catalogues",
        )
        star_fig.update_layout(
            xaxis=dict(range=[360, 0]),
            yaxis_title="Declination (deg)",
            xaxis_title="Right Ascension (deg)",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(star_fig, use_container_width=True)

    if base_data is not None and not base_data.empty:
        st.subheader("Preview of Combined Dataset")
        st.dataframe(base_data.head(50), use_container_width=True)
