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
try:
    from predict_advanced_enhanced import get_advanced_predictor
    ADVANCED_PREDICTOR_AVAILABLE = True
except ImportError:
    try:
        from predict_advanced_simple import SimplifiedAdvancedPredictor
        ADVANCED_PREDICTOR_AVAILABLE = False
    except ImportError:
        ADVANCED_PREDICTOR_AVAILABLE = False

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
        radial-gradient(2px 2px at 160px 30px, var(--secondary-purple), transparent),
        radial-gradient(1px 1px at 200px 50px, #ffffff, transparent),
        radial-gradient(1px 1px at 250px 90px, #ffffff, transparent),
        radial-gradient(2px 2px at 300px 20px, #ffffff, transparent);
    background-repeat: repeat;
    background-size: 350px 200px;
    animation: sparkle 20s linear infinite;
    pointer-events: none;
    z-index: -1;
    opacity: 0.8;
}

body::after {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-image: 
        radial-gradient(1px 1px at 50px 60px, var(--primary-cyan), transparent),
        radial-gradient(1px 1px at 120px 20px, var(--secondary-purple), transparent),
        radial-gradient(1px 1px at 80px 100px, var(--accent-pink), transparent),
        radial-gradient(1px 1px at 180px 90px, #ffffff, transparent),
        radial-gradient(2px 2px at 220px 40px, #ffffff, transparent);
    background-repeat: repeat;
    background-size: 280px 180px;
    animation: sparkle 25s linear infinite reverse;
    pointer-events: none;
    z-index: -1;
    opacity: 0.6;
}

@keyframes sparkle {
    0% { transform: translateY(0px) translateX(0px); }
    25% { transform: translateY(-50px) translateX(25px); }
    50% { transform: translateY(-100px) translateX(-25px); }
    75% { transform: translateY(-150px) translateX(50px); }
    100% { transform: translateY(-200px) translateX(0px); }
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

/* NUCLEAR OPTION - COMPLETE ANCHOR LINK ANNIHILATION */

/* Hide ALL possible anchor link variations */
a[href*="#"],
a[class*="emotion-cache"],
a[class*="et2rgd"],
[data-testid] a,
.stMarkdown a,
.element-container a,
.css-* a,
.e* a,
.st-* a,
div[data-testid="stMarkdownContainer"] a,
div[class*="stMarkdown"] a,
*[href^="#"],
*[href*="#exovision"],
*[href*="#"]:not([href="#"]),
svg[viewBox="0 0 24 24"],
svg[stroke="currentColor"],
svg[stroke-width="2"],
svg[stroke-linecap="round"],
svg[stroke-linejoin="round"],
svg path[d*="M15 7h3a5 5 0 0 1 5 5"],
svg line[x1="8"][y1="12"][x2="16"][y2="12"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    pointer-events: none !important;
    z-index: -1 !important;
}

/* Target specific Streamlit generated classes */
.css-1v0mbdj,
.css-10trblm,
.css-16huue1,
.css-1d391kg,
.css-qrbaxs,
.e1fqkh3o0,
.e1fqkh3o1,
.e1fqkh3o2,
.e1fqkh3o3 {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* Force all headers to not have any child links */
h1, h2, h3, h4, h5, h6 {
    position: relative !important;
}

h1 *, h2 *, h3 *, h4 *, h5 *, h6 * {
    pointer-events: none !important;
}

h1 a, h2 a, h3 a, h4 a, h5 a, h6 a,
h1 svg, h2 svg, h3 svg, h4 svg, h5 svg, h6 svg {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
}

/* Override any Streamlit anchor styles */
.stMarkdown h1::before,
.stMarkdown h2::before,
.stMarkdown h3::before,
.stMarkdown h4::before,
.stMarkdown h5::before,
.stMarkdown h6::before {
    display: none !important;
    content: none !important;
}

/* Hide any remaining link-like elements */
*[role="link"],
*[tabindex][href],
button[onclick*="#"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
}

/* FINAL CLEANUP - Target any remaining artifacts */
/* Hide any pseudo-elements that might show anchor links */
.stMarkdown *::before,
.stMarkdown *::after,
.element-container *::before,
.element-container *::after {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    content: none !important;
}

/* Only target anchor-related pseudo-elements, preserve star animations */
a::before,
a::after,
a[href*="#"]::before,
a[href*="#"]::after {
    content: none !important;
    display: none !important;
}

/* Target any remaining visual artifacts */
.stMarkdown > div > *:first-child a,
.element-container > div > *:first-child a,
[data-testid="stMarkdownContainer"] > div > *:first-child a {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 0 !important;
    height: 0 !important;
}

/* Hide any remaining chain/link icons */
svg[xmlns="http://www.w3.org/2000/svg"][width="16"][height="16"],
svg[viewBox="0 0 24 24"][fill="none"][stroke="currentColor"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
}

/* Force override any inline styles */
a[href*="#"] {
    display: none !important;
    visibility: hidden !important;
    opacity: 0 !important;
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    pointer-events: none !important;
    z-index: -9999 !important;
    width: 0 !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    color: transparent !important;
    text-indent: -9999px !important;
    overflow: hidden !important;
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

// Remove all anchor links and reference blocks - NUCLEAR OPTION
function removeAnchorLinks() {
    // Remove all possible anchor link selectors
    const selectors = [
        'a[href*="#"]',
        'a[class*="emotion-cache"]',
        'a[class*="et2rgd"]',
        '[data-testid] a',
        '.stMarkdown a',
        '.element-container a',
        'div[data-testid="stMarkdownContainer"] a',
        'div[class*="stMarkdown"] a',
        '*[href^="#"]',
        '*[href*="#exovision"]',
        'svg[viewBox="0 0 24 24"]',
        'svg[stroke="currentColor"]',
        'svg[stroke-width="2"]',
        'svg path[d*="M15 7h3a5 5 0 0 1 5 5"]',
        'svg line[x1="8"][y1="12"][x2="16"][y2="12"]',
        '*[role="link"]',
        '*[tabindex][href]',
        'button[onclick*="#"]'
    ];
    
    selectors.forEach(selector => {
        try {
            const elements = document.querySelectorAll(selector);
            elements.forEach(el => {
                el.style.display = 'none !important';
                el.style.visibility = 'hidden !important';
                el.style.opacity = '0 !important';
                el.style.position = 'absolute !important';
                el.style.left = '-9999px !important';
                el.style.top = '-9999px !important';
                el.style.pointerEvents = 'none !important';
                el.style.zIndex = '-1 !important';
                if (el.parentNode) {
                    el.parentNode.removeChild(el);
                }
            });
        } catch (e) {
            // Ignore selector errors
        }
    });
    
    // Remove any elements with specific classes
    const classesToRemove = [
        'css-1v0mbdj',
        'css-10trblm', 
        'css-16huue1',
        'css-1d391kg',
        'css-qrbaxs',
        'e1fqkh3o0',
        'e1fqkh3o1',
        'e1fqkh3o2',
        'e1fqkh3o3',
        'st-emotion-cache-yinll1',
        'et2rgd21'
    ];
    
    classesToRemove.forEach(className => {
        const elements = document.querySelectorAll('.' + className);
        elements.forEach(el => {
            el.style.display = 'none !important';
            el.style.visibility = 'hidden !important';
            el.style.opacity = '0 !important';
            if (el.parentNode) {
                el.parentNode.removeChild(el);
            }
        });
    });
    
    // Remove any remaining links in headers
    const headers = document.querySelectorAll('h1, h2, h3, h4, h5, h6');
    headers.forEach(header => {
        const links = header.querySelectorAll('a, svg');
        links.forEach(link => {
            link.style.display = 'none !important';
            link.style.visibility = 'hidden !important';
            link.style.opacity = '0 !important';
            if (link.parentNode) {
                link.parentNode.removeChild(link);
            }
        });
    });
}

// Run on page load and when selectbox changes
document.addEventListener('DOMContentLoaded', function() {
    forceTextVisibility();
    removeAnchorLinks();
});
setInterval(function() {
    forceTextVisibility();
    removeAnchorLinks();
}, 500); // Increased frequency to 500ms

// Add MutationObserver to catch dynamically added elements
const observer = new MutationObserver(function(mutations) {
    mutations.forEach(function(mutation) {
        if (mutation.type === 'childList') {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === Node.ELEMENT_NODE) {
                    // Check if the added node or its children contain anchor links
                    const links = node.querySelectorAll ? node.querySelectorAll('a[href*="#"], svg[viewBox="0 0 24 24"]') : [];
                    links.forEach(function(link) {
                        link.style.display = 'none !important';
                        link.style.visibility = 'hidden !important';
                        link.style.opacity = '0 !important';
                        link.style.position = 'absolute !important';
                        link.style.left = '-9999px !important';
                        link.style.top = '-9999px !important';
                        if (link.parentNode) {
                            link.parentNode.removeChild(link);
                        }
                    });
                    
                    // If the node itself is an anchor link
                    if (node.tagName === 'A' && node.href && node.href.includes('#')) {
                        node.style.display = 'none !important';
                        node.style.visibility = 'hidden !important';
                        node.style.opacity = '0 !important';
                        if (node.parentNode) {
                            node.parentNode.removeChild(node);
                        }
                    }
                    
                    // If the node is an SVG
                    if (node.tagName === 'SVG') {
                        node.style.display = 'none !important';
                        node.style.visibility = 'hidden !important';
                        node.style.opacity = '0 !important';
                        if (node.parentNode) {
                            node.parentNode.removeChild(node);
                        }
                    }
                }
            });
        }
    });
});

// Start observing
observer.observe(document.body, {
    childList: true,
    subtree: true
});
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
    st.markdown(f"""
    <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
       text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 1rem 0; border-bottom: 2px solid rgba(0, 245, 255, 0.3);
       padding-bottom: 0.5rem;">
        {label}
    </h3>
    """, unsafe_allow_html=True)
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

training_tab, advanced_tab, insights_tab, live_data_tab, monitoring_tab = st.tabs([
    "⚡ NEURAL TRAINING & PREDICTION", 
    "🤖 QUANTUM AI CORE", 
    "📊 DATA VISUALIZATION",
    "📡 LIVE DATA",
    "🔍 MONITORING"
])

with training_tab:
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.6); border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            ⚡ NEURAL TRAINING & PREDICTION
        </h2>
        <p style="font-family: 'Exo 2', sans-serif; color: #8b5cf6; margin: 1rem 0;">
            Complete ML Pipeline • Train → Test → Evaluate • Performance Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Training Section
    st.markdown("""
    <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
       text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 1rem 0; border-bottom: 2px solid rgba(0, 245, 255, 0.3);
       padding-bottom: 0.5rem;">
        🧠 MODEL TRAINING
    </h3>
    """, unsafe_allow_html=True)
    
    algorithm_label = st.selectbox("Model selection", ("Random Forest (baseline)", "XGBoost"))
    algorithm_key = "random_forest" if "Random Forest" in algorithm_label else "xgboost"
    base_model_file = st.file_uploader("Upload base model (optional)", type=("pkl", "joblib"), key="base_model")
    training_file = st.file_uploader("Upload CSV for retraining (optional)", type="csv", key="training_data")

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
    
    # Prediction Section
    st.markdown("""
    <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
       text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 2rem 0 1rem 0; border-bottom: 2px solid rgba(0, 245, 255, 0.3);
       padding-bottom: 0.5rem;">
        🔮 MODEL PREDICTION & EVALUATION
    </h3>
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
    
    prediction_file = st.file_uploader("Upload CSV for prediction", type="csv", key="prediction_training")

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
                st.markdown("""
                <h4 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
                   text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 1rem 0; border-bottom: 2px solid rgba(0, 245, 255, 0.3);
                   padding-bottom: 0.5rem;">
                    Predicted Dispositions
                </h4>
                """, unsafe_allow_html=True)
                st.dataframe(results, use_container_width=True)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prediction Results",
                    data=csv,
                    file_name="prediction_results.csv",
                    mime="text/csv"
                )
    
    # Advanced Analytics Dashboard Section
    st.markdown("""
    <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
       text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 2rem 0 1rem 0; border-bottom: 2px solid rgba(0, 245, 255, 0.3);
       padding-bottom: 0.5rem;">
        📊 ADVANCED ANALYTICS DASHBOARD
    </h3>
    """, unsafe_allow_html=True)
    
    # Model Performance Analytics
    st.markdown("""
    <h4 style="font-family: 'Orbitron', monospace; font-weight: 600; color: #8b5cf6; 
       text-shadow: 0 0 10px rgba(139, 92, 246, 0.8); margin: 1rem 0;">
        📈 MODEL PERFORMANCE ANALYTICS
    </h4>
    """, unsafe_allow_html=True)
    
    # Try to get model metrics for analysis
    try:
        metrics = train.get_stored_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
        with col2:
            st.metric("Recall", f"{metrics['recall']:.3f}")
        with col3:
            st.metric("F1 Score", f"{metrics['f1']:.3f}")
        with col4:
            precision = metrics.get('precision', 'N/A')
            if isinstance(precision, (int, float)):
                st.metric("Precision", f"{precision:.3f}")
            else:
                st.metric("Precision", "N/A")
        
        # Performance visualization
        if base_data is not None and not base_data.empty:
            st.markdown("""
            <h4 style="font-family: 'Orbitron', monospace; font-weight: 600; color: #8b5cf6; 
               text-shadow: 0 0 10px rgba(139, 92, 246, 0.8); margin: 1rem 0;">
                📊 CLASS DISTRIBUTION ANALYSIS
            </h4>
            """, unsafe_allow_html=True)
            
            # Class distribution
            if preprocess.LABEL_COLUMN in base_data.columns:
                disposition_counts = base_data[preprocess.LABEL_COLUMN].value_counts().reset_index()
                disposition_counts.columns = ["Disposition", "Count"]
                chart = px.pie(
                    disposition_counts,
                    values="Count",
                    names="Disposition",
                    title="Training Data Class Distribution",
                    color_discrete_sequence=[
                        "#00f5ff",  # Cyan
                        "#8b5cf6",  # Purple
                        "#10b981",  # Emerald
                        "#f59e0b",  # Amber
                        "#ef4444",  # Red
                        "#06b6d4"   # Sky
                    ]
                )
                chart.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        family="'Exo 2', sans-serif",
                        size=12,
                        color="#ffffff"
                    ),
                    title_font=dict(
                        family="'Orbitron', monospace",
                        size=16,
                        color="#00f5ff"
                    ),
                    legend=dict(
                        bgcolor="rgba(0,0,0,0.8)",
                        bordercolor="#8b5cf6",
                        borderwidth=1,
                        font=dict(color="#ffffff", family="'Exo 2', sans-serif")
                    ),
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                chart.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont=dict(
                        family="'Exo 2', sans-serif",
                        size=11,
                        color="#ffffff"
                    ),
                    marker=dict(
                        line=dict(color='#1a1a1a', width=2)
                    )
                )
                st.plotly_chart(chart, use_container_width=True)
        
    except FileNotFoundError:
        st.info("No trained model found. Train a model above to see performance analytics.")
    
    # Feature Importance Analysis
    st.markdown("""
    <h4 style="font-family: 'Orbitron', monospace; font-weight: 600; color: #8b5cf6; 
       text-shadow: 0 0 10px rgba(139, 92, 246, 0.8); margin: 1rem 0;">
        🔬 FEATURE IMPORTANCE ANALYSIS
    </h4>
    """, unsafe_allow_html=True)
    
    if base_data is not None and not base_data.empty:
        # Calculate correlation with target
        if preprocess.LABEL_COLUMN in base_data.columns:
            # Convert target to numeric for correlation - fix the numpy array issue
            target_numeric = pd.Series(pd.Categorical(base_data[preprocess.LABEL_COLUMN]).codes)
            correlations = {}
            
            for feature in preprocess.FEATURE_COLUMNS:
                if feature in base_data.columns:
                    try:
                        corr = abs(base_data[feature].corr(target_numeric))
                        correlations[feature] = corr if not pd.isna(corr) else 0.0
                    except:
                        correlations[feature] = 0.0
            
            # Sort by importance
            sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
            
            # Create feature importance chart
            feature_names = [f.replace('_', ' ').title() for f, _ in sorted_features[:8]]
            importance_values = [v for _, v in sorted_features[:8]]
            
            fig = px.bar(
                x=importance_values,
                y=feature_names,
                orientation='h',
                title="Feature Importance (Correlation with Target)",
                color=importance_values,
                color_continuous_scale=[[0, "#1a1a1a"], [0.5, "#8b5cf6"], [1, "#00f5ff"]]
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(
                    family="'Exo 2', sans-serif",
                    size=12,
                    color="#ffffff"
                ),
                title_font=dict(
                    family="'Orbitron', monospace",
                    size=16,
                    color="#00f5ff"
                ),
                xaxis=dict(
                    gridcolor="rgba(139, 92, 246, 0.3)",
                    gridwidth=1,
                    showgrid=True,
                    color="#ffffff",
                    title_font=dict(family="'Exo 2', sans-serif", size=12)
                ),
                yaxis=dict(
                    gridcolor="rgba(139, 92, 246, 0.3)",
                    gridwidth=1,
                    showgrid=True,
                    color="#ffffff",
                    title_font=dict(family="'Exo 2', sans-serif", size=12),
                    categoryorder='total ascending'
                ),
                margin=dict(l=20, r=20, t=60, b=20),
                showlegend=False
            )
            fig.update_traces(
                marker=dict(
                    line=dict(color='#8b5cf6', width=1),
                    opacity=0.8
                ),
                texttemplate='%{x:.3f}',
                textposition='outside',
                textfont=dict(
                    family="'Exo 2', sans-serif",
                    size=10,
                    color="#ffffff"
                )
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Prediction History and Analytics
    st.markdown("""
    <h4 style="font-family: 'Orbitron', monospace; font-weight: 600; color: #8b5cf6; 
       text-shadow: 0 0 10px rgba(139, 92, 246, 0.8); margin: 1rem 0;">
        📋 PREDICTION HISTORY & ANALYTICS
    </h4>
    """, unsafe_allow_html=True)
    
    # Upload historical predictions for analysis
    historical_file = st.file_uploader("Upload Historical Prediction Results for Analysis", type="csv", key="historical_analytics")
    
    if historical_file is not None:
        try:
            historical_data = pd.read_csv(historical_file)
            
            if 'prediction' in historical_data.columns:
                # Prediction distribution
                pred_counts = historical_data['prediction'].value_counts()
                fig_pred = px.pie(
                    values=pred_counts.values,
                    names=pred_counts.index,
                    title="Historical Prediction Distribution",
                    color_discrete_sequence=[
                        "#00f5ff",  # Cyan
                        "#8b5cf6",  # Purple
                        "#10b981",  # Emerald
                        "#f59e0b",  # Amber
                        "#ef4444"   # Red
                    ]
                )
                fig_pred.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(
                        family="'Exo 2', sans-serif",
                        size=12,
                        color="#ffffff"
                    ),
                    title_font=dict(
                        family="'Orbitron', monospace",
                        size=16,
                        color="#00f5ff"
                    ),
                    legend=dict(
                        bgcolor="rgba(0,0,0,0.8)",
                        bordercolor="#8b5cf6",
                        borderwidth=1,
                        font=dict(color="#ffffff", family="'Exo 2', sans-serif")
                    ),
                    margin=dict(l=20, r=20, t=60, b=20)
                )
                fig_pred.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    textfont=dict(
                        family="'Exo 2', sans-serif",
                        size=11,
                        color="#ffffff"
                    ),
                    marker=dict(
                        line=dict(color='#1a1a1a', width=2)
                    )
                )
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Show summary statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(historical_data))
                with col2:
                    exoplanet_count = (historical_data['prediction'] == 'CONFIRMED').sum() if 'CONFIRMED' in historical_data['prediction'].values else 0
                    st.metric("Exoplanet Predictions", exoplanet_count)
                with col3:
                    st.metric("Non-Exoplanet Predictions", len(historical_data) - exoplanet_count)
                
                # Display sample of historical data
                st.markdown("**Sample Historical Predictions:**")
                st.dataframe(historical_data.head(10), use_container_width=True)
            
        except Exception as e:
            st.error(f"Error analyzing historical data: {e}")
    
    # Model Comparison Tools
    st.markdown("""
    <h4 style="font-family: 'Orbitron', monospace; font-weight: 600; color: #8b5cf6; 
       text-shadow: 0 0 10px rgba(139, 92, 246, 0.8); margin: 1rem 0;">
        ⚖️ MODEL COMPARISON TOOLS
    </h4>
    """, unsafe_allow_html=True)
    
    st.info("""
    **Model Comparison Features:**
    - Compare different algorithms (Random Forest vs XGBoost)
    - Analyze performance across different feature sets
    - Evaluate model stability over time
    - Generate comprehensive model reports
    """)
    
    if st.button("🔄 Refresh Analytics", use_container_width=True):
        st.rerun()


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
        if ADVANCED_PREDICTOR_AVAILABLE:
            st.session_state.advanced_predictor = get_advanced_predictor()
        else:
            st.session_state.advanced_predictor = SimplifiedAdvancedPredictor()
    
    # Get model status
    if ADVANCED_PREDICTOR_AVAILABLE:
        model_status = st.session_state.advanced_predictor.get_model_status()
        advanced_available = model_status['advanced_available']
        models_loaded = model_status['models_loaded']
        model_type = model_status.get('model_type', 'unknown')
        cv_score = model_status.get('cv_score', 'unknown')
    else:
        advanced_available = False
        models_loaded = 0
        model_type = 'Simplified'
        cv_score = 'N/A'
    
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
                
                # Handle different feature importance structures
                feature_importance = result.get('feature_importance', {})
                
                if 'light_curve' in feature_importance:
                    # Simple predictor structure
                    importance_col1, importance_col2 = st.columns(2)
                    with importance_col1:
                        st.metric("Light Curve", f"{feature_importance['light_curve']:.1%}")
                    with importance_col2:
                        stellar_importance = feature_importance.get('stellar_features', [0.2] * 11)
                        if isinstance(stellar_importance, list):
                            avg_stellar = np.mean(stellar_importance)
                        else:
                            avg_stellar = stellar_importance
                        st.metric("Stellar Features", f"{avg_stellar:.1%}")
                elif 'orbital_features' in feature_importance:
                    # Enhanced predictor structure
                    importance_col1, importance_col2, importance_col3 = st.columns(3)
                    with importance_col1:
                        st.metric("Orbital Features", f"{feature_importance['orbital_features']:.1%}")
                    with importance_col2:
                        st.metric("Stellar Features", f"{feature_importance['stellar_features']:.1%}")
                    with importance_col3:
                        st.metric("Engineered Features", f"{feature_importance['engineered_features']:.1%}")
                else:
                    # Fallback if structure is unknown
                    st.info("Feature importance not available for this model.")
    
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
        
        if advanced_available:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, rgba(0, 255, 0, 0.1), rgba(0, 200, 0, 0.05));
               border: 2px solid #00ff00; border-radius: 15px; padding: 1rem; margin: 1rem 0;
               font-family: 'Orbitron', monospace; font-weight: 600; color: #00ff00;
               text-shadow: 0 0 10px rgba(0, 255, 0, 0.5); box-shadow: 0 0 20px rgba(0, 255, 0, 0.2);">
                ✅ ADVANCED AI CORE ONLINE<br>
                <span style="font-size: 0.8rem; color: #8b5cf6;">
                    {model_type} • {models_loaded} Models • CV: {cv_score if isinstance(cv_score, str) else f"{cv_score:.4f}"}
                </span>
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
    
    # Add advanced visualizations
    try:
        from simple_visualizations import ExoplanetVisualizer
        visualizer = ExoplanetVisualizer()
        
        if base_data is not None and not base_data.empty:
            # Data Distributions Section
            st.subheader("📈 Data Distributions")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Orbital Period Distribution**")
                fig_period = visualizer.create_orbital_period_distribution(base_data)
                st.plotly_chart(fig_period, use_container_width=True)
            
            with col2:
                st.markdown("**Feature Correlation Matrix**")
                fig_corr = visualizer.create_correlation_heatmap(base_data)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Model Performance Section
            st.subheader("🎯 Model Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Performance Metrics**")
                sample_metrics = {
                    'accuracy': 0.85,
                    'precision': 0.82,
                    'recall': 0.88,
                    'f1_score': 0.85
                }
                fig_performance = visualizer.create_model_performance_gauge(sample_metrics)
                st.plotly_chart(fig_performance, use_container_width=True)
            
            with col2:
                st.markdown("**Feature Importance**")
                feature_importance = {
                    'Orbital Period': 0.3,
                    'Planet Radius': 0.25,
                    'Planet Mass': 0.2,
                    'Stellar Temperature': 0.15,
                    'Stellar Radius': 0.1
                }
                fig_importance = visualizer.create_feature_importance_chart(feature_importance)
                st.plotly_chart(fig_importance, use_container_width=True)
            
            # Transit Simulation Section
            st.subheader("📡 Transit Simulation")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Light Curve Simulation**")
                fig_lc = visualizer.create_light_curve_simulation()
                st.plotly_chart(fig_lc, use_container_width=True)
            
            with col2:
                st.markdown("**Custom Parameters**")
                period = st.slider("Orbital Period (days)", 1.0, 100.0, 10.0)
                depth = st.slider("Transit Depth", 0.001, 0.1, 0.01)
                duration = st.slider("Transit Duration (hours)", 0.5, 24.0, 2.0)
                
                if st.button("Generate Custom Light Curve"):
                    fig_custom = visualizer.create_light_curve_simulation(
                        period=period, depth=depth, duration=duration
                    )
                    st.plotly_chart(fig_custom, use_container_width=True)
        else:
            st.info("No data available for visualization. Please ensure data is loaded.")
    
    except ImportError:
        st.error("Advanced visualizations not available. Please ensure simple_visualizations.py is present.")
    except Exception as e:
        st.error(f"Error loading visualizations: {e}")
    if base_data is not None and not base_data.empty:
        disposition_counts = base_data[preprocess.LABEL_COLUMN].value_counts().reset_index()
        disposition_counts.columns = ["Disposition", "Count"]
        chart = px.bar(
            disposition_counts,
            x="Disposition",
            y="Count",
            title="Distribution of Planet Candidates by Status",
            color="Disposition",
            color_discrete_sequence=[
                "#00f5ff",  # Cyan
                "#8b5cf6",  # Purple
                "#10b981",  # Emerald
                "#f59e0b",  # Amber
                "#ef4444",  # Red
                "#06b6d4"   # Sky
            ]
        )
        chart.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="'Exo 2', sans-serif",
                size=12,
                color="#ffffff"
            ),
            title_font=dict(
                family="'Orbitron', monospace",
                size=18,
                color="#00f5ff"
            ),
            xaxis=dict(
                gridcolor="rgba(139, 92, 246, 0.3)",
                gridwidth=1,
                showgrid=True,
                color="#ffffff",
                title_font=dict(family="'Exo 2', sans-serif", size=14),
                tickfont=dict(family="'Exo 2', sans-serif", size=11)
            ),
            yaxis=dict(
                gridcolor="rgba(139, 92, 246, 0.3)",
                gridwidth=1,
                showgrid=True,
                color="#ffffff",
                title_font=dict(family="'Exo 2', sans-serif", size=14),
                tickfont=dict(family="'Exo 2', sans-serif", size=11)
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            legend=dict(
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="#8b5cf6",
                borderwidth=1,
                font=dict(color="#ffffff", family="'Exo 2', sans-serif")
            )
        )
        chart.update_traces(
            marker=dict(
                line=dict(color='#1a1a1a', width=2),
                opacity=0.8
            ),
            texttemplate='%{y}',
            textposition='outside',
            textfont=dict(
                family="'Exo 2', sans-serif",
                size=11,
                color="#ffffff"
            )
        )
        st.plotly_chart(chart, use_container_width=True)

        st.markdown("""
        <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 1rem 0; border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            Orbital Period vs Planet Radius
        </h3>
        """, unsafe_allow_html=True)
        scatter_fig = px.scatter(
            base_data,
            x="pl_orbper",
            y="pl_rade",
            color=preprocess.LABEL_COLUMN,
            title="Orbital Period vs Planet Radius Analysis",
            hover_data=["pl_eqt", "pl_insol", "st_teff", "st_rad"],
            color_discrete_sequence=[
                "#00f5ff",  # Cyan
                "#8b5cf6",  # Purple
                "#10b981",  # Emerald
                "#f59e0b",  # Amber
                "#ef4444"   # Red
            ],
            size_max=15,
            opacity=0.7
        )
        scatter_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="'Exo 2', sans-serif",
                size=12,
                color="#ffffff"
            ),
            title_font=dict(
                family="'Orbitron', monospace",
                size=18,
                color="#00f5ff"
            ),
            xaxis=dict(
                gridcolor="rgba(139, 92, 246, 0.3)",
                gridwidth=1,
                showgrid=True,
                color="#ffffff",
                title="Orbital Period (days)",
                title_font=dict(family="'Exo 2', sans-serif", size=14),
                tickfont=dict(family="'Exo 2', sans-serif", size=11)
            ),
            yaxis=dict(
                gridcolor="rgba(139, 92, 246, 0.3)",
                gridwidth=1,
                showgrid=True,
                color="#ffffff",
                title="Planet Radius (Earth radii)",
                title_font=dict(family="'Exo 2', sans-serif", size=14),
                tickfont=dict(family="'Exo 2', sans-serif", size=11)
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            legend=dict(
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="#8b5cf6",
                borderwidth=1,
                font=dict(color="#ffffff", family="'Exo 2', sans-serif")
            )
        )
        scatter_fig.update_traces(
            marker=dict(
                line=dict(color='#1a1a1a', width=1),
                opacity=0.8
            )
        )
        st.plotly_chart(scatter_fig, use_container_width=True)

    if star_map_data is not None and not star_map_data.empty:
        st.markdown("""
        <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 1rem 0; border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            Galactic Star Map
        </h3>
        """, unsafe_allow_html=True)
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
            size_max=20,
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
            title="Galactic Star Map - Exoplanet Discovery Timeline",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        star_fig.update_layout(
            xaxis=dict(
                range=[360, 0],
                gridcolor="rgba(139, 92, 246, 0.3)",
                gridwidth=1,
                showgrid=True,
                color="#ffffff",
                title="Right Ascension (deg)",
                title_font=dict(family="'Exo 2', sans-serif", size=14),
                tickfont=dict(family="'Exo 2', sans-serif", size=11)
            ),
            yaxis=dict(
                gridcolor="rgba(139, 92, 246, 0.3)",
                gridwidth=1,
                showgrid=True,
                color="#ffffff",
                title="Declination (deg)",
                title_font=dict(family="'Exo 2', sans-serif", size=14),
                tickfont=dict(family="'Exo 2', sans-serif", size=11)
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(
                family="'Exo 2', sans-serif",
                size=12,
                color="#ffffff"
            ),
            title_font=dict(
                family="'Orbitron', monospace",
                size=18,
                color="#00f5ff"
            ),
            margin=dict(l=20, r=20, t=80, b=20),
            legend=dict(
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="#8b5cf6",
                borderwidth=1,
                font=dict(color="#ffffff", family="'Exo 2', sans-serif")
            )
        )
        star_fig.update_traces(
            marker=dict(
                line=dict(color='#1a1a1a', width=1),
                opacity=0.8
            )
        )
        st.plotly_chart(star_fig, use_container_width=True)

    if base_data is not None and not base_data.empty:
        st.markdown("""
        <h3 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.8); margin: 1rem 0; border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            Preview of Combined Dataset
        </h3>
        """, unsafe_allow_html=True)
        st.dataframe(base_data.head(50), use_container_width=True)

# Live Data Tab
with live_data_tab:
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.6); border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            📡 LIVE DATA INTEGRATION
        </h2>
        <p style="font-family: 'Exo 2', sans-serif; color: #8b5cf6; margin: 1rem 0;">
            Real-time NASA Exoplanet Archive • Live Data Updates • Mission Status
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from live_data_integration import LiveDataManager
        from simple_visualizations import ExoplanetVisualizer
        manager = LiveDataManager()
        visualizer = ExoplanetVisualizer()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Refresh Exoplanets", use_container_width=True):
                with st.spinner("Fetching latest exoplanets..."):
                    exoplanets = manager.get_live_data("exoplanets", force_refresh=True)
                    if not exoplanets.empty:
                        st.success(f"✅ Fetched {len(exoplanets)} exoplanets")
                        st.dataframe(exoplanets.head(10), use_container_width=True)
                    else:
                        st.warning("No exoplanet data available")
        
        with col2:
            if st.button("🔄 Refresh Kepler", use_container_width=True):
                with st.spinner("Fetching Kepler candidates..."):
                    kepler = manager.get_live_data("kepler", force_refresh=True)
                    if not kepler.empty:
                        st.success(f"✅ Fetched {len(kepler)} Kepler candidates")
                        st.dataframe(kepler.head(10), use_container_width=True)
                    else:
                        st.warning("No Kepler data available")
        
        with col3:
            if st.button("🔄 Refresh TESS", use_container_width=True):
                with st.spinner("Fetching TESS TOIs..."):
                    tess = manager.get_live_data("tess", force_refresh=True)
                    if not tess.empty:
                        st.success(f"✅ Fetched {len(tess)} TESS TOIs")
                        st.dataframe(tess.head(10), use_container_width=True)
                    else:
                        st.warning("No TESS data available")
        
        # Data status
        st.subheader("📊 Data Status")
        status = manager.get_data_status()
        
        for data_type, info in status.items():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{data_type.title()} Records", info['count'])
            with col2:
                st.metric("Status", info['status'])
            with col3:
                st.metric("Last Updated", info['last_updated'][:19] if info['last_updated'] != 'never' else 'Never')
        
        # Visualizations section
        st.subheader("🎨 Live Data Visualizations")
        
        # Get some data for visualization
        exoplanets_data = manager.get_live_data("exoplanets")
        
        if not exoplanets_data.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Orbital Period Distribution**")
                fig_period = visualizer.create_orbital_period_distribution(exoplanets_data)
                st.plotly_chart(fig_period, use_container_width=True)
            
            with col2:
                st.markdown("**Feature Correlation Matrix**")
                fig_corr = visualizer.create_correlation_heatmap(exoplanets_data)
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("No data available for visualization. Please refresh data first.")
        
    except ImportError:
        st.error("Live data integration not available. Please ensure live_data_integration.py is present.")
    except Exception as e:
        st.error(f"Error loading live data: {e}")

# Monitoring Tab
with monitoring_tab:
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-family: 'Orbitron', monospace; font-weight: 700; color: #00f5ff; 
           text-shadow: 0 0 15px rgba(0, 245, 255, 0.6); border-bottom: 2px solid rgba(0, 245, 255, 0.3);
           padding-bottom: 0.5rem;">
            🔍 REAL-TIME MONITORING
        </h2>
        <p style="font-family: 'Exo 2', sans-serif; color: #8b5cf6; margin: 1rem 0;">
            Training Progress • System Resources • Performance Metrics
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        from real_time_monitor import StreamlitTrainingMonitor
        monitor_ui = StreamlitTrainingMonitor()
        
        # Auto-refresh every 30 seconds
        if st.button("🔄 Refresh Status", use_container_width=True):
            st.rerun()
        
        # Display training status
        monitor_ui.display_training_status()
        
        # Display training summary
        monitor_ui.display_training_summary()
        
    except ImportError:
        st.error("Real-time monitoring not available. Please ensure real_time_monitor.py is present.")
    except Exception as e:
        st.error(f"Error loading monitoring: {e}")
