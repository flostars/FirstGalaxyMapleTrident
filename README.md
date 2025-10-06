# 🌌 ExoVision AI Platform
## A World Away: Hunting for Exoplanets with AI

[![NASA Space Apps Challenge 2025](https://img.shields.io/badge/NASA%20Space%20Apps-2025-blue.svg)](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/?tab=details)
[![Python](https://img.shields.io/badge/Python-3.11+-green.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-blue.svg)](https://fastapi.tiangolo.com)

> **An advanced multi-modal exoplanet detection and analysis platform powered by AI, designed for the NASA Space Apps Challenge 2025.**

## 🚀 Project Overview

ExoVision AI is a comprehensive platform that combines machine learning, real-time data analysis, and advanced visualizations to revolutionize exoplanet discovery and analysis. Built for the [NASA Space Apps Challenge 2025 "A World Away: Hunting for Exoplanets with AI"](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/?tab=details), this platform demonstrates cutting-edge AI applications in space science.

### 🎯 Key Features

- **🤖 Advanced AI Core**: Multi-algorithm ensemble (Random Forest, XGBoost) with neural network processing
- **📡 Live Data Integration**: Real-time NASA Exoplanet Archive API connectivity
- **📊 Enhanced Visualizations**: Interactive charts with dual-scale analysis and improved clarity
- **🌌 Galactic Star Mapping**: Dual-panel celestial mapping with discovery timeline analysis
- **🔬 Cross-Mission Learning**: TESS, Kepler, and K2 mission data integration
- **⚡ Real-Time Analysis**: Instant predictions and model retraining capabilities
- **🎨 Cyberpunk UI**: Modern, responsive interface with space-themed styling

## 🏗️ Architecture

```
ExoVision AI Platform
├── 🎨 Frontend (Streamlit UI)
│   ├── Data Visualization Dashboard
│   ├── Live Data Integration
│   ├── Model Training Interface
│   └── Prediction Engine
├── ⚙️ Backend (FastAPI)
│   ├── Prediction API (/predict)
│   ├── Retraining API (/retrain)
│   ├── Statistics API (/stats)
│   └── NASA Data Integration
├── 🧠 AI Core
│   ├── Random Forest Classifier
│   ├── XGBoost Ensemble
│   ├── Feature Engineering
│   └── Model Persistence
└── 📊 Enhanced Visualizations
    ├── Orbital Period Analysis
    ├── Galactic Star Mapping
    ├── Correlation Heatmaps
    └── Data Quality Dashboards
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Git

### Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/flostars/FirstGalaxyMapleTrident.git
   cd FirstGalaxyMapleTrident
   ```

2. **Install dependencies**
   ```bash
   pip install pandas scikit-learn xgboost fastapi uvicorn streamlit plotly joblib torch transformers
   ```

3. **Launch the platform**
   ```bash
   # Option 1: Use the batch file (Windows)
   .\run_platform.bat
   
   # Option 2: Manual launch
   python -m streamlit run ui.py
   ```

4. **Access the platform**
   - **Local**: http://localhost:8501
   - **Network**: http://10.0.0.85:8501

## 📊 Enhanced Visualizations

### 🌟 Key Improvements Made

#### 1. **Orbital Period Distribution Analysis**
- **Dual-scale visualization** (linear + logarithmic)
- **Statistical annotations** with median and mean indicators
- **Outlier filtering** for better data clarity
- **Fixed text overlap** issues for professional presentation

#### 2. **Galactic Star Map**
- **Dual-panel layout**: Discovery timeline + Planet size distribution
- **Decade-based grouping** for temporal analysis
- **Planet size categories**: Earth-like, Super-Earth, Neptune-like, Jupiter-like, Giant
- **Enhanced hover data** with coordinates, discovery info, and planet characteristics
- **Reference lines** for celestial equator and galactic plane

#### 3. **Orbital Period vs Planet Radius**
- **Logarithmic scales** for better data representation
- **Jittering** to reduce point overlap
- **Clear color coding** by planet classification
- **Reference lines** for Earth and Jupiter comparisons

#### 4. **Data Quality Dashboard**
- **Comprehensive data analysis** tools
- **Missing data visualization**
- **Feature correlation analysis**
- **Statistical summaries**

## 🔧 API Endpoints

### FastAPI Backend (`api.py`)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Upload CSV for exoplanet predictions |
| `/retrain` | POST | Retrain model with new data |
| `/stats` | GET | Retrieve latest training metrics |

### Example API Usage

```python
import requests

# Make a prediction
response = requests.post('http://localhost:8000/predict', 
                        files={'file': open('exoplanet_data.csv', 'rb')})
predictions = response.json()

# Get model statistics
stats = requests.get('http://localhost:8000/stats').json()
```

## 🧠 Machine Learning Features

### Algorithms Supported
- **Random Forest**: Robust baseline classifier
- **XGBoost**: Gradient boosting ensemble
- **Neural Networks**: Deep learning capabilities (via transformers)

### Feature Engineering
- **Orbital parameters**: Period, semi-major axis, eccentricity, inclination
- **Stellar properties**: Temperature, radius, mass, surface gravity
- **System characteristics**: Distance, visual magnitude, equilibrium temperature
- **Discovery metadata**: Method, facility, year, controversy flags

### Model Performance
- **Cross-validation**: 5-fold validation for robust metrics
- **Feature importance**: Automated feature ranking
- **Real-time retraining**: Continuous learning from new data
- **Model persistence**: Automatic model saving and loading

## 📁 Project Structure

```
FirstGalaxyMapleTrident/
├── 📊 Data Visualization
│   ├── ui.py                          # Main Streamlit interface
│   ├── improved_visualizations.py     # Enhanced visualization engine
│   └── simple_visualizations.py       # Original visualization functions
├── ⚙️ Backend Services
│   ├── api.py                         # FastAPI prediction service
│   ├── train.py                       # Model training utilities
│   └── preprocess.py                  # Data preprocessing helpers
├── 🧠 AI Core
│   ├── models/                        # Trained model storage
│   └── quick_train_advanced.py        # Quick model training
├── 🚀 Deployment
│   ├── run_platform.bat               # Windows launch script
│   └── requirements.txt               # Python dependencies
└── 📚 Documentation
    ├── README.md                      # This file
    └── data/                          # Sample datasets
```

## 🌟 NASA Space Apps Challenge Alignment

This project directly addresses the [NASA Space Apps Challenge 2025 "A World Away: Hunting for Exoplanets with AI"](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/?tab=details) by:

### 🎯 Challenge Requirements Met
- **AI-Powered Detection**: Advanced machine learning algorithms for exoplanet classification
- **Real-Time Analysis**: Live data integration with NASA Exoplanet Archive
- **Multi-Mission Data**: Integration of TESS, Kepler, and K2 mission data
- **User-Friendly Interface**: Intuitive Streamlit dashboard for researchers and enthusiasts
- **Scalable Architecture**: FastAPI backend for production deployment
- **Advanced Visualizations**: Interactive charts for data exploration and analysis

### 🚀 Innovation Highlights
- **Dual-panel Galactic Mapping**: Unique visualization combining temporal and size analysis
- **Enhanced Data Clarity**: Improved visualizations with better scaling and information density
- **Real-Time Learning**: Continuous model improvement with new data
- **Professional UI**: Cyberpunk-themed interface for engaging user experience

## 🔬 Scientific Impact

### Research Applications
- **Exoplanet Classification**: Automated identification of planet types
- **Discovery Pattern Analysis**: Temporal trends in exoplanet discoveries
- **Stellar System Analysis**: Understanding planet-star relationships
- **Data Quality Assessment**: Comprehensive data validation tools

### Educational Value
- **Interactive Learning**: Hands-on exploration of exoplanet data
- **Visual Data Science**: Intuitive understanding of complex astronomical data
- **Real-Time Updates**: Live connection to NASA's latest discoveries
- **Open Source**: Fully accessible codebase for learning and modification

## 🛠️ Development

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Local Development
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest

# Start development server
streamlit run ui.py --server.runOnSave true
```

## 📈 Recent Updates

### Version 2.0 - Enhanced Visualizations (Latest)
- ✅ **Improved Orbital Period Distribution** with dual-scale analysis
- ✅ **Enhanced Galactic Star Map** with dual-panel layout
- ✅ **Better Scatter Plots** with logarithmic scaling
- ✅ **Removed unnecessary sections** for cleaner interface
- ✅ **Fixed text overlap** issues in visualizations
- ✅ **Professional UI improvements** with better color schemes

### Version 1.0 - Initial Release
- ✅ **Core AI platform** with Random Forest and XGBoost
- ✅ **NASA API integration** for live data
- ✅ **Basic visualizations** and user interface
- ✅ **FastAPI backend** for predictions and retraining

## 📞 Support & Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/flostars/FirstGalaxyMapleTrident/issues)
- **NASA Space Apps**: [Challenge Details](https://www.spaceappschallenge.org/2025/challenges/a-world-away-hunting-for-exoplanets-with-ai/?tab=details)
- **Documentation**: See inline code comments and docstrings

## 📄 License

This project is developed for the NASA Space Apps Challenge 2025. Please refer to NASA's terms and conditions for usage guidelines.

---

**🌌 Built with ❤️ for the NASA Space Apps Challenge 2025**

*"Exploring the cosmos, one exoplanet at a time."*