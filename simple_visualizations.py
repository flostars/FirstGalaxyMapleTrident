"""
Simple and Reliable Visualizations for ExoVision AI
Focused on practical, working visualizations for the platform
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json

class ExoplanetVisualizer:
    """Simple and reliable visualization tools for exoplanet data"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#00f5ff',
            'secondary': '#8b5cf6', 
            'accent': '#f472b6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444'
        }
    
    def create_planet_size_comparison(self, df: pd.DataFrame) -> go.Figure:
        """Create planet radius distribution chart"""
        
        try:
            # Check for available columns
            name_cols = ['kepoi_name', 'kepler_name', 'pl_name']
            radius_cols = ['koi_prad', 'pl_rade']
            temp_cols = ['koi_steff', 'st_teff']
            
            name_col = None
            radius_col = None
            temp_col = None
            
            for col in name_cols:
                if col in df.columns:
                    name_col = col
                    break
            
            for col in radius_cols:
                if col in df.columns:
                    radius_col = col
                    break
            
            for col in temp_cols:
                if col in df.columns:
                    temp_col = col
                    break
            
            if not name_col or not radius_col:
                return self._create_empty_chart(f"Missing required columns. Available: {list(df.columns)[:10]}...")
            
            # Create planet data
            planet_data = df[[name_col, radius_col]].dropna()
            if temp_col:
                planet_data[temp_col] = df[temp_col]
            
            if planet_data.empty:
                return self._create_empty_chart("No planet data available after filtering")
        except Exception as e:
            return self._create_empty_chart(f"Error processing data: {str(e)}")
        
        # Create radius distribution chart
        fig = go.Figure()
        
        # Add planets as bars
        fig.add_trace(go.Bar(
            x=planet_data[name_col].head(20),  # Show first 20 planets
            y=planet_data[radius_col].head(20),
            marker=dict(
                color=planet_data[temp_col].head(20) if temp_col else [5000] * min(20, len(planet_data)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Stellar Temperature (K)"),
                opacity=0.7
            ),
            hovertemplate='<b>%{x}</b><br>Radius: %{y:.2f} R⊕<br>Star Temp: %{marker.color:.0f}K<extra></extra>',
            name='Exoplanets'
        ))
        
        # Add Earth reference line
        fig.add_hline(
            y=1,
            line_dash="dash",
            line_color="red",
            annotation_text="Earth (1 R⊕)"
        )
        
        fig.update_layout(
            title="Planet Radius Distribution",
            xaxis_title="Planet Name",
            yaxis_title="Planet Radius (Earth radii)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500,
            xaxis=dict(tickangle=45)
        )
        
        return fig
    
    def create_orbital_period_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create orbital period distribution histogram"""
        
        try:
            # Try different column names for orbital period
            period_cols = ['pl_orbper', 'koi_period', 'orbital_period']
            period_col = None
            
            for col in period_cols:
                if col in df.columns:
                    period_col = col
                    break
            
            if period_col is None:
                return self._create_empty_chart(f"Missing orbital period column. Available: {list(df.columns)[:10]}...")
            
            # Filter data
            periods = df[period_col].dropna()
            
            if periods.empty:
                return self._create_empty_chart("No orbital period data available")
        except Exception as e:
            return self._create_empty_chart(f"Error processing orbital period data: {str(e)}")
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=periods,
            nbinsx=30,
            marker_color=self.color_scheme['primary'],
            opacity=0.7,
            name='Orbital Periods'
        ))
        
        # Add median line
        median_period = periods.median()
        fig.add_vline(
            x=median_period,
            line_dash="dash",
            line_color=self.color_scheme['accent'],
            annotation_text=f"Median: {median_period:.1f} days"
        )
        
        fig.update_layout(
            title="Distribution of Orbital Periods",
            xaxis_title="Orbital Period (days)",
            yaxis_title="Number of Planets",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def create_habitable_zone_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create habitable zone visualization"""
        
        try:
            # Try different column names for stellar data
            temp_cols = ['st_teff', 'koi_steff', 'stellar_teff']
            rad_cols = ['st_rad', 'koi_srad', 'stellar_rad']
            
            temp_col = None
            rad_col = None
            
            for col in temp_cols:
                if col in df.columns:
                    temp_col = col
                    break
            
            for col in rad_cols:
                if col in df.columns:
                    rad_col = col
                    break
            
            if temp_col is None or rad_col is None:
                return self._create_empty_chart(f"Missing stellar data columns. Available: {list(df.columns)[:10]}...")
            
            # Calculate habitable zone for each star
            stellar_temp = df[temp_col].dropna()
            stellar_rad = df[rad_col].dropna()
            
            if stellar_temp.empty or stellar_rad.empty:
                return self._create_empty_chart("No stellar data available")
        except Exception as e:
            return self._create_empty_chart(f"Error processing stellar data: {str(e)}")
        
        # Calculate luminosity and habitable zone
        luminosity = (stellar_rad ** 2) * ((stellar_temp / 5778) ** 4)
        hz_inner = 0.95 * np.sqrt(luminosity)
        hz_outer = 1.37 * np.sqrt(luminosity)
        
        # Get planet data - try different column names
        name_cols = ['kepoi_name', 'kepler_name', 'pl_name']
        orbit_cols = ['pl_orbsmax', 'koi_period']  # Use period as proxy for distance
        radius_cols = ['pl_rade', 'koi_prad']
        
        name_col = None
        orbit_col = None
        radius_col = None
        
        for col in name_cols:
            if col in df.columns:
                name_col = col
                break
        
        for col in orbit_cols:
            if col in df.columns:
                orbit_col = col
                break
        
        for col in radius_cols:
            if col in df.columns:
                radius_col = col
                break
        
        if not name_col or not orbit_col or not radius_col:
            return self._create_empty_chart(f"Missing planet data columns. Available: {list(df.columns)[:10]}...")
        
        planet_data = df[[name_col, orbit_col, radius_col]].dropna()
        
        if planet_data.empty:
            return self._create_empty_chart("No planet data available")
        
        fig = go.Figure()
        
        # Add habitable zone for each star (simplified)
        if len(hz_inner) > 0:
            fig.add_trace(go.Scatter(
                x=[hz_inner.iloc[0], hz_outer.iloc[0], hz_outer.iloc[0], hz_inner.iloc[0], hz_inner.iloc[0]],
                y=[0, 0, 1, 1, 0],
                fill='toself',
                fillcolor='rgba(0, 255, 0, 0.2)',
                line=dict(color='rgba(0, 255, 0, 0.8)', width=2),
                name='Habitable Zone',
                hoverinfo='skip'
            ))
        
        # Add planets (show first 10 for clarity)
        for idx, planet in planet_data.head(10).iterrows():
            # Use orbital period as proxy for distance (simplified)
            period = planet[orbit_col]
            semi_major = (period / 365.25) ** (2/3)  # Rough conversion from period to AU
            planet_name = planet[name_col]
            radius = planet[radius_col]
            
            # Determine if in habitable zone (simplified)
            in_hz = len(hz_inner) > 0 and hz_inner.iloc[0] <= semi_major <= hz_outer.iloc[0]
            color = self.color_scheme['success'] if in_hz else self.color_scheme['primary']
            
            fig.add_trace(go.Scatter(
                x=[semi_major],
                y=[0.5],
                mode='markers',
                marker=dict(
                    size=max(8, min(20, radius * 3)),
                    color=color,
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                name=planet_name,
                hovertemplate=f'<b>{planet_name}</b><br>Distance: {semi_major:.2f} AU<br>Radius: {radius:.2f} R⊕<br>In HZ: {"Yes" if in_hz else "No"}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Habitable Zone Analysis",
            xaxis_title="Semi-Major Axis (AU)",
            yaxis_title="",
            yaxis=dict(showticklabels=False, range=[-0.1, 1.1]),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Create horizontal bar chart for feature importance"""
        
        if not feature_importance:
            return self._create_empty_chart("No feature importance data available")
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=features,
            x=importance,
            orientation='h',
            marker_color=self.color_scheme['primary'],
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=max(300, len(features) * 25)
        )
        
        return fig
    
    def create_model_performance_gauge(self, metrics: Dict[str, float]) -> go.Figure:
        """Create gauge chart for model performance"""
        
        accuracy = metrics.get('accuracy', 0) * 100
        precision = metrics.get('precision', 0) * 100
        recall = metrics.get('recall', 0) * 100
        f1 = metrics.get('f1_score', 0) * 100
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        # Accuracy gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=accuracy,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['primary']},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ), row=1, col=1)
        
        # Precision gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=precision,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['secondary']},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ), row=1, col=2)
        
        # Recall gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=recall,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['accent']},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ), row=2, col=1)
        
        # F1-Score gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=f1,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['success']},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ), row=2, col=2)
        
        fig.update_layout(
            title="Model Performance Metrics",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap"""
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return self._create_empty_chart("Not enough numeric columns for correlation")
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Correlation Matrix",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )
        
        return fig
    
    def create_light_curve_simulation(self, period: float = 10, depth: float = 0.01, 
                                    duration: float = 2, noise: float = 0.001) -> go.Figure:
        """Create simulated light curve"""
        
        # Generate time series
        time = np.linspace(0, period * 3, 1000)
        
        # Create transit signal
        transit_center = period / 2
        transit_width = duration / 24  # Convert hours to days
        
        light_curve = np.ones_like(time)
        transit_mask = (time >= transit_center - transit_width/2) & (time <= transit_center + transit_width/2)
        light_curve[transit_mask] = 1 - depth
        
        # Add noise
        light_curve += np.random.normal(0, noise, len(time))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=time,
            y=light_curve,
            mode='lines',
            line=dict(color=self.color_scheme['primary'], width=2),
            name='Light Curve',
            hovertemplate='Time: %{x:.2f} days<br>Flux: %{y:.4f}<extra></extra>'
        ))
        
        # Highlight transit
        fig.add_vrect(
            x0=transit_center - transit_width/2,
            x1=transit_center + transit_width/2,
            fillcolor="red",
            opacity=0.2,
            annotation_text="Transit",
            annotation_position="top"
        )
        
        fig.update_layout(
            title="Simulated Light Curve",
            xaxis_title="Time (days)",
            yaxis_title="Relative Flux",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="white")
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300
        )
        return fig


def test_simple_visualizations():
    """Test the simple visualization functions"""
    print("🎨 Testing Simple Visualizations")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'pl_name': ['Kepler-1b', 'Kepler-2b', 'Kepler-3b', 'Kepler-4b', 'Kepler-5b'],
        'pl_orbper': [3.5, 10.2, 15.8, 2.1, 8.9],
        'pl_rade': [1.2, 2.1, 0.8, 1.5, 1.9],
        'pl_bmasse': [5.2, 12.1, 2.3, 8.7, 15.3],
        'pl_orbsmax': [0.05, 0.1, 0.15, 0.03, 0.08],
        'st_teff': [5500, 6000, 5000, 5800, 6200],
        'st_rad': [1.0, 1.2, 0.9, 1.1, 1.3]
    })
    
    visualizer = ExoplanetVisualizer()
    
    # Test planet size comparison
    print("Creating planet size comparison...")
    fig_size = visualizer.create_planet_size_comparison(sample_data)
    print("✅ Planet size comparison created")
    
    # Test orbital period distribution
    print("Creating orbital period distribution...")
    fig_period = visualizer.create_orbital_period_distribution(sample_data)
    print("✅ Orbital period distribution created")
    
    # Test habitable zone plot
    print("Creating habitable zone plot...")
    fig_hz = visualizer.create_habitable_zone_plot(sample_data)
    print("✅ Habitable zone plot created")
    
    # Test feature importance chart
    print("Creating feature importance chart...")
    feature_importance = {
        'pl_orbper': 0.3,
        'pl_rade': 0.25,
        'pl_bmasse': 0.2,
        'st_teff': 0.15,
        'st_rad': 0.1
    }
    fig_importance = visualizer.create_feature_importance_chart(feature_importance)
    print("✅ Feature importance chart created")
    
    # Test model performance gauge
    print("Creating model performance gauge...")
    metrics = {
        'accuracy': 0.85,
        'precision': 0.82,
        'recall': 0.88,
        'f1_score': 0.85
    }
    fig_performance = visualizer.create_model_performance_gauge(metrics)
    print("✅ Model performance gauge created")
    
    # Test correlation heatmap
    print("Creating correlation heatmap...")
    fig_corr = visualizer.create_correlation_heatmap(sample_data)
    print("✅ Correlation heatmap created")
    
    # Test light curve simulation
    print("Creating light curve simulation...")
    fig_lc = visualizer.create_light_curve_simulation()
    print("✅ Light curve simulation created")
    
    print("\n🎉 All simple visualizations created successfully!")
    print("\n📊 Available Visualizations:")
    print("1. 🌍 Planet Size Comparison - Interactive bubble chart")
    print("2. 📈 Orbital Period Distribution - Histogram with statistics")
    print("3. 🌟 Habitable Zone Plot - Visual analysis of planet positions")
    print("4. 📊 Feature Importance Chart - Horizontal bar chart")
    print("5. 🎯 Model Performance Gauge - Multi-metric dashboard")
    print("6. 🔥 Correlation Heatmap - Feature relationship analysis")
    print("7. 📡 Light Curve Simulation - Transit signal visualization")


if __name__ == "__main__":
    test_simple_visualizations()
