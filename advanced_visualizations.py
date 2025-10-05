"""
Advanced Visualizations for ExoVision AI
Interactive 3D plots, animations, and advanced charts
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json
from datetime import datetime

class ExoplanetVisualizer:
    """Advanced visualization tools for exoplanet data"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#00f5ff',
            'secondary': '#8b5cf6', 
            'accent': '#f472b6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444'
        }
    
    def create_3d_system_visualization(self, df: pd.DataFrame, 
                                     star_name: str = None) -> go.Figure:
        """Create 3D visualization of exoplanet system"""
        
        if star_name and star_name in df['hostname'].values:
            system_data = df[df['hostname'] == star_name]
        else:
            system_data = df.head(5)  # Show first 5 planets
        
        fig = go.Figure()
        
        # Add star at center
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0],
            mode='markers',
            marker=dict(
                size=20,
                color=self.color_scheme['warning'],
                symbol='circle'
            ),
            name='Star',
            hovertemplate='<b>%{text}</b><br>Temperature: %{customdata[0]}K<br>Radius: %{customdata[1]}R☉',
            text=[system_data['hostname'].iloc[0] if not system_data.empty else 'Star'],
            customdata=[[system_data['st_teff'].iloc[0], system_data['st_rad'].iloc[0]] if not system_data.empty else [0, 0]]
        ))
        
        # Add planets
        for idx, planet in system_data.iterrows():
            # Calculate orbital position (simplified)
            period = planet.get('pl_orbper', 365)  # days
            semi_major = planet.get('pl_orbsmax', 1)  # AU
            
            # Create orbital path
            theta = np.linspace(0, 2*np.pi, 100)
            x_orbit = semi_major * np.cos(theta)
            y_orbit = semi_major * np.sin(theta)
            z_orbit = np.zeros_like(theta)
            
            # Add orbital path
            fig.add_trace(go.Scatter3d(
                x=x_orbit, y=y_orbit, z=z_orbit,
                mode='lines',
                line=dict(color=self.color_scheme['primary'], width=1, dash='dot'),
                name=f"{planet.get('pl_name', 'Planet')} Orbit",
                showlegend=False
            ))
            
            # Add planet
            planet_size = max(5, min(20, planet.get('pl_rade', 1) * 5))
            planet_color = self._get_planet_color(planet)
            
            fig.add_trace(go.Scatter3d(
                x=[semi_major], y=[0], z=[0],
                mode='markers',
                marker=dict(
                    size=planet_size,
                    color=planet_color,
                    opacity=0.8
                ),
                name=planet.get('pl_name', f'Planet {idx}'),
                hovertemplate='<b>%{text}</b><br>Period: %{customdata[0]} days<br>Radius: %{customdata[1]}R⊕<br>Mass: %{customdata[2]}M⊕',
                text=[planet.get('pl_name', f'Planet {idx}')],
                customdata=[[period, planet.get('pl_rade', 0), planet.get('pl_bmasse', 0)]]
            ))
        
        # Update layout
        fig.update_layout(
            title=f"3D Exoplanet System: {star_name or 'Sample System'}",
            scene=dict(
                xaxis_title="Distance (AU)",
                yaxis_title="Distance (AU)", 
                zaxis_title="Distance (AU)",
                bgcolor='rgba(0,0,0,0)',
                xaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                yaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)'),
                zaxis=dict(backgroundcolor='rgba(0,0,0,0)', gridcolor='rgba(255,255,255,0.1)')
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            width=800,
            height=600
        )
        
        return fig
    
    def create_habitable_zone_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create habitable zone visualization"""
        
        # Calculate habitable zone boundaries
        stellar_temp = df['st_teff'].median()
        stellar_lum = (df['st_rad'] ** 2) * ((df['st_teff'] / 5778) ** 4)
        
        # Conservative habitable zone (0.95-1.37 AU for Sun-like star)
        hz_inner = 0.95 * np.sqrt(stellar_lum)
        hz_outer = 1.37 * np.sqrt(stellar_lum)
        
        fig = go.Figure()
        
        # Add habitable zone
        fig.add_trace(go.Scatter(
            x=[hz_inner, hz_outer, hz_outer, hz_inner, hz_inner],
            y=[0, 0, 1, 1, 0],
            fill='toself',
            fillcolor='rgba(0, 255, 0, 0.3)',
            line=dict(color='rgba(0, 255, 0, 0.8)', width=2),
            name='Habitable Zone',
            hovertemplate='Habitable Zone<br>Inner: %{x:.2f} AU<br>Outer: %{x:.2f} AU'
        ))
        
        # Add planets
        for idx, planet in df.iterrows():
            semi_major = planet.get('pl_orbsmax', 1)
            planet_name = planet.get('pl_name', f'Planet {idx}')
            
            # Determine if in habitable zone
            in_hz = hz_inner <= semi_major <= hz_outer
            color = self.color_scheme['success'] if in_hz else self.color_scheme['primary']
            
            fig.add_trace(go.Scatter(
                x=[semi_major],
                y=[0.5],
                mode='markers',
                marker=dict(
                    size=15,
                    color=color,
                    symbol='circle'
                ),
                name=planet_name,
                hovertemplate=f'<b>{planet_name}</b><br>Distance: {semi_major:.2f} AU<br>In HZ: {"Yes" if in_hz else "No"}'
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
    
    def create_light_curve_animation(self, period: float = 10, depth: float = 0.01, 
                                   duration: float = 2, noise: float = 0.001) -> go.Figure:
        """Create animated light curve"""
        
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
        
        # Create animation frames
        frames = []
        for i in range(0, len(time), 10):
            frames.append(go.Frame(
                data=[go.Scatter(
                    x=time[:i+1],
                    y=light_curve[:i+1],
                    mode='lines',
                    line=dict(color=self.color_scheme['primary'], width=2)
                )],
                name=f"frame_{i}"
            ))
        
        fig = go.Figure(
            data=[go.Scatter(
                x=time,
                y=light_curve,
                mode='lines',
                line=dict(color=self.color_scheme['primary'], width=2),
                name='Light Curve'
            )],
            frames=frames
        )
        
        # Add buttons
        fig.update_layout(
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}}]
                    }
                ]
            }],
            title="Animated Light Curve",
            xaxis_title="Time (days)",
            yaxis_title="Relative Flux",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )
        
        return fig
    
    def create_feature_importance_heatmap(self, feature_importance: Dict[str, float]) -> go.Figure:
        """Create feature importance heatmap"""
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_features)
        
        # Create heatmap data
        heatmap_data = np.array(importance).reshape(1, -1)
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data,
            x=features,
            y=['Importance'],
            colorscale='Viridis',
            showscale=True,
            hovertemplate='Feature: %{x}<br>Importance: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Feature Importance Heatmap",
            xaxis_title="Features",
            yaxis_title="",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=300
        )
        
        return fig
    
    def create_model_performance_dashboard(self, metrics: Dict[str, Any]) -> go.Figure:
        """Create model performance dashboard"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision vs Recall', 'Confusion Matrix', 'ROC Curve'),
            specs=[[{"type": "indicator"}, {"type": "scatter"}],
                   [{"type": "heatmap"}, {"type": "scatter"}]]
        )
        
        # Accuracy gauge
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=metrics.get('accuracy', 0) * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Accuracy (%)"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': self.color_scheme['primary']},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "darkgray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ), row=1, col=1)
        
        # Precision vs Recall scatter
        precision = metrics.get('precision', 0)
        recall = metrics.get('recall', 0)
        fig.add_trace(go.Scatter(
            x=[recall], y=[precision],
            mode='markers',
            marker=dict(size=20, color=self.color_scheme['accent']),
            name='Model Performance'
        ), row=1, col=2)
        
        # Confusion Matrix
        cm = metrics.get('confusion_matrix', [[0, 0], [0, 0]])
        fig.add_trace(go.Heatmap(
            z=cm,
            x=['Predicted No', 'Predicted Yes'],
            y=['Actual No', 'Actual Yes'],
            colorscale='Blues',
            showscale=False
        ), row=2, col=1)
        
        # ROC Curve (simplified)
        fpr = np.linspace(0, 1, 100)
        tpr = np.linspace(0, 1, 100)  # Simplified - would use actual ROC data
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            line=dict(color=self.color_scheme['success'], width=3),
            name='ROC Curve'
        ), row=2, col=2)
        
        fig.update_layout(
            title="Model Performance Dashboard",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )
        
        return fig
    
    def _get_planet_color(self, planet: pd.Series) -> str:
        """Get color based on planet properties"""
        radius = planet.get('pl_rade', 1)
        mass = planet.get('pl_bmasse', 1)
        
        # Color based on size
        if radius < 1.5:
            return self.color_scheme['primary']  # Small planets
        elif radius < 4:
            return self.color_scheme['secondary']  # Medium planets
        else:
            return self.color_scheme['accent']  # Large planets
    
    def create_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        """Create correlation matrix heatmap"""
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
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


def test_visualizations():
    """Test the visualization functions"""
    print("🎨 Testing Advanced Visualizations")
    print("=" * 50)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'pl_name': ['Kepler-1b', 'Kepler-2b', 'Kepler-3b'],
        'hostname': ['Kepler-1', 'Kepler-2', 'Kepler-3'],
        'pl_orbper': [3.5, 10.2, 15.8],
        'pl_rade': [1.2, 2.1, 0.8],
        'pl_bmasse': [5.2, 12.1, 2.3],
        'pl_orbsmax': [0.05, 0.1, 0.15],
        'st_teff': [5500, 6000, 5000],
        'st_rad': [1.0, 1.2, 0.9]
    })
    
    visualizer = ExoplanetVisualizer()
    
    # Test 3D system visualization
    print("Creating 3D system visualization...")
    fig_3d = visualizer.create_3d_system_visualization(sample_data)
    print("✅ 3D visualization created")
    
    # Test habitable zone plot
    print("Creating habitable zone plot...")
    fig_hz = visualizer.create_habitable_zone_plot(sample_data)
    print("✅ Habitable zone plot created")
    
    # Test light curve animation
    print("Creating light curve animation...")
    fig_lc = visualizer.create_light_curve_animation()
    print("✅ Light curve animation created")
    
    # Test feature importance heatmap
    print("Creating feature importance heatmap...")
    feature_importance = {
        'pl_orbper': 0.3,
        'pl_rade': 0.25,
        'pl_bmasse': 0.2,
        'st_teff': 0.15,
        'st_rad': 0.1
    }
    fig_heatmap = visualizer.create_feature_importance_heatmap(feature_importance)
    print("✅ Feature importance heatmap created")
    
    print("\n🎉 All visualizations created successfully!")


if __name__ == "__main__":
    test_visualizations()
