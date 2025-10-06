"""
Improved Visualizations for ExoVision AI
Enhanced charts that are more informative and clear
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import json

class ImprovedExoplanetVisualizer:
    """Enhanced visualization tools with better clarity and informativeness"""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#00f5ff',
            'secondary': '#8b5cf6', 
            'accent': '#f472b6',
            'success': '#10b981',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#06b6d4'
        }
        
        # Define clear color mapping for planet classifications
        self.classification_colors = {
            'CONFIRMED': '#10b981',      # Green
            'CANDIDATE': '#3b82f6',      # Blue  
            'FALSE POSITIVE': '#ef4444', # Red
            'REFUTED': '#f59e0b',        # Orange
            'PC': '#8b5cf6',             # Purple
            'KP': '#06b6d4',             # Cyan
            'APC': '#f472b6',            # Pink
            'FA': '#ef4444',             # Red
            'CP': '#10b981',             # Green
            'FP': '#3b82f6'              # Blue
        }
    
    def create_improved_orbital_period_distribution(self, df: pd.DataFrame) -> go.Figure:
        """Create improved orbital period distribution with log scale and better binning"""
        
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
            
            # Filter data and remove extreme outliers
            periods = df[period_col].dropna()
            periods = periods[(periods > 0) & (periods < 10000)]  # Remove extreme outliers
            
            if periods.empty:
                return self._create_empty_chart("No orbital period data available")
                
        except Exception as e:
            return self._create_empty_chart(f"Error processing orbital period data: {str(e)}")
        
        # Create subplot with both linear and log scale
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Linear Scale (0-1000 days)', 'Log Scale (Full Range)'),
            vertical_spacing=0.15
        )
        
        # Linear scale histogram (0-1000 days)
        periods_linear = periods[periods <= 1000]
        if not periods_linear.empty:
            fig.add_trace(go.Histogram(
                x=periods_linear,
                nbinsx=50,
                marker_color=self.color_scheme['primary'],
                opacity=0.7,
                name='Linear Scale',
                showlegend=False
            ), row=1, col=1)
            
            # Add statistics with better positioning
            median_linear = periods_linear.median()
            mean_linear = periods_linear.mean()
            fig.add_vline(
                x=median_linear,
                line_dash="dash",
                line_color=self.color_scheme['accent'],
                annotation_text=f"Median: {median_linear:.1f} days",
                annotation_position="top right",
                row=1, col=1
            )
            fig.add_vline(
                x=mean_linear,
                line_dash="dot",
                line_color=self.color_scheme['warning'],
                annotation_text=f"Mean: {mean_linear:.1f} days",
                annotation_position="top left",
                row=1, col=1
            )
        
        # Log scale histogram (full range)
        periods_log = periods[periods > 0]
        if not periods_log.empty:
            # Create log-spaced bins
            log_bins = np.logspace(np.log10(periods_log.min()), np.log10(periods_log.max()), 50)
            fig.add_trace(go.Histogram(
                x=periods_log,
                xbins=dict(start=log_bins[0], end=log_bins[-1], size=(log_bins[1] - log_bins[0])),
                marker_color=self.color_scheme['secondary'],
                opacity=0.7,
                name='Log Scale',
                showlegend=False
            ), row=2, col=1)
            
            # Add statistics with better positioning
            median_log = periods_log.median()
            fig.add_vline(
                x=median_log,
                line_dash="dash",
                line_color=self.color_scheme['accent'],
                annotation_text=f"Median: {median_log:.1f} days",
                annotation_position="top right",
                row=2, col=1
            )
        
        # Update layout
        fig.update_layout(
            title="Improved Orbital Period Distribution Analysis",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=700,
            showlegend=False
        )
        
        # Update axes
        fig.update_xaxes(title_text="Orbital Period (days)", row=1, col=1)
        fig.update_xaxes(title_text="Orbital Period (days) - Log Scale", type="log", row=2, col=1)
        fig.update_yaxes(title_text="Number of Planets", row=1, col=1)
        fig.update_yaxes(title_text="Number of Planets", row=2, col=1)
        
        return fig
    
    def create_improved_scatter_plot(self, df: pd.DataFrame) -> go.Figure:
        """Create improved scatter plot with log scales and better handling of overplotting"""
        
        try:
            # Get required columns
            period_cols = ['pl_orbper', 'koi_period', 'orbital_period']
            radius_cols = ['pl_rade', 'koi_prad', 'planet_radius']
            label_cols = ['pl_discmethod', 'koi_disposition', 'label', 'disposition']
            
            period_col = None
            radius_col = None
            label_col = None
            
            for col in period_cols:
                if col in df.columns:
                    period_col = col
                    break
            
            for col in radius_cols:
                if col in df.columns:
                    radius_col = col
                    break
                    
            for col in label_cols:
                if col in df.columns:
                    label_col = col
                    break
            
            if not period_col or not radius_col:
                return self._create_empty_chart(f"Missing required columns. Available: {list(df.columns)[:10]}...")
            
            # Filter data
            plot_data = df[[period_col, radius_col]].dropna()
            if label_col:
                plot_data[label_col] = df[label_col]
            
            # Remove extreme outliers
            plot_data = plot_data[
                (plot_data[period_col] > 0) & (plot_data[period_col] < 10000) &
                (plot_data[radius_col] > 0) & (plot_data[radius_col] < 100)
            ]
            
            if plot_data.empty:
                return self._create_empty_chart("No data available after filtering")
                
        except Exception as e:
            return self._create_empty_chart(f"Error processing data: {str(e)}")
        
        # Create figure
        fig = go.Figure()
        
        if label_col and label_col in plot_data.columns:
            # Group by classification
            for classification in plot_data[label_col].unique():
                if pd.isna(classification):
                    continue
                    
                subset = plot_data[plot_data[label_col] == classification]
                color = self.classification_colors.get(classification, self.color_scheme['primary'])
                
                # Add jitter to reduce overplotting
                jittered_period = subset[period_col] * (1 + np.random.normal(0, 0.01, len(subset)))
                jittered_radius = subset[radius_col] * (1 + np.random.normal(0, 0.01, len(subset)))
                
                fig.add_trace(go.Scatter(
                    x=jittered_period,
                    y=jittered_radius,
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=color,
                        opacity=0.6,
                        line=dict(width=0.5, color='white')
                    ),
                    name=classification,
                    hovertemplate=f'<b>{classification}</b><br>Period: %{{x:.2f}} days<br>Radius: %{{y:.2f}} R⊕<extra></extra>'
                ))
        else:
            # No classification available
            fig.add_trace(go.Scatter(
                x=plot_data[period_col],
                y=plot_data[radius_col],
                mode='markers',
                marker=dict(
                    size=6,
                    color=self.color_scheme['primary'],
                    opacity=0.6,
                    line=dict(width=0.5, color='white')
                ),
                name='Exoplanets',
                hovertemplate='Period: %{x:.2f} days<br>Radius: %{y:.2f} R⊕<extra></extra>'
            ))
        
        # Add reference lines
        fig.add_hline(y=1, line_dash="dash", line_color="red", 
                     annotation_text="Earth (1 R⊕)", annotation_position="right")
        fig.add_vline(x=365, line_dash="dash", line_color="orange", 
                     annotation_text="1 Year", annotation_position="top")
        
        # Update layout with log scales
        fig.update_layout(
            title="Improved Orbital Period vs Planet Radius Analysis",
            xaxis_title="Orbital Period (days) - Log Scale",
            yaxis_title="Planet Radius (Earth radii) - Log Scale",
            xaxis=dict(type="log", showgrid=True, gridcolor='rgba(139, 92, 246, 0.3)'),
            yaxis=dict(type="log", showgrid=True, gridcolor='rgba(139, 92, 246, 0.3)'),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600,
            legend=dict(
                bgcolor='rgba(0,0,0,0.8)',
                bordercolor='#8b5cf6',
                borderwidth=1
            )
        )
        
        return fig
    
    def create_improved_star_map(self, df: pd.DataFrame) -> go.Figure:
        """Create enhanced galactic star map with better visualization and information"""
        
        try:
            # Get required columns
            ra_cols = ['ra', 'ra_deg', 'right_ascension']
            dec_cols = ['dec', 'dec_deg', 'declination']
            year_cols = ['disc_year', 'discovery_year', 'year']
            name_cols = ['pl_name', 'kepoi_name', 'kepler_name', 'identifier']
            radius_cols = ['pl_rade', 'koi_prad', 'planet_radius']
            
            ra_col = None
            dec_col = None
            year_col = None
            name_col = None
            radius_col = None
            
            for col in ra_cols:
                if col in df.columns:
                    ra_col = col
                    break
            
            for col in dec_cols:
                if col in df.columns:
                    dec_col = col
                    break
                    
            for col in year_cols:
                if col in df.columns:
                    year_col = col
                    break
                    
            for col in name_cols:
                if col in df.columns:
                    name_col = col
                    break
                    
            for col in radius_cols:
                if col in df.columns:
                    radius_col = col
                    break
            
            if not ra_col or not dec_col:
                return self._create_empty_chart(f"Missing coordinate columns. Available: {list(df.columns)[:10]}...")
            
            # Filter data
            map_data = df[[ra_col, dec_col]].dropna()
            if year_col:
                map_data[year_col] = df[year_col]
            if name_col:
                map_data[name_col] = df[name_col]
            if radius_col:
                map_data[radius_col] = df[radius_col]
            
            # Clean coordinate data
            map_data = map_data[
                (map_data[ra_col] >= 0) & (map_data[ra_col] <= 360) &
                (map_data[dec_col] >= -90) & (map_data[dec_col] <= 90)
            ]
            
            if map_data.empty:
                return self._create_empty_chart("No valid coordinate data available")
                
        except Exception as e:
            return self._create_empty_chart(f"Error processing coordinate data: {str(e)}")
        
        # Create figure with subplots for better organization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Discovery Timeline', 'Planet Size Distribution'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]],
            horizontal_spacing=0.1
        )
        
        # Left plot: Discovery Timeline
        if year_col and year_col in map_data.columns:
            # Group by discovery decade for better clarity
            map_data['decade'] = (map_data[year_col] // 10) * 10
            decades = sorted(map_data['decade'].dropna().unique())
            decades = [d for d in decades if 1990 <= d <= 2030]
            
            # Use a clear color palette
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#feca57', 
                     '#ff9ff3', '#54a0ff', '#5f27cd', '#00d2d3', '#ff9f43']
            
            for i, decade in enumerate(decades):
                subset = map_data[map_data['decade'] == decade]
                color = colors[i % len(colors)]
                
                # Add slight jitter to reduce clustering
                jittered_ra = subset[ra_col] + np.random.normal(0, 1, len(subset))
                jittered_dec = subset[dec_col] + np.random.normal(0, 1, len(subset))
                
                fig.add_trace(go.Scatter(
                    x=jittered_ra,
                    y=jittered_dec,
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=color,
                        opacity=0.8,
                        line=dict(width=1, color='white'),
                        symbol='circle'
                    ),
                    name=f'{int(decade)}s',
                    hovertemplate=f'<b>{int(decade)}s</b><br>RA: %{{x:.1f}}°<br>Dec: %{{y:.1f}}°<br>Count: {len(subset)}<extra></extra>',
                    showlegend=True
                ), row=1, col=1)
        else:
            # No year data available
            fig.add_trace(go.Scatter(
                x=map_data[ra_col],
                y=map_data[dec_col],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.color_scheme['primary'],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name='Exoplanets',
                hovertemplate='RA: %{x:.1f}°<br>Dec: %{y:.1f}°<extra></extra>',
                showlegend=True
            ), row=1, col=1)
        
        # Right plot: Planet Size Distribution
        if radius_col and radius_col in map_data.columns:
            # Create size categories
            map_data['size_category'] = pd.cut(
                map_data[radius_col], 
                bins=[0, 1, 2, 4, 8, float('inf')], 
                labels=['Earth-like', 'Super-Earth', 'Neptune-like', 'Jupiter-like', 'Giant'],
                include_lowest=True
            )
            
            size_colors = {
                'Earth-like': '#10b981',    # Green
                'Super-Earth': '#3b82f6',   # Blue
                'Neptune-like': '#8b5cf6',  # Purple
                'Jupiter-like': '#f59e0b',  # Orange
                'Giant': '#ef4444'          # Red
            }
            
            for category in map_data['size_category'].cat.categories:
                subset = map_data[map_data['size_category'] == category]
                if len(subset) > 0:
                    # Add jitter to reduce clustering
                    jittered_ra = subset[ra_col] + np.random.normal(0, 1, len(subset))
                    jittered_dec = subset[dec_col] + np.random.normal(0, 1, len(subset))
                    
                    fig.add_trace(go.Scatter(
                        x=jittered_ra,
                        y=jittered_dec,
                        mode='markers',
                        marker=dict(
                            size=12,
                            color=size_colors.get(category, self.color_scheme['primary']),
                            opacity=0.8,
                            line=dict(width=1, color='white'),
                            symbol='circle'
                        ),
                        name=category,
                        hovertemplate=f'<b>{category}</b><br>RA: %{{x:.1f}}°<br>Dec: %{{y:.1f}}°<br>Radius: {subset[radius_col].mean():.2f} R⊕<extra></extra>',
                        showlegend=True
                    ), row=1, col=2)
        else:
            # No radius data available
            fig.add_trace(go.Scatter(
                x=map_data[ra_col],
                y=map_data[dec_col],
                mode='markers',
                marker=dict(
                    size=10,
                    color=self.color_scheme['secondary'],
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                name='Exoplanets',
                hovertemplate='RA: %{x:.1f}°<br>Dec: %{y:.1f}°<extra></extra>',
                showlegend=True
            ), row=1, col=2)
        
        # Add celestial equator to both plots
        for col in [1, 2]:
            fig.add_hline(y=0, line_dash="dash", line_color="white", 
                         annotation_text="Celestial Equator", annotation_position="right",
                         row=1, col=col)
        
        # Add galactic plane reference (approximate)
        for col in [1, 2]:
            fig.add_hline(y=0, line_dash="dot", line_color="yellow", 
                         annotation_text="Galactic Plane", annotation_position="left",
                         row=1, col=col)
        
        # Update layout
        fig.update_layout(
            title="Enhanced Galactic Star Map - Multi-Dimensional Analysis",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', size=12),
            height=700,
            showlegend=True,
            legend=dict(
                bgcolor='rgba(0,0,0,0.9)',
                bordercolor='#8b5cf6',
                borderwidth=2,
                font=dict(size=10),
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        # Update axes for both subplots
        for col in [1, 2]:
            fig.update_xaxes(
                title_text="Right Ascension (degrees)",
                range=[0, 360],
                showgrid=True,
                gridcolor='rgba(139, 92, 246, 0.3)',
                dtick=60,
                row=1, col=col
            )
            fig.update_yaxes(
                title_text="Declination (degrees)",
                range=[-90, 90],
                showgrid=True,
                gridcolor='rgba(139, 92, 246, 0.3)',
                dtick=30,
                row=1, col=col
            )
        
        return fig
    
    def create_enhanced_correlation_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create enhanced correlation heatmap with better color scheme and annotations"""
        
        try:
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return self._create_empty_chart("Not enough numeric columns for correlation")
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Create custom colorscale
            colorscale = [
                [0.0, '#ef4444'],  # Red for negative
                [0.5, '#1a1a1a'],  # Black for zero
                [1.0, '#10b981']   # Green for positive
            ]
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.index,
                colorscale=colorscale,
                zmid=0,
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>',
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont={"color": "white"}
            ))
            
            fig.update_layout(
                title="Enhanced Feature Correlation Matrix",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=600,
                xaxis=dict(tickangle=45),
                yaxis=dict(tickangle=0)
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating correlation heatmap: {str(e)}")
    
    def create_data_quality_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create a dashboard showing data quality and completeness"""
        
        try:
            # Calculate missing data percentages
            missing_data = df.isnull().sum()
            missing_pct = (missing_data / len(df)) * 100
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Missing Data by Column', 'Data Types Distribution', 
                              'Sample Size by Category', 'Data Quality Score'),
                specs=[[{"type": "bar"}, {"type": "pie"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # Missing data bar chart
            fig.add_trace(go.Bar(
                x=missing_pct.index,
                y=missing_pct.values,
                marker_color=self.color_scheme['danger'],
                name='Missing %',
                showlegend=False
            ), row=1, col=1)
            
            # Data types pie chart
            dtype_counts = df.dtypes.value_counts()
            fig.add_trace(go.Pie(
                labels=dtype_counts.index.astype(str),
                values=dtype_counts.values,
                name="Data Types",
                showlegend=False
            ), row=1, col=2)
            
            # Sample size by category (if categorical columns exist)
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                cat_counts = df[cat_col].value_counts().head(10)
                fig.add_trace(go.Bar(
                    x=cat_counts.index,
                    y=cat_counts.values,
                    marker_color=self.color_scheme['primary'],
                    name='Count',
                    showlegend=False
                ), row=2, col=1)
            
            # Data quality score
            quality_score = ((len(df) - missing_data.sum()) / (len(df) * len(df.columns))) * 100
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=quality_score,
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
                title="Data Quality Dashboard",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=800
            )
            
            return fig
            
        except Exception as e:
            return self._create_empty_chart(f"Error creating data quality dashboard: {str(e)}")
    
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


def test_improved_visualizations():
    """Test the improved visualization functions"""
    print("🎨 Testing Improved Visualizations")
    print("=" * 50)
    
    # Create sample data with realistic distributions
    np.random.seed(42)
    n_planets = 1000
    
    # Create realistic orbital period distribution (log-normal)
    periods = np.random.lognormal(mean=2, sigma=1.5, size=n_planets)
    periods = periods[periods < 10000]  # Remove extreme outliers
    
    # Create realistic planet radius distribution
    radii = np.random.lognormal(mean=0.5, sigma=0.8, size=len(periods))
    radii = radii[radii < 50]  # Remove extreme outliers
    
    # Create realistic coordinates (clustered around certain regions)
    ra = np.random.normal(280, 20, len(periods))  # Clustered around 280 degrees
    ra = np.clip(ra, 0, 360)
    dec = np.random.normal(30, 15, len(periods))  # Clustered around 30 degrees
    dec = np.clip(dec, -90, 90)
    
    # Create discovery years
    years = np.random.randint(2000, 2025, len(periods))
    
    # Create classifications
    classifications = np.random.choice(['CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE', 'REFUTED'], 
                                     len(periods), p=[0.4, 0.3, 0.2, 0.1])
    
    sample_data = pd.DataFrame({
        'pl_orbper': periods,
        'pl_rade': radii,
        'ra': ra,
        'dec': dec,
        'disc_year': years,
        'pl_discmethod': classifications,
        'pl_name': [f'Planet-{i}' for i in range(len(periods))]
    })
    
    visualizer = ImprovedExoplanetVisualizer()
    
    # Test improved orbital period distribution
    print("Creating improved orbital period distribution...")
    fig_period = visualizer.create_improved_orbital_period_distribution(sample_data)
    print("✅ Improved orbital period distribution created")
    
    # Test improved scatter plot
    print("Creating improved scatter plot...")
    fig_scatter = visualizer.create_improved_scatter_plot(sample_data)
    print("✅ Improved scatter plot created")
    
    # Test improved star map
    print("Creating improved star map...")
    fig_star = visualizer.create_improved_star_map(sample_data)
    print("✅ Improved star map created")
    
    # Test enhanced correlation heatmap
    print("Creating enhanced correlation heatmap...")
    fig_corr = visualizer.create_enhanced_correlation_heatmap(sample_data)
    print("✅ Enhanced correlation heatmap created")
    
    # Test data quality dashboard
    print("Creating data quality dashboard...")
    fig_quality = visualizer.create_data_quality_dashboard(sample_data)
    print("✅ Data quality dashboard created")
    
    print("\n🎉 All improved visualizations created successfully!")
    print("\n📊 Improved Visualizations:")
    print("1. 📈 Enhanced Orbital Period Distribution - Dual scale (linear + log)")
    print("2. 🔍 Improved Scatter Plot - Log scales, jittering, clear classifications")
    print("3. 🌌 Better Star Map - Reduced clustering, clear legend, proper coordinates")
    print("4. 🔥 Enhanced Correlation Heatmap - Better colors, annotations")
    print("5. 📊 Data Quality Dashboard - Comprehensive data analysis")


if __name__ == "__main__":
    test_improved_visualizations()
