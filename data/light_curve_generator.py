"""Light curve generation and processing utilities."""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from scipy import signal
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class LightCurveGenerator:
    """Generate synthetic light curves from transit parameters."""
    
    def __init__(self, n_points: int = 1000, noise_level: float = 0.001):
        self.n_points = n_points
        self.noise_level = noise_level
        
    def create_transit_light_curve(self, 
                                 period: float,
                                 depth: float, 
                                 duration: float,
                                 impact_param: float = 0.0,
                                 limb_darkening: float = 0.6) -> np.ndarray:
        """Create a realistic transit light curve."""
        
        # Time array (3 orbital periods)
        time = np.linspace(0, period * 3, self.n_points)
        
        # Transit parameters
        transit_center = period / 2
        transit_width = duration / 24  # Convert hours to days
        
        # Create transit signal
        light_curve = np.ones_like(time)
        
        # Transit mask
        transit_mask = (time >= transit_center - transit_width/2) & (time <= transit_center + transit_width/2)
        
        if np.any(transit_mask):
            # Create realistic transit shape
            transit_times = time[transit_mask]
            transit_center_time = transit_center
            
            # Distance from transit center (normalized)
            x = (transit_times - transit_center_time) / (transit_width / 2)
            
            # Limb darkening effect
            if impact_param < 1.0:  # Only if planet crosses stellar disk
                # Simple limb darkening model
                limb_factor = 1 - limb_darkening * (1 - np.sqrt(1 - x**2))
                limb_factor = np.maximum(limb_factor, 0.1)  # Minimum brightness
                
                # Apply transit depth with limb darkening
                transit_signal = 1 - depth * limb_factor
            else:
                # No transit (planet doesn't cross stellar disk)
                transit_signal = np.ones_like(x)
            
            light_curve[transit_mask] = transit_signal
        
        return light_curve
    
    def add_stellar_activity(self, light_curve: np.ndarray, 
                           period: float, 
                           activity_strength: float = 0.0005) -> np.ndarray:
        """Add stellar activity (spots, rotation) to light curve."""
        
        time = np.linspace(0, period * 3, len(light_curve))
        
        # Stellar rotation period (typically 10-30 days)
        rotation_period = period * np.random.uniform(5, 15)
        
        # Add multiple spot cycles
        activity = np.zeros_like(light_curve)
        
        # Primary spot cycle
        activity += activity_strength * np.sin(2 * np.pi * time / rotation_period)
        
        # Secondary spot cycle (different phase)
        activity += 0.3 * activity_strength * np.sin(2 * np.pi * time / rotation_period + np.pi/3)
        
        # Add some random variability
        activity += 0.1 * activity_strength * np.random.randn(len(light_curve))
        
        return light_curve + activity
    
    def add_instrumental_noise(self, light_curve: np.ndarray, 
                             noise_level: Optional[float] = None) -> np.ndarray:
        """Add realistic instrumental noise."""
        
        if noise_level is None:
            noise_level = self.noise_level
        
        # Gaussian noise
        gaussian_noise = np.random.normal(0, noise_level, len(light_curve))
        
        # Systematic noise (long-term trends)
        systematic_noise = 0.1 * noise_level * np.sin(2 * np.pi * np.arange(len(light_curve)) / 100)
        
        # Shot noise (Poisson-like)
        shot_noise = np.random.poisson(1000, len(light_curve)) / 1000 - 1
        shot_noise *= noise_level * 0.5
        
        total_noise = gaussian_noise + systematic_noise + shot_noise
        
        return light_curve + total_noise
    
    def add_secondary_eclipse(self, light_curve: np.ndarray, 
                            period: float, 
                            secondary_depth: float = 0.0001) -> np.ndarray:
        """Add secondary eclipse (planet behind star)."""
        
        time = np.linspace(0, period * 3, len(light_curve))
        
        # Secondary eclipse occurs at phase 0.5
        secondary_center = period * 0.5
        secondary_width = period * 0.01  # Much shorter than primary transit
        
        secondary_mask = (time >= secondary_center - secondary_width/2) & (time <= secondary_center + secondary_width/2)
        
        if np.any(secondary_mask):
            light_curve[secondary_mask] -= secondary_depth
        
        return light_curve
    
    def generate_light_curve(self, 
                           period: float,
                           depth: float,
                           duration: float,
                           impact_param: float = 0.0,
                           add_activity: bool = True,
                           add_secondary: bool = False,
                           add_noise: bool = True) -> np.ndarray:
        """Generate complete light curve with all effects."""
        
        # Create base transit
        light_curve = self.create_transit_light_curve(
            period, depth, duration, impact_param
        )
        
        # Add stellar activity
        if add_activity:
            light_curve = self.add_stellar_activity(light_curve, period)
        
        # Add secondary eclipse
        if add_secondary:
            light_curve = self.add_secondary_eclipse(light_curve, period)
        
        # Add instrumental noise
        if add_noise:
            light_curve = self.add_instrumental_noise(light_curve)
        
        return light_curve


class LightCurveProcessor:
    """Process and analyze light curves."""
    
    def __init__(self):
        self.generator = LightCurveGenerator()
    
    def detect_transits(self, light_curve: np.ndarray, 
                       threshold: float = 0.001) -> List[Dict]:
        """Detect transit events in light curve."""
        
        # Smooth the light curve
        smoothed = signal.savgol_filter(light_curve, 21, 3)
        
        # Find dips below threshold
        dips = smoothed < (1 - threshold)
        
        # Find transit boundaries
        transit_starts = []
        transit_ends = []
        
        in_transit = False
        for i, is_dip in enumerate(dips):
            if is_dip and not in_transit:
                transit_starts.append(i)
                in_transit = True
            elif not is_dip and in_transit:
                transit_ends.append(i)
                in_transit = False
        
        # Create transit events
        transits = []
        for start, end in zip(transit_starts, transit_ends):
            if end - start > 5:  # Minimum transit duration
                transit_depth = 1 - np.min(light_curve[start:end])
                transit_duration = end - start
                
                transits.append({
                    'start': start,
                    'end': end,
                    'depth': transit_depth,
                    'duration': transit_duration,
                    'center': (start + end) / 2
                })
        
        return transits
    
    def estimate_period(self, light_curve: np.ndarray, 
                       time: np.ndarray) -> float:
        """Estimate orbital period using autocorrelation."""
        
        # Remove mean
        detrended = light_curve - np.mean(light_curve)
        
        # Compute autocorrelation
        autocorr = np.correlate(detrended, detrended, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Find peaks (excluding the first one at lag 0)
        peaks, _ = signal.find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))
        
        if len(peaks) > 0:
            # Convert peak index to time
            dt = time[1] - time[0]
            period = (peaks[0] + 1) * dt
            return period
        
        return 0.0
    
    def normalize_light_curve(self, light_curve: np.ndarray) -> np.ndarray:
        """Normalize light curve to unit mean."""
        return light_curve / np.mean(light_curve)
    
    def detrend_light_curve(self, light_curve: np.ndarray, 
                          window_size: int = 100) -> np.ndarray:
        """Remove long-term trends from light curve."""
        
        # Use moving average to estimate trend
        trend = signal.savgol_filter(light_curve, window_size, 3)
        
        # Remove trend
        detrended = light_curve - trend + 1
        
        return detrended


class BatchLightCurveGenerator:
    """Generate batches of light curves for training."""
    
    def __init__(self, n_points: int = 1000):
        self.generator = LightCurveGenerator(n_points)
        self.processor = LightCurveProcessor()
    
    def generate_batch(self, 
                      periods: np.ndarray,
                      depths: np.ndarray,
                      durations: np.ndarray,
                      impact_params: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate batch of light curves."""
        
        batch_size = len(periods)
        light_curves = np.zeros((batch_size, self.generator.n_points))
        
        if impact_params is None:
            impact_params = np.random.uniform(0, 0.9, batch_size)
        
        for i in range(batch_size):
            light_curve = self.generator.generate_light_curve(
                period=periods[i],
                depth=depths[i],
                duration=durations[i],
                impact_param=impact_params[i]
            )
            light_curves[i] = light_curve
        
        return light_curves
    
    def generate_from_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        """Generate light curves from DataFrame with transit parameters."""
        
        # Extract parameters
        periods = df.get('pl_orbper', df.get('koi_period', 10.0)).values
        depths = df.get('pl_trandep', df.get('koi_depth', 1000.0)).values / 1e6  # Convert ppm to fraction
        durations = df.get('pl_trandurh', df.get('koi_duration', 2.0)).values
        
        # Generate light curves
        light_curves = self.generate_batch(periods, depths, durations)
        
        return light_curves


def create_light_curve_dataset(df: pd.DataFrame, 
                             n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Create light curve dataset from exoplanet DataFrame."""
    
    generator = BatchLightCurveGenerator(n_points)
    
    # Generate light curves
    light_curves = generator.generate_from_dataframe(df)
    
    # Extract stellar features
    stellar_features = df[['pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse', 
                          'pl_eqt', 'pl_insol', 'st_teff', 'st_rad', 'st_mass', 
                          'st_met', 'st_logg']].values
    
    return light_curves, stellar_features
