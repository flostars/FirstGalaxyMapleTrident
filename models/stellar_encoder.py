"""Stellar parameter encoder for exoplanet detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class StellarEncoder(nn.Module):
    """Deep neural network for encoding stellar parameters."""
    
    def __init__(self, input_dim: int = 11, hidden_dim: int = 256, dropout: float = 0.2):
        super(StellarEncoder, self).__init__()
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_dim)
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        # Feature importance attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through stellar encoder."""
        # Normalize input
        x = self.input_norm(x)
        
        # Encode features
        encoded = self.encoder(x)
        
        # Apply self-attention
        attended, _ = self.attention(
            encoded.unsqueeze(1),  # [batch, 1, hidden_dim]
            encoded.unsqueeze(1),
            encoded.unsqueeze(1)
        )
        
        return attended.squeeze(1)  # [batch, hidden_dim]


class StellarFeatureProcessor:
    """Process and normalize stellar parameters."""
    
    def __init__(self, feature_columns: List[str]):
        self.feature_columns = feature_columns
        self.scalers = {}
        self.feature_stats = {}
        
    def fit(self, data: torch.Tensor) -> 'StellarFeatureProcessor':
        """Fit scalers to the data."""
        for i, col in enumerate(self.feature_columns):
            values = data[:, i]
            self.feature_stats[col] = {
                'mean': values.mean().item(),
                'std': values.std().item(),
                'min': values.min().item(),
                'max': values.max().item()
            }
        return self
    
    def transform(self, data: torch.Tensor) -> torch.Tensor:
        """Transform data using fitted scalers."""
        normalized_data = data.clone()
        
        for i, col in enumerate(self.feature_columns):
            if col in self.feature_stats:
                stats = self.feature_stats[col]
                # Z-score normalization
                normalized_data[:, i] = (data[:, i] - stats['mean']) / (stats['std'] + 1e-8)
        
        return normalized_data
    
    def inverse_transform(self, data: torch.Tensor) -> torch.Tensor:
        """Inverse transform normalized data."""
        denormalized_data = data.clone()
        
        for i, col in enumerate(self.feature_columns):
            if col in self.feature_stats:
                stats = self.feature_stats[col]
                # Inverse Z-score normalization
                denormalized_data[:, i] = data[:, i] * stats['std'] + stats['mean']
        
        return denormalized_data


class StellarParameterValidator:
    """Validate stellar parameters for physical consistency."""
    
    @staticmethod
    def validate_effective_temperature(teff: float) -> bool:
        """Validate stellar effective temperature."""
        return 2000 <= teff <= 10000  # K
    
    @staticmethod
    def validate_surface_gravity(logg: float) -> bool:
        """Validate stellar surface gravity."""
        return 0.0 <= logg <= 6.0  # cgs
    
    @staticmethod
    def validate_radius(radius: float) -> bool:
        """Validate stellar radius."""
        return 0.1 <= radius <= 100.0  # Solar radii
    
    @staticmethod
    def validate_mass(mass: float) -> bool:
        """Validate stellar mass."""
        return 0.1 <= mass <= 10.0  # Solar masses
    
    @staticmethod
    def validate_metallicity(metallicity: float) -> bool:
        """Validate stellar metallicity."""
        return -2.0 <= metallicity <= 1.0  # dex
    
    @classmethod
    def validate_all(cls, stellar_params: dict) -> dict:
        """Validate all stellar parameters."""
        validation_results = {}
        
        if 'st_teff' in stellar_params:
            validation_results['teff_valid'] = cls.validate_effective_temperature(stellar_params['st_teff'])
        
        if 'st_logg' in stellar_params:
            validation_results['logg_valid'] = cls.validate_surface_gravity(stellar_params['st_logg'])
        
        if 'st_rad' in stellar_params:
            validation_results['rad_valid'] = cls.validate_radius(stellar_params['st_rad'])
        
        if 'st_mass' in stellar_params:
            validation_results['mass_valid'] = cls.validate_mass(stellar_params['st_mass'])
        
        if 'st_met' in stellar_params:
            validation_results['met_valid'] = cls.validate_metallicity(stellar_params['st_met'])
        
        return validation_results


def create_stellar_features(teff: float, logg: float, radius: float, mass: float, 
                          metallicity: float, **kwargs) -> torch.Tensor:
    """Create stellar feature tensor from parameters."""
    
    # Base stellar parameters
    features = torch.tensor([teff, logg, radius, mass, metallicity], dtype=torch.float32)
    
    # Add derived features
    if 'pl_orbper' in kwargs and 'st_mass' in kwargs:
        # Orbital velocity
        orbital_velocity = (2 * 3.14159 * kwargs['pl_orbper'] / 365.25) * (kwargs['st_mass'] ** 0.5)
        features = torch.cat([features, torch.tensor([orbital_velocity])])
    
    if 'pl_insol' in kwargs:
        # Insolation
        features = torch.cat([features, torch.tensor([kwargs['pl_insol']])])
    
    if 'pl_eqt' in kwargs:
        # Equilibrium temperature
        features = torch.cat([features, torch.tensor([kwargs['pl_eqt']])])
    
    # Pad or truncate to standard size
    target_size = 11
    if len(features) < target_size:
        padding = torch.zeros(target_size - len(features))
        features = torch.cat([features, padding])
    elif len(features) > target_size:
        features = features[:target_size]
    
    return features
