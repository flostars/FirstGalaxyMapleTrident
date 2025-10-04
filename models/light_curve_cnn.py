"""Light curve processing CNN for exoplanet detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResNet1D(nn.Module):
    """1D ResNet for light curve analysis."""
    
    def __init__(self, input_dim: int = 1000, num_classes: int = 2):
        super(ResNet1D, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1):
        """Create a residual layer."""
        layers = []
        
        # Downsample if needed
        if stride != 1 or in_channels != out_channels:
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride))
            layers.append(nn.BatchNorm1d(out_channels))
        
        # Add residual blocks
        layers.append(ResidualBlock(out_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
            
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for 1D ResNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = F.relu(out)
        
        return out


class LightCurveCNN(nn.Module):
    """Enhanced CNN for light curve analysis with attention."""
    
    def __init__(self, input_dim: int = 1000, hidden_dim: int = 256):
        super(LightCurveCNN, self).__init__()
        
        # Feature extraction layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.projection = nn.Linear(256, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through light curve CNN."""
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        # Extract features
        features = self.conv_layers(x)  # [batch, 256, 1]
        features = features.squeeze(-1)  # [batch, 256]
        
        # Apply attention (self-attention)
        features_attended, _ = self.attention(
            features.unsqueeze(1),  # [batch, 1, 256]
            features.unsqueeze(1),
            features.unsqueeze(1)
        )
        features_attended = features_attended.squeeze(1)  # [batch, 256]
        
        # Project to hidden dimension
        output = self.projection(features_attended)
        
        return output


def create_light_curve(period: float, depth: float, duration: float, 
                      noise_level: float = 0.001, n_points: int = 1000) -> torch.Tensor:
    """Create synthetic light curve for training/testing."""
    
    # Time array (3 orbital periods)
    time = torch.linspace(0, period * 3, n_points)
    
    # Create transit signal
    light_curve = torch.ones_like(time)
    
    # Transit parameters
    transit_center = period / 2
    transit_width = duration / 24  # Convert hours to days
    
    # Create transit dip
    transit_mask = (time >= transit_center - transit_width/2) & (time <= transit_center + transit_width/2)
    light_curve[transit_mask] = 1 - depth
    
    # Add noise
    noise = torch.randn_like(time) * noise_level
    light_curve += noise
    
    # Add stellar activity (long-term trends)
    activity_period = period * 10  # Stellar rotation
    activity = 0.0005 * torch.sin(2 * torch.pi * time / activity_period)
    light_curve += activity
    
    return light_curve
