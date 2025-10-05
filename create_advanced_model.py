#!/usr/bin/env python3
"""
Create Advanced ExoVision AI Model
Simple script to create and save an advanced model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

def create_simple_advanced_model():
    """Create a simple advanced model for demonstration."""
    print("Creating Advanced ExoVision AI Model...")
    
    # Create a simple neural network
    class SimpleExoVisionAI(nn.Module):
        def __init__(self):
            super().__init__()
            # Light curve processing
            self.light_curve_conv = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(100)
            )
            
            # Stellar features processing
            self.stellar_fc = nn.Sequential(
                nn.Linear(11, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU()
            )
            
            # Fusion and classification
            self.fusion = nn.Sequential(
                nn.Linear(64 * 100 + 32, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 2)
            )
            
            # Uncertainty estimation
            self.uncertainty = nn.Sequential(
                nn.Linear(64 * 100 + 32, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        
        def forward(self, light_curves, stellar_features, missions=None):
            # Process light curves
            batch_size = light_curves.size(0)
            light_curves = light_curves.unsqueeze(1)  # Add channel dimension
            lc_features = self.light_curve_conv(light_curves)
            lc_features = lc_features.view(batch_size, -1)
            
            # Process stellar features
            stellar_features = self.stellar_fc(stellar_features)
            
            # Fuse features
            fused = torch.cat([lc_features, stellar_features], dim=1)
            
            # Classification
            logits = self.fusion(fused)
            
            # Uncertainty
            uncertainty = self.uncertainty(fused)
            
            return {
                'logits': logits,
                'uncertainty': uncertainty.squeeze()
            }
    
    # Create model
    model = SimpleExoVisionAI()
    
    # Create dummy data for initialization
    dummy_lc = torch.randn(1, 1000)
    dummy_stellar = torch.randn(1, 11)
    
    # Test forward pass
    with torch.no_grad():
        result = model(dummy_lc, dummy_stellar)
        print(f"Model created successfully!")
        print(f"Output shape: {result['logits'].shape}")
        print(f"Uncertainty shape: {result['uncertainty'].shape}")
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "exovision_ai_quick.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Advanced model saved to {model_path}")
    
    return model

if __name__ == "__main__":
    create_simple_advanced_model()
