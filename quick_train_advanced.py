#!/usr/bin/env python3
"""
Quick Advanced Training Script
Trains the advanced ExoVision AI model with minimal configuration
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our advanced models
import sys
sys.path.append(str(Path(__file__).parent))
from models.exovision_ai import create_exovision_model
from data.light_curve_generator import create_light_curve_dataset
from exoplanet_app.preprocess import load_datasets


def quick_advanced_training():
    """Quick training of the advanced ExoVision AI model."""
    print("🚀 Quick Advanced ExoVision AI Training")
    print("=" * 50)
    
    # Load data
    print("📊 Loading datasets...")
    df = load_datasets()
    print(f"  Loaded {len(df)} records")
    
    # Generate light curves and stellar features
    print("🔄 Generating synthetic light curves...")
    light_curves, stellar_features = create_light_curve_dataset(df)
    
    # Create labels
    labels = (df['label'] == 'CONFIRMED').astype(int).values
    
    # Create simple dataset
    from torch.utils.data import TensorDataset, DataLoader
    light_curves_tensor = torch.FloatTensor(light_curves)
    stellar_features_tensor = torch.FloatTensor(stellar_features)
    labels_tensor = torch.LongTensor(labels)
    
    dataset = TensorDataset(light_curves_tensor, stellar_features_tensor, labels_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    print("🧠 Creating ExoVision AI model...")
    model = create_exovision_model(
        light_curve_dim=1000,
        stellar_dim=stellar_features.shape[1],
        hidden_dim=128,  # Smaller for quick training
        num_classes=2
    )
    
    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Quick training loop
    print("🚀 Starting quick training...")
    model.train()
    
    for epoch in range(10):  # Quick training
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (light_curves, stellar_features, labels) in enumerate(dataloader):
            light_curves = light_curves.to(device)
            stellar_features = stellar_features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            result = model(light_curves, stellar_features)
            loss = criterion(result['logits'], labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(result['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")
    
    # Save model
    model_path = "models/exovision_ai_quick.pth"
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"✅ Quick model saved to {model_path}")
    
    # Test prediction
    print("🧪 Testing prediction...")
    model.eval()
    with torch.no_grad():
        test_light_curve = light_curves_tensor[:1].to(device)
        test_stellar = stellar_features_tensor[:1].to(device)
        result = model(test_light_curve, test_stellar)
        prediction = torch.argmax(result['logits'], dim=1).item()
        confidence = 1.0 - result['uncertainty'].item()
        
        print(f"Test prediction: {'EXOPLANET' if prediction == 1 else 'NO_EXOPLANET'}")
        print(f"Confidence: {confidence:.2%}")
    
    print("\n🎉 Quick Advanced Training Complete!")
    print("The model is now ready for use in the web interface.")


if __name__ == "__main__":
    quick_advanced_training()
