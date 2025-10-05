"""
Advanced Deep Learning Models for Exoplanet Classification
Implements state-of-the-art architectures for superior performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism for feature relationships"""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.size()
        
        # Linear transformations
        Q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Output projection and residual connection
        output = self.w_o(context)
        return self.layer_norm(x + self.dropout(output))


class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        
        return F.relu(out + residual)


class ExoTransformer(nn.Module):
    """Transformer-based architecture for exoplanet classification"""
    
    def __init__(self, input_dim: int, n_classes: int = 2, 
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 100, d_model))
        
        # Transformer layers
        self.attention_layers = nn.ModuleList([
            MultiHeadSelfAttention(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, n_classes)
        )
        
    def forward(self, x):
        # Input projection and positional encoding
        x = self.input_projection(x.unsqueeze(1))  # Add sequence dimension
        x = x + self.positional_encoding[:, :x.size(1), :]
        
        # Transformer layers
        for attn, ffn in zip(self.attention_layers, self.ffn_layers):
            x = attn(x)
            residual = x
            x = ffn(x)
            x = x + residual
        
        # Global average pooling and classification
        x = x.mean(dim=1)
        return self.classifier(x)


class AdvancedResNet(nn.Module):
    """Advanced ResNet architecture with attention and feature fusion"""
    
    def __init__(self, input_dim: int, n_classes: int = 2, 
                 hidden_dims: List[int] = [512, 256, 128, 64],
                 dropout: float = 0.1):
        super().__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.feature_extractor.append(ResidualBlock(prev_dim, hidden_dim, dropout))
            prev_dim = hidden_dim
        
        # Attention mechanism
        self.attention = MultiHeadSelfAttention(prev_dim, n_heads=4, dropout=dropout)
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(prev_dim // 2),
            nn.Dropout(dropout),
            nn.Linear(prev_dim // 2, prev_dim // 4),
            nn.ReLU(),
            nn.BatchNorm1d(prev_dim // 4),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(prev_dim // 4, n_classes)
        
    def forward(self, x):
        # Feature extraction
        for layer in self.feature_extractor:
            x = layer(x)
        
        # Apply attention
        x = self.attention(x.unsqueeze(1)).squeeze(1)
        
        # Feature fusion and classification
        x = self.feature_fusion(x)
        return self.classifier(x)


class EnsembleModel(nn.Module):
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, input_dim: int, n_classes: int = 2):
        super().__init__()
        
        # Multiple model architectures
        self.transformer = ExoTransformer(input_dim, n_classes)
        self.resnet = AdvancedResNet(input_dim, n_classes)
        
        # Additional specialized models
        self.deep_net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )
        
        # Ensemble fusion
        self.fusion = nn.Sequential(
            nn.Linear(n_classes * 3, n_classes * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(n_classes * 2, n_classes)
        )
        
    def forward(self, x):
        # Get predictions from all models
        transformer_out = self.transformer(x)
        resnet_out = self.resnet(x)
        deep_out = self.deep_net(x)
        
        # Ensemble fusion
        combined = torch.cat([transformer_out, resnet_out, deep_out], dim=1)
        return self.fusion(combined)


class AdvancedExoplanetClassifier:
    """Advanced classifier with multiple model architectures and training techniques"""
    
    def __init__(self, input_dim: int, model_type: str = "ensemble", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model_type = model_type
        
        # Initialize model based on type
        if model_type == "transformer":
            self.model = ExoTransformer(input_dim)
        elif model_type == "resnet":
            self.model = AdvancedResNet(input_dim)
        elif model_type == "ensemble":
            self.model = EnsembleModel(input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model.to(self.device)
        
        # Training components
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Single training step"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        return {
            'val_loss': total_loss / len(val_loader),
            'val_accuracy': correct / total
        }
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """Make predictions"""
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.model(x)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
    
    def get_feature_importance(self, x: torch.Tensor) -> np.ndarray:
        """Get feature importance using gradient-based attribution"""
        self.model.eval()
        x = x.to(self.device)
        x.requires_grad_(True)
        
        outputs = self.model(x)
        # Use the max probability class for attribution
        max_class = outputs.argmax(dim=1)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=outputs[0, max_class[0]], 
            inputs=x,
            retain_graph=True
        )[0]
        
        # Return absolute gradients as importance scores
        return torch.abs(gradients).mean(dim=0).cpu().numpy()
    
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_type': self.model_type
        }, path)
    
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def create_advanced_model(input_dim: int, model_type: str = "ensemble") -> AdvancedExoplanetClassifier:
    """Factory function to create advanced models"""
    return AdvancedExoplanetClassifier(input_dim, model_type)


if __name__ == "__main__":
    # Test the models
    input_dim = 11
    batch_size = 32
    
    # Create test data
    x = torch.randn(batch_size, input_dim)
    y = torch.randint(0, 2, (batch_size,))
    
    # Test different model types
    for model_type in ["transformer", "resnet", "ensemble"]:
        print(f"\nTesting {model_type} model:")
        model = create_advanced_model(input_dim, model_type)
        
        # Forward pass
        predictions = model.predict(x)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Sample predictions: {predictions[:3]}")
        
        # Training step
        loss = model.train_step(x, y)
        print(f"Training loss: {loss:.4f}")
        
        print(f"{model_type.capitalize()} model test completed successfully!")
