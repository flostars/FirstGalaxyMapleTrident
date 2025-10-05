"""Multi-modal attention fusion for exoplanet detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class AttentionFusion(nn.Module):
    """Multi-modal attention fusion layer."""
    
    def __init__(self, light_curve_dim: int = 256, stellar_dim: int = 256, 
                 hidden_dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super(AttentionFusion, self).__init__()
        
        self.light_curve_dim = light_curve_dim
        self.stellar_dim = stellar_dim
        self.hidden_dim = hidden_dim
        
        # Projection layers
        self.light_curve_proj = nn.Linear(light_curve_dim, hidden_dim)
        self.stellar_proj = nn.Linear(stellar_dim, hidden_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, light_curve_features: torch.Tensor, 
                stellar_features: torch.Tensor) -> torch.Tensor:
        """Fuse light curve and stellar features using attention."""
        
        # Project features to common dimension
        lc_proj = self.light_curve_proj(light_curve_features)  # [batch, hidden_dim]
        st_proj = self.stellar_proj(stellar_features)  # [batch, hidden_dim]
        
        # Stack features for attention
        features = torch.stack([lc_proj, st_proj], dim=1)  # [batch, 2, hidden_dim]
        
        # Self-attention
        attended, attention_weights = self.attention(features, features, features)
        
        # Residual connection and layer norm
        attended = self.norm1(attended + features)
        
        # Feed-forward network
        ffn_out = self.ffn(attended)
        ffn_out = self.norm2(ffn_out + attended)
        
        # Global average pooling
        fused_features = ffn_out.mean(dim=1)  # [batch, hidden_dim]
        
        # Output projection
        output = self.output_proj(fused_features)
        
        return output, attention_weights


class CrossModalAttention(nn.Module):
    """Cross-modal attention between light curves and stellar parameters."""
    
    def __init__(self, light_curve_dim: int = 256, stellar_dim: int = 256, 
                 hidden_dim: int = 256, num_heads: int = 8):
        super(CrossModalAttention, self).__init__()
        
        # Projection layers
        self.lc_proj = nn.Linear(light_curve_dim, hidden_dim)
        self.st_proj = nn.Linear(stellar_dim, hidden_dim)
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, light_curve_features: torch.Tensor, 
                stellar_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply cross-modal attention."""
        
        # Project features
        lc_proj = self.lc_proj(light_curve_features).unsqueeze(1)  # [batch, 1, hidden_dim]
        st_proj = self.st_proj(stellar_features).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention: light curve queries, stellar keys/values
        lc_attended, lc_weights = self.cross_attention(lc_proj, st_proj, st_proj)
        
        # Cross-attention: stellar queries, light curve keys/values
        st_attended, st_weights = self.cross_attention(st_proj, lc_proj, lc_proj)
        
        # Combine attended features
        combined = torch.cat([lc_attended.squeeze(1), st_attended.squeeze(1)], dim=1)
        combined = self.norm(combined)
        
        return combined, (lc_weights, st_weights)


class TemporalAttention(nn.Module):
    """Temporal attention for light curve sequences."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 256, 
                 num_heads: int = 8, max_length: int = 1000):
        super(TemporalAttention, self).__init__()
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(max_length, input_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal attention to light curve."""
        
        batch_size, seq_len = x.shape
        
        # Add positional encoding
        if seq_len <= self.pos_encoding.size(0):
            pos_enc = self.pos_encoding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)
            x = x.unsqueeze(-1) + pos_enc
        else:
            # Truncate or pad as needed
            if seq_len > self.pos_encoding.size(0):
                x = x[:, :self.pos_encoding.size(0)]
            pos_enc = self.pos_encoding.unsqueeze(0).expand(batch_size, -1, -1)
            x = x.unsqueeze(-1) + pos_enc
        
        # Apply transformer
        attended = self.transformer(x)
        
        # Global average pooling
        output = attended.mean(dim=1)
        
        return output


class FeatureImportanceAttention(nn.Module):
    """Attention mechanism for feature importance weighting."""
    
    def __init__(self, input_dim: int, num_features: int):
        super(FeatureImportanceAttention, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, num_features),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute feature importance weights."""
        
        # Compute attention weights
        weights = self.attention(features)  # [batch, num_features]
        
        # Apply weights to features
        weighted_features = features * weights.unsqueeze(-1)
        
        return weighted_features, weights


def compute_attention_visualization(attention_weights: torch.Tensor, 
                                  feature_names: list) -> dict:
    """Compute attention visualization for interpretability."""
    
    # Average attention weights across heads
    avg_weights = attention_weights.mean(dim=1)  # [batch, seq_len, seq_len]
    
    # Get attention for each sample
    visualizations = []
    for i in range(avg_weights.size(0)):
        sample_attention = avg_weights[i]
        
        # Create attention map
        attention_map = {
            'feature_names': feature_names,
            'attention_matrix': sample_attention.cpu().numpy().tolist(),
            'max_attention': sample_attention.max().item(),
            'min_attention': sample_attention.min().item()
        }
        
        visualizations.append(attention_map)
    
    return visualizations
