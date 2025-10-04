"""Mission-specific adapters for cross-mission learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MissionAdapter(nn.Module):
    """Adapter layer for mission-specific feature adaptation."""
    
    def __init__(self, input_dim: int = 256, adapter_dim: int = 64, 
                 dropout: float = 0.1):
        super(MissionAdapter, self).__init__()
        
        self.input_dim = input_dim
        self.adapter_dim = adapter_dim
        
        # Down-projection
        self.down_proj = nn.Sequential(
            nn.Linear(input_dim, adapter_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Up-projection
        self.up_proj = nn.Sequential(
            nn.Linear(adapter_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter."""
        residual = x
        
        # Down-project
        x = self.down_proj(x)
        
        # Up-project
        x = self.up_proj(x)
        
        # Residual connection and normalization
        x = self.norm(x + residual)
        
        return x


class MissionAdapters(nn.Module):
    """Collection of mission-specific adapters."""
    
    def __init__(self, input_dim: int = 256, adapter_dim: int = 64):
        super(MissionAdapters, self).__init__()
        
        # Mission-specific adapters
        self.adapters = nn.ModuleDict({
            'kepler': MissionAdapter(input_dim, adapter_dim),
            'k2': MissionAdapter(input_dim, adapter_dim),
            'tess': MissionAdapter(input_dim, adapter_dim),
            'neossat': MissionAdapter(input_dim, adapter_dim)
        })
        
        # Mission embedding
        self.mission_embedding = nn.Embedding(4, input_dim)
        
        # Cross-mission attention
        self.cross_mission_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, features: torch.Tensor, mission: str) -> torch.Tensor:
        """Apply mission-specific adaptation."""
        
        # Get mission adapter
        if mission in self.adapters:
            adapter = self.adapters[mission]
            adapted_features = adapter(features)
        else:
            # Default adapter (no adaptation)
            adapted_features = features
        
        return adapted_features
    
    def forward_all_missions(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Apply all mission adapters for cross-mission learning."""
        
        adapted_features = {}
        
        for mission_name, adapter in self.adapters.items():
            adapted_features[mission_name] = adapter(features)
        
        return adapted_features
    
    def cross_mission_fusion(self, mission_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse features from all missions."""
        
        # Stack features from all missions
        stacked_features = torch.stack(list(mission_features.values()), dim=1)  # [batch, num_missions, dim]
        
        # Apply cross-mission attention
        fused_features, attention_weights = self.cross_mission_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Global average pooling
        final_features = fused_features.mean(dim=1)  # [batch, dim]
        
        return final_features, attention_weights


class DomainAdaptationLayer(nn.Module):
    """Domain adaptation layer for cross-mission transfer learning."""
    
    def __init__(self, input_dim: int = 256, hidden_dim: int = 128):
        super(DomainAdaptationLayer, self).__init__()
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 4),  # 4 missions
            nn.Softmax(dim=-1)
        )
        
        # Gradient reversal layer
        self.gradient_reversal = GradientReversalLayer()
        
    def forward(self, features: torch.Tensor, mission: str, 
                alpha: float = 1.0) -> torch.Tensor:
        """Apply domain adaptation."""
        
        # Reverse gradients for domain adaptation
        reversed_features = self.gradient_reversal(features, alpha)
        
        # Domain classification
        domain_pred = self.domain_classifier(reversed_features)
        
        return domain_pred


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation."""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class MissionSpecificNormalization(nn.Module):
    """Mission-specific normalization layer."""
    
    def __init__(self, input_dim: int = 256, num_missions: int = 4):
        super(MissionSpecificNormalization, self).__init__()
        
        # Mission-specific batch normalization
        self.mission_bn = nn.ModuleDict({
            'kepler': nn.BatchNorm1d(input_dim),
            'k2': nn.BatchNorm1d(input_dim),
            'tess': nn.BatchNorm1d(input_dim),
            'neossat': nn.BatchNorm1d(input_dim)
        })
        
        # Mission-specific scaling
        self.mission_scale = nn.Parameter(torch.ones(num_missions, input_dim))
        self.mission_shift = nn.Parameter(torch.zeros(num_missions, input_dim))
        
    def forward(self, features: torch.Tensor, mission: str) -> torch.Tensor:
        """Apply mission-specific normalization."""
        
        # Get mission index
        mission_idx = {'kepler': 0, 'k2': 1, 'tess': 2, 'neossat': 3}[mission]
        
        # Apply mission-specific batch normalization
        if mission in self.mission_bn:
            normalized = self.mission_bn[mission](features)
        else:
            normalized = features
        
        # Apply mission-specific scaling and shifting
        scale = self.mission_scale[mission_idx]
        shift = self.mission_shift[mission_idx]
        
        scaled = normalized * scale + shift
        
        return scaled


class CrossMissionValidation(nn.Module):
    """Cross-mission validation for ensemble predictions."""
    
    def __init__(self, input_dim: int = 256, num_classes: int = 2):
        super(CrossMissionValidation, self).__init__()
        
        # Mission-specific classifiers
        self.mission_classifiers = nn.ModuleDict({
            'kepler': nn.Linear(input_dim, num_classes),
            'k2': nn.Linear(input_dim, num_classes),
            'tess': nn.Linear(input_dim, num_classes)
        })
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, features: torch.Tensor, mission: str) -> Dict[str, torch.Tensor]:
        """Get mission-specific prediction and confidence."""
        
        # Get mission-specific prediction
        if mission in self.mission_classifiers:
            prediction = self.mission_classifiers[mission](features)
        else:
            # Default prediction
            prediction = torch.zeros(features.size(0), 2)
        
        # Estimate confidence
        confidence = self.confidence_estimator(features)
        
        return {
            'prediction': prediction,
            'confidence': confidence.squeeze(-1)
        }
    
    def ensemble_predict(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get ensemble prediction from all missions."""
        
        predictions = []
        confidences = []
        
        for mission in ['kepler', 'k2', 'tess']:
            result = self.forward(features, mission)
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
        
        # Stack predictions and confidences
        stacked_predictions = torch.stack(predictions, dim=1)  # [batch, num_missions, num_classes]
        stacked_confidences = torch.stack(confidences, dim=1)  # [batch, num_missions]
        
        # Weighted ensemble
        weights = F.softmax(stacked_confidences, dim=1)  # [batch, num_missions]
        ensemble_prediction = torch.sum(stacked_predictions * weights.unsqueeze(-1), dim=1)
        
        # Average confidence
        avg_confidence = stacked_confidences.mean(dim=1)
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': stacked_predictions,
            'individual_confidences': stacked_confidences,
            'ensemble_confidence': avg_confidence,
            'weights': weights
        }
