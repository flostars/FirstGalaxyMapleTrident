"""Main ExoVision AI model for multi-modal exoplanet detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union

from .light_curve_cnn import LightCurveCNN, create_light_curve
from .stellar_encoder import StellarEncoder, StellarFeatureProcessor
from .attention_fusion import AttentionFusion, CrossModalAttention
from .mission_adapters import MissionAdapters, CrossMissionValidation


class ExoVisionAI(nn.Module):
    """Main ExoVision AI model for multi-modal exoplanet detection."""
    
    def __init__(self, 
                 light_curve_dim: int = 1000,
                 stellar_dim: int = 11,
                 hidden_dim: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super(ExoVisionAI, self).__init__()
        
        # Light curve processing
        self.light_curve_cnn = LightCurveCNN(
            input_dim=light_curve_dim,
            hidden_dim=hidden_dim
        )
        
        # Stellar parameter encoding
        self.stellar_encoder = StellarEncoder(
            input_dim=stellar_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Multi-modal attention fusion
        self.attention_fusion = AttentionFusion(
            light_curve_dim=hidden_dim,
            stellar_dim=hidden_dim,
            hidden_dim=hidden_dim * 2,
            dropout=dropout
        )
        
        # Mission-specific adapters
        self.mission_adapters = MissionAdapters(
            input_dim=hidden_dim * 2,
            adapter_dim=hidden_dim // 2
        )
        
        # Cross-mission validation
        self.cross_mission_validation = CrossMissionValidation(
            input_dim=hidden_dim * 2,
            num_classes=num_classes
        )
        
        # Final classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Feature importance
        self.feature_importance = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, stellar_dim + 1),  # +1 for light curve
            nn.Softmax(dim=-1)
        )
        
    def forward(self, 
                light_curves: torch.Tensor,
                stellar_features: torch.Tensor,
                mission: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the complete model."""
        
        # Process light curves
        lc_features = self.light_curve_cnn(light_curves)  # [batch, hidden_dim]
        
        # Process stellar features
        stellar_features_encoded = self.stellar_encoder(stellar_features)  # [batch, hidden_dim]
        
        # Multi-modal attention fusion
        fused_features, attention_weights = self.attention_fusion(
            lc_features, stellar_features_encoded
        )  # [batch, hidden_dim * 2]
        
        # Mission-specific adaptation
        if mission is not None:
            adapted_features = self.mission_adapters(fused_features, mission)
        else:
            adapted_features = fused_features
        
        # Classification
        logits = self.classifier(adapted_features)
        probabilities = F.softmax(logits, dim=-1)
        
        # Uncertainty estimation
        uncertainty = self.uncertainty_estimator(adapted_features)
        
        # Feature importance
        importance_weights = self.feature_importance(adapted_features)
        
        # Cross-mission validation (if mission specified)
        cross_mission_results = None
        if mission is not None:
            cross_mission_results = self.cross_mission_validation.forward(
                adapted_features, mission
            )
        
        return {
            'logits': logits,
            'probabilities': probabilities,
            'uncertainty': uncertainty.squeeze(-1),
            'attention_weights': attention_weights,
            'feature_importance': importance_weights,
            'cross_mission_results': cross_mission_results,
            'fused_features': adapted_features
        }
    
    def ensemble_predict(self, 
                        light_curves: torch.Tensor,
                        stellar_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get ensemble prediction across all missions."""
        
        # Get predictions from all missions
        mission_predictions = {}
        mission_confidences = {}
        
        for mission in ['kepler', 'k2', 'tess']:
            result = self.forward(light_curves, stellar_features, mission)
            mission_predictions[mission] = result['probabilities']
            mission_confidences[mission] = 1.0 - result['uncertainty']  # Convert uncertainty to confidence
        
        # Stack predictions and confidences
        stacked_predictions = torch.stack(list(mission_predictions.values()), dim=1)  # [batch, 3, num_classes]
        stacked_confidences = torch.stack(list(mission_confidences.values()), dim=1)  # [batch, 3]
        
        # Weighted ensemble
        weights = F.softmax(stacked_confidences, dim=1)  # [batch, 3]
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
    
    def get_interpretability(self, 
                           light_curves: torch.Tensor,
                           stellar_features: torch.Tensor,
                           mission: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Get interpretability features for explainability."""
        
        # Forward pass
        result = self.forward(light_curves, stellar_features, mission)
        
        # Feature importance breakdown
        importance = result['feature_importance']
        lc_importance = importance[:, 0]  # Light curve importance
        stellar_importance = importance[:, 1:]  # Stellar features importance
        
        # Attention visualization
        attention_weights = result['attention_weights']
        
        return {
            'light_curve_importance': lc_importance,
            'stellar_importance': stellar_importance,
            'attention_weights': attention_weights,
            'uncertainty': result['uncertainty'],
            'feature_importance': importance
        }


class ExoVisionTrainer:
    """Training utilities for ExoVision AI."""
    
    def __init__(self, model: ExoVisionAI, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        self.scheduler = None
        
    def setup_training(self, learning_rate: float = 1e-4, weight_decay: float = 1e-5):
        """Setup optimizer and scheduler."""
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        
        # Move to device
        light_curves = batch['light_curves'].to(self.device)
        stellar_features = batch['stellar_features'].to(self.device)
        labels = batch['labels'].to(self.device)
        missions = batch.get('missions', None)
        
        # Forward pass
        result = self.model(light_curves, stellar_features, missions)
        
        # Compute loss
        classification_loss = F.cross_entropy(result['logits'], labels)
        uncertainty_loss = torch.mean(result['uncertainty'])  # Encourage low uncertainty
        total_loss = classification_loss + 0.1 * uncertainty_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Compute metrics
        predictions = torch.argmax(result['logits'], dim=-1)
        accuracy = (predictions == labels).float().mean()
        
        return {
            'total_loss': total_loss.item(),
            'classification_loss': classification_loss.item(),
            'uncertainty_loss': uncertainty_loss.item(),
            'accuracy': accuracy.item()
        }
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single validation step."""
        self.model.eval()
        
        with torch.no_grad():
            # Move to device
            light_curves = batch['light_curves'].to(self.device)
            stellar_features = batch['stellar_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            missions = batch.get('missions', None)
            
            # Forward pass
            result = self.model(light_curves, stellar_features, missions)
            
            # Compute loss
            classification_loss = F.cross_entropy(result['logits'], labels)
            uncertainty_loss = torch.mean(result['uncertainty'])
            total_loss = classification_loss + 0.1 * uncertainty_loss
            
            # Compute metrics
            predictions = torch.argmax(result['logits'], dim=-1)
            accuracy = (predictions == labels).float().mean()
            
            return {
                'total_loss': total_loss.item(),
                'classification_loss': classification_loss.item(),
                'uncertainty_loss': uncertainty_loss.item(),
                'accuracy': accuracy.item()
            }


def create_exovision_model(light_curve_dim: int = 1000,
                          stellar_dim: int = 11,
                          hidden_dim: int = 256,
                          num_classes: int = 2) -> ExoVisionAI:
    """Create ExoVision AI model with default parameters."""
    
    return ExoVisionAI(
        light_curve_dim=light_curve_dim,
        stellar_dim=stellar_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    )


def load_pretrained_model(model_path: str, device: str = 'cpu') -> ExoVisionAI:
    """Load pre-trained ExoVision AI model."""
    
    model = create_exovision_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def save_model(model: ExoVisionAI, path: str):
    """Save ExoVision AI model."""
    torch.save(model.state_dict(), path)
