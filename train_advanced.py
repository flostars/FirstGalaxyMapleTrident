#!/usr/bin/env python3
"""
Advanced ExoVision AI Training Pipeline
Implements multi-modal deep learning with cross-mission learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import our advanced models
import sys
sys.path.append(str(Path(__file__).parent))
from models.exovision_ai import ExoVisionAI, ExoVisionTrainer, create_exovision_model
from data.light_curve_generator import BatchLightCurveGenerator, create_light_curve_dataset


class ExoplanetDataset(Dataset):
    """PyTorch Dataset for exoplanet detection with light curves and stellar features."""
    
    def __init__(self, light_curves, stellar_features, labels, missions=None):
        self.light_curves = torch.FloatTensor(light_curves)
        self.stellar_features = torch.FloatTensor(stellar_features)
        self.labels = torch.LongTensor(labels)
        self.missions = missions
        
    def __len__(self):
        return len(self.light_curves)
    
    def __getitem__(self, idx):
        sample = {
            'light_curves': self.light_curves[idx],
            'stellar_features': self.stellar_features[idx],
            'labels': self.labels[idx]
        }
        
        if self.missions is not None:
            sample['missions'] = self.missions[idx]
            
        return sample


class AdvancedExoVisionTrainer:
    """Advanced training pipeline for ExoVision AI."""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.trainer = None
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
        
    def prepare_data(self, df, test_size=0.2, val_size=0.2):
        """Prepare data for training with light curves and stellar features."""
        print("🔄 Preparing advanced dataset...")
        
        # Generate light curves
        print("  📊 Generating synthetic light curves...")
        light_curves, stellar_features = create_light_curve_dataset(df)
        
        # Create labels
        labels = self._create_labels(df)
        
        # Create mission labels
        missions = self._create_mission_labels(df)
        
        # Split data
        X_temp, X_test, y_temp, y_test, m_temp, m_test = train_test_split(
            light_curves, labels, missions, test_size=test_size, 
            random_state=42, stratify=labels
        )
        
        X_train, X_val, y_train, y_val, m_train, m_val = train_test_split(
            X_temp, y_temp, m_temp, test_size=val_size/(1-test_size), 
            random_state=42, stratify=y_temp
        )
        
        # Create datasets
        train_dataset = ExoplanetDataset(X_train, stellar_features[:len(X_train)], y_train, m_train)
        val_dataset = ExoplanetDataset(X_val, stellar_features[len(X_train):len(X_train)+len(X_val)], y_val, m_val)
        test_dataset = ExoplanetDataset(X_test, stellar_features[len(X_train)+len(X_val):], y_test, m_test)
        
        print(f"  ✅ Dataset prepared:")
        print(f"     Train: {len(train_dataset)} samples")
        print(f"     Validation: {len(val_dataset)} samples")
        print(f"     Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_labels(self, df):
        """Create binary labels for exoplanet detection."""
        # Use the existing label column or create from disposition
        if 'label' in df.columns:
            labels = (df['label'] == 'CONFIRMED').astype(int)
        else:
            # Create labels from disposition columns
            labels = np.zeros(len(df))
            for col in ['disposition', 'koi_disposition', 'tfopwg_disp']:
                if col in df.columns:
                    confirmed_mask = df[col].str.contains('CONFIRMED', case=False, na=False)
                    labels[confirmed_mask] = 1
                    break
        
        return labels.values
    
    def _create_mission_labels(self, df):
        """Create mission labels for cross-mission learning."""
        missions = []
        for _, row in df.iterrows():
            if 'kepid' in row and pd.notna(row['kepid']):
                missions.append('kepler')
            elif 'tid' in row and pd.notna(row['tid']):
                missions.append('tess')
            else:
                missions.append('k2')
        return np.array(missions)
    
    def train_model(self, train_dataset, val_dataset, epochs=50, batch_size=32, learning_rate=1e-4):
        """Train the advanced ExoVision AI model."""
        print("🚀 Training Advanced ExoVision AI...")
        print("=" * 50)
        
        # Create model
        stellar_dim = train_dataset.stellar_features.shape[1]
        self.model = create_exovision_model(
            light_curve_dim=1000,
            stellar_dim=stellar_dim,
            hidden_dim=256,
            num_classes=2
        )
        
        # Create trainer
        self.trainer = ExoVisionTrainer(self.model, self.device)
        self.trainer.setup_training(learning_rate=learning_rate)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Training loop
        best_val_f1 = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training
            train_metrics = self._train_epoch(train_loader)
            
            # Validation
            val_metrics = self._validate_epoch(val_loader)
            
            # Store history
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
            self.training_history['train_f1'].append(train_metrics.get('f1', 0))
            self.training_history['val_f1'].append(val_metrics.get('f1', 0))
            
            # Save best model
            if val_metrics.get('f1', 0) > best_val_f1:
                best_val_f1 = val_metrics.get('f1', 0)
                best_model_state = self.model.state_dict().copy()
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_metrics['total_loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
                print(f"  Val Loss: {val_metrics['total_loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
                print("-" * 50)
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"✅ Best model loaded (Val F1: {best_val_f1:.4f})")
        
        return self.training_history
    
    def _train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        for batch in train_loader:
            # Move to device
            light_curves = batch['light_curves'].to(self.device)
            stellar_features = batch['stellar_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            missions = batch.get('missions', None)
            
            # Forward pass
            result = self.model(light_curves, stellar_features, missions)
            
            # Compute loss
            classification_loss = nn.CrossEntropyLoss()(result['logits'], labels)
            uncertainty_loss = torch.mean(result['uncertainty'])
            total_loss_batch = classification_loss + 0.1 * uncertainty_loss
            
            # Backward pass
            self.trainer.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.trainer.optimizer.step()
            
            total_loss += total_loss_batch.item()
            
            # Store predictions
            predictions = torch.argmax(result['logits'], dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'total_loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                light_curves = batch['light_curves'].to(self.device)
                stellar_features = batch['stellar_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                missions = batch.get('missions', None)
                
                # Forward pass
                result = self.model(light_curves, stellar_features, missions)
                
                # Compute loss
                classification_loss = nn.CrossEntropyLoss()(result['logits'], labels)
                uncertainty_loss = torch.mean(result['uncertainty'])
                total_loss_batch = classification_loss + 0.1 * uncertainty_loss
                
                total_loss += total_loss_batch.item()
                
                # Store predictions
                predictions = torch.argmax(result['logits'], dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        return {
            'total_loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1
        }
    
    def evaluate_model(self, test_dataset):
        """Evaluate the trained model."""
        print("🧪 Evaluating Advanced ExoVision AI...")
        
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                light_curves = batch['light_curves'].to(self.device)
                stellar_features = batch['stellar_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                missions = batch.get('missions', None)
                
                result = self.model(light_curves, stellar_features, missions)
                
                predictions = torch.argmax(result['logits'], dim=1)
                probabilities = torch.softmax(result['logits'], dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print(f"📊 Test Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return metrics, all_predictions, all_probabilities
    
    def save_model(self, path="models/exovision_ai_advanced.pth"):
        """Save the trained model."""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
            print(f"✅ Advanced model saved to {path}")
            
            # Save training history
            history_path = path.replace('.pth', '_history.json')
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
            print(f"✅ Training history saved to {history_path}")
    
    def plot_training_history(self, save_path="models/training_history_advanced.png"):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.training_history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.training_history['val_acc'], label='Val Acc')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.training_history['train_f1'], label='Train F1')
        axes[1, 0].plot(self.training_history['val_f1'], label='Val F1')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning curve
        axes[1, 1].text(0.5, 0.5, 'Advanced ExoVision AI\nTraining Complete!', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=axes[1, 1].transAxes, fontsize=16)
        axes[1, 1].set_title('Status')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Training history plot saved to {save_path}")


def main():
    """Main training pipeline for Advanced ExoVision AI."""
    print("🚀 Advanced ExoVision AI Training Pipeline")
    print("=" * 60)
    
    # Initialize trainer
    trainer = AdvancedExoVisionTrainer()
    
    # Load data
    print("📊 Loading datasets...")
    from exoplanet_app.preprocess import load_datasets
    df = load_datasets()
    print(f"  Loaded {len(df)} records")
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = trainer.prepare_data(df)
    
    # Train model
    history = trainer.train_model(train_dataset, val_dataset, epochs=100)
    
    # Evaluate model
    metrics, predictions, probabilities = trainer.evaluate_model(test_dataset)
    
    # Save model
    trainer.save_model()
    
    # Plot training history
    trainer.plot_training_history()
    
    print("\n🎉 Advanced ExoVision AI Training Complete!")
    print("=" * 60)
    print(f"Final Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
