"""
Advanced Training Pipeline for Exoplanet Classification
Implements state-of-the-art training techniques and optimization strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from models.advanced_classifier import AdvancedExoplanetClassifier, create_advanced_model
except ImportError:
    from advanced_classifier import AdvancedExoplanetClassifier, create_advanced_model


class AdvancedDataProcessor:
    """Advanced data preprocessing with feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
            'pl_orbincl', 'pl_tranmid', 'st_teff', 'st_rad', 'st_mass', 'st_logg'
        ]
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced engineered features"""
        df_eng = df.copy()
        
        # Planetary features
        df_eng['pl_density'] = df_eng['pl_bmasse'] / (df_eng['pl_rade'] ** 3)
        df_eng['pl_escape_velocity'] = np.sqrt(2 * df_eng['pl_bmasse'] / df_eng['pl_rade'])
        df_eng['pl_surface_gravity'] = df_eng['pl_bmasse'] / (df_eng['pl_rade'] ** 2)
        
        # Stellar features
        df_eng['st_luminosity'] = (df_eng['st_rad'] ** 2) * ((df_eng['st_teff'] / 5778) ** 4)
        df_eng['st_surface_gravity'] = 10 ** df_eng['st_logg']
        
        # Orbital features
        df_eng['pl_insolation'] = df_eng['st_luminosity'] / (df_eng['pl_orbsmax'] ** 2)
        df_eng['pl_equilibrium_temp'] = df_eng['st_teff'] * np.sqrt(df_eng['st_rad'] / (2 * df_eng['pl_orbsmax']))
        df_eng['pl_hill_sphere'] = df_eng['pl_orbsmax'] * ((df_eng['pl_bmasse'] / (3 * df_eng['st_mass'])) ** (1/3))
        
        # Habitability indicators
        df_eng['habitable_zone_distance'] = np.sqrt(df_eng['st_luminosity'])
        df_eng['habitable_zone_ratio'] = df_eng['pl_orbsmax'] / df_eng['habitable_zone_distance']
        
        # Transit features
        df_eng['transit_depth'] = (df_eng['pl_rade'] / df_eng['st_rad']) ** 2
        df_eng['transit_duration'] = (df_eng['pl_orbper'] / np.pi) * np.arcsin(df_eng['st_rad'] / df_eng['pl_orbsmax'])
        
        # Interaction features
        df_eng['mass_radius_ratio'] = df_eng['pl_bmasse'] / df_eng['pl_rade']
        df_eng['stellar_planet_ratio'] = df_eng['st_mass'] / df_eng['pl_bmasse']
        df_eng['orbital_velocity'] = np.sqrt(df_eng['st_mass'] / df_eng['pl_orbsmax'])
        
        # Fill infinite and NaN values
        df_eng = df_eng.replace([np.inf, -np.inf], np.nan)
        df_eng = df_eng.fillna(df_eng.median())
        
        return df_eng
    
    def create_augmented_data(self, X: np.ndarray, y: np.ndarray, augment_factor: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Create augmented training data with noise injection"""
        augmented_X = []
        augmented_y = []
        
        for i in range(len(X)):
            # Original sample
            augmented_X.append(X[i])
            augmented_y.append(y[i])
            
            # Generate augmented samples
            for _ in range(augment_factor):
                # Add Gaussian noise
                noise = np.random.normal(0, 0.05, X[i].shape)
                augmented_sample = X[i] + noise
                
                # Add small random scaling
                scale_factor = np.random.uniform(0.95, 1.05, X[i].shape)
                augmented_sample = augmented_sample * scale_factor
                
                augmented_X.append(augmented_sample)
                augmented_y.append(y[i])
        
        return np.array(augmented_X), np.array(augmented_y)


class AdvancedTrainer:
    """Advanced training pipeline with cross-validation and hyperparameter optimization"""
    
    def __init__(self, model_type: str = "ensemble", cv_folds: int = 5):
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.processor = AdvancedDataProcessor()
        self.models = []
        self.cv_scores = []
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'koi_disposition') -> Tuple[np.ndarray, np.ndarray]:
        """Prepare and engineer features from raw data"""
        # Engineer features
        df_eng = self.processor.engineer_features(df)
        
        # Select features (original + engineered)
        feature_cols = self.processor.feature_names + [
            'pl_density', 'pl_escape_velocity', 'pl_surface_gravity',
            'st_luminosity', 'st_surface_gravity', 'pl_insolation',
            'pl_equilibrium_temp', 'pl_hill_sphere', 'habitable_zone_distance',
            'habitable_zone_ratio', 'transit_depth', 'transit_duration',
            'mass_radius_ratio', 'stellar_planet_ratio', 'orbital_velocity'
        ]
        
        # Filter available columns
        available_cols = [col for col in feature_cols if col in df_eng.columns]
        X = df_eng[available_cols].values
        
        # Prepare target
        if target_col in df.columns:
            # Convert to binary classification
            y = (df[target_col] == 'CONFIRMED').astype(int).values
        else:
            # Generate synthetic labels for demo
            y = np.random.randint(0, 2, len(X))
        
        # Scale features
        X = self.processor.scaler.fit_transform(X)
        
        return X, y
    
    def create_data_loaders(self, X: np.ndarray, y: np.ndarray, 
                          batch_size: int = 64, augment: bool = True) -> Tuple[DataLoader, DataLoader]:
        """Create balanced data loaders with augmentation"""
        
        if augment:
            X, y = self.processor.create_augmented_data(X, y)
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create balanced sampling for training
        class_counts = np.bincount(y_train)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_train]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        # Create datasets and loaders
        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train_single_model(self, X: np.ndarray, y: np.ndarray, 
                          epochs: int = 100, patience: int = 15) -> Dict[str, float]:
        """Train a single model with early stopping"""
        
        # Create model
        model = create_advanced_model(X.shape[1], self.model_type)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(X, y)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Training
            epoch_train_loss = 0
            model.model.train()
            
            for batch_x, batch_y in train_loader:
                loss = model.train_step(batch_x, batch_y)
                epoch_train_loss += loss
            
            # Validation
            val_metrics = model.validate(val_loader)
            
            # Learning rate scheduling
            model.scheduler.step()
            
            # Track metrics
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_metrics['val_loss'])
            val_accuracies.append(val_metrics['val_accuracy'])
            
            # Early stopping
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                patience_counter = 0
                # Save best model
                best_model_state = model.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}, "
                      f"Val Acc: {val_metrics['val_accuracy']:.4f}")
        
        # Load best model
        model.model.load_state_dict(best_model_state)
        
        # Final evaluation
        final_metrics = model.validate(val_loader)
        
        return {
            'model': model,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'final_accuracy': final_metrics['val_accuracy'],
            'final_loss': final_metrics['val_loss']
        }
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, epochs: int = 100) -> Dict[str, float]:
        """Perform cross-validation training"""
        
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\nTraining Fold {fold + 1}/{self.cv_folds}")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            # Train model on fold
            result = self.train_single_model(
                np.vstack([X_train_fold, X_val_fold]), 
                np.hstack([y_train_fold, y_val_fold]),
                epochs=epochs
            )
            
            cv_scores.append(result['final_accuracy'])
            self.models.append(result['model'])
            
            print(f"Fold {fold + 1} Accuracy: {result['final_accuracy']:.4f}")
        
        self.cv_scores = cv_scores
        
        return {
            'mean_cv_score': np.mean(cv_scores),
            'std_cv_score': np.std(cv_scores),
            'cv_scores': cv_scores
        }
    
    def create_ensemble_predictions(self, X: np.ndarray) -> np.ndarray:
        """Create ensemble predictions from all CV models"""
        if not self.models:
            raise ValueError("No trained models available. Run cross_validate first.")
        
        predictions = []
        X_tensor = torch.FloatTensor(X)
        
        for model in self.models:
            pred = model.predict(X_tensor)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Comprehensive model evaluation"""
        predictions = self.create_ensemble_predictions(X_test)
        y_pred = np.argmax(predictions, axis=1)
        y_pred_proba = predictions[:, 1]  # Probability of positive class
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1_score': report['1']['f1-score'],
            'classification_report': report
        }
    
    def save_models(self, save_dir: str):
        """Save all trained models"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = save_path / f"advanced_model_fold_{i}.pth"
            model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'cv_folds': self.cv_folds,
            'cv_scores': self.cv_scores,
            'mean_cv_score': np.mean(self.cv_scores) if self.cv_scores else 0
        }
        
        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Models saved to {save_path}")


def train_advanced_model(data_path: str, model_type: str = "ensemble", 
                        epochs: int = 100, cv_folds: int = 5) -> Dict[str, float]:
    """Main training function"""
    
    print(f"Training Advanced {model_type.upper()} Model")
    print("=" * 50)
    
    # Load data
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} samples from {data_path}")
    else:
        # Create synthetic data for demo
        print("Creating synthetic data for demonstration...")
        np.random.seed(42)
        n_samples = 5000
        
        feature_names = [
            'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
            'pl_orbincl', 'pl_tranmid', 'st_teff', 'st_rad', 'st_mass', 'st_logg'
        ]
        
        # Generate realistic synthetic data
        data = {}
        data['pl_orbper'] = np.random.lognormal(2, 1, n_samples)  # Orbital period
        data['pl_rade'] = np.random.lognormal(0, 0.5, n_samples)  # Planet radius
        data['pl_bmasse'] = np.random.lognormal(0, 1, n_samples)  # Planet mass
        data['pl_orbsmax'] = np.random.lognormal(0, 0.5, n_samples)  # Semi-major axis
        data['pl_orbeccen'] = np.random.beta(0.5, 2, n_samples)  # Eccentricity
        data['pl_orbincl'] = np.random.uniform(0, 180, n_samples)  # Inclination
        data['pl_tranmid'] = np.random.uniform(2450000, 2460000, n_samples)  # Transit time
        data['st_teff'] = np.random.normal(5778, 1000, n_samples)  # Stellar temperature
        data['st_rad'] = np.random.lognormal(0, 0.3, n_samples)  # Stellar radius
        data['st_mass'] = np.random.lognormal(0, 0.3, n_samples)  # Stellar mass
        data['st_logg'] = np.random.normal(4.4, 0.5, n_samples)  # Surface gravity
        
        df = pd.DataFrame(data)
        df['koi_disposition'] = np.random.choice(['CONFIRMED', 'FALSE POSITIVE'], n_samples, p=[0.3, 0.7])
    
    # Initialize trainer
    trainer = AdvancedTrainer(model_type=model_type, cv_folds=cv_folds)
    
    # Prepare data
    X, y = trainer.prepare_data(df)
    print(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")
    
    # Cross-validation training
    cv_results = trainer.cross_validate(X, y, epochs=epochs)
    
    print("\nCross-Validation Results:")
    print(f"Mean CV Accuracy: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")
    
    # Save models
    models_dir = Path(__file__).parent.parent / "models" / "advanced"
    trainer.save_models(str(models_dir))
    
    return cv_results


if __name__ == "__main__":
    # Test different model types
    model_types = ["transformer", "resnet", "ensemble"]
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*60}")
        
        results = train_advanced_model(
            data_path="dummy_data.csv",  # Will create synthetic data
            model_type=model_type,
            epochs=50,  # Reduced for testing
            cv_folds=3   # Reduced for testing
        )
        
        print(f"{model_type.upper()} Results:")
        print(f"Mean Accuracy: {results['mean_cv_score']:.4f}")
        print(f"Std Accuracy: {results['std_cv_score']:.4f}")
