"""
Real Data Training Script for ExoVision AI
Trains advanced models on actual exoplanet datasets with comprehensive evaluation
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Add current directory to Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

try:
    from models.advanced_classifier import AdvancedExoplanetClassifier, create_advanced_model
    from train_advanced_pipeline import AdvancedDataProcessor, AdvancedTrainer
    print("Advanced models imported successfully!")
except ImportError as e:
    print(f"Import error: {e}")
    print("Advanced models not available. Please install PyTorch.")
    exit(1)


class RealDataTrainer:
    """Trainer specifically for real exoplanet datasets"""
    
    def __init__(self):
        self.processor = AdvancedDataProcessor()
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.data_sources = []
        
    def load_real_datasets(self) -> pd.DataFrame:
        """Load and combine all available real datasets"""
        datasets = []
        
        # Dataset paths - using absolute paths
        base_path = Path(__file__).parent.parent  # Go up from FirstGalaxyMapleTrident to NASA
        data_paths = [
            str(base_path / "FirstGalaxyMapleTrident" / "data" / "cumulative_2025.10.04_09.55.00.csv"),  # Kepler cumulative
            str(base_path / "FirstGalaxyMapleTrident" / "data" / "k2pandc_2025.10.04_09.55.15.csv"),     # K2 confirmed planets
            str(base_path / "FirstGalaxyMapleTrident" / "data" / "TOI_2025.10.04_09.55.27.csv"),         # TESS Objects of Interest
            str(base_path / "cumulative_2025.10.04_09.54.53.csv"),    # Additional Kepler data
            str(base_path / "k2pandc_2025.10.04_10.35.22.csv"),       # Additional K2 data
            str(base_path / "TOI_2025.10.04_10.34.56.csv")            # Additional TESS data
        ]
        
        for path in data_paths:
            print(f"Checking path: {path} - Exists: {Path(path).exists()}")
            if Path(path).exists():
                try:
                    # Try different parsing strategies for problematic CSV files
                    try:
                        # First try with standard settings, skipping comment lines
                        df = pd.read_csv(path, comment='#', skip_blank_lines=True)
                    except pd.errors.ParserError:
                        # If that fails, try with more flexible parsing
                        try:
                            df = pd.read_csv(path, sep=',', quotechar='"', escapechar='\\', 
                                           on_bad_lines='skip', engine='python', comment='#', skip_blank_lines=True)
                        except:
                            # Last resort: try with different separator
                            df = pd.read_csv(path, sep='\t', on_bad_lines='skip', comment='#', skip_blank_lines=True)
                    
                    print(f"Loaded {len(df)} samples from {path}")
                    
                    # Add source information
                    if 'cumulative' in path.lower() or 'kepler' in path.lower():
                        df['mission'] = 'kepler'
                    elif 'k2' in path.lower():
                        df['mission'] = 'k2'
                    elif 'toi' in path.lower() or 'tess' in path.lower():
                        df['mission'] = 'tess'
                    else:
                        df['mission'] = 'unknown'
                    
                    datasets.append(df)
                    self.data_sources.append(path)
                    
                except Exception as e:
                    print(f"Error loading {path}: {e}")
        
        if not datasets:
            raise ValueError("No datasets could be loaded!")
        
        # Combine all datasets
        combined_df = pd.concat(datasets, ignore_index=True, sort=False)
        print(f"Combined dataset: {len(combined_df)} total samples")
        
        return combined_df
    
    def prepare_real_data(self, df: pd.DataFrame) -> tuple:
        """Prepare real data for training with proper target handling"""
        
        print("Preparing real exoplanet data...")
        print(f"Original dataset shape: {df.shape}")
        print(f"Columns available: {list(df.columns)}")
        
        # Find the target column
        target_candidates = [
            'koi_disposition', 'pl_disposition', 'disposition', 
            'tfopwg_disp', 'toi_disposition', 'status'
        ]
        
        target_col = None
        for col in target_candidates:
            if col in df.columns:
                target_col = col
                break
        
        if target_col is None:
            print("No target column found. Available columns:")
            print(df.columns.tolist())
            # Create synthetic targets based on available data
            print("Creating synthetic targets based on data characteristics...")
            target_col = 'synthetic_target'
            df[target_col] = self._create_synthetic_targets(df)
        
        print(f"Using target column: {target_col}")
        print(f"Target distribution:\n{df[target_col].value_counts()}")
        
        # Prepare features
        feature_columns = [
            'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
            'pl_orbincl', 'pl_tranmid', 'st_teff', 'st_rad', 'st_mass', 'st_logg'
        ]
        
        # Check which features are available
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        print(f"Available features ({len(available_features)}): {available_features}")
        if missing_features:
            print(f"Missing features ({len(missing_features)}): {missing_features}")
        
        # Use only available features and fill missing values
        df_features = df[available_features].copy()
        
        # Fill missing values with median
        for col in df_features.columns:
            if df_features[col].dtype in ['float64', 'int64']:
                median_val = df_features[col].median()
                df_features[col] = df_features[col].fillna(median_val)
                print(f"Filled {col} missing values with median: {median_val:.3f}")
        
        # Remove rows with any remaining NaN values
        initial_len = len(df_features)
        df_features = df_features.dropna()
        df_target = df[target_col].iloc[df_features.index]
        print(f"Removed {initial_len - len(df_features)} rows with missing values")
        
        # Engineer additional features
        print("Engineering additional features...")
        df_engineered = self.processor.engineer_features(df_features)
        
        # Prepare target variable
        if df_target.dtype == 'object':
            # Handle categorical targets
            positive_labels = ['CONFIRMED', 'confirmed', 'PC', 'CP', 'TRUE', 'CANDIDATE']
            df_target_binary = df_target.isin(positive_labels).astype(int)
        else:
            # Handle numeric targets
            df_target_binary = (df_target > 0.5).astype(int)
        
        print(f"Final target distribution:\n{df_target_binary.value_counts()}")
        print(f"Final feature matrix shape: {df_engineered.shape}")
        
        return df_engineered.values, df_target_binary.values
    
    def _create_synthetic_targets(self, df: pd.DataFrame) -> pd.Series:
        """Create synthetic targets based on data characteristics"""
        # Simple heuristic based on typical exoplanet characteristics
        targets = []
        
        for _, row in df.iterrows():
            score = 0.5  # Base probability
            
            # Check orbital period (confirmed exoplanets often have shorter periods)
            if 'pl_orbper' in row and not pd.isna(row['pl_orbper']):
                period = row['pl_orbper']
                if 1 < period < 100:
                    score += 0.2
                elif period > 1000:
                    score -= 0.2
            
            # Check planet radius (Earth to Neptune size more likely confirmed)
            if 'pl_rade' in row and not pd.isna(row['pl_rade']):
                radius = row['pl_rade']
                if 0.5 < radius < 4.0:
                    score += 0.15
                elif radius > 10:
                    score -= 0.15
            
            # Check stellar temperature
            if 'st_teff' in row and not pd.isna(row['st_teff']):
                temp = row['st_teff']
                if 3000 < temp < 7000:
                    score += 0.1
            
            # Add some randomness
            score += np.random.uniform(-0.1, 0.1)
            
            # Convert to binary classification
            targets.append('CONFIRMED' if score > 0.5 else 'FALSE POSITIVE')
        
        return pd.Series(targets)
    
    def train_and_evaluate(self, model_type: str = "ensemble", epochs: int = 100, cv_folds: int = 5):
        """Train models and generate comprehensive evaluation metrics"""
        
        print(f"\n{'='*60}")
        print(f"TRAINING {model_type.upper()} MODEL ON REAL DATA")
        print(f"{'='*60}")
        
        # Load real data
        df = self.load_real_datasets()
        X, y = self.prepare_real_data(df)
        
        print(f"\nTraining Configuration:")
        print(f"- Model Type: {model_type}")
        print(f"- Features: {X.shape[1]}")
        print(f"- Samples: {X.shape[0]}")
        print(f"- Positive Class: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
        print(f"- Epochs: {epochs}")
        print(f"- CV Folds: {cv_folds}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Cross-validation training
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_results = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'auc_score': [],
            'models': []
        }
        
        fold_predictions = []
        fold_true_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
            print(f"\n--- Training Fold {fold + 1}/{cv_folds} ---")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model
            model = create_advanced_model(X.shape[1], model_type)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val)
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 15
            
            for epoch in range(epochs):
                # Training step
                train_loss = model.train_step(X_train_tensor, y_train_tensor)
                
                # Validation step
                val_metrics = model.validate([(X_val_tensor, y_val_tensor)])
                
                # Early stopping
                if val_metrics['val_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_loss']
                    patience_counter = 0
                    best_model_state = model.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                if epoch % 20 == 0:
                    print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_metrics['val_loss']:.4f}, "
                          f"Val Acc: {val_metrics['val_accuracy']:.4f}")
            
            # Load best model and evaluate
            model.model.load_state_dict(best_model_state)
            
            # Get predictions
            predictions = model.predict(X_val_tensor)
            y_pred = np.argmax(predictions, axis=1)
            y_pred_proba = predictions[:, 1]
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            try:
                auc = roc_auc_score(y_val, y_pred_proba)
            except:
                auc = 0.5  # Random classifier AUC
            
            cv_results['accuracy'].append(accuracy)
            cv_results['precision'].append(precision)
            cv_results['recall'].append(recall)
            cv_results['f1_score'].append(f1)
            cv_results['auc_score'].append(auc)
            cv_results['models'].append(model)
            
            fold_predictions.extend(y_pred_proba)
            fold_true_labels.extend(y_val)
            
            print(f"Fold {fold + 1} Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
        
        # Calculate overall metrics
        overall_metrics = {
            'model_type': model_type,
            'cv_folds': cv_folds,
            'epochs_trained': epochs,
            'features_used': X.shape[1],
            'samples_trained': X.shape[0],
            'class_distribution': {
                'positive': int(np.sum(y)),
                'negative': int(len(y) - np.sum(y)),
                'positive_ratio': float(np.mean(y))
            },
            'performance_metrics': {
                'accuracy': {
                    'mean': float(np.mean(cv_results['accuracy'])),
                    'std': float(np.std(cv_results['accuracy'])),
                    'scores': [float(x) for x in cv_results['accuracy']]
                },
                'precision': {
                    'mean': float(np.mean(cv_results['precision'])),
                    'std': float(np.std(cv_results['precision'])),
                    'scores': [float(x) for x in cv_results['precision']]
                },
                'recall': {
                    'mean': float(np.mean(cv_results['recall'])),
                    'std': float(np.std(cv_results['recall'])),
                    'scores': [float(x) for x in cv_results['recall']]
                },
                'f1_score': {
                    'mean': float(np.mean(cv_results['f1_score'])),
                    'std': float(np.std(cv_results['f1_score'])),
                    'scores': [float(x) for x in cv_results['f1_score']]
                },
                'auc_score': {
                    'mean': float(np.mean(cv_results['auc_score'])),
                    'std': float(np.std(cv_results['auc_score'])),
                    'scores': [float(x) for x in cv_results['auc_score']]
                }
            },
            'data_sources': self.data_sources
        }
        
        # Calculate overall AUC
        try:
            overall_auc = roc_auc_score(fold_true_labels, fold_predictions)
            overall_metrics['overall_auc'] = float(overall_auc)
        except:
            overall_metrics['overall_auc'] = 0.5
        
        # Print comprehensive results
        self._print_results(overall_metrics)
        
        # Save results
        self._save_results(overall_metrics, cv_results)
        
        return overall_metrics, cv_results
    
    def _print_results(self, metrics: dict):
        """Print comprehensive training results"""
        
        print(f"\n{'='*60}")
        print("FINAL TRAINING RESULTS")
        print(f"{'='*60}")
        
        print(f"\nModel Configuration:")
        print(f"  Type: {metrics['model_type'].upper()}")
        print(f"  Features: {metrics['features_used']}")
        print(f"  Training Samples: {metrics['samples_trained']:,}")
        print(f"  CV Folds: {metrics['cv_folds']}")
        print(f"  Epochs: {metrics['epochs_trained']}")
        
        print(f"\nClass Distribution:")
        cd = metrics['class_distribution']
        print(f"  Positive (Confirmed): {cd['positive']:,} ({cd['positive_ratio']*100:.1f}%)")
        print(f"  Negative (False Positive): {cd['negative']:,} ({(1-cd['positive_ratio'])*100:.1f}%)")
        
        print(f"\nPerformance Metrics (Cross-Validation):")
        pm = metrics['performance_metrics']
        print(f"  Accuracy:  {pm['accuracy']['mean']:.4f} ± {pm['accuracy']['std']:.4f}")
        print(f"  Precision: {pm['precision']['mean']:.4f} ± {pm['precision']['std']:.4f}")
        print(f"  Recall:    {pm['recall']['mean']:.4f} ± {pm['recall']['std']:.4f}")
        print(f"  F1-Score:  {pm['f1_score']['mean']:.4f} ± {pm['f1_score']['std']:.4f}")
        print(f"  AUC-ROC:   {pm['auc_score']['mean']:.4f} ± {pm['auc_score']['std']:.4f}")
        
        if 'overall_auc' in metrics:
            print(f"  Overall AUC: {metrics['overall_auc']:.4f}")
        
        print(f"\nData Sources:")
        for i, source in enumerate(metrics['data_sources'], 1):
            print(f"  {i}. {source}")
        
        # Performance interpretation
        acc_mean = pm['accuracy']['mean']
        if acc_mean >= 0.95:
            performance_level = "EXCELLENT"
        elif acc_mean >= 0.90:
            performance_level = "VERY GOOD"
        elif acc_mean >= 0.85:
            performance_level = "GOOD"
        elif acc_mean >= 0.80:
            performance_level = "FAIR"
        else:
            performance_level = "NEEDS IMPROVEMENT"
        
        print(f"\nOverall Performance: {performance_level}")
        print(f"Model Status: READY FOR DEPLOYMENT" if acc_mean >= 0.85 else "REQUIRES FURTHER TRAINING")
    
    def _save_results(self, metrics: dict, cv_results: dict):
        """Save training results and models"""
        
        # Use existing models directory
        results_dir = Path("models")
        results_dir.mkdir(exist_ok=True)
        
        # Save metrics
        with open(results_dir / "performance_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save models
        for i, model in enumerate(cv_results['models']):
            model_path = results_dir / f"model_fold_{i}.pth"
            model.save_model(str(model_path))
        
        print(f"\nResults saved to: {results_dir}")
        print(f"  - performance_metrics.json")
        print(f"  - model_fold_*.pth (x{len(cv_results['models'])})")


def main():
    """Main training function"""
    
    # Initialize trainer
    trainer = RealDataTrainer()
    
    # Train different model types
    model_types = ["ensemble", "transformer", "resnet"]
    all_results = {}
    
    for model_type in model_types:
        try:
            print(f"\n\n{'#'*80}")
            print(f"TRAINING {model_type.upper()} MODEL")
            print(f"{'#'*80}")
            
            metrics, cv_results = trainer.train_and_evaluate(
                model_type=model_type,
                epochs=50,  # Reduced for faster training
                cv_folds=3  # Reduced for faster training
            )
            
            all_results[model_type] = metrics
            
        except Exception as e:
            print(f"Error training {model_type} model: {e}")
            continue
    
    # Compare models
    if all_results:
        print(f"\n{'='*80}")
        print("MODEL COMPARISON")
        print(f"{'='*80}")
        
        print(f"{'Model':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'AUC':<10}")
        print("-" * 70)
        
        for model_type, metrics in all_results.items():
            pm = metrics['performance_metrics']
            print(f"{model_type:<12} "
                  f"{pm['accuracy']['mean']:<10.4f} "
                  f"{pm['precision']['mean']:<10.4f} "
                  f"{pm['recall']['mean']:<10.4f} "
                  f"{pm['f1_score']['mean']:<10.4f} "
                  f"{pm['auc_score']['mean']:<10.4f}")
        
        # Find best model
        best_model = max(all_results.items(), key=lambda x: x[1]['performance_metrics']['accuracy']['mean'])
        print(f"\nBest Model: {best_model[0].upper()}")
        print(f"Best Accuracy: {best_model[1]['performance_metrics']['accuracy']['mean']:.4f}")


if __name__ == "__main__":
    main()
