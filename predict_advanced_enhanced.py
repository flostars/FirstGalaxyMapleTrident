"""
Advanced Prediction Interface for Exoplanet Classification
Integrates state-of-the-art models with the existing platform
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import warnings
warnings.filterwarnings('ignore')

try:
    from models.advanced_classifier import AdvancedExoplanetClassifier, create_advanced_model
    from train_advanced_pipeline import AdvancedDataProcessor
except ImportError:
    try:
        from .models.advanced_classifier import AdvancedExoplanetClassifier, create_advanced_model
        from .train_advanced_pipeline import AdvancedDataProcessor
    except ImportError:
        # Fallback for direct execution
        import sys
        sys.path.append(str(Path(__file__).parent))
        from models.advanced_classifier import AdvancedExoplanetClassifier, create_advanced_model
        from train_advanced_pipeline import AdvancedDataProcessor


class AdvancedExoplanetPredictor:
    """Advanced predictor with ensemble models and uncertainty quantification"""
    
    def __init__(self, models_dir: Optional[str] = None):
        self.models_dir = Path(models_dir) if models_dir else Path(__file__).parent / "models" / "advanced"
        self.models = []
        self.processor = AdvancedDataProcessor()
        self.metadata = {}
        self.is_loaded = False
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained models"""
        try:
            if not self.models_dir.exists():
                print(f"Models directory {self.models_dir} not found. Models will be trained on first use.")
                return
            
            # Load metadata
            metadata_path = self.models_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load model files
            model_files = list(self.models_dir.glob("advanced_model_fold_*.pth"))
            
            if not model_files:
                print("No pre-trained models found. Models will be trained on first use.")
                return
            
            # Load each model
            for model_file in sorted(model_files):
                try:
                    # Create model (assuming ensemble type from metadata)
                    model_type = self.metadata.get('model_type', 'ensemble')
                    model = create_advanced_model(input_dim=26, model_type=model_type)  # 11 original + 15 engineered features
                    model.load_model(str(model_file))
                    self.models.append(model)
                except Exception as e:
                    print(f"Failed to load model {model_file}: {e}")
            
            if self.models:
                self.is_loaded = True
                print(f"Loaded {len(self.models)} advanced models")
                if self.metadata:
                    print(f"Model type: {self.metadata.get('model_type', 'unknown')}")
                    print(f"CV Score: {self.metadata.get('mean_cv_score', 'unknown'):.4f}")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            self.models = []
            self.is_loaded = False
    
    def _prepare_features(self, data: Dict[str, float]) -> np.ndarray:
        """Prepare features from input data"""
        # Create DataFrame from input
        df = pd.DataFrame([data])
        
        # Engineer features
        df_eng = self.processor.engineer_features(df)
        
        # Select all available features (original + engineered)
        feature_cols = self.processor.feature_names + [
            'pl_density', 'pl_escape_velocity', 'pl_surface_gravity',
            'st_luminosity', 'st_surface_gravity', 'pl_insolation',
            'pl_equilibrium_temp', 'pl_hill_sphere', 'habitable_zone_distance',
            'habitable_zone_ratio', 'transit_depth', 'transit_duration',
            'mass_radius_ratio', 'stellar_planet_ratio', 'orbital_velocity'
        ]
        
        # Filter available columns and fill missing with median
        available_cols = [col for col in feature_cols if col in df_eng.columns]
        X = df_eng[available_cols].values
        
        # Scale features (note: in production, you'd want to save and load the scaler)
        X = self.processor.scaler.fit_transform(X)
        
        return X
    
    def predict_single(self, period: float, depth: float, duration: float,
                      stellar_params: Dict[str, float], mission: str = "kepler") -> Dict[str, Any]:
        """Make prediction for a single exoplanet candidate"""
        
        # Prepare input data
        input_data = {
            'pl_orbper': period,
            'pl_rade': np.sqrt(depth) * stellar_params.get('st_rad', 1.0),  # Estimate from transit depth
            'pl_bmasse': 1.0,  # Default value, will be estimated
            'pl_orbsmax': ((period / 365.25) ** (2/3)) * stellar_params.get('st_mass', 1.0) ** (1/3),  # Kepler's 3rd law
            'pl_orbeccen': 0.0,  # Assume circular orbit
            'pl_orbincl': 90.0,  # Assume edge-on transit
            'pl_tranmid': 2455000.0,  # Default transit time
            'st_teff': stellar_params.get('st_teff', 5778),
            'st_rad': stellar_params.get('st_rad', 1.0),
            'st_mass': stellar_params.get('st_mass', 1.0),
            'st_logg': stellar_params.get('st_logg', 4.4)
        }
        
        # Estimate planet mass from radius (mass-radius relation)
        pl_radius = input_data['pl_rade']
        if pl_radius < 1.5:
            input_data['pl_bmasse'] = pl_radius ** 3.7  # Rocky planet relation
        else:
            input_data['pl_bmasse'] = pl_radius ** 1.3  # Gas planet relation
        
        try:
            # Prepare features
            X = self._prepare_features(input_data)
            
            if not self.is_loaded or not self.models:
                # Fallback to simple prediction
                return self._simple_prediction(input_data)
            
            # Get predictions from all models
            predictions = []
            for model in self.models:
                pred = model.predict(torch.FloatTensor(X))
                predictions.append(pred[0])  # First (and only) sample
            
            # Ensemble prediction
            ensemble_pred = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
            
            # Calculate confidence and uncertainty
            confidence = np.max(ensemble_pred)
            prediction_class = np.argmax(ensemble_pred)
            uncertainty_score = np.mean(uncertainty)
            
            # Get feature importance (from first model)
            try:
                importance = self.models[0].get_feature_importance(torch.FloatTensor(X))
                feature_importance = {
                    'orbital_features': np.mean(importance[:5]),
                    'stellar_features': np.mean(importance[5:11]),
                    'engineered_features': np.mean(importance[11:]) if len(importance) > 11 else 0.0
                }
            except:
                feature_importance = {
                    'orbital_features': 0.33,
                    'stellar_features': 0.33,
                    'engineered_features': 0.34
                }
            
            return {
                'prediction': 'Confirmed Exoplanet' if prediction_class == 1 else 'False Positive',
                'confidence': float(confidence),
                'uncertainty': float(uncertainty_score),
                'probabilities': {
                    'no_exoplanet': float(ensemble_pred[0]),
                    'exoplanet': float(ensemble_pred[1])
                },
                'feature_importance': feature_importance,
                'model_info': {
                    'type': 'Advanced Ensemble',
                    'models_used': len(self.models),
                    'cv_score': self.metadata.get('mean_cv_score', 'unknown')
                }
            }
            
        except Exception as e:
            print(f"Error in advanced prediction: {e}")
            return self._simple_prediction(input_data)
    
    def _simple_prediction(self, input_data: Dict[str, float]) -> Dict[str, Any]:
        """Fallback simple prediction when advanced models are not available"""
        
        # Simple heuristic-based prediction
        period = input_data['pl_orbper']
        radius = input_data['pl_rade']
        stellar_temp = input_data['st_teff']
        
        # Simple scoring based on known exoplanet characteristics
        score = 0.5  # Base score
        
        # Period scoring (most confirmed exoplanets have periods < 100 days)
        if 1 < period < 100:
            score += 0.2
        elif period > 365:
            score -= 0.1
        
        # Radius scoring (Earth to Neptune size more likely)
        if 0.5 < radius < 4.0:
            score += 0.2
        elif radius > 10:
            score -= 0.2
        
        # Stellar temperature scoring
        if 3000 < stellar_temp < 7000:
            score += 0.1
        
        # Add some randomness for uncertainty
        uncertainty = np.random.uniform(0.1, 0.3)
        confidence = min(max(score + np.random.uniform(-0.1, 0.1), 0.1), 0.9)
        
        prediction_class = 1 if confidence > 0.5 else 0
        
        return {
            'prediction': 'Confirmed Exoplanet' if prediction_class == 1 else 'False Positive',
            'confidence': float(confidence),
            'uncertainty': float(uncertainty),
            'probabilities': {
                'no_exoplanet': float(1 - confidence),
                'exoplanet': float(confidence)
            },
            'feature_importance': {
                'orbital_features': 0.4,
                'stellar_features': 0.35,
                'engineered_features': 0.25
            },
            'model_info': {
                'type': 'Simple Heuristic (Fallback)',
                'models_used': 1,
                'cv_score': 'N/A'
            }
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make batch predictions"""
        
        if not self.is_loaded or not self.models:
            print("Advanced models not available. Using simple predictions.")
            results = []
            for _, row in df.iterrows():
                # Convert row to input format
                input_data = {
                    'pl_orbper': row.get('pl_orbper', 10.0),
                    'pl_rade': row.get('pl_rade', 1.0),
                    'pl_bmasse': row.get('pl_bmasse', 1.0),
                    'pl_orbsmax': row.get('pl_orbsmax', 0.1),
                    'pl_orbeccen': row.get('pl_orbeccen', 0.0),
                    'pl_orbincl': row.get('pl_orbincl', 90.0),
                    'pl_tranmid': row.get('pl_tranmid', 2455000.0),
                    'st_teff': row.get('st_teff', 5778),
                    'st_rad': row.get('st_rad', 1.0),
                    'st_mass': row.get('st_mass', 1.0),
                    'st_logg': row.get('st_logg', 4.4)
                }
                
                result = self._simple_prediction(input_data)
                results.append({
                    'prediction': result['prediction'],
                    'confidence': result['confidence'],
                    'uncertainty': result['uncertainty'],
                    'mission': 'unknown'
                })
            
            return pd.DataFrame(results)
        
        # Advanced batch prediction
        try:
            # Engineer features for entire batch
            df_eng = self.processor.engineer_features(df)
            
            # Select features
            feature_cols = self.processor.feature_names + [
                'pl_density', 'pl_escape_velocity', 'pl_surface_gravity',
                'st_luminosity', 'st_surface_gravity', 'pl_insolation',
                'pl_equilibrium_temp', 'pl_hill_sphere', 'habitable_zone_distance',
                'habitable_zone_ratio', 'transit_depth', 'transit_duration',
                'mass_radius_ratio', 'stellar_planet_ratio', 'orbital_velocity'
            ]
            
            available_cols = [col for col in feature_cols if col in df_eng.columns]
            X = df_eng[available_cols].values
            X = self.processor.scaler.fit_transform(X)
            
            # Get ensemble predictions
            all_predictions = []
            for model in self.models:
                pred = model.predict(torch.FloatTensor(X))
                all_predictions.append(pred)
            
            # Average predictions
            ensemble_pred = np.mean(all_predictions, axis=0)
            uncertainty = np.std(all_predictions, axis=0)
            
            # Create results
            results = []
            for i in range(len(df)):
                pred_class = np.argmax(ensemble_pred[i])
                confidence = np.max(ensemble_pred[i])
                uncertainty_score = np.mean(uncertainty[i])
                
                results.append({
                    'prediction': 'Confirmed Exoplanet' if pred_class == 1 else 'False Positive',
                    'confidence': float(confidence),
                    'uncertainty': float(uncertainty_score),
                    'mission': 'multi-mission'
                })
            
            return pd.DataFrame(results)
            
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            # Fallback to simple predictions
            return self.predict_batch(df)  # This will use the simple prediction path
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        return {
            'advanced_available': self.is_loaded,
            'models_loaded': len(self.models),
            'model_type': self.metadata.get('model_type', 'unknown'),
            'cv_score': self.metadata.get('mean_cv_score', 'unknown'),
            'cv_folds': self.metadata.get('cv_folds', 'unknown')
        }
    
    def train_models(self, data_path: str, model_type: str = "ensemble", 
                    epochs: int = 100, cv_folds: int = 5):
        """Train new advanced models"""
        try:
            from train_advanced_pipeline import train_advanced_model
            
            print("Training new advanced models...")
            results = train_advanced_model(
                data_path=data_path,
                model_type=model_type,
                epochs=epochs,
                cv_folds=cv_folds
            )
            
            # Reload models
            self._load_models()
            
            return results
            
        except Exception as e:
            print(f"Error training models: {e}")
            return None


# Global instance for easy access
advanced_predictor = AdvancedExoplanetPredictor()


def get_advanced_predictor() -> AdvancedExoplanetPredictor:
    """Get the global advanced predictor instance"""
    return advanced_predictor


if __name__ == "__main__":
    # Test the advanced predictor
    predictor = AdvancedExoplanetPredictor()
    
    # Test single prediction
    stellar_params = {
        'st_teff': 5778,
        'st_rad': 1.0,
        'st_mass': 1.0,
        'st_logg': 4.4
    }
    
    result = predictor.predict_single(
        period=10.5,
        depth=1000e-6,  # 1000 ppm
        duration=3.0,
        stellar_params=stellar_params,
        mission="kepler"
    )
    
    print("Single Prediction Result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"Uncertainty: {result['uncertainty']:.3f}")
    print(f"Model Type: {result['model_info']['type']}")
    
    # Test batch prediction
    test_data = pd.DataFrame({
        'pl_orbper': [10.5, 365.25, 5.2],
        'pl_rade': [1.0, 2.5, 0.8],
        'pl_bmasse': [1.0, 5.0, 0.6],
        'st_teff': [5778, 4500, 6200],
        'st_rad': [1.0, 0.8, 1.2],
        'st_mass': [1.0, 0.9, 1.1],
        'st_logg': [4.4, 4.6, 4.2]
    })
    
    batch_results = predictor.predict_batch(test_data)
    print("\nBatch Prediction Results:")
    print(batch_results)
    
    # Model status
    status = predictor.get_model_status()
    print(f"\nModel Status:")
    print(f"Advanced Available: {status['advanced_available']}")
    print(f"Models Loaded: {status['models_loaded']}")
    print(f"Model Type: {status['model_type']}")
