#!/usr/bin/env python3
"""
Advanced ExoVision AI Prediction Pipeline
Real-time exoplanet detection with multi-modal AI
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import our advanced models
import sys
sys.path.append(str(Path(__file__).parent))
from models.exovision_ai import ExoVisionAI, create_exovision_model, load_pretrained_model
from data.light_curve_generator import LightCurveGenerator, LightCurveProcessor


class AdvancedExoVisionPredictor:
    """Advanced prediction pipeline for ExoVision AI."""
    
    def __init__(self, model_path: str = "models/exovision_ai_advanced.pth", device: str = 'cpu'):
        self.device = device
        self.model = None
        self.light_curve_generator = LightCurveGenerator()
        self.light_curve_processor = LightCurveProcessor()
        self.feature_columns = [
            'pl_orbper', 'pl_orbsmax', 'pl_rade', 'pl_bmasse',
            'pl_eqt', 'pl_insol', 'st_teff', 'st_rad', 'st_mass',
            'st_met', 'st_logg'
        ]
        
        # Load model
        self.load_model(model_path)
    
    def load_model(self, model_path: str):
        """Load the trained ExoVision AI model."""
        try:
            import torch
            if Path(model_path).exists():
                self.model = create_exovision_model()
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                print(f"✅ Advanced model loaded from {model_path}")
            else:
                print(f"⚠️  Model file not found: {model_path}")
                print("   Using basic model fallback...")
                self.model = None
        except ImportError:
            print("⚠️  PyTorch not available. Using basic model fallback...")
            self.model = None
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
    
    def predict_single(self, 
                      period: float,
                      depth: float,
                      duration: float,
                      stellar_params: Dict[str, float],
                      mission: str = 'kepler') -> Dict:
        """Predict exoplanet for a single candidate."""
        
        if self.model is None:
            return self._fallback_prediction()
        
        try:
            # Generate light curve
            light_curve = self.light_curve_generator.generate_light_curve(
                period=period,
                depth=depth,
                duration=duration
            )
            
            # Prepare stellar features
            stellar_features = self._prepare_stellar_features(stellar_params)
            
            # Convert to tensors
            light_curve_tensor = torch.FloatTensor(light_curve).unsqueeze(0)
            stellar_features_tensor = torch.FloatTensor(stellar_features).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                result = self.model(light_curve_tensor, stellar_features_tensor, mission)
            
            # Extract results
            prediction = torch.argmax(result['logits'], dim=1).item()
            probabilities = torch.softmax(result['logits'], dim=1).squeeze().numpy()
            confidence = 1.0 - result['uncertainty'].item()
            
            # Get interpretability features
            interpretability = self.model.get_interpretability(
                light_curve_tensor, stellar_features_tensor, mission
            )
            
            return {
                'prediction': 'EXOPLANET' if prediction == 1 else 'NO_EXOPLANET',
                'confidence': confidence,
                'probabilities': {
                    'no_exoplanet': float(probabilities[0]),
                    'exoplanet': float(probabilities[1])
                },
                'uncertainty': float(result['uncertainty'].item()),
                'feature_importance': {
                    'light_curve': float(interpretability['light_curve_importance'].item()),
                    'stellar_features': interpretability['stellar_importance'].numpy().tolist()
                },
                'attention_weights': interpretability['attention_weights'],
                'mission': mission
            }
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return self._fallback_prediction()
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict exoplanets for a batch of candidates."""
        
        if self.model is None:
            return self._fallback_batch_prediction(df)
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                # Extract parameters
                period = row.get('pl_orbper', row.get('koi_period', 10.0))
                depth = row.get('pl_trandep', row.get('koi_depth', 1000.0)) / 1e6
                duration = row.get('pl_trandurh', row.get('koi_duration', 2.0))
                
                # Determine mission
                mission = 'kepler'
                if 'tid' in row and pd.notna(row['tid']):
                    mission = 'tess'
                elif 'kepid' not in row or pd.isna(row['kepid']):
                    mission = 'k2'
                
                # Extract stellar parameters
                stellar_params = {}
                for col in self.feature_columns:
                    if col in row and pd.notna(row[col]):
                        stellar_params[col] = float(row[col])
                    else:
                        stellar_params[col] = 0.0
                
                # Make prediction
                prediction = self.predict_single(
                    period=period,
                    depth=depth,
                    duration=duration,
                    stellar_params=stellar_params,
                    mission=mission
                )
                
                # Add to results
                result_row = row.to_dict()
                result_row.update({
                    'prediction': prediction['prediction'],
                    'confidence': prediction['confidence'],
                    'uncertainty': prediction['uncertainty'],
                    'mission': prediction['mission']
                })
                results.append(result_row)
                
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                # Add fallback result
                result_row = row.to_dict()
                result_row.update({
                    'prediction': 'UNKNOWN',
                    'confidence': 0.0,
                    'uncertainty': 1.0,
                    'mission': 'unknown'
                })
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def ensemble_predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get ensemble predictions across all missions."""
        
        if self.model is None:
            return self._fallback_batch_prediction(df)
        
        results = []
        
        for idx, row in df.iterrows():
            try:
                # Extract parameters
                period = row.get('pl_orbper', row.get('koi_period', 10.0))
                depth = row.get('pl_trandep', row.get('koi_depth', 1000.0)) / 1e6
                duration = row.get('pl_trandurh', row.get('koi_duration', 2.0))
                
                # Extract stellar parameters
                stellar_params = {}
                for col in self.feature_columns:
                    if col in row and pd.notna(row[col]):
                        stellar_params[col] = float(row[col])
                    else:
                        stellar_params[col] = 0.0
                
                # Generate light curve
                light_curve = self.light_curve_generator.generate_light_curve(
                    period=period,
                    depth=depth,
                    duration=duration
                )
                
                # Prepare stellar features
                stellar_features = self._prepare_stellar_features(stellar_params)
                
                # Convert to tensors
                light_curve_tensor = torch.FloatTensor(light_curve).unsqueeze(0)
                stellar_features_tensor = torch.FloatTensor(stellar_features).unsqueeze(0)
                
                # Get ensemble prediction
                with torch.no_grad():
                    ensemble_result = self.model.ensemble_predict(
                        light_curve_tensor, stellar_features_tensor
                    )
                
                # Extract results
                prediction = torch.argmax(ensemble_result['ensemble_prediction'], dim=1).item()
                probabilities = torch.softmax(ensemble_result['ensemble_prediction'], dim=1).squeeze().numpy()
                confidence = ensemble_result['ensemble_confidence'].item()
                
                # Get individual mission predictions
                individual_predictions = {}
                for i, mission in enumerate(['kepler', 'k2', 'tess']):
                    mission_pred = torch.argmax(ensemble_result['individual_predictions'][0, i], dim=0).item()
                    mission_conf = ensemble_result['individual_confidences'][0, i].item()
                    individual_predictions[mission] = {
                        'prediction': 'EXOPLANET' if mission_pred == 1 else 'NO_EXOPLANET',
                        'confidence': float(mission_conf)
                    }
                
                # Add to results
                result_row = row.to_dict()
                result_row.update({
                    'prediction': 'EXOPLANET' if prediction == 1 else 'NO_EXOPLANET',
                    'confidence': float(confidence),
                    'probabilities': {
                        'no_exoplanet': float(probabilities[0]),
                        'exoplanet': float(probabilities[1])
                    },
                    'individual_predictions': individual_predictions,
                    'ensemble_weights': ensemble_result['weights'][0].numpy().tolist()
                })
                results.append(result_row)
                
            except Exception as e:
                print(f"❌ Error processing row {idx}: {e}")
                # Add fallback result
                result_row = row.to_dict()
                result_row.update({
                    'prediction': 'UNKNOWN',
                    'confidence': 0.0,
                    'individual_predictions': {}
                })
                results.append(result_row)
        
        return pd.DataFrame(results)
    
    def _prepare_stellar_features(self, stellar_params: Dict[str, float]) -> np.ndarray:
        """Prepare stellar features for the model."""
        features = []
        for col in self.feature_columns:
            features.append(stellar_params.get(col, 0.0))
        return np.array(features)
    
    def _fallback_prediction(self) -> Dict:
        """Fallback prediction when model is not available."""
        return {
            'prediction': 'UNKNOWN',
            'confidence': 0.0,
            'probabilities': {'no_exoplanet': 0.5, 'exoplanet': 0.5},
            'uncertainty': 1.0,
            'feature_importance': {'light_curve': 0.0, 'stellar_features': [0.0] * 11},
            'attention_weights': None,
            'mission': 'unknown'
        }
    
    def _fallback_batch_prediction(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback batch prediction when model is not available."""
        results = []
        for idx, row in df.iterrows():
            result_row = row.to_dict()
            result_row.update({
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'uncertainty': 1.0,
                'mission': 'unknown'
            })
            results.append(result_row)
        return pd.DataFrame(results)


def main():
    """Test the advanced prediction pipeline."""
    print("🚀 Advanced ExoVision AI Prediction Pipeline")
    print("=" * 50)
    
    # Initialize predictor
    predictor = AdvancedExoVisionPredictor()
    
    # Test single prediction
    print("🧪 Testing single prediction...")
    test_params = {
        'pl_orbper': 10.5,
        'pl_orbsmax': 0.1,
        'pl_rade': 1.2,
        'pl_bmasse': 2.1,
        'pl_eqt': 300.0,
        'pl_insol': 1.0,
        'st_teff': 5800.0,
        'st_rad': 1.0,
        'st_mass': 1.0,
        'st_met': 0.0,
        'st_logg': 4.4
    }
    
    result = predictor.predict_single(
        period=10.5,
        depth=0.001,
        duration=3.0,
        stellar_params=test_params,
        mission='kepler'
    )
    
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Uncertainty: {result['uncertainty']:.4f}")
    
    print("\n✅ Advanced prediction pipeline ready!")


if __name__ == "__main__":
    main()
