"""
Live Data Integration for ExoVision AI
Connects to NASA Exoplanet Archive API for real-time data updates
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class NASAExoplanetAPI:
    """NASA Exoplanet Archive API integration"""
    
    def __init__(self):
        self.base_url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ExoVision-AI/1.0 (Educational Research)'
        })
        
    def get_latest_exoplanets(self, limit: int = 1000) -> pd.DataFrame:
        """Get latest confirmed exoplanets"""
        query = f"""
        SELECT 
            pl_name, hostname, discoverymethod, disc_year, disc_facility,
            pl_orbper, pl_rade, pl_bmasse, pl_orbsmax, pl_orbeccen,
            pl_orbincl, pl_tranmid, st_teff, st_rad, st_mass, st_logg,
            sy_dist, sy_vmag, pl_eqt, pl_insol, pl_controv_flag
        FROM ps 
        WHERE default_flag = 1 
        AND pl_name IS NOT NULL
        ORDER BY rowupdate DESC
        LIMIT {limit}
        """
        
        try:
            response = self.session.get(
                self.base_url,
                params={
                    'query': query,
                    'format': 'json'
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            
            # Clean and process data
            df = self._clean_data(df)
            return df
            
        except Exception as e:
            print(f"Error fetching data from NASA API: {e}")
            return pd.DataFrame()
    
    def get_kepler_candidates(self, limit: int = 500) -> pd.DataFrame:
        """Get latest Kepler candidates"""
        query = f"""
        SELECT 
            kepoi_name, koi_disposition, koi_score, koi_period, koi_depth,
            koi_duration, koi_impact, koi_steff, koi_slogg, koi_srad,
            koi_smass, koi_kepmag, ra, dec
        FROM q1_q17_dr24_koi 
        WHERE koi_disposition IN ('CONFIRMED', 'CANDIDATE', 'FALSE POSITIVE')
        ORDER BY koi_score DESC
        LIMIT {limit}
        """
        
        try:
            response = self.session.get(
                self.base_url,
                params={
                    'query': query,
                    'format': 'json'
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            print(f"Error fetching Kepler data: {e}")
            return pd.DataFrame()
    
    def get_tess_tois(self, limit: int = 500) -> pd.DataFrame:
        """Get latest TESS TOIs"""
        query = f"""
        SELECT 
            toi, tid, tfopwg_disp, pl_orbper, pl_rade, pl_bmasse,
            pl_orbsmax, pl_orbeccen, st_teff, st_rad, st_mass, st_logg,
            st_tmag, st_dist, pl_trandurh, pl_trandep
        FROM toi 
        WHERE tfopwg_disp IN ('PC', 'KP', 'CP', 'FP')
        ORDER BY toi
        LIMIT {limit}
        """
        
        try:
            response = self.session.get(
                self.base_url,
                params={
                    'query': query,
                    'format': 'json'
                },
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data)
            return df
            
        except Exception as e:
            print(f"Error fetching TESS data: {e}")
            return pd.DataFrame()
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize data"""
        if df.empty:
            return df
            
        # Convert numeric columns
        numeric_cols = [
            'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
            'pl_orbincl', 'pl_tranmid', 'st_teff', 'st_rad', 'st_mass', 'st_logg',
            'sy_dist', 'sy_vmag', 'pl_eqt', 'pl_insol'
        ]
        
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add mission column
        df['mission'] = 'nasa_archive'
        
        # Add data source timestamp
        df['data_source'] = 'nasa_api'
        df['last_updated'] = datetime.now().isoformat()
        
        return df
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data"""
        try:
            # Get counts from different sources
            exoplanets = self.get_latest_exoplanets(limit=10)
            kepler = self.get_kepler_candidates(limit=10)
            tess = self.get_tess_tois(limit=10)
            
            return {
                'exoplanets_count': len(exoplanets),
                'kepler_count': len(kepler),
                'tess_count': len(tess),
                'last_updated': datetime.now().isoformat(),
                'api_status': 'active'
            }
        except Exception as e:
            return {
                'exoplanets_count': 0,
                'kepler_count': 0,
                'tess_count': 0,
                'last_updated': datetime.now().isoformat(),
                'api_status': f'error: {str(e)}'
            }


class LiveDataManager:
    """Manages live data updates and caching"""
    
    def __init__(self, cache_dir: str = "data/live"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.api = NASAExoplanetAPI()
        self.cache_duration = timedelta(hours=1)  # Cache for 1 hour
        
    def get_cached_data(self, data_type: str) -> Optional[pd.DataFrame]:
        """Get cached data if still fresh"""
        cache_file = self.cache_dir / f"{data_type}_cache.json"
        
        if not cache_file.exists():
            return None
            
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still fresh
            cache_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cache_time > self.cache_duration:
                return None
                
            # Load cached DataFrame
            df = pd.DataFrame(cache_data['data'])
            return df
            
        except Exception as e:
            print(f"Error loading cached data: {e}")
            return None
    
    def cache_data(self, data_type: str, df: pd.DataFrame):
        """Cache data with timestamp"""
        cache_file = self.cache_dir / f"{data_type}_cache.json"
        
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': df.to_dict('records')
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            print(f"Error caching data: {e}")
    
    def get_live_data(self, data_type: str = "exoplanets", force_refresh: bool = False) -> pd.DataFrame:
        """Get live data with caching"""
        
        # Try to get cached data first
        if not force_refresh:
            cached_data = self.get_cached_data(data_type)
            if cached_data is not None:
                print(f"Using cached {data_type} data")
                return cached_data
        
        # Fetch fresh data
        print(f"Fetching fresh {data_type} data from NASA API...")
        
        if data_type == "exoplanets":
            df = self.api.get_latest_exoplanets()
        elif data_type == "kepler":
            df = self.api.get_kepler_candidates()
        elif data_type == "tess":
            df = self.api.get_tess_tois()
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        if not df.empty:
            # Cache the data
            self.cache_data(data_type, df)
            print(f"Successfully fetched and cached {len(df)} {data_type} records")
        else:
            print(f"No {data_type} data available")
            
        return df
    
    def get_all_live_data(self) -> Dict[str, pd.DataFrame]:
        """Get all available live data"""
        data_types = ["exoplanets", "kepler", "tess"]
        results = {}
        
        for data_type in data_types:
            results[data_type] = self.get_live_data(data_type)
            
        return results
    
    def get_data_status(self) -> Dict:
        """Get status of all data sources"""
        status = {}
        
        for data_type in ["exoplanets", "kepler", "tess"]:
            cached_data = self.get_cached_data(data_type)
            if cached_data is not None:
                status[data_type] = {
                    'status': 'cached',
                    'count': len(cached_data),
                    'last_updated': cached_data.get('last_updated', 'unknown')
                }
            else:
                status[data_type] = {
                    'status': 'not_cached',
                    'count': 0,
                    'last_updated': 'never'
                }
        
        return status


def test_live_data_integration():
    """Test the live data integration"""
    print("🚀 Testing Live Data Integration")
    print("=" * 50)
    
    # Test API connection
    api = NASAExoplanetAPI()
    summary = api.get_data_summary()
    print(f"API Status: {summary}")
    
    # Test data manager
    manager = LiveDataManager()
    
    # Test fetching exoplanets
    print("\n📡 Fetching latest exoplanets...")
    exoplanets = manager.get_live_data("exoplanets")
    print(f"Fetched {len(exoplanets)} exoplanets")
    
    if not exoplanets.empty:
        print(f"Sample columns: {list(exoplanets.columns)[:10]}")
        print(f"Sample data:\n{exoplanets.head(2)}")
    
    # Test data status
    status = manager.get_data_status()
    print(f"\n📊 Data Status: {status}")


if __name__ == "__main__":
    test_live_data_integration()
