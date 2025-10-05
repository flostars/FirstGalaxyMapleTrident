#!/usr/bin/env python3
"""
Clean CSV data files by removing comment lines and fixing parsing issues
"""

import pandas as pd
from pathlib import Path
import re

def clean_csv_file(input_path: str, output_path: str = None) -> str:
    """Clean a CSV file by removing comment lines and fixing parsing issues"""
    if output_path is None:
        output_path = input_path.replace('.csv', '_cleaned.csv')
    
    print(f"Cleaning {input_path}...")
    
    # Read the file line by line to handle comments
    cleaned_lines = []
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line_num, line in enumerate(f, 1):
            # Skip comment lines and empty lines
            if line.strip().startswith('#') or not line.strip():
                continue
            
            # Clean the line - remove any problematic characters
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
    
    # Write cleaned data
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(cleaned_lines))
    
    print(f"Cleaned {len(cleaned_lines)} data lines, saved to {output_path}")
    return output_path

def main():
    """Clean all CSV files in the data directory"""
    data_dir = Path("data")
    if not data_dir.exists():
        print("Data directory not found!")
        return
    
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("No CSV files found in data directory!")
        return
    
    print(f"Found {len(csv_files)} CSV files to clean:")
    for file in csv_files:
        print(f"  - {file.name}")
    
    # Clean each file
    cleaned_files = []
    for csv_file in csv_files:
        try:
            cleaned_file = clean_csv_file(str(csv_file))
            cleaned_files.append(cleaned_file)
        except Exception as e:
            print(f"Error cleaning {csv_file}: {e}")
    
    print(f"\nSuccessfully cleaned {len(cleaned_files)} files!")
    
    # Test loading one of the cleaned files
    if cleaned_files:
        test_file = cleaned_files[0]
        try:
            df = pd.read_csv(test_file)
            print(f"\nTest load successful! Loaded {len(df)} rows with {len(df.columns)} columns")
            print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
        except Exception as e:
            print(f"Error testing cleaned file: {e}")

if __name__ == "__main__":
    main()
