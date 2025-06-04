import csv
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class PhotonicsDataset:
    """Represents a single dataset with N_s, N_i, N_c columns."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.dark_counts = {}  # N_s, N_i, N_c dark counts
        self.piezo_data = {}   # piezo_position -> {N_s, N_i, N_c}
        self.metadata = {}     # metadata key -> value
    
    def __repr__(self):
        return f"PhotonicsDataset(name='{self.name}', {len(self.piezo_data)} piezo positions)"


def parse_photonics_csv(filename: str) -> List[PhotonicsDataset]:
    """
    Parse a CSV file containing photonics datasets.
    
    Returns a list of PhotonicsDataset objects, one for each set of N_s, N_i, N_c columns.
    """
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows:
        return []
    
    # Parse header to find dataset columns
    header = rows[0]
    datasets = []
    
    # Find groups of N_s, N_i, N_c columns
    i = 1  # Skip first column (piezo motor position)
    dataset_idx = 0
    
    while i < len(header):
        if i + 2 < len(header):  # Need at least 3 columns for N_s, N_i, N_c
            # Look for N_s, N_i, N_c pattern (allowing for empty headers)
            dataset = PhotonicsDataset(name=f"Dataset_{dataset_idx + 1}")
            datasets.append(dataset)
            i += 3  # Move to next potential dataset
            dataset_idx += 1
        else:
            break
    
    if not datasets:
        return []
    
    # Parse data rows
    data_section = True
    metadata_section = False
    
    for row_idx, row in enumerate(rows[1:], 1):  # Skip header
        if not row or all(cell.strip() == '' for cell in row):
            # Empty row indicates transition to metadata
            data_section = False
            metadata_section = True
            continue
        
        first_col = row[0].strip() if row else ""
        
        if data_section:
            if first_col == "Pump blocked":
                # Dark counts row
                col_idx = 1
                for dataset in datasets:
                    if col_idx + 2 < len(row):
                        try:
                            dataset.dark_counts['N_s'] = int(row[col_idx]) if row[col_idx].strip() else 0
                            dataset.dark_counts['N_i'] = int(row[col_idx + 1]) if row[col_idx + 1].strip() else 0
                            dataset.dark_counts['N_c'] = int(row[col_idx + 2]) if row[col_idx + 2].strip() else 0
                        except (ValueError, IndexError):
                            pass
                    col_idx += 3
            
            elif first_col.isdigit():
                # Piezo position data row
                piezo_pos = int(first_col)
                col_idx = 1
                
                for dataset in datasets:
                    if col_idx + 2 < len(row):
                        try:
                            ns = int(row[col_idx]) if row[col_idx].strip() else 0
                            ni = int(row[col_idx + 1]) if row[col_idx + 1].strip() else 0
                            nc = int(row[col_idx + 2]) if row[col_idx + 2].strip() else 0
                            
                            dataset.piezo_data[piezo_pos] = {
                                'N_s': ns,
                                'N_i': ni, 
                                'N_c': nc
                            }
                        except (ValueError, IndexError):
                            pass
                    col_idx += 3
        
        elif metadata_section and first_col:
            # Metadata row - find which column has the value for each dataset
            col_idx = 1
            for dataset in datasets:
                if col_idx + 2 < len(row):
                    # Check all three columns for this dataset to find the value
                    for offset in range(3):
                        if col_idx + offset < len(row) and row[col_idx + offset].strip():
                            try:
                                # Try to parse as number first
                                value = float(row[col_idx + offset])
                                if value.is_integer():
                                    value = int(value)
                            except ValueError:
                                # Keep as string if not a number
                                value = row[col_idx + offset].strip()
                            
                            dataset.metadata[first_col] = value
                            break
                col_idx += 3
    
    return datasets


def datasets_to_dataframe(datasets: List[PhotonicsDataset]) -> pd.DataFrame:
    """
    Convert parsed datasets to a pandas DataFrame for analysis.
    
    Returns a DataFrame with columns: dataset, piezo_position, N_s, N_i, N_c
    """
    rows = []
    
    for i, dataset in enumerate(datasets):
        for piezo_pos, counts in dataset.piezo_data.items():
            rows.append({
                'dataset': i,
                'dataset_name': dataset.name,
                'piezo_position': piezo_pos,
                'N_s': counts['N_s'],
                'N_i': counts['N_i'],
                'N_c': counts['N_c']
            })
    
    return pd.DataFrame(rows)


def print_dataset_summary(datasets: List[PhotonicsDataset]):
    """Print a summary of the parsed datasets."""
    print(f"Parsed {len(datasets)} datasets:")
    print()
    
    for i, dataset in enumerate(datasets):
        print(f"Dataset {i + 1} ({dataset.name}):")
        print(f"  Dark counts: {dataset.dark_counts}")
        print(f"  Piezo positions: {len(dataset.piezo_data)} points")
        if dataset.piezo_data:
            positions = sorted(dataset.piezo_data.keys())
            print(f"  Position range: {positions[0]} to {positions[-1]}")
        print(f"  Metadata: {len(dataset.metadata)} items")
        for key, value in dataset.metadata.items():
            print(f"    {key}: {value}")
        print()


if __name__ == "__main__":
    # Example usage
    datasets = parse_photonics_csv("2025-06-02.csv")
    print_dataset_summary(datasets)
    
    # Convert to DataFrame for analysis
    df = datasets_to_dataframe(datasets)
    print("DataFrame shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
