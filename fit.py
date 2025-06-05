#!/usr/bin/env python3

import csv
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import curve_fit
import argparse
import matplotlib.pyplot as plt


class PhotonicsDataset:
    """Represents a single dataset with N_s, N_i, N_c columns."""

    def __init__(self, name: str = ""):
        self.name = name
        self.dark_counts = {}  # N_s, N_i, N_c dark counts
        self.piezo_pos = np.array([])
        self.N_s = np.array([])
        self.N_i = np.array([])
        self.N_c = np.array([])
        self.metadata = {}     # metadata key -> value

    def __repr__(self):
        return f"PhotonicsDataset(name='{self.name}', {len(self.piezo_pos)} piezo positions)"


def parse_photonics_csv(filepath: str) -> List[PhotonicsDataset]:
    """
    Parse a CSV file containing photonics datasets.

    Format:
    - Header row: "Piezo motor position", "N_s", "N_i", "N_c", "N_s", "N_i", "N_c", ...
    - Dark counts row: "Pump blocked", values for each dataset
    - Data rows: piezo position, followed by N_s, N_i, N_c values for each dataset
    - Empty rows separate data from metadata
    - Metadata rows: label in first column, values in subsequent columns

    Returns:
        List of PhotonicsDataset objects
    """
    datasets = []

    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return datasets

    # Parse header to determine number of datasets
    header = rows[0]
    # Count triplets of N_s, N_i, N_c columns after the first column
    num_datasets = (len(header) - 1) // 3

    # Initialize datasets
    for i in range(num_datasets):
        datasets.append(PhotonicsDataset(name=f"Dataset_{i+1}"))

    # Find where data ends and metadata begins
    data_end_idx = 1  # Start after header
    for i, row in enumerate(rows[1:], 1):
        if not row[0] or row[0].strip() == '':
            data_end_idx = i
            break
        # Check if this looks like a numeric piezo position or "Pump blocked"
        if row[0] not in ["Pump blocked"] and not _is_numeric_or_empty(row[0]):
            data_end_idx = i
            break

    # Parse dark counts (row with "Pump blocked")
    dark_counts_row = None
    for i, row in enumerate(rows[1:data_end_idx], 1):
        if row[0] == "Pump blocked":
            dark_counts_row = row
            break

    if dark_counts_row:
        for dataset_idx in range(num_datasets):
            col_start = 1 + dataset_idx * 3
            datasets[dataset_idx].dark_counts = {
                'N_s': _parse_float(dark_counts_row[col_start]) if col_start < len(dark_counts_row) else None,
                'N_i': _parse_float(dark_counts_row[col_start + 1]) if col_start + 1 < len(dark_counts_row) else None,
                'N_c': _parse_float(dark_counts_row[col_start + 2]) if col_start + 2 < len(dark_counts_row) else None,
            }

    # Parse data rows (piezo positions and counts)
    piezo_positions = []
    data_rows = []

    for row in rows[1:data_end_idx]:
        if not row[0] or row[0] == "Pump blocked":
            continue

        piezo_pos = _parse_float(row[0])
        if piezo_pos is not None:
            piezo_positions.append(piezo_pos)
            data_rows.append(row)

    # Extract data for each dataset
    for dataset_idx in range(num_datasets):
        col_start = 1 + dataset_idx * 3

        N_s_values = []
        N_i_values = []
        N_c_values = []

        for row in data_rows:
            N_s = _parse_float(row[col_start]) if col_start < len(row) else None
            N_i = _parse_float(row[col_start + 1]) if col_start + 1 < len(row) else None
            N_c = _parse_float(row[col_start + 2]) if col_start + 2 < len(row) else None

            N_s_values.append(N_s if N_s is not None else 0)
            N_i_values.append(N_i if N_i is not None else 0)
            N_c_values.append(N_c if N_c is not None else 0)

        datasets[dataset_idx].piezo_pos = np.array(piezo_positions)
        datasets[dataset_idx].N_s = np.array(N_s_values)
        datasets[dataset_idx].N_i = np.array(N_i_values)
        datasets[dataset_idx].N_c = np.array(N_c_values)

    # Parse metadata rows
    metadata_start_idx = data_end_idx
    while metadata_start_idx < len(rows) and (not rows[metadata_start_idx] or rows[metadata_start_idx][0].strip() == ''):
        metadata_start_idx += 1

    for row in rows[metadata_start_idx:]:
        if not row or not row[0]:
            continue

        metadata_key = row[0].strip()
        if not metadata_key:
            continue

        # Find the value for each dataset (can be in any of the 3 columns for that dataset)
        for dataset_idx in range(num_datasets):
            col_start = 1 + dataset_idx * 3
            value = None

            # Check all 3 columns for this dataset to find the value
            for col_offset in range(3):
                col_idx = col_start + col_offset
                if col_idx < len(row) and row[col_idx].strip():
                    value = row[col_idx].strip()
                    break

            if value is not None:
                # Try to parse as number, otherwise keep as string
                numeric_value = _parse_float(value)
                datasets[dataset_idx].metadata[metadata_key] = numeric_value if numeric_value is not None else value

    return datasets


def _parse_float(value: str) -> Optional[float]:
    """Parse a string as float, return None if not possible."""
    if not value or not isinstance(value, str):
        return None

    value = value.strip()
    if not value:
        return None

    try:
        return float(value)
    except ValueError:
        return None


def _is_numeric_or_empty(value: str) -> bool:
    """Check if a string is numeric or empty."""
    if not value or not isinstance(value, str):
        return True

    value = value.strip()
    if not value:
        return True

    try:
        float(value)
        return True
    except ValueError:
        return False


def main():
    """Test the CSV parser with the provided data file."""
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = "2025-06-02.csv"

    if not os.path.exists(filepath):
        print(f"File {filepath} not found.")
        return

    datasets = parse_photonics_csv(filepath)

    print(f"Parsed {len(datasets)} datasets from {filepath}")
    print()

    for i, dataset in enumerate(datasets):
        print(f"Dataset {i+1}:")
        print(f"  Dark counts: {dataset.dark_counts}")
        print(f"  Data points: {len(dataset.piezo_pos)}")
        print(f"  Piezo range: {dataset.piezo_pos.min():.1f} to {dataset.piezo_pos.max():.1f}")
        print(f"  N_s range: {dataset.N_s.min():.0f} to {dataset.N_s.max():.0f}")
        print(f"  N_i range: {dataset.N_i.min():.0f} to {dataset.N_i.max():.0f}")
        print(f"  N_c range: {dataset.N_c.min():.0f} to {dataset.N_c.max():.0f}")
        print(f"  Metadata: {dataset.metadata}")
        print()


if __name__ == "__main__":
    main()
