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

    def print(self):
        """Print a summary of this dataset."""
        print(f"Dataset {self.name}:")
        print(f"  Dark counts: {self.dark_counts}")
        print(f"  Data points: {len(self.piezo_pos)}")
        print(f"  Piezo range: {self.piezo_pos.min():.1f} to {self.piezo_pos.max():.1f}")
        if self.N_s is not None:
            print(f"  N_s range: {self.N_s.min():.0f} to {self.N_s.max():.0f} (mean: {self.N_s.mean():.1f})")
        else:
            print(f"  N_s: all zeros")

        if self.N_i is not None:
            print(f"  N_i range: {self.N_i.min():.0f} to {self.N_i.max():.0f} (mean: {self.N_i.mean():.1f})")
        else:
            print(f"  N_i: all zeros")

        if self.N_c is not None:
            print(f"  N_c range: {self.N_c.min():.0f} to {self.N_c.max():.0f} (mean: {self.N_c.mean():.1f})")
        else:
            print(f"  N_c: all zeros")
        print(f"  Metadata: {self.metadata}")
        print()


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

        # Subtract dark counts from data
        dark_N_s = datasets[dataset_idx].dark_counts.get('N_s', 0) or 0
        dark_N_i = datasets[dataset_idx].dark_counts.get('N_i', 0) or 0
        dark_N_c = datasets[dataset_idx].dark_counts.get('N_c', 0) or 0

        datasets[dataset_idx].N_s = np.maximum(0, datasets[dataset_idx].N_s - dark_N_s)
        datasets[dataset_idx].N_i = np.maximum(0, datasets[dataset_idx].N_i - dark_N_i)
        datasets[dataset_idx].N_c = np.maximum(0, datasets[dataset_idx].N_c - dark_N_c)

        # Set to None if all values are zero
        if np.all(datasets[dataset_idx].N_s == 0):
            datasets[dataset_idx].N_s = None
        if np.all(datasets[dataset_idx].N_i == 0):
            datasets[dataset_idx].N_i = None
        if np.all(datasets[dataset_idx].N_c == 0):
            datasets[dataset_idx].N_c = None

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

    # Update dataset names using 'name' metadata if available
    for dataset in datasets:
        if 'name' in dataset.metadata:
            dataset.name = str(dataset.metadata['name'])

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


# ---------- Fitting utilities ----------
def _cos_model(x: np.ndarray, A: float, k: float, phi: float, C: float) -> np.ndarray:
    """
    Cosine model: y = A * cos(k * x + phi) + C

    k is the angular wavenumber, k = 2π / period.
    """
    return A * np.cos(k * x + phi) + C


def _initial_guess(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """Return initial guesses for A, k, phi, C."""
    C = np.mean(y)
    A = (np.max(y) - np.min(y)) / 2 if np.max(y) != np.min(y) else 1.0
    span = x.max() - x.min() if x.max() != x.min() else 1.0
    guess_period = 22
    k = 2 * np.pi / guess_period
    phi = 0.0
    return A, k, phi, C


def _fit_single_series(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Fit one N_i or N_c series allowing its own phase, amplitude and
    offset, but yielding an angular wavenumber k that should be common
    to all series.  Returns (k_fit, k_var).
    """
    # Keep only finite data
    mask = np.isfinite(y)
    if not np.any(mask):
        return np.nan, np.nan

    x = x[mask]
    y = y[mask]

    p0 = _initial_guess(x, y)
    try:
        popt, pcov = curve_fit(_cos_model, x, y, p0=p0, maxfev=10000)
        k_fit = popt[1]
        k_var = pcov[1, 1] if pcov.size >= 4 else np.nan
        return k_fit, k_var
    except (RuntimeError, ValueError):
        # curve_fit failed
        return np.nan, np.nan


def fit_period(datasets: List[PhotonicsDataset]) -> Tuple[float, float]:
    """
    Fit a cosine to all N_i and N_c counts from every dataset.

    Returns
    -------
    period : float
        Piezo position change corresponding to a 2π phase shift.
    period_std : float
        One-sigma uncertainty derived from the covariance of the fit.
    """
    # --- Fit each N_i and N_c series separately, allowing an
    #     independent phase for every series but extracting its k ---
    periods: List[float] = []
    k_values: List[float] = []
    k_vars: List[float] = []

    for ds in datasets:
        for attr in ("N_i", "N_c"):
            counts = getattr(ds, attr)
            if counts is None:
                continue

            k_fit, k_var = _fit_single_series(ds.piezo_pos, counts)
            if not np.isfinite(k_fit):
                continue  # skip series that failed to fit

            k_values.append(k_fit)
            k_vars.append(k_var if np.isfinite(k_var) and k_var > 0 else np.nan)
            periods.append(2 * np.pi / k_fit)

    if not k_values:
        raise RuntimeError("No usable N_i or N_c data found for fitting.")

    period_array = np.array(periods)

    # Use inverse-variance weighting when variances are available
    if np.all(np.isfinite(k_vars)):
        weights = 1 / np.array(k_vars)
        k_mean = np.average(k_values, weights=weights)
        period_mean = 2 * np.pi / k_mean
        # Propagate uncertainty of weighted mean
        k_mean_var = 1 / np.sum(weights)
        period_std = (2 * np.pi) * np.sqrt(k_mean_var) / (k_mean ** 2)
    else:
        # Fall back to simple mean / sample-std
        period_mean = float(np.mean(period_array))
        period_std = float(np.std(period_array, ddof=1)) if len(period_array) > 1 else np.nan

    return float(period_mean), float(period_std)


def fit_phases(datasets: List[PhotonicsDataset], period: float) -> Dict[str, Any]:
    """
    Simultaneously fit all N_i and N_c data using shared phase parameters.
    
    N_i is fit to: A_i * (1 + cos(k * x + phi)) 
    N_c is fit to: A_c * (1 + cos(k * x + phi + phi_c))
    
    Where k = 2π / period is fixed from the period fit.
    
    Parameters
    ----------
    datasets : List[PhotonicsDataset]
        List of datasets containing N_i and N_c data
    period : float
        Period in piezo steps for 2π phase change
        
    Returns
    -------
    dict
        Dictionary containing fit results:
        - 'phi': global phase offset (radians)
        - 'phi_c': coincidence phase offset (radians) 
        - 'phi_std': uncertainty in phi
        - 'phi_c_std': uncertainty in phi_c
        - 'amplitudes': dict of fitted amplitudes for each series
        - 'fit_success': boolean indicating if fit converged
    """
    from scipy.optimize import curve_fit
    
    # Fixed wavenumber from period
    k = 2 * np.pi / period
    
    # Collect all data points
    all_x = []
    all_y = []
    series_info = []  # (start_idx, end_idx, series_type, dataset_name)
    
    current_idx = 0
    for ds in datasets:
        for attr in ("N_i", "N_c"):
            counts = getattr(ds, attr)
            if counts is None:
                continue
                
            # Filter out non-finite data
            mask = np.isfinite(counts)
            if not np.any(mask):
                continue
                
            x_data = ds.piezo_pos[mask]
            y_data = counts[mask]
            
            start_idx = current_idx
            end_idx = current_idx + len(y_data)
            
            all_x.extend(x_data)
            all_y.extend(y_data)
            series_info.append((start_idx, end_idx, attr, ds.name))
            
            current_idx = end_idx
    
    if not all_x:
        raise RuntimeError("No usable data found for phase fitting.")
    
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    
    # Create the combined model function
    def combined_model(x, phi, phi_c, *amplitudes):
        """
        Combined model for all series.
        Parameters: phi, phi_c, then one amplitude per series
        """
        result = np.zeros_like(x)
        
        for i, (start_idx, end_idx, series_type, _) in enumerate(series_info):
            A = amplitudes[i]
            x_segment = x[start_idx:end_idx]
            
            if series_type == "N_i":
                y_segment = A * (1 + np.cos(k * x_segment + phi))
            else:  # N_c
                y_segment = A * (1 + np.cos(k * x_segment + phi + phi_c))
                
            result[start_idx:end_idx] = y_segment
            
        return result
    
    # Initial parameter guesses
    phi_guess = 0.0
    phi_c_guess = 0.0
    
    # Estimate amplitude for each series
    amp_guesses = []
    for start_idx, end_idx, series_type, _ in series_info:
        y_segment = all_y[start_idx:end_idx]
        amp_guess = np.mean(y_segment) / 2 if len(y_segment) > 0 else 1.0
        amp_guesses.append(amp_guess)
    
    p0 = [phi_guess, phi_c_guess] + amp_guesses
    
    try:
        # Perform the fit
        popt, pcov = curve_fit(combined_model, all_x, all_y, p0=p0, maxfev=10000)
        
        phi_fit = popt[0]
        phi_c_fit = popt[1]
        amplitude_fits = popt[2:]
        
        # Extract uncertainties
        param_stds = np.sqrt(np.diag(pcov))
        phi_std = param_stds[0]
        phi_c_std = param_stds[1]
        
        # Store amplitudes by series
        amplitudes = {}
        for i, (_, _, series_type, dataset_name) in enumerate(series_info):
            key = f"{dataset_name}_{series_type}"
            amplitudes[key] = amplitude_fits[i]
        
        return {
            'phi': float(phi_fit),
            'phi_c': float(phi_c_fit),
            'phi_std': float(phi_std),
            'phi_c_std': float(phi_c_std),
            'amplitudes': amplitudes,
            'fit_success': True,
            'period_used': period,
            'k_used': k
        }
        
    except (RuntimeError, ValueError) as e:
        return {
            'phi': np.nan,
            'phi_c': np.nan,
            'phi_std': np.nan,
            'phi_c_std': np.nan,
            'amplitudes': {},
            'fit_success': False,
            'error': str(e),
            'period_used': period,
            'k_used': k
        }


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
        dataset.print()

    # Fit cosine to all N_i and N_c data to estimate the piezo step change for a 2π phase shift
    try:
        period, period_std = fit_period(datasets)
        print(f"Estimated period (Δpiezo for 2π): {period:.2f} ± {period_std:.2f} (piezo steps)")
        
        # Now fit the global phases using the determined period
        phase_results = fit_phases(datasets, period)
        
        if phase_results['fit_success']:
            print(f"\nGlobal phase fit results:")
            print(f"  phi (global phase): {phase_results['phi']:.3f} ± {phase_results['phi_std']:.3f} rad")
            print(f"  phi_c (coincidence phase): {phase_results['phi_c']:.3f} ± {phase_results['phi_c_std']:.3f} rad")
            print(f"  Period used: {phase_results['period_used']:.2f} piezo steps")
            print(f"\nFitted amplitudes:")
            for series, amplitude in phase_results['amplitudes'].items():
                print(f"    {series}: {amplitude:.1f}")
        else:
            print(f"\nGlobal phase fit failed: {phase_results.get('error', 'Unknown error')}")
            
    except RuntimeError as exc:
        print(f"Could not determine period: {exc}")


if __name__ == "__main__":
    main()
