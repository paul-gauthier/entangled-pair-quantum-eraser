#!/usr/bin/env python3

import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any
from scipy.optimize import curve_fit

from readcsv import PhotonicsDataset, parse_photonics_csv


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
            phi_deg = np.degrees(phase_results['phi'])
            phi_std_deg = np.degrees(phase_results['phi_std'])
            phi_c_deg = np.degrees(phase_results['phi_c'])
            phi_c_std_deg = np.degrees(phase_results['phi_c_std'])
            
            print(f"\nGlobal phase fit results:")
            print(f"  phi (global phase): {phase_results['phi']:.3f} ± {phase_results['phi_std']:.3f} rad ({phi_deg:.1f} ± {phi_std_deg:.1f}°)")
            print(f"  phi_c (coincidence phase): {phase_results['phi_c']:.3f} ± {phase_results['phi_c_std']:.3f} rad ({phi_c_deg:.1f} ± {phi_c_std_deg:.1f}°)")
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
