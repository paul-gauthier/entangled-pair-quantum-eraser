#!/usr/bin/env python3

import csv
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple, Any, Optional
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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


def cosine_model(x, amplitude, offset, period, phase):
    """Cosine model: amplitude * cos(2π * x / period + phase) + offset"""
    return amplitude * np.cos(2 * np.pi * x / period + phase) + offset


def determine_piezo_period(dataset: PhotonicsDataset, count_type: str = 'N_i') -> Tuple[float, dict]:
    """
    Determine the piezo period (steps per 2π) by fitting a cosine to the data.

    Args:
        dataset: PhotonicsDataset to analyze
        count_type: Which count type to use ('N_s', 'N_i', or 'N_c')

    Returns:
        Tuple of (period_in_steps, fit_info_dict)
    """
    if not dataset.piezo_data:
        raise ValueError("No piezo data available")

    # Extract data
    positions = np.array(sorted(dataset.piezo_data.keys()))
    counts = np.array([dataset.piezo_data[pos][count_type] for pos in positions])

    # Subtract dark counts if available
    if count_type in dataset.dark_counts:
        counts = counts - dataset.dark_counts[count_type]

    # Initial parameter guesses
    amplitude_guess = (np.max(counts) - np.min(counts)) / 2
    offset_guess = np.mean(counts)
    period_guess = (positions[-1] - positions[0]) / 2  # Assume ~2 cycles in the data
    phase_guess = 0

    initial_guess = [amplitude_guess, offset_guess, period_guess, phase_guess]

    try:
        # Fit the cosine model
        popt, pcov = curve_fit(cosine_model, positions, counts, p0=initial_guess)
        amplitude, offset, period, phase = popt

        # Calculate fit quality metrics
        fitted_counts = cosine_model(positions, *popt)
        residuals = counts - fitted_counts
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((counts - np.mean(counts)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Parameter uncertainties
        param_errors = np.sqrt(np.diag(pcov))

        fit_info = {
            'amplitude': amplitude,
            'offset': offset,
            'period': period,
            'phase': phase,
            'amplitude_error': param_errors[0],
            'offset_error': param_errors[1],
            'period_error': param_errors[2],
            'phase_error': param_errors[3],
            'r_squared': r_squared,
            'positions': positions,
            'counts': counts,
            'fitted_counts': fitted_counts,
            'residuals': residuals
        }

        return abs(period), fit_info

    except Exception as e:
        raise RuntimeError(f"Failed to fit cosine model: {e}")


def plot_piezo_period_fit(fit_info: dict, dataset_name: str = "", count_type: str = 'N_i',
                         output_filename: str = None, show: bool = True):
    """Plot the piezo period fit results."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    positions = fit_info['positions']
    counts = fit_info['counts']
    fitted_counts = fit_info['fitted_counts']
    residuals = fit_info['residuals']

    # Main plot
    ax1.scatter(positions, counts, alpha=0.7, label='Data', color='blue')
    ax1.plot(positions, fitted_counts, 'r-', label='Cosine fit', linewidth=2)
    ax1.set_xlabel('Piezo position (steps)')
    ax1.set_ylabel(f'{count_type} counts')
    ax1.set_title(f'Piezo Period Determination - {dataset_name}\n'
                  f'Period = {fit_info["period"]:.2f} ± {fit_info["period_error"]:.2f} steps, '
                  f'R² = {fit_info["r_squared"]:.4f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Residuals plot
    ax2.scatter(positions, residuals, alpha=0.7, color='red')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Piezo position (steps)')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Fit Residuals')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_filename:
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_filename}")

    if show:
        plt.show()

    return fig


def analyze_all_periods(datasets: List[PhotonicsDataset], count_type: str = 'N_i') -> dict:
    """Analyze piezo periods for all datasets."""
    results = {}

    for i, dataset in enumerate(datasets):
        try:
            period, fit_info = determine_piezo_period(dataset, count_type)
            results[i] = {
                'dataset_name': dataset.name,
                'period': period,
                'period_error': fit_info['period_error'],
                'r_squared': fit_info['r_squared'],
                'fit_info': fit_info
            }
            print(f"Dataset {i+1} ({dataset.name}): Period = {period:.2f} ± {fit_info['period_error']:.2f} steps, R² = {fit_info['r_squared']:.4f}")
        except Exception as e:
            print(f"Failed to analyze dataset {i+1} ({dataset.name}): {e}")
            results[i] = None

    return results


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
    if len(sys.argv) != 2:
        print("Usage: fit.py <csv_filename>", file=sys.stderr)
        sys.exit(1)

    csv_filename = sys.argv[1]

    try:
        datasets = parse_photonics_csv(csv_filename)
        print_dataset_summary(datasets)

        # Convert to DataFrame for analysis
        df = datasets_to_dataframe(datasets)
        print("DataFrame shape:", df.shape)
        print("\nFirst few rows:")
        print(df.head())

        print("\n" + "="*60)
        print("PIEZO PERIOD ANALYSIS")
        print("="*60)

        # Analyze piezo periods for all datasets
        period_results = analyze_all_periods(datasets, count_type='N_i')

        # Calculate average period across all successful fits
        successful_periods = [r['period'] for r in period_results.values() if r is not None]
        if successful_periods:
            avg_period = np.mean(successful_periods)
            std_period = np.std(successful_periods)
            print(f"\nAverage period across all datasets: {avg_period:.2f} ± {std_period:.2f} steps per 2π")
            print(f"This corresponds to {2*np.pi/avg_period:.4f} radians per step")

        # Create output directory based on CSV filename
        csv_basename = os.path.splitext(os.path.basename(csv_filename))[0]
        output_dir = csv_basename
        os.makedirs(output_dir, exist_ok=True)

        # Plot the first successful fit as an example
        for i, result in period_results.items():
            if result is not None:
                output_path = os.path.join(output_dir, f"piezo_period_fit_dataset_{i+1}.pdf")
                plot_piezo_period_fit(
                    result['fit_info'],
                    dataset_name=result['dataset_name'],
                    output_filename=output_path,
                    show=False
                )
                break

    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)
