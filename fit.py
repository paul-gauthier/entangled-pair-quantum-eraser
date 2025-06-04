#!/usr/bin/env python3

import csv
import pandas as pd
import numpy as np
import sys
from typing import Dict, List, Tuple, Any, Optional


# Consider a model for the N_i and N_c counts from an experiment:

# `f_before` the fraction of signal photons that are lost before they reach the eraser.
# The which way info on the signal has decohered into the environment.
# The idler can not self-interfere.
#
# f_before is a fixed attribute of the apparatus.

# `f_eraser` represents the fraction of signals which are absorbed/lost during
# the erasing operation. Their which way info has decohered into the environment.
# Their idler partner can not self-interfere.
#
# Erasing is performed with an LP at a specific angle.
# Typically the angles are 0, 45 or 90, all of which will absorb half the
# arriving Phi+ signals and transmit the other half. So f_eraser will usually
# be 0.5 in this case, but can be set to other constants for other experimental
# setups.
#
# `f_eraser` is a constant for a given experiment, not fitted from data.

# `f_after` is the fraction of signal photons lost after the eraser before reaching
# the signal detector.
#
# f_after is a fixed attribute of the apparatus.

# `e` is the fraction of idlers which are being erased in the experiment. This
# depends on the setting of the signal LP, but also on imperfections in the apparatus.
# In real experimental results, e will never be 1 or 0. Hopefully, when the eraser is on
# e will approach 1 and when it's off, it will approach 0. But pragmatically, it's not
# likely to be especially close to these extremes.
#
# e varies depending on specific experimental settings like the signal LP angle.

# `delta` is the phase difference in the MZI arms introduced by the piezo mirror stage.
#
# Each experiment sweeps a range of piezo mirror positions to collect counts at varying delta.

# `phi` is the offset to align delta with the phase of the other arm
#
# phi is different for each experiment, because thermal drift can alter the relative
# path lengths of the MZI arms.

# `phi_c` is the extra phase seen only by coincidences
#
# phi_c is a fixed attribute of the apparatus.

# `R` is the rate of entangled pair production.
#
# R is a fixed constant that doesn't vary between experiments.

N__i = R * (
    # Signals lost before the eraser can't interfere, half go out each MZI port
    + 1/2 * f_before * f_eraser

    # Non-erased pairs can't interfere either, half go out each MZI port
    + 1/2 * (1 - f_before * f_eraser) * (1 - e)

    # Signals that reached the eraser and were erased
    # ... their idlers will oscillate between all and none going out each MZI port
    + 1/2 * (1 - f_before * f_eraser) * e * (cos(delta + phi) + 1)
)

N_c = R * (1 - f_before * f_eraser) * (1 - f_after) * ( # signals that pass the eraser and reach the detector
    # Non-erased pairs can't interfere, half go out each MZI port
    + 1/2 * (1 - e)

    # erased pairs oscillate between all and none going out each MZI port
    + 1/2 * e * (cos(delta + phi + phi_c) + 1)
)

# When fitting data to these models, we can use these approaches:
#
# • R cancels out of every visibility and cosine amplitude ratio, so
# it can only be inferred from the absolute level of the idler
# singles.  That is fine, but its value is irrelevant to everything
# else; you could fix R = 2 × mean(N_i) and make the fit more stable.
#
# • f_before * f_eraser and f_after never appear separately in any modulation
# depth – they enter only through the product g ≡ (1–f_before * f_eraser)(1–f_after).
# – From N_c∕N_i one gets g directly.
# – From the idler visibility V_i^run = (1–f_before * f_eraser) e_run and the coincidence
#   visibility V_c^run = e_run one can solve algebraically for (1–f_before * f_eraser).


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
    except FileNotFoundError:
        print(f"Error: File '{csv_filename}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing file: {e}", file=sys.stderr)
        sys.exit(1)
