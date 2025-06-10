#!/usr/bin/env python

import argparse
import json
import os
import sys

import numpy as np

from plot_utils import fit_steps_per_2pi, plot_counts

# ---------------------------------------------------------------------------
# Dark–count handling
# ---------------------------------------------------------------------------
ACCIDENTAL_WINDOW = 25e-9  # 25 ns coincidence window


def load_jsonl_data(filename):
    """Load data from a JSONL file and return arrays of piezo steps and counts."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Sort by step to ensure proper ordering
    data.sort(key=lambda x: x.get("stage_position", x.get("step")))

    # Extract arrays
    piezo_steps = np.array([d.get("stage_position", d.get("step")) for d in data])
    Ns = np.array([d["N_s"] for d in data])
    Ni = np.array([d["N_i"] for d in data])
    Nc = np.array([d["N_c"] for d in data])

    return piezo_steps, Ns, Ni, Nc


def load_and_correct_datasets(jsonl_filename):
    """
    Load datasets from a single JSONL file and apply dark count corrections.

    The file contains multiple datasets separated by dark count records with dark=True.
    Each dataset is corrected using the dark record that immediately follows it.

    Parameters
    ----------
    jsonl_filename : str
        Path to the JSONL file containing multiple datasets.

    Returns
    -------
    list of dict
        List of dictionaries, one for each dataset.
        Each dictionary contains:
        - 'filename': The original filename of the dataset.
        - 'dataset_index': Index of the dataset within the file.
        - 'piezo_steps', 'Ns': Raw data arrays.
        - 'Ni_corr', 'Nc_corr': Dark-corrected data arrays.
    """
    if not os.path.exists(jsonl_filename):
        print(f"Warning: File {jsonl_filename} not found.")
        return []

    # Load all records from the file
    data = []
    with open(jsonl_filename, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Split into datasets at dark=True records
    datasets = []
    current_dataset = []

    for record in data:
        if record.get("dark", False):
            # Found a dark record - save current dataset if it exists
            if current_dataset:
                datasets.append((current_dataset, record))
                current_dataset = []
        else:
            current_dataset.append(record)

    # Discard any final records after the last dark=True
    if current_dataset:
        print(f"  Warning: Discarding {len(current_dataset)} records after last dark measurement")

    if not datasets:
        print(f"  Warning: No datasets found in {jsonl_filename}")
        return []

    # Get reference stage positions from first dataset
    first_dataset_data, _ = datasets[0]
    first_dataset_data.sort(key=lambda x: x.get("stage_position", x.get("step")))
    reference_positions = [d.get("stage_position", d.get("step")) for d in first_dataset_data]

    print(f"  Found {len(datasets)} datasets in {jsonl_filename}")
    print(
        f"  Reference dataset has {len(reference_positions)} stage positions: {reference_positions}"
    )

    corrected_datasets = []

    for dataset_index, (dataset_data, dark_record) in enumerate(datasets):
        # Sort dataset by stage position
        dataset_data.sort(key=lambda x: x.get("stage_position", x.get("step")))

        # Check if this dataset has the same stage positions as the first
        positions = [d.get("stage_position", d.get("step")) for d in dataset_data]

        if positions != reference_positions:
            print(f"  Warning: Dataset {dataset_index} has different stage positions, skipping")
            print(f"    Expected: {reference_positions}")
            print(f"    Got: {positions}")
            continue

        # Extract arrays from this dataset
        piezo_steps = np.array(positions)
        Ns = np.array([d["N_s"] for d in dataset_data])
        Ni = np.array([d["N_i"] for d in dataset_data])
        Nc = np.array([d["N_c"] for d in dataset_data])

        # Apply dark correction using the dark record
        Ni_dark = dark_record["N_i"]
        Ns_dark = dark_record["N_s"]

        Ni_corr = np.clip(Ni - Ni_dark, 0, None)
        accidental = Ns_dark * Ni_dark * ACCIDENTAL_WINDOW
        Nc_corr = np.clip(Nc - accidental, 0, None)

        corrected_datasets.append(
            {
                "filename": jsonl_filename,
                "dataset_index": dataset_index,
                "piezo_steps": piezo_steps,
                "Ns": Ns,
                "Ni": Ni,
                "Nc": Nc,
                "Ni_corr": Ni_corr,
                "Nc_corr": Nc_corr,
            }
        )

        print(
            f"  Dataset {dataset_index}: Applied dark correction (Ni_dark={Ni_dark},"
            f" Ns_dark={Ns_dark})"
        )

    return corrected_datasets


def main():
    parser = argparse.ArgumentParser(description="Plot Mach-Zehnder interferometer data.")
    parser.add_argument("jsonl_file", help="JSONL data file to process.")
    parser.add_argument(
        "--max-phase",
        type=float,
        metavar="X",
        help="Only use data up to a phase delay of X*π.",
    )
    parser.add_argument(
        "--steps-per-two-pi",
        type=float,
        help="Use this value for STEPS_PER_2PI instead of fitting it from the data.",
    )
    args = parser.parse_args()

    datasets = load_and_correct_datasets(args.jsonl_file)
    if not datasets:
        print("No valid datasets found!")
        sys.exit(1)

    if args.steps_per_two_pi:
        steps_per_2pi = args.steps_per_two_pi
        print(f"Using provided STEPS_PER_2PI = {steps_per_2pi:.3f} for all plots\n")
    else:
        # First pass: collect all data for fitting STEPS_PER_2PI
        print("Collecting data to fit STEPS_PER_2PI...")
        datasets_for_fitting = [(ds["piezo_steps"], ds["Nc_corr"]) for ds in datasets]

        # Fit STEPS_PER_2PI from all datasets
        steps_per_2pi = fit_steps_per_2pi(datasets_for_fitting)
        print(f"Using STEPS_PER_2PI = {steps_per_2pi:.3f} for all plots\n")

    # Second pass: generate plots with fitted parameter
    for ds in datasets:
        jsonl_filename = ds["filename"]
        dataset_index = ds["dataset_index"]
        print(f"Processing dataset {dataset_index} from {jsonl_filename}...")

        # Make copies to avoid modifying the dict in place
        piezo_steps = ds["piezo_steps"].copy()
        Ns = ds["Ns"].copy()
        Ni_corr = ds["Ni_corr"].copy()
        Nc_corr = ds["Nc_corr"].copy()

        # Generate output filename by replacing .jsonl with .pdf and adding dataset index
        base_filename = os.path.splitext(jsonl_filename)[0]
        output_filename = f"{base_filename}_dataset_{dataset_index}.pdf"

        # Extract a label from the filename (remove path and extension) and add dataset index
        basename = os.path.splitext(os.path.basename(jsonl_filename))[0]
        label_suffix = f"{basename}_dataset_{dataset_index}"

        # Plot and save
        plot_counts(
            piezo_steps,
            Ns,
            Ni_corr,
            Nc_corr,
            steps_per_2pi,
            output_filename=output_filename,
            label_suffix=label_suffix,
        )


if __name__ == "__main__":
    main()
