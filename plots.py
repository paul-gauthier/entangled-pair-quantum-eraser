#!/usr/bin/env python

import argparse
import json
import os
import sys

import numpy as np

from plot_utils import delta_from_steps, fit_steps_per_2pi, plot_counts

# ---------------------------------------------------------------------------
# Dark–count handling
# ---------------------------------------------------------------------------
ACCIDENTAL_WINDOW = 25e-9  # 25 ns coincidence window


def _load_dark_data(filename):
    """
    Load a *dark* JSONL file and return three 1-D numpy arrays:
    piezo_steps_dark, Ns_dark, Ni_dark.
    """
    piezo_steps_d, Ns_d, Ni_d, _ = load_jsonl_data(filename)
    return piezo_steps_d, Ns_d, Ni_d


def _nearest_dark(idx_step, dark_steps):
    """Return index of the dark-data row whose step is closest to `idx_step`."""
    return int(np.abs(dark_steps - idx_step).argmin())


def apply_dark_correction(piezo_steps, Ns, Ni, Nc, dark_data):
    """
    Subtract dark counts from Ni and accidental coincidences from Nc.

    Parameters
    ----------
    piezo_steps, Ns, Ni, Nc : np.ndarray
        Raw experimental data.
    dark_data : tuple | None
        (piezo_steps_dark, Ns_dark, Ni_dark) or ``None``.

    Returns
    -------
    Ni_corr, Nc_corr : np.ndarray
        Dark-subtracted arrays (Ns is returned unchanged).
    """
    if dark_data is None:
        # No dark file – return originals unchanged
        return Ni.copy(), Nc.copy()

    dark_steps, Ns_dark, Ni_dark = dark_data
    Ni_corr = np.empty_like(Ni, dtype=float)
    Nc_corr = np.empty_like(Nc, dtype=float)

    for i, step in enumerate(piezo_steps):
        j = _nearest_dark(step, dark_steps)
        Ni_dark_val = Ni_dark[j]
        Ns_dark_val = Ns_dark[j]

        Ni_corr[i] = Ni[i] - Ni_dark_val
        accidental = Ns_dark_val * Ni_dark_val * ACCIDENTAL_WINDOW
        Nc_corr[i] = Nc[i] - accidental

    # Clip any negative values to zero
    Ni_corr = np.clip(Ni_corr, 0, None)
    Nc_corr = np.clip(Nc_corr, 0, None)
    return Ni_corr, Nc_corr


def load_jsonl_data(filename):
    """Load data from a JSONL file and return arrays of piezo steps and counts."""
    data = []
    with open(filename, "r") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # Sort by step to ensure proper ordering
    data.sort(key=lambda x: x["step"])

    # Extract arrays
    piezo_steps = np.array([d["step"] for d in data])
    Ns = np.array([d["N_s"] for d in data])
    Ni = np.array([d["N_i"] for d in data])
    Nc = np.array([d["N_c"] for d in data])

    return piezo_steps, Ns, Ni, Nc


def load_and_correct_datasets(jsonl_files):
    """
    Load datasets from JSONL files and apply dark count corrections.

    Returns a list of dictionaries, one for each main (non-dark) dataset.
    Each dictionary contains:
    - 'filename': The original filename of the dataset.
    - 'piezo_steps', 'Ns': Raw data arrays.
    - 'Ni_corr', 'Nc_corr': Dark-corrected data arrays.
    """
    supplied_files = set(jsonl_files)
    datasets = []

    main_files = [f for f in jsonl_files if "-dark-" not in os.path.basename(f)]

    for jsonl_filename in main_files:
        if not os.path.exists(jsonl_filename):
            print(f"Warning: File {jsonl_filename} not found, skipping.")
            continue

        piezo_steps, Ns, Ni, Nc = load_jsonl_data(jsonl_filename)

        # Locate a matching dark file (must also have been supplied).
        dark_filename = None
        if jsonl_filename.endswith("-on.jsonl"):
            cand = jsonl_filename.replace("-on.jsonl", "-dark-on.jsonl")
            dark_filename = cand if cand in supplied_files else None
        elif jsonl_filename.endswith("-off.jsonl"):
            cand = jsonl_filename.replace("-off.jsonl", "-dark-off.jsonl")
            dark_filename = cand if cand in supplied_files else None

        dark_data = None
        if dark_filename:
            try:
                dark_data = _load_dark_data(dark_filename)
                print(f"  Using dark data from {dark_filename} for {jsonl_filename}")
            except FileNotFoundError:
                print(f"  Warning: dark file {dark_filename} missing – no correction.")
                dark_data = None
        else:
            print(f"  Warning: no matching dark file for {jsonl_filename}")

        Ni_corr, Nc_corr = apply_dark_correction(piezo_steps, Ns, Ni, Nc, dark_data)

        datasets.append(
            {
                "filename": jsonl_filename,
                "piezo_steps": piezo_steps,
                "Ns": Ns,
                "Ni_corr": Ni_corr,
                "Nc_corr": Nc_corr,
            }
        )
    return datasets


def main():
    parser = argparse.ArgumentParser(description="Plot Mach-Zehnder interferometer data.")
    parser.add_argument("jsonl_files", nargs="+", help="One or more JSONL data files to process.")
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
    parser.add_argument(
        "--nc-phi", type=float, help="Fix the phase phi for the Nc fit (in radians)."
    )
    parser.add_argument(
        "--ni-phi", type=float, help="Fix the phase phi for the Ni fit (in radians)."
    )
    args = parser.parse_args()

    datasets = load_and_correct_datasets(args.jsonl_files)
    if not datasets:
        print("No valid data files found!")
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
        print(f"Processing {jsonl_filename}...")

        # Make copies to avoid modifying the dict in place
        piezo_steps = ds["piezo_steps"].copy()
        Ns = ds["Ns"].copy()
        Ni_corr = ds["Ni_corr"].copy()
        Nc_corr = ds["Nc_corr"].copy()

        if args.max_phase is not None:
            delta = delta_from_steps(piezo_steps, steps_per_2pi)
            mask = delta <= args.max_phase * np.pi

            if not np.any(mask):
                print(
                    f"  Warning: --max-phase filter removed all data from {jsonl_filename}."
                    " Skipping plot."
                )
                continue

            piezo_steps = piezo_steps[mask]
            Ns = Ns[mask]
            Ni_corr = Ni_corr[mask]
            Nc_corr = Nc_corr[mask]
            print(
                f"  Filtered data to max phase {args.max_phase}π, {len(piezo_steps)} points"
                " remaining."
            )

        # Generate output filename by replacing .jsonl with .pdf
        output_filename = os.path.splitext(jsonl_filename)[0] + ".pdf"

        # Extract a label from the filename (remove path and extension)
        basename = os.path.splitext(os.path.basename(jsonl_filename))[0]

        # Plot and save
        plot_counts(
            piezo_steps,
            Ns,
            Ni_corr,
            Nc_corr,
            steps_per_2pi,
            output_filename=output_filename,
            label_suffix=basename,
            nc_phi=args.nc_phi,
            ni_phi=args.ni_phi,
        )


if __name__ == "__main__":
    main()
