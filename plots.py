#!/usr/bin/env python

import argparse
import json
import os
import sys

import numpy as np

from plot_utils import delta_from_steps, fit_steps_per_2pi, plot_counts


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
    args = parser.parse_args()

    if args.steps_per_two_pi:
        steps_per_2pi = args.steps_per_two_pi
        print(f"Using provided STEPS_PER_2PI = {steps_per_2pi:.3f} for all plots\n")
    else:
        # First pass: collect all data for fitting STEPS_PER_2PI
        print("Collecting data to fit STEPS_PER_2PI...")
        datasets_for_fitting = []

        for jsonl_filename in args.jsonl_files:
            if not os.path.exists(jsonl_filename):
                print(f"Warning: File {jsonl_filename} not found, skipping.")
                continue

            piezo_steps, Ns, Ni, Nc = load_jsonl_data(jsonl_filename)
            # Use coincidence counts for fitting as they typically have the clearest oscillation
            datasets_for_fitting.append((piezo_steps, Nc))

        if not datasets_for_fitting:
            print("No valid data files found!")
            sys.exit(1)

        # Fit STEPS_PER_2PI from all datasets
        steps_per_2pi = fit_steps_per_2pi(datasets_for_fitting)
        print(f"Using STEPS_PER_2PI = {steps_per_2pi:.3f} for all plots\n")

    # Second pass: generate plots with fitted parameter
    for jsonl_filename in args.jsonl_files:
        if not os.path.exists(jsonl_filename):
            print(f"Warning: File {jsonl_filename} not found, skipping.")
            continue

        print(f"Processing {jsonl_filename}...")

        # Load data from JSONL file
        piezo_steps, Ns, Ni, Nc = load_jsonl_data(jsonl_filename)

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
            Ni = Ni[mask]
            Nc = Nc[mask]
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
            Ni,
            Nc,
            steps_per_2pi,
            output_filename=output_filename,
            label_suffix=basename,
        )


if __name__ == "__main__":
    main()
