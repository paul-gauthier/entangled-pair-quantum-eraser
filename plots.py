#!/usr/bin/env python

import json
import os
import sys

import numpy as np

from plot_utils import plot_counts


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
    if len(sys.argv) < 2:
        print("Usage: python plots.py <jsonl_file1> [jsonl_file2] ...")
        sys.exit(1)

    for jsonl_filename in sys.argv[1:]:
        if not os.path.exists(jsonl_filename):
            print(f"Warning: File {jsonl_filename} not found, skipping.")
            continue

        print(f"Processing {jsonl_filename}...")

        # Load data from JSONL file
        piezo_steps, Ns, Ni, Nc = load_jsonl_data(jsonl_filename)

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
            output_filename=output_filename,
            label_suffix=basename,
        )


if __name__ == "__main__":
    main()
