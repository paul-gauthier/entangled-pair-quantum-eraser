#!/usr/bin/env python

import json

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import plot_coincidence_counts_only


def load_jsonl_data(filename):
    """Load data from JSONL file and return arrays of steps, N_i, and N_c."""
    steps = []
    N_i = []
    N_c = []

    with open(filename, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            steps.append(data["step"])
            N_i.append(data["N_i"])
            N_c.append(data["N_c"])

    return np.array(steps), np.array(N_i), np.array(N_c)


if __name__ == "__main__":
    # Load data from JSONL files
    steps_eraser_on, Ni_eraser_on, Nc_eraser_on = load_jsonl_data(
        "../lab-code/data/2025-06-06-16-02-40--eraser-on.jsonl"
    )
    steps_eraser_off, Ni_eraser_off, Nc_eraser_off = load_jsonl_data(
        "../lab-code/data/2025-06-06-16-02-40--eraser-off.jsonl"
    )

    # Use the steps from either dataset (they should be the same)
    piezo_steps = steps_eraser_on

    # For "Signal blocked" we use N_i from eraser_off data (idler counts when signal is effectively blocked)
    Nc_eraser_on = Nc_eraser_on
    Nc_eraser_off = Nc_eraser_off

    # Plot for "Eraser on"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_eraser_on,
        output_filename="coincidence_counts_eraser_on.pdf",
        label_suffix="Coincidence counts, eraser-on",
    )

    # Plot for "Eraser off"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_eraser_off,
        output_filename="coincidence_counts_eraser_off.pdf",
        label_suffix="Coincidence counts, eraser-off",
    )
