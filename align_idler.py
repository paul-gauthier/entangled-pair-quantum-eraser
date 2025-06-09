#!/usr/bin/env python3

import json
import re
import sys
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

from dump import dump

def read_jsonl(fname):
    """Read idler data from a jsonl file."""
    mzi_lp_angles, idler_counts = [], []

    with open(fname, "r") as f:
        for line in f:
            data = json.loads(line)
            if "mzi_lp" in data and "N_i" in data:
                mzi_lp_angles.append(data["mzi_lp"])
                idler_counts.append(data["N_i"])

    return (
        np.array(mzi_lp_angles),
        np.array(idler_counts),
    )


def cos_squared_model(theta, A, B, phi):
    """Cosine squared model: A * cos²(theta + phi) + B"""
    return A * np.cos(np.radians(theta + phi)) ** 2 + B


def fit_idler_data(angles, counts):
    """Fit cosine squared curve to idler data and return fit parameters."""
    if len(angles) < 3:  # curve_fit needs at least as many points as parameters
        return None

    # Initial parameter guesses
    # For idler: amplitude, offset, phase shift
    if np.all(counts == counts[0]):  # All counts are the same, no curve to fit
        return None
    guess = [np.max(counts) - np.min(counts), np.min(counts), 0]

    try:
        # Fit the curve with A ≥ 0 so that φ is uniquely defined
        params, cov = curve_fit(
            cos_squared_model,
            angles,
            counts,
            p0=guess,
            #bounds=([0, -np.inf, -180], [np.inf, np.inf, 180]),
        )
        return params
    except RuntimeError:
        # Fit failed
        return None


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <jsonl_files...>")
        sys.exit(1)

    files = sys.argv[1:]

    # Group files by (hwpoff, lpoff)
    file_groups = defaultdict(dict)
    pattern = re.compile(r"hwp(0|45)-hwpoff(-?[\d.]+)-lpoff(-?\d+)")

    for fname in files:
        match = pattern.search(fname)
        if match:
            hwp, hwpoff, lpoff = match.groups()
            hwp = int(hwp)
            hwpoff = float(hwpoff)
            lpoff = int(lpoff)
            file_groups[(hwpoff, lpoff)][hwp] = fname

    # Process each group and print results
    print(f"{'hwpoff':>6s}, {'lpoff':>6s}, {'phi(hwp0)':>10s}, {'phi(hwp45)':>10s}")
    for (hwpoff, lpoff), hwp_files in sorted(file_groups.items()):

        phi0, phi45 = float("nan"), float("nan")

        if 0 in hwp_files:
            angles0, counts0 = read_jsonl(hwp_files[0])
            params0 = fit_idler_data(angles0, counts0)
            if params0 is not None:
                phi0 = params0[2]

        if 45 in hwp_files:
            angles45, counts45 = read_jsonl(hwp_files[45])
            params45 = fit_idler_data(angles45, counts45)
            if params45 is not None:
                phi45 = params45[2]

        phi0_str = f"{phi0:10.1f}" if not np.isnan(phi0) else ""
        phi45_str = f"{phi45:10.1f}" if not np.isnan(phi45) else ""

        print(f"{hwpoff:6.1f}, {lpoff:6d}, {phi0_str:>10s}, {phi45_str:>10s}")


if __name__ == "__main__":
    main()
