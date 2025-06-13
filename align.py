#!/usr/bin/env python3

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def read_jsonl(fname):
    """Read data from a jsonl file."""
    signal_lp_angles, signal_counts = [], []
    mzi_lp_angles, idler_counts = [], []
    mzi_hwp_angles = []

    with open(fname, "r") as f:
        for line in f:
            data = json.loads(line)
            if "signal_lp" in data and "N_s" in data:
                signal_lp_angles.append(data["signal_lp"])
                signal_counts.append(data["N_s"])
            if "mzi_lp" in data and "N_i" in data:
                mzi_lp_angles.append(data["mzi_lp"])
                idler_counts.append(data["N_i"])
                # Also collect mzi_hwp if available
                if "mzi_hwp" in data:
                    mzi_hwp_angles.append(data["mzi_hwp"])

    return (
        np.array(signal_lp_angles),
        np.array(signal_counts),
        np.array(mzi_lp_angles),
        np.array(idler_counts),
        np.array(mzi_hwp_angles),
    )


def cos_squared_model(theta, A, B, phi):
    """Cosine squared model: A * cos²(theta + phi) + B"""
    return A * np.cos(np.radians(theta + phi)) ** 2 + B


def fit_and_plot(signal_angles, signal_counts, mzi_angles, idler_counts, fname, use_mzi_hwp=False):
    """Fit cosine squared curves to signal and idler data."""
    # Initial parameter guesses
    # For signal: amplitude, offset, phase shift
    signal_guess = [np.max(signal_counts) - np.min(signal_counts), np.min(signal_counts), 0]
    idler_guess = [np.max(idler_counts) - np.min(idler_counts), np.min(idler_counts), 0]

    # Fit the curves
    signal_params, signal_cov = curve_fit(
        cos_squared_model, signal_angles, signal_counts, p0=signal_guess
    )
    idler_params, idler_cov = curve_fit(cos_squared_model, mzi_angles, idler_counts, p0=idler_guess)

    # Print fit parameters
    print("Signal fit parameters:")
    print(f"  Amplitude (A): {signal_params[0]:.1f} ± {np.sqrt(signal_cov[0,0]):.1f}")
    print(f"  Offset (B): {signal_params[1]:.1f} ± {np.sqrt(signal_cov[1,1]):.1f}")
    print(f"  Phase shift (φ): {signal_params[2]:.1f}° ± {np.sqrt(signal_cov[2,2]):.1f}°")
    print()

    print("Idler fit parameters:")
    print(f"  Amplitude (A): {idler_params[0]:.1f} ± {np.sqrt(idler_cov[0,0]):.1f}")
    print(f"  Offset (B): {idler_params[1]:.1f} ± {np.sqrt(idler_cov[1,1]):.1f}")
    print(f"  Phase shift (φ): {idler_params[2]:.1f}° ± {np.sqrt(idler_cov[2,2]):.1f}°")
    print()

    # Generate smooth curves for plotting
    signal_angles_smooth = np.linspace(np.min(signal_angles), np.max(signal_angles), 200)
    mzi_angles_smooth = np.linspace(np.min(mzi_angles), np.max(mzi_angles), 200)
    signal_fit = cos_squared_model(signal_angles_smooth, *signal_params)
    idler_fit = cos_squared_model(mzi_angles_smooth, *idler_params)

    # Plot the data and fits
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(signal_angles, signal_counts, color="blue", label="Signal LP", alpha=0.7)
    plt.plot(
        signal_angles_smooth,
        signal_fit,
        "b-",
        label=f"Fit: {signal_params[0]:.0f}cos²(θ+{signal_params[2]:.1f}°)+{signal_params[1]:.0f}",
    )
    plt.xlabel("Signal LP Angle (degrees)")
    plt.ylabel("Signal counts (N_s)")
    plt.title(f"Signal vs Signal LP Angle\n({fname})")
    plt.xticks(np.arange(-45, 180, 45))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    mzi_label = "MZI HWP" if use_mzi_hwp else "MZI LP"
    plt.scatter(mzi_angles, idler_counts, color="red", label=mzi_label, alpha=0.7)
    plt.plot(
        mzi_angles_smooth,
        idler_fit,
        "r-",
        label=f"Fit: {idler_params[0]:.0f}cos²(θ+{idler_params[2]:.1f}°)+{idler_params[1]:.0f}",
    )
    plt.xlabel(f"{mzi_label} Angle (degrees)")
    plt.ylabel("Idler counts (N_i)")
    plt.title(f"Idler vs {mzi_label} Angle\n({fname})")
    plt.xticks(np.arange(-45, 180, 45))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("polarizer_alignment_fits.png")
    plt.show()

    return signal_params, idler_params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit and plot polarizer alignment data")
    parser.add_argument("jsonl_file", help="JSONL file containing the data")
    parser.add_argument("--x-mzi-hwp", action="store_true", 
                       help="Use MZI HWP angle on x-axis instead of MZI LP angle")
    
    args = parser.parse_args()
    
    signal_angles, signal_counts, mzi_lp_angles, idler_counts, mzi_hwp_angles = read_jsonl(args.jsonl_file)
    
    if args.x_mzi_hwp:
        if len(mzi_hwp_angles) == 0:
            print("Error: --x-mzi-hwp specified but no mzi_hwp data found in file")
            sys.exit(1)
        mzi_angles = mzi_hwp_angles
    else:
        mzi_angles = mzi_lp_angles
    
    fit_and_plot(signal_angles, signal_counts, mzi_angles, idler_counts, args.jsonl_file, args.x_mzi_hwp)
