#!/usr/bin/env python

import argparse
import json
import os
import sys

import numpy as np

from joint_plot import plot_joint_counts
from plot_utils import (
    fit_steps_per_2pi,
    global_cosine_fit,
    global_joint_cosine_fit,
    plot_counts,
)

# ---------------------------------------------------------------------------
# Dark–count handling
# ---------------------------------------------------------------------------
ACCIDENTAL_WINDOW = 25e-9  # 25 ns coincidence window


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
        duration = sum(d.get("acq_time", 0) for d in dataset_data)
        dark_duration = dark_record.get("acq_time", 0)

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
                "duration": duration + dark_duration,
            }
        )

        print(
            f"  Dataset {dataset_index}: Applied dark correction (Ni_dark={Ni_dark},"
            f" Ns_dark={Ns_dark}, Nc_accidental={accidental:.2f})"
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

    # Fit STEPS_PER_2PI for each dataset individually (using raw Nc) and
    # combine the results with a 1/σ²-weighted average
    print("Fitting STEPS_PER_2PI for each dataset ...")
    sp2pi_vals = []
    sp2pi_errs = []
    for ds in datasets:
        try:
            sp2pi, sp2pi_err = fit_steps_per_2pi(ds["piezo_steps"], ds["Nc_corr"], ds["Nc"])
            sp2pi_vals.append(sp2pi)
            sp2pi_errs.append(sp2pi_err)
        except RuntimeError:
            print(f"  Failed to fit STEPS_PER_2PI for dataset {ds['dataset_index']}, skipping.")

    if not sp2pi_vals:
        print("\nCould not fit STEPS_PER_2PI for any dataset. Exiting.")
        sys.exit(1)

    weights = 1.0 / np.square(sp2pi_errs)
    steps_per_2pi = float(np.sum(weights * sp2pi_vals) / np.sum(weights))
    combined_err = float(1.0 / np.sqrt(np.sum(weights)))
    print(
        f"Computed weighted STEPS_PER_2PI = {steps_per_2pi:.3f} ± {combined_err:.3f} for all"
        " plots\n"
    )

    if args.steps_per_two_pi:
        steps_per_2pi = args.steps_per_two_pi
        print(f"Using provided STEPS_PER_2PI = {steps_per_2pi:.3f} for all plots\n")

    # Second pass: generate plots with fitted parameter and collect visibilities
    V_i_list, V_i_err_list, V_c_list, V_c_err_list = [], [], [], []
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

        try:
            # Plot and save, collecting visibility metrics
            _, metrics = plot_counts(
                piezo_steps,
                Ns,
                Ni_corr,
                Nc_corr,
                steps_per_2pi,
                output_filename=output_filename,
                label_suffix=label_suffix,
                Nc_raw=ds["Nc"],
                Ni_raw=ds["Ni"],
                return_metrics=True,
            )
            V_i_list.append(metrics["V_vis_i"])
            V_i_err_list.append(metrics["V_err_i"])
            V_c_list.append(metrics["V_vis_c"])
            V_c_err_list.append(metrics["V_err_c"])
        except RuntimeError:
            print("Failed to fit dataset, skipping")

    # ------------------------------------------------------------------
    # Joint plot across all datasets using the global joint fit
    # ------------------------------------------------------------------
    try:
        base = os.path.splitext(args.jsonl_file)[0]
        joint_pdf = f"{base}_joint.pdf"
        os.makedirs(os.path.dirname(joint_pdf) or ".", exist_ok=True)
        plot_joint_counts(datasets, steps_per_2pi, out=joint_pdf, title=os.path.basename(base))
    except RuntimeError as e:
        print(f"Joint plot failed: {e}")

    # ------------------------------------------------------------------
    # Combine visibilities across all datasets using inverse-variance weighting
    # ------------------------------------------------------------------
    def _weighted_mean(values: np.ndarray, errors: np.ndarray) -> tuple[float, float, float]:
        weights = 1.0 / np.square(errors)
        mean = float(np.sum(weights * values) / np.sum(weights))
        err = float(1.0 / np.sqrt(np.sum(weights)))
        red_chi2 = (
            float(np.sum(weights * np.square(values - mean)) / (len(values) - 1))
            if len(values) > 1
            else float("nan")
        )
        return mean, err, red_chi2

    if V_i_list:
        V_i_comb, V_i_comb_err, red_chi2_i = _weighted_mean(
            np.array(V_i_list), np.array(V_i_err_list)
        )
        V_c_comb, V_c_comb_err, red_chi2_c = _weighted_mean(
            np.array(V_c_list), np.array(V_c_err_list)
        )

        print("\nCombined visibility estimates (inverse-variance weighted):")
        print(
            f"  Idler:       Vi = {V_i_comb:.4f} ± {V_i_comb_err:.4f} "
            f" [{V_i_comb - V_i_comb_err:.4f}, {V_i_comb + V_i_comb_err:.4f}]   (reduced χ² ="
            f" {red_chi2_i:.2f})"
        )
        print(
            f"  Coincidence: Vc = {V_c_comb:.4f} ± {V_c_comb_err:.4f} "
            f" [{V_c_comb - V_c_comb_err:.4f}, {V_c_comb + V_c_comb_err:.4f}]   (reduced χ² ="
            f" {red_chi2_c:.2f})"
        )

        # ------------------------------------------------------------------
        # Independent global cosine fits for Ni and Nc (legacy analysis)
        # ------------------------------------------------------------------
        try:
            global_cosine_fit(
                datasets,
                steps_per_2pi,
                counts_key="Ni_corr",
                raw_key="Ni",
                label="Idler",
            )
            global_cosine_fit(
                datasets,
                steps_per_2pi,
                counts_key="Nc_corr",
                raw_key="Nc",
                label="Coincidence",
            )
        except RuntimeError as e:
            print(f"\nIndependent global fits failed: {e}")

        # ------------------------------------------------------------------
        # Global hierarchical fit with shared φ_ic between Ni & Nc
        # ------------------------------------------------------------------
        try:
            global_joint_cosine_fit(
                datasets,
                steps_per_2pi,
                ni_key="Ni_corr",
                nc_key="Nc_corr",
                ni_raw_key="Ni",
                nc_raw_key="Nc",
            )
        except RuntimeError as e:
            print(f"\nGlobal joint fit failed: {e}")

    # ------------------------------------------------------------------
    # Final summary statistics
    # ------------------------------------------------------------------
    total_signals = sum(np.sum(ds["Ns"]) for ds in datasets)
    total_idlers = sum(np.sum(ds["Ni"]) for ds in datasets)
    total_coincidences = sum(np.sum(ds["Nc"]) for ds in datasets)
    total_photons = total_signals + total_idlers
    total_duration_s = sum(ds.get("duration", 0) for ds in datasets)

    td_int = int(total_duration_s)
    hours, remainder = divmod(td_int, 3600)
    minutes, seconds = divmod(remainder, 60)
    duration_hms = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    total_phase_scanned_pi = 0
    if "steps_per_2pi" in locals() and steps_per_2pi > 0:
        for ds in datasets:
            steps_range = np.max(ds["piezo_steps"]) - np.min(ds["piezo_steps"])
            phase_range_pi = 2 * steps_range / steps_per_2pi
            total_phase_scanned_pi += phase_range_pi

    print("\n--- Summary Statistics ---")
    print(f"Total signals:      {int(total_signals):,}")
    print(f"Total idlers:       {int(total_idlers):,}")
    print(f"Total coincidences: {int(total_coincidences):,}")
    print(f"Total photons (S+I): {int(total_photons):,}")
    print(f"Total duration:     {duration_hms}")
    print(f"Total datasets:     {len(datasets)}")
    if total_phase_scanned_pi > 0:
        print(
            f"Total phase scanned: {total_phase_scanned_pi:.2f}π"
            f" ({total_phase_scanned_pi/2:.2f} periods)"
        )


if __name__ == "__main__":
    main()
