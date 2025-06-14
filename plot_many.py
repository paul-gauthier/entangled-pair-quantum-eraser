#!/usr/bin/env python
"""
Fit and summarize Mach-Zehnder interferometer data from multiple JSONL files.

This script processes a list of JSONL files, each containing one or more
datasets. For each dataset, it performs a global joint cosine fit to the
idler (N_i) and coincidence (N_c) counts and extracts key parameters.

The final output is a sorted summary table of the fitted parameters and
experimental settings for all datasets.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

from plot_utils import global_joint_cosine_fit
from plots import _fit_and_assign_steps_per_2pi, load_and_correct_datasets


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Fit and summarize Mach-Zehnder interferometer data from multiple files."
    )
    parser.add_argument("jsonl_files", nargs="+", help="JSONL data files to process.")
    parser.add_argument(
        "--steps-per-two-pi",
        type=float,
        help="Use this value for STEPS_PER_2PI instead of fitting it from the data.",
    )
    args = parser.parse_args()

    all_results = []

    for jsonl_file in args.jsonl_files:
        if not os.path.exists(jsonl_file):
            print(f"File not found: {jsonl_file}", file=sys.stderr)
            continue

        print(f"Processing {jsonl_file}...")
        datasets = load_and_correct_datasets(jsonl_file)
        if not datasets:
            print(f"  No valid datasets found in {jsonl_file}", file=sys.stderr)
            continue

        _fit_and_assign_steps_per_2pi(datasets, args.steps_per_two_pi)

        for ds in datasets:
            try:
                # global_joint_cosine_fit expects a list of datasets
                fit_results = global_joint_cosine_fit(
                    [ds],  # fit each dataset individually
                    ni_key="Ni_corr",
                    nc_key="Nc_corr",
                    ni_raw_key="Ni",
                    nc_raw_key="Nc",
                )

                result = {
                    "signal_lp": ds.get("signal_lp"),
                    "mzi_hwp": ds.get("mzi_hwp"),
                    "mzi_lp": ds.get("mzi_lp"),
                    "V_i": fit_results["V_i"],
                    "V_c": fit_results["V_c"],
                    "A_i": fit_results["A_i"],
                    "A_c": fit_results["A_c"],
                }
                all_results.append(result)

            except RuntimeError as e:
                print(
                    f"  Could not fit dataset {ds['dataset_index']} from {jsonl_file}: {e}",
                    file=sys.stderr,
                )
                continue

    if not all_results:
        print("\nNo results to display.")
        return

    df = pd.DataFrame(all_results)

    # Add _off columns, calculated as angle mod 45
    for col in ["signal_lp", "mzi_hwp", "mzi_lp"]:
        if col in df.columns:
            df[f"{col}_off"] = df[col].apply(lambda x: x % 45 if x is not None else None)

    # Sort by angle settings
    sort_cols = ["signal_lp", "mzi_hwp", "mzi_lp"]
    df = df.sort_values(by=[c for c in sort_cols if c in df.columns])

    # Define and filter columns for the final output table
    output_cols = [
        "signal_lp",
        "signal_lp_off",
        "mzi_hwp",
        "mzi_hwp_off",
        "mzi_lp",
        "mzi_lp_off",
        "V_i",
        "V_c",
        "A_i",
        "A_c",
    ]
    final_cols = [c for c in output_cols if c in df.columns]

    print("\n--- Summary Table ---")
    print(df.to_string(columns=final_cols, index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
