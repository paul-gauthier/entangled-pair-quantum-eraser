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

from dump import dump  # noqa
from joint_plot import plot_joint_counts
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
    parser.add_argument(
        "--plots-dir",
        type=str,
        help="Directory to save joint plots for each dataset.",
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

                if args.plots_dir:
                    os.makedirs(args.plots_dir, exist_ok=True)
                    base_name = os.path.splitext(os.path.basename(jsonl_file))[0]
                    dataset_index = ds["dataset_index"]
                    plot_filename = os.path.join(args.plots_dir, f"{base_name}_{dataset_index}.pdf")
                    title = f"Dataset {dataset_index} from {os.path.basename(jsonl_file)}"
                    plot_joint_counts(
                        [ds],
                        out=plot_filename,
                        title=title,
                    )

                A_i = fit_results["A_i"]
                A_c = fit_results["A_c"]
                result = {
                    "signal_lp": ds.get("signal_lp"),
                    "mzi_hwp": ds.get("mzi_hwp"),
                    "mzi_lp": ds.get("mzi_lp"),
                    "V_i": fit_results["V_i"],
                    "V_i_err": fit_results["V_i_err"],
                    "V_c": fit_results["V_c"],
                    "A_i": A_i,
                    "A_i_err": fit_results["A_i_err"],
                    "A_c": A_c,
                    "A_c_err": fit_results["A_c_err"],
                    "Ai/Ac": A_i / A_c if A_c else np.nan,
                    "acq_dur": ds.get("acq_time"),
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

    # Add _off columns, calculated as offset from nearest multiple of 45
    for col in ["signal_lp", "mzi_hwp", "mzi_lp"]:
        if col in df.columns:
            df[f"{col}_off"] = df[col].apply(
                lambda x: x - round(x / 45) * 45 if x is not None else None
            )

    # Sort by angle settings
    sort_cols = ["signal_lp", "mzi_hwp", "mzi_lp"]
    df = df.sort_values(by=[c for c in sort_cols if c in df.columns])

    if "A_i" in df.columns:
        df["A_i_rolling_avg"] = df["A_i"].rolling(window=3, center=True).mean()
        if len(df) > 1:
            df.loc[df.index[0], "A_i_rolling_avg"] = df["A_i"].iloc[0:2].mean()
            df.loc[df.index[-1], "A_i_rolling_avg"] = df["A_i"].iloc[-2:].mean()

    if "V_i" in df.columns:
        df["V_i_rolling_avg"] = df["V_i"].rolling(window=3, center=True).mean()
        if len(df) > 1:
            df.loc[df.index[0], "V_i_rolling_avg"] = df["V_i"].iloc[0:2].mean()
            df.loc[df.index[-1], "V_i_rolling_avg"] = df["V_i"].iloc[-2:].mean()

    if "V_i" in df.columns and "V_i_err" in df.columns:
        df["V_i"] = df.apply(lambda r: f"{f'{r.V_i:.4f}':>5} {f'±{r.V_i_err:.4f}':>5}", axis=1)
    if "A_i" in df.columns and "A_i_err" in df.columns:
        df["A_i"] = df.apply(lambda r: f"{f'{r.A_i:.1f}':>7} {f'±{r.A_i_err:.0f}':>4}", axis=1)
    if "A_c" in df.columns and "A_c_err" in df.columns:
        df["A_c"] = df.apply(lambda r: f"{f'{r.A_c:.1f}':>7} {f'±{r.A_c_err:.0f}':>4}", axis=1)

    # Define and filter columns for the final output table
    output_cols = [
        "signal_lp",
        "signal_lp_off",
        "mzi_hwp",
        "mzi_hwp_off",
        "mzi_lp",
        "mzi_lp_off",
        "V_i",
        "V_i_rolling_avg",
        "V_c",
        "A_i",
        "A_i_rolling_avg",
        "A_c",
        "Ai/Ac",
        "acq_dur",
    ]
    final_cols = [c for c in output_cols if c in df.columns]

    print("\n--- Summary Table ---")

    if df.empty:
        print(df.to_string(columns=final_cols, index=False))
        return

    elided_cols = {}
    cols_to_print = list(final_cols)

    if len(df) > 1:
        for col in final_cols:
            if df[col].nunique(dropna=False) == 1:
                elided_cols[col] = df[col].iloc[0]
                cols_to_print.remove(col)

    for col, val in elided_cols.items():
        if pd.isna(val):
            val_str = "None"
        elif isinstance(val, float):
            val_str = f"{val:.4f}"
        else:
            val_str = str(val)
        print(f"{col}: {val_str}")

    if elided_cols and cols_to_print:
        print("-" * 20)

    if cols_to_print:
        print(df.to_string(columns=cols_to_print, index=False, float_format="%.4f"))
    elif elided_cols:
        print(f"({len(df)} rows with above values)")


if __name__ == "__main__":
    main()
