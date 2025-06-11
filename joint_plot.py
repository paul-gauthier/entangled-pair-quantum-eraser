"""
Joint plot of all idler (N_i) and coincidence (N_c) scans on a single pair of
axes using parameters from the global joint cosine fit.

The function `plot_joint_counts` performs the global fit (or accepts an
already-computed fit) and produces a two-panel PDF figure.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from plot_utils import delta_from_steps, global_joint_cosine_fit


def plot_joint_counts(
    datasets: list[dict],
    steps_per_2pi: float,
    *,
    out: str = "all_datasets_joint.pdf",
    show: bool = False,
    align_phase: bool = True,
) -> str:
    """
    Generate a figure showing every dataset’s N_i and N_c points on common
    axes together with the best-fit cosine curves obtained from a single
    global joint fit.

    Parameters
    ----------
    datasets :
        List of dataset dictionaries as produced by `plots.load_and_correct_datasets`.
    steps_per_2pi :
        Conversion factor from piezo steps to 2π phase delay.
    out :
        Output PDF/PNG filename.
    show :
        If True also display the figure interactively.

    Returns
    -------
    str
        The output filename.
    """
    # ------------------------------------------------------------------
    # Perform the hierarchical fit across ALL datasets
    # ------------------------------------------------------------------
    fit = global_joint_cosine_fit(
        datasets,
        steps_per_2pi,
        ni_key="Ni_corr",
        nc_key="Nc_corr",
        ni_raw_key="Ni",
        nc_raw_key="Nc",
    )

    A_i, C0_i = fit["A_i"], fit["C0_i"]
    A_c, C0_c = fit["A_c"], fit["C0_c"]
    phi_ic = fit["phi_ic"]
    phis = fit["phis"]

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, (ax_i, ax_c) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    for k, ds in enumerate(datasets):
        δ = delta_from_steps(ds["piezo_steps"], steps_per_2pi)
        if align_phase:
            δ = δ + phis[k]  # shift each scan by its fitted phase
        col = colours[k % len(colours)]

        # --- Data points ---------------------------------------------------
        ax_i.errorbar(
            δ,
            ds["Ni_corr"],
            yerr=np.sqrt(np.maximum(ds["Ni"], 1)),
            fmt="o",
            ms=4,
            capsize=3,
            color=col,
            label=f"$N_i$ #{k}",
        )
        ax_c.errorbar(
            δ,
            ds["Nc_corr"],
            yerr=np.sqrt(np.maximum(ds["Nc"], 1)),
            fmt="x",
            ms=5,
            capsize=3,
            color=col,
            label=f"$N_c$ #{k}",
        )

        # --- Smooth fitted curves -----------------------------------------
        δ_fine = np.linspace(δ.min(), δ.max(), 400)
        if align_phase:
            ax_i.plot(
                δ_fine,
                C0_i + A_i * (1 + np.cos(δ_fine)) / 2,
                color=col,
                ls="--",
            )
            ax_c.plot(
                δ_fine,
                C0_c + A_c * (1 + np.cos(δ_fine + phi_ic)) / 2,
                color=col,
                ls="--",
            )
        else:
            ax_i.plot(
                δ_fine,
                C0_i + A_i * (1 + np.cos(δ_fine + phis[k])) / 2,
                color=col,
                ls="--",
            )
            ax_c.plot(
                δ_fine,
                C0_c + A_c * (1 + np.cos(δ_fine + phis[k] + phi_ic)) / 2,
                color=col,
                ls="--",
            )

    # Labels, grid, legend --------------------------------------------------
    ax_i.set_ylabel(r"Idler counts $N_i$")
    ax_c.set_ylabel(r"Coinc. counts $N_c$")
    ax_c.set_xlabel(r"Phase delay $\delta$ (rad)")
    ax_i.grid(True, ls=":")
    ax_c.grid(True, ls=":")
    ax_i.legend(fontsize=9, ncol=2)
    ax_c.legend(fontsize=9, ncol=2)

    # -------- π–tick labels on the shared x–axis ---------------------------
    delta_all = np.hstack(
        [
            delta_from_steps(ds["piezo_steps"], steps_per_2pi) + (phis[k] if align_phase else 0)
            for k, ds in enumerate(datasets)
        ]
    )
    start_tick = np.ceil(delta_all.min() / np.pi) * np.pi
    xticks = np.arange(start_tick, delta_all.max() + np.pi / 2, np.pi)

    xticklabels = []
    for tick in xticks:
        m = int(round(tick / np.pi))
        if m == 0:
            xticklabels.append("0")
        elif m == 1:
            xticklabels.append(r"$\pi$")
        elif m == -1:
            xticklabels.append(r"$-\pi$")
        else:
            xticklabels.append(rf"{m}$\pi$")

    ax_c.set_xticks(xticks)
    ax_c.set_xticklabels(xticklabels)

    fig.suptitle("Global joint cosine fit across all datasets")
    fig.tight_layout()

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plt.savefig(out)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Joint plot written to {out}")
    return out
