"""
Reusable plotting utilities for Mach–Zehnder interferometer count data.

The main entry-point is `plot_counts`, which generates a two-panel plot of
signal, idler, and coincidence counts versus phase delay.  A secondary
x-axis shows the corresponding piezo-stage displacement in nanometres.

Example
-------
>>> from plot_utils import plot_counts
>>> plot_counts(piezo_steps, Ns, Ni, Nc, output_filename="my_plot.pdf")
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Conversions between piezo stage steps, phase delay δ, and nanometres (nm)
# ---------------------------------------------------------------------------
STEPS_PER_2PI = 22.0          # 22 piezo steps correspond to 2π phase change
NM_PER_STEP = 26.03           # Piezo stage moves 26.03 nm per step


def delta_from_steps(steps: np.ndarray | float) -> np.ndarray | float:
    """Convert piezo steps → phase delay δ (radians)."""
    return steps * (2 * np.pi / STEPS_PER_2PI)


def steps_from_delta(delta: np.ndarray | float) -> np.ndarray | float:
    """Convert phase delay δ (radians) → piezo steps."""
    return delta * STEPS_PER_2PI / (2 * np.pi)


def steps_to_nm(steps: np.ndarray | float) -> np.ndarray | float:
    """Convert piezo steps → nanometres of stage travel."""
    return steps * NM_PER_STEP


def nm_to_steps(nm: np.ndarray | float) -> np.ndarray | float:
    """Convert nanometres of stage travel → piezo steps."""
    return nm / NM_PER_STEP


def delta_to_nm(delta: np.ndarray | float) -> np.ndarray | float:
    """Convert phase delay δ (radians) → nanometres of stage travel."""
    return steps_to_nm(steps_from_delta(delta))


def nm_to_delta(nm: np.ndarray | float) -> np.ndarray | float:
    """Convert nanometres of stage travel → phase delay δ (radians)."""
    return delta_from_steps(nm_to_steps(nm))


# ---------------------------------------------------------------------------
# Model function for fitting
# ---------------------------------------------------------------------------
def _cos_model(d, A, C0, phi):
    """Cosine model for fitting interference patterns."""
    return C0 + A * (1 + np.cos(d + phi)) / 2


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_counts(
    piezo_steps: np.ndarray,
    Ns: np.ndarray,
    Ni: np.ndarray,
    Nc: np.ndarray,
    *,
    output_filename: str = "counts_vs_phase_delay.pdf",
    label_suffix: str = "",
    show: bool = False,
) -> str:
    """
    Plot Ns, Ni, and Nc versus phase delay and save the figure.

    Parameters
    ----------
    piezo_steps :
        Array of piezo stage positions (integer steps).
    Ns, Ni, Nc :
        Arrays of signal, idler, and coincidence counts (same length as
        ``piezo_steps``).
    output_filename :
        Path where the PDF/PNG will be written.
    label_suffix :
        Optional suffix appended to trace labels, useful when plotting
        multiple datasets on shared axes.
    show :
        If ``True`` also display the figure interactively.

    Returns
    -------
    str
        The `output_filename` path for convenience.
    """

    print()

    # Poisson (√N) uncertainties
    Ns_err = np.sqrt(Ns)
    Ni_err = np.sqrt(Ni)
    Nc_err = np.sqrt(Nc)

    # Phase delay for x-axis
    delta = delta_from_steps(piezo_steps)

    # ------------------------------------------------------------------
    # Fit coincidence counts with ½(1+cos(δ+φ)) model
    # ------------------------------------------------------------------
    p0 = [np.ptp(Nc), np.min(Nc), 0.0]  # initial guesses
    popt, _ = curve_fit(
        _cos_model,
        delta,
        Nc,
        p0=p0,
        sigma=Nc_err,
        absolute_sigma=True,
    )

    # ------------------------------------------------------------------
    # Convert optimiser output to physically meaningful parameters
    # ------------------------------------------------------------------
    A_fit, C0_fit, phi_fit = popt
    if A_fit < 0:               # enforce non-negative modulation depth
        A_fit = -A_fit
        phi_fit += np.pi        # keep model invariant
        C0_fit -= A_fit         # compensate offset to preserve model
    # Wrap phase into (−π, π]
    phi_fit = (phi_fit + np.pi) % (2 * np.pi) - np.pi

    delta_fine = np.linspace(delta.min(), delta.max(), 500)
    Nc_fit = _cos_model(delta_fine, A_fit, C0_fit, phi_fit)

    # Print fitted parameters
    print(f"Fit results for {output_filename}:")
    print(f"  C0 = {C0_fit:.2f}")
    print(f"  A = {A_fit:.2f}")
    print(f"  phi = {phi_fit:.2f} rad")
    V_vis = A_fit / (A_fit + 2 * C0_fit)
    print(f"  Visibility V = {V_vis:.3f}")
    print(f"  LaTeX: $C_0 = {C0_fit:.2f},  A = {A_fit:.2f},  \\phi = {phi_fit:.2f}\\," \
          f"\\text{{rad}}$. and $V = {V_vis:.3f}$")

    # Style ------------------------------------------------------------------
    plt.rcParams.update({"font.size": 16})
    color_ns, color_nc, color_ni = "tab:blue", "tab:red", "tab:green"

    # Figure & axes ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Add title
    if label_suffix:
        fig.suptitle(f"Counts vs Phase Delay: {label_suffix}", fontsize=20)

    # Top panel: Ni only -----------------------------------------------------
    ax1.set_ylabel(r"Counts/sec", fontsize=18)
    ax1.errorbar(
        delta,
        Ni,
        yerr=Ni_err,
        fmt="s",
        color=color_ni,
        #linestyle=":",
        label=fr"Idler ($N_{{i,{label_suffix}}}$)",
        capsize=3,
    )
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="center right")

    # Bottom panel: Nc -------------------------------------------------------
    ax2.set_ylabel(r"Counts/sec", fontsize=18)
    ax2.errorbar(
        delta,
        Nc,
        yerr=Nc_err,
        fmt="x",
        color=color_nc,
        #linestyle="--",
        label=fr"Coincidence ($N_{{c,{label_suffix}}}$, $V={V_vis:.3f}$) with fit",
        capsize=3,
    )
    # Overlay best-fit cosine curve
    ax2.plot(
        delta_fine,
        Nc_fit,
        linestyle="--",
        color=color_nc,
        lw=1,
    )
    ax2.grid(True, linestyle=":", alpha=0.7)
    ax2.set_xlabel(r"Phase Delay $\delta$ (rad)", fontsize=18)
    ax2.legend(loc='upper right')

    # Shared x-axis tick labels ---------------------------------------------
    xticks = np.arange(0, np.max(delta) + np.pi / 2, np.pi)
    xticklabels = ["0"] + [f"{i}$\\pi$" for i in range(1, len(xticks))]
    xticklabels = [s.replace("1$\\pi$", "$\\pi$") for s in xticklabels]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)

    # Set ymax for the coincidence graph (ax2) as per user request
    nc_min_val = np.min(Nc)
    nc_max_val = np.max(Nc)
    nc_max_val += np.sqrt(nc_max_val)
    new_ymax_for_ax2 = nc_max_val + (nc_max_val - nc_min_val) * 0.2
    current_ax2_ymin, _ = ax2.get_ylim()  # Preserve current ymin
    ax2.set_ylim(bottom=current_ax2_ymin, top=new_ymax_for_ax2)

    # Layout & save ----------------------------------------------------------
    fig.tight_layout()
    plt.savefig(output_filename)
    if show:
        plt.show()
    plt.close(fig)

    print(f"Plot saved as {output_filename}")
    return output_filename


def plot_coincidence_counts_only(
    piezo_steps: np.ndarray,
    Nc: np.ndarray,
    *,
    output_filename: str = "coincidence_counts_vs_phase_delay.pdf",
    label_suffix: str = "",
    show: bool = False,
) -> str:
    """
    Plot Nc versus phase delay, fit with a cosine model, and save the figure.

    Parameters
    ----------
    piezo_steps :
        Array of piezo stage positions (integer steps).
    Nc :
        Array of coincidence counts (same length as ``piezo_steps``).
    output_filename :
        Path where the PDF/PNG will be written.
    label_suffix :
        Optional suffix appended to trace labels and included in the plot title.
    show :
        If ``True`` also display the figure interactively.

    Returns
    -------
    str
        The `output_filename` path for convenience.
    """
    print()

    # Poisson (√N) uncertainties
    Nc_err = np.sqrt(Nc)

    # Phase delay for x-axis
    delta = delta_from_steps(piezo_steps)

    # Fit coincidence counts with ½(1+cos(δ+φ)) model
    p0 = [np.ptp(Nc), np.min(Nc), 0.0]  # initial guesses
    popt, _ = curve_fit(
        _cos_model,
        delta,
        Nc,
        p0=p0,
        sigma=Nc_err,
        absolute_sigma=True,
    )

    # Convert optimiser output to physically meaningful parameters
    A_fit, C0_fit, phi_fit = popt
    if A_fit < 0:  # enforce non-negative modulation depth
        A_fit = -A_fit
        phi_fit += np.pi  # keep model invariant
        C0_fit -= A_fit  # compensate offset to preserve model
    phi_fit = (phi_fit + np.pi) % (2 * np.pi) - np.pi  # Wrap phase into (−π, π]

    delta_fine = np.linspace(delta.min(), delta.max(), 500)
    Nc_fit_curve = _cos_model(delta_fine, A_fit, C0_fit, phi_fit)

    # Print fitted parameters
    print(f"Fit results for {output_filename}:")
    print(f"  C0 = {C0_fit:.2f}")
    print(f"  A = {A_fit:.2f}")
    print(f"  phi = {phi_fit:.2f} rad")
    V_vis = A_fit / (A_fit + 2 * C0_fit) if (A_fit + 2 * C0_fit) != 0 else 0
    print(f"  Visibility V = {V_vis:.3f}")
    print(f"  LaTeX: $C_0 = {C0_fit:.2f},  A = {A_fit:.2f},  \\phi = {phi_fit:.2f}\\," \
          f"\\text{{rad}}$. and $V = {V_vis:.3f}$")

    # Style
    plt.rcParams.update({"font.size": 16})
    color_nc = "tab:red"

    # Figure & axes
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    title = "Counts vs Phase Delay"
    if label_suffix:
        title += f": {label_suffix}"
    fig.suptitle(title, fontsize=20)

    # Plot Nc
    ax.set_ylabel(r"Counts/sec", fontsize=18)
    ax.errorbar(
        delta,
        Nc,
        yerr=Nc_err,
        fmt="x",
        color=color_nc,
        label=fr"{label_suffix}, $V={V_vis:.3f}$) with fit",
        capsize=3,
    )
    ax.plot(delta_fine, Nc_fit_curve, linestyle="--", color=color_nc, lw=1)
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.set_xlabel(r"Phase Delay $\delta$ (rad)", fontsize=18)
    ax.legend(loc='upper right')

    xticks = np.arange(0, np.max(delta) + np.pi / 2, np.pi)
    xticklabels = ["0"] + [f"{i}$\\pi$" for i in range(1, len(xticks))]
    xticklabels = [s.replace("1$\\pi$", "$\\pi$") for s in xticklabels]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    ax_nm = ax.secondary_xaxis("top", functions=(delta_to_nm, nm_to_delta))
    ax_nm.set_xlabel("Piezo Displacement (nm)", fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for suptitle
    plt.savefig(output_filename)
    if show:
        plt.show()
    plt.close(fig)

    print(f"Plot saved as {output_filename}")
    return output_filename
