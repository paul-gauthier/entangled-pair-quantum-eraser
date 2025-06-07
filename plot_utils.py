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

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Global variable to store the fitted value
_fitted_steps_per_2pi = None


def delta_from_steps(steps: np.ndarray | float) -> np.ndarray | float:
    """Convert piezo steps → phase delay δ (radians)."""
    if _fitted_steps_per_2pi is None:
        raise ValueError("STEPS_PER_2PI has not been fitted. Call fit_steps_per_2pi() first.")
    return steps * (2 * np.pi / _fitted_steps_per_2pi)


def steps_from_delta(delta: np.ndarray | float) -> np.ndarray | float:
    """Convert phase delay δ (radians) → piezo steps."""
    if _fitted_steps_per_2pi is None:
        raise ValueError("STEPS_PER_2PI has not been fitted. Call fit_steps_per_2pi() first.")
    return delta * _fitted_steps_per_2pi / (2 * np.pi)


# ---------------------------------------------------------------------------
# Model function for fitting
# ---------------------------------------------------------------------------
def _cos_model(d, A, C0, phi):
    """Cosine model for fitting interference patterns."""
    return C0 + A * (1 + np.cos(d + phi)) / 2


def _cos_model_with_period(steps, A, C0, phi, steps_per_2pi):
    """Cosine model with variable period for fitting STEPS_PER_2PI."""
    delta = steps * (2 * np.pi / steps_per_2pi)
    return C0 + A * (1 + np.cos(delta + phi)) / 2


def fit_steps_per_2pi(datasets):
    """
    Fit STEPS_PER_2PI from multiple datasets.

    Parameters
    ----------
    datasets : list of tuples
        Each tuple should be (piezo_steps, counts) where counts can be Ni or Nc

    Returns
    -------
    float
        The fitted STEPS_PER_2PI value
    """
    global _fitted_steps_per_2pi

    all_steps = []
    all_counts = []
    all_weights = []

    for piezo_steps, counts in datasets:
        all_steps.extend(piezo_steps)
        all_counts.extend(counts)
        all_weights.extend(1.0 / np.sqrt(np.maximum(counts, 1)))  # Poisson weights

    all_steps = np.array(all_steps)
    all_counts = np.array(all_counts)
    all_weights = np.array(all_weights)

    # Initial guess
    p0 = [np.ptp(all_counts), np.min(all_counts), 0.0, 22.0]

    try:
        popt, _ = curve_fit(
            _cos_model_with_period,
            all_steps,
            all_counts,
            p0=p0,
            sigma=all_weights,
            absolute_sigma=True,
            bounds=([0, 0, -np.pi, 10], [np.inf, np.inf, np.pi, 50]),
        )

        _, _, _, fitted_steps_per_2pi = popt
        _fitted_steps_per_2pi = fitted_steps_per_2pi

        print(f"Fitted STEPS_PER_2PI = {fitted_steps_per_2pi:.3f}")
        return fitted_steps_per_2pi

    except Exception as e:
        raise RuntimeError(f"Could not fit STEPS_PER_2PI: {e}")


def set_steps_per_2pi(value):
    """Manually set the STEPS_PER_2PI value."""
    global _fitted_steps_per_2pi
    _fitted_steps_per_2pi = value


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
    Ni_err = np.sqrt(Ni)
    Nc_err = np.sqrt(Nc)

    # Phase delay for x-axis
    delta = delta_from_steps(piezo_steps)

    # ------------------------------------------------------------------
    # Fit coincidence counts with ½(1+cos(δ+φ)) model
    # ------------------------------------------------------------------
    p0_c = [np.ptp(Nc), np.min(Nc), 0.0]  # initial guesses
    popt_c, pcov_c = curve_fit(
        _cos_model,
        delta,
        Nc,
        p0=p0_c,
        sigma=Nc_err,
        absolute_sigma=True,
    )

    # ------------------------------------------------------------------
    # Convert optimiser output to physically meaningful parameters (Nc)
    # ------------------------------------------------------------------
    A_fit_c, C0_fit_c, phi_fit_c = popt_c
    A_err_c, C0_err_c, phi_err_c = np.sqrt(np.diag(pcov_c))
    if A_fit_c < 0:  # enforce non-negative modulation depth
        A_fit_c = -A_fit_c
        phi_fit_c += np.pi  # keep model invariant
        C0_fit_c -= A_fit_c  # compensate offset to preserve model
    # Wrap phase into [0, 2π)
    phi_fit_c = phi_fit_c % (2 * np.pi)

    # ------------------------------------------------------------------
    # Fit idler counts with ½(1+cos(δ+φ)) model
    # ------------------------------------------------------------------
    p0_i = [np.ptp(Ni), np.min(Ni), 0.0]  # initial guesses
    popt_i, pcov_i = curve_fit(
        _cos_model,
        delta,
        Ni,
        p0=p0_i,
        sigma=Ni_err,
        absolute_sigma=True,
    )

    # ------------------------------------------------------------------
    # Convert optimiser output to physically meaningful parameters (Ni)
    # ------------------------------------------------------------------
    A_fit_i, C0_fit_i, phi_fit_i = popt_i
    A_err_i, C0_err_i, phi_err_i = np.sqrt(np.diag(pcov_i))
    if A_fit_i < 0:  # enforce non-negative modulation depth
        A_fit_i = -A_fit_i
        phi_fit_i += np.pi  # keep model invariant
        C0_fit_i -= A_fit_i  # compensate offset to preserve model
    # Wrap phase into [0, 2π)
    phi_fit_i = phi_fit_i % (2 * np.pi)

    delta_fine = np.linspace(delta.min(), delta.max(), 500)
    Nc_fit = _cos_model(delta_fine, A_fit_c, C0_fit_c, phi_fit_c)
    Ni_fit = _cos_model(delta_fine, A_fit_i, C0_fit_i, phi_fit_i)

    # Calculate visibility uncertainties using error propagation
    V_vis_c = A_fit_c / (A_fit_c + 2 * C0_fit_c)
    V_vis_i = A_fit_i / (A_fit_i + 2 * C0_fit_i)

    # Error propagation for visibility V = A/(A + 2*C0)
    # dV/dA = 2*C0/(A + 2*C0)^2, dV/dC0 = -2*A/(A + 2*C0)^2
    denom_c = (A_fit_c + 2 * C0_fit_c) ** 2
    denom_i = (A_fit_i + 2 * C0_fit_i) ** 2
    V_err_c = np.sqrt(
        (2 * C0_fit_c / denom_c * A_err_c) ** 2 + (-2 * A_fit_c / denom_c * C0_err_c) ** 2
    )
    V_err_i = np.sqrt(
        (2 * C0_fit_i / denom_i * A_err_i) ** 2 + (-2 * A_fit_i / denom_i * C0_err_i) ** 2
    )

    # Print fitted parameters
    print(f"Fit results for {output_filename}:")
    print("  Coincidence counts:")
    print(f"    C0 = {C0_fit_c:.2f} ± {C0_err_c:.2f}")
    print(f"    A = {A_fit_c:.2f} ± {A_err_c:.2f}")
    print(f"    phi = {phi_fit_c:.2f} ± {phi_err_c:.2f} rad ({np.degrees(phi_fit_c):.1f}°)")
    print(f"    Visibility V = {V_vis_c:.3f} ± {V_err_c:.3f}")
    print("  Idler counts:")
    print(f"    C0 = {C0_fit_i:.2f} ± {C0_err_i:.2f}")
    print(f"    A = {A_fit_i:.2f} ± {A_err_i:.2f}")
    print(f"    phi = {phi_fit_i:.2f} ± {phi_err_i:.2f} rad ({np.degrees(phi_fit_i):.1f}°)")
    print(f"    Visibility V = {V_vis_i:.3f} ± {V_err_i:.3f}")
    print(
        f"  LaTeX: Coincidence: $C_0 = {C0_fit_c:.2f},  A = {A_fit_c:.2f},  \\phi ="
        f" {phi_fit_c:.2f}\\,\\text{{rad}}$. and $V = {V_vis_c:.3f}$"
    )
    print(
        f"  LaTeX: Idler: $C_0 = {C0_fit_i:.2f},  A = {A_fit_i:.2f},  \\phi = {phi_fit_i:.2f}\\,"
        f"\\text{{rad}}$. and $V = {V_vis_i:.3f}$"
    )

    # Style ------------------------------------------------------------------
    plt.rcParams.update({"font.size": 16})
    color_nc, color_ni = "tab:red", "tab:green"

    # Figure & axes ----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Add title
    if label_suffix:
        title = label_suffix
    else:
        title = "Counts vs Phase Delay"
    fig.suptitle(title, fontsize=20)

    # Top panel: Ni only -----------------------------------------------------
    ax1.set_ylabel(r"Counts/sec", fontsize=18)
    ax1.errorbar(
        delta,
        Ni,
        yerr=Ni_err,
        fmt="s",
        color=color_ni,
        # linestyle=":",
        label=r"N_i",
        capsize=3,
    )
    # Overlay best-fit cosine curve for idler
    ax1.plot(
        delta_fine,
        Ni_fit,
        linestyle="--",
        color=color_ni,
        lw=1,
    )
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(loc="upper right")

    # Add fit parameters text box for idler
    textstr_i = (
        f"$C_0 = {C0_fit_i:.1f} \\pm {C0_err_i:.1f}$\n"
        f"$A = {A_fit_i:.1f} \\pm {A_err_i:.1f}$\n"
        f"$\\phi = {phi_fit_i:.2f} \\pm {phi_err_i:.2f}$ rad\n"
        f"$V = {V_vis_i:.3f} \\pm {V_err_i:.3f}$"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax1.text(
        0.02,
        0.98,
        textstr_i,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

    # Bottom panel: Nc -------------------------------------------------------
    ax2.set_ylabel(r"Counts/sec", fontsize=18)
    ax2.errorbar(
        delta,
        Nc,
        yerr=Nc_err,
        fmt="x",
        color=color_nc,
        # linestyle="--",
        label=r"N_c",
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
    ax2.legend(loc="upper right")

    # Add fit parameters text box for coincidence
    textstr_c = (
        f"$C_0 = {C0_fit_c:.1f} \\pm {C0_err_c:.1f}$\n"
        f"$A = {A_fit_c:.1f} \\pm {A_err_c:.1f}$\n"
        f"$\\phi = {phi_fit_c:.2f} \\pm {phi_err_c:.2f}$ rad\n"
        f"$V = {V_vis_c:.3f} \\pm {V_err_c:.3f}$"
    )
    props = dict(boxstyle="round", facecolor="lightblue", alpha=0.8)
    ax2.text(
        0.02,
        0.98,
        textstr_c,
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=props,
    )

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
