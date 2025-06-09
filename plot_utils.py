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


def delta_from_steps(steps: np.ndarray | float, steps_per_2pi: float) -> np.ndarray | float:
    """Convert piezo steps → phase delay δ (radians)."""
    return steps * (2 * np.pi / steps_per_2pi)


def steps_from_delta(delta: np.ndarray | float, steps_per_2pi: float) -> np.ndarray | float:
    """Convert phase delay δ (radians) → piezo steps."""
    return delta * steps_per_2pi / (2 * np.pi)


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

        print(f"Fitted STEPS_PER_2PI = {fitted_steps_per_2pi:.3f}")
        return fitted_steps_per_2pi

    except Exception as e:
        raise RuntimeError(f"Could not fit STEPS_PER_2PI: {e}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_counts(
    piezo_steps: np.ndarray,
    Ns: np.ndarray,
    Ni: np.ndarray,
    Nc: np.ndarray,
    steps_per_2pi: float,
    *,
    output_filename: str = "counts_vs_phase_delay.pdf",
    label_suffix: str = "",
    show: bool = False,
    nc_phi: float | None = None,
    ni_phi: float | None = None,
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
    steps_per_2pi :
        Conversion factor from piezo steps to 2π phase delay.
    output_filename :
        Path where the PDF/PNG will be written.
    label_suffix :
        Optional suffix appended to trace labels, useful when plotting
        multiple datasets on shared axes.
    show :
        If ``True`` also display the figure interactively.
    nc_phi :
        If provided, fix the phase of the coincidence count fit to this
        value (in radians).
    ni_phi :
        If provided, fix the phase of the idler count fit to this value
        (in radians).

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
    delta = delta_from_steps(piezo_steps, steps_per_2pi)

    # ------------------------------------------------------------------
    # Fit coincidence counts with ½(1+cos(δ+φ)) model
    # ------------------------------------------------------------------
    if nc_phi is not None:

        def model_c(d, A, C0):
            return _cos_model(d, A, C0, nc_phi)

        p0_c = [np.ptp(Nc), np.min(Nc)]
        bounds_c = ([0, 0], [np.inf, np.inf])
        popt_c_fit, pcov_c_fit = curve_fit(
            model_c, delta, Nc, p0=p0_c, sigma=Nc_err, absolute_sigma=True, bounds=bounds_c
        )
        popt_c = [popt_c_fit[0], popt_c_fit[1], nc_phi]
        pcov_c = np.zeros((3, 3))
        pcov_c[:2, :2] = pcov_c_fit
    else:
        p0_c = [np.ptp(Nc), np.min(Nc), 0.0]  # initial guesses
        bounds_c = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        popt_c, pcov_c = curve_fit(
            _cos_model,
            delta,
            Nc,
            p0=p0_c,
            sigma=Nc_err,
            absolute_sigma=True,
            bounds=bounds_c,
        )

    # ------------------------------------------------------------------
    # Convert optimiser output to physically meaningful parameters (Nc)
    # ------------------------------------------------------------------
    A_fit_c, C0_fit_c, phi_fit_c = popt_c
    A_err_c, C0_err_c, phi_err_c = np.sqrt(np.diag(pcov_c))
    # Wrap phase into [0, 2π)
    phi_fit_c = phi_fit_c % (2 * np.pi)

    # ------------------------------------------------------------------
    # Fit idler counts with ½(1+cos(δ+φ)) model
    # ------------------------------------------------------------------
    if ni_phi is not None:

        def model_i(d, A, C0):
            return _cos_model(d, A, C0, ni_phi)

        p0_i = [np.ptp(Ni), np.min(Ni)]
        bounds_i = ([0, 0], [np.inf, np.inf])
        popt_i_fit, pcov_i_fit = curve_fit(
            model_i, delta, Ni, p0=p0_i, sigma=Ni_err, absolute_sigma=True, bounds=bounds_i
        )
        popt_i = [popt_i_fit[0], popt_i_fit[1], ni_phi]
        pcov_i = np.zeros((3, 3))
        pcov_i[:2, :2] = pcov_i_fit
    else:
        p0_i = [np.ptp(Ni), np.min(Ni), 0.0]  # initial guesses
        bounds_i = ([0, 0, -np.inf], [np.inf, np.inf, np.inf])
        popt_i, pcov_i = curve_fit(
            _cos_model,
            delta,
            Ni,
            p0=p0_i,
            sigma=Ni_err,
            absolute_sigma=True,
            bounds=bounds_i,
        )

    # ------------------------------------------------------------------
    # Convert optimiser output to physically meaningful parameters (Ni)
    # ------------------------------------------------------------------
    A_fit_i, C0_fit_i, phi_fit_i = popt_i
    A_err_i, C0_err_i, phi_err_i = np.sqrt(np.diag(pcov_i))
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

    # ------------------------------------------------------------------
    # Goodness-of-fit: reduced χ² for each data series
    # ------------------------------------------------------------------
    resid_c = (Nc - _cos_model(delta, *popt_c)) / Nc_err
    chi2_c = np.sum(resid_c**2)
    dof_c = len(Nc) - len(popt_c)
    red_chi2_c = chi2_c / dof_c

    resid_i = (Ni - _cos_model(delta, *popt_i)) / Ni_err
    chi2_i = np.sum(resid_i**2)
    dof_i = len(Ni) - len(popt_i)
    red_chi2_i = chi2_i / dof_i

    # Calculate average counts (C0 + A/2) and uncertainties
    avg_c = C0_fit_c + A_fit_c / 2
    avg_i = C0_fit_i + A_fit_i / 2
    # Error propagation for average: d(avg)/dC0 = 1, d(avg)/dA = 1/2
    avg_err_c = np.sqrt(C0_err_c**2 + (A_err_c / 2) ** 2)
    avg_err_i = np.sqrt(C0_err_i**2 + (A_err_i / 2) ** 2)

    # Print fitted parameters
    print(f"Fit results for {output_filename}:")
    num_points = len(piezo_steps)
    delta_range_pi = (np.max(delta) - np.min(delta)) / np.pi
    print(f"  {num_points} data points spanning {delta_range_pi:.2f}π radians.")
    print("  Coincidence counts:")
    print(f"    C0 = {C0_fit_c:.2f} ± {C0_err_c:.2f}")
    print(
        f"    A = {A_fit_c:.2f} ± {A_err_c:.2f}  [{A_fit_c - A_err_c:.2f}, {A_fit_c + A_err_c:.2f}]"
    )
    print(f"    Average = {avg_c:.2f} ± {avg_err_c:.2f}")
    print(
        f"    phi = {phi_fit_c:.2f} ± {phi_err_c:.2f} rad ({np.degrees(phi_fit_c):.1f} ±"
        f" {np.degrees(phi_err_c):.1f}°)"
    )
    print(
        f"    Visibility V = {V_vis_c:.4f} ± {V_err_c:.4f}  [{V_vis_c - V_err_c:.4f},"
        f" {V_vis_c + V_err_c:.4f}]"
    )
    print(f"    reduced χ² = {red_chi2_c:.2f}")
    print("  Idler counts:")
    print(f"    C0 = {C0_fit_i:.2f} ± {C0_err_i:.2f}")
    print(
        f"    A = {A_fit_i:.2f} ± {A_err_i:.2f}  [{A_fit_i - A_err_i:.2f}, {A_fit_i + A_err_i:.2f}]"
    )
    print(f"    Average = {avg_i:.2f} ± {avg_err_i:.2f}")
    print(
        f"    phi = {phi_fit_i:.2f} ± {phi_err_i:.2f} rad ({np.degrees(phi_fit_i):.1f} ±"
        f" {np.degrees(phi_err_i):.1f}°)"
    )
    print(
        f"    Visibility V = {V_vis_i:.4f} ± {V_err_i:.4f}  [{V_vis_i - V_err_i:.4f},"
        f" {V_vis_i + V_err_i:.4f}]"
    )
    print(f"    reduced χ² = {red_chi2_i:.2f}")

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
    ax1.set_ylabel(r"Counts", fontsize=18)
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
        f"$\\mathrm{{Avg}} = {avg_i:.1f} \\pm {avg_err_i:.1f}$\n"
        f"$\\phi = {phi_fit_i:.2f} \\pm {phi_err_i:.2f}$ rad\n"
        f"$\\phi = {np.degrees(phi_fit_i):.1f} \\pm {np.degrees(phi_err_i):.1f}°$\n"
        f"$V = {V_vis_i:.4f} \\pm {V_err_i:.4f}$"
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
    ax2.set_ylabel(r"Counts", fontsize=18)
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
        f"$\\mathrm{{Avg}} = {avg_c:.1f} \\pm {avg_err_c:.1f}$\n"
        f"$\\phi = {phi_fit_c:.2f} \\pm {phi_err_c:.2f}$ rad\n"
        f"$\\phi = {np.degrees(phi_fit_c):.1f} \\pm {np.degrees(phi_err_c):.1f}°$\n"
        f"$V = {V_vis_c:.4f} \\pm {V_err_c:.4f}$"
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
    start_tick = np.ceil(np.min(delta) / np.pi) * np.pi
    xticks = np.arange(start_tick, np.max(delta) + np.pi / 2, np.pi)

    xticklabels = []
    for tick in xticks:
        multiple = int(round(tick / np.pi))
        if multiple == 0:
            xticklabels.append("0")
        elif multiple == 1:
            xticklabels.append("$\\pi$")
        elif multiple == -1:
            xticklabels.append("$-\\pi$")
        else:
            xticklabels.append(f"{multiple}$\\pi$")

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
