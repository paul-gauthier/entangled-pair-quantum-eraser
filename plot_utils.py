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
from scipy.optimize import curve_fit, least_squares

# Maximum number of function evaluations allowed in each curve_fit call.
# Increase this if you still encounter “Optimal parameters not found”.
MAXFEV = 20_000


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


def fit_steps_per_2pi(
    piezo_steps: np.ndarray,
    counts_corr: np.ndarray,
    counts_raw: np.ndarray | None = None,
    *,
    label: str = "",
) -> tuple[float, float]:
    """
    Fit the phase-delay calibration (STEPS_PER_2PI) for a *single* dataset.

    Parameters
    ----------
    piezo_steps :
        1-D array of piezo stage positions (integer steps).
    counts_corr :
        1-D array of dark / accidental-corrected coincidence counts.
    counts_raw :
        Optional 1-D array of *raw* coincidence counts used only for
        Poisson uncertainties (σ = √N).  If ``None``, ``counts_corr`` is
        also used for σ.

    Returns
    -------
    tuple
        (fitted_steps_per_2pi, one_sigma_uncertainty)
    """

    # Poisson σ = √N (shot-noise) taken from *raw* counts
    if counts_raw is None:
        counts_raw = counts_corr
    sigma = np.sqrt(np.maximum(counts_raw, 1))

    # Initial guess: [amplitude, offset, phase, steps_per_2pi]
    p0 = [np.ptp(counts_corr), np.min(counts_corr), 0.0, 22.0]

    popt, pcov = curve_fit(
        _cos_model_with_period,
        piezo_steps,
        counts_corr,
        p0=p0,
        sigma=sigma,
        absolute_sigma=True,
        bounds=([0, 0, -np.pi, 10], [np.inf, np.inf, np.pi, 50]),
    )

    fitted_steps_per_2pi = popt[3]
    se_steps_per_2pi = np.sqrt(pcov[3, 3])

    label_str = f" for {label}" if label else ""
    print(f"  Fitted STEPS_PER_2PI{label_str} = {fitted_steps_per_2pi:.3f} ± {se_steps_per_2pi:.3f}")
    return fitted_steps_per_2pi, se_steps_per_2pi


# ---------------------------------------------------------------------------
# Global (hierarchical) cosine fit across multiple datasets
# ---------------------------------------------------------------------------
def global_cosine_fit(
    datasets: list[dict],
    *,
    counts_key: str,
    raw_key: str,
    label: str = "Series",
) -> dict[str, float]:
    """
    Simultaneously fit all datasets with shared amplitude ``A`` and offset
    ``C0`` while allowing each dataset its own phase shift ``φ_k``.

    The model for each point *j* in dataset *k* is::

        y_j = C0 + A * (1 + cos(δ_j + φ_k)) / 2

    Poisson shot-noise (σ = √N) from the *raw* counts is used for weighting.

    Returns a dictionary containing the best-fit parameters, their
    1 σ errors, the derived visibility and the reduced χ² of the fit.
    """
    # -------- Flatten all scans into one big array ------------------------
    deltas, y, sigma, idx = [], [], [], []
    for k, ds in enumerate(datasets):
        δ = delta_from_steps(ds["piezo_steps"], ds["steps_per_2pi"])
        y_k = ds[counts_key]
        σ_k = np.sqrt(np.maximum(ds[raw_key], 1))
        n_k = len(y_k)

        deltas.append(δ)
        y.append(y_k)
        sigma.append(σ_k)
        idx.append(np.full(n_k, k, dtype=int))

    delta_all = np.concatenate(deltas)
    y_all = np.concatenate(y)
    sigma_all = np.concatenate(sigma)
    idx_all = np.concatenate(idx)

    m = len(datasets)  # number of individual phase parameters

    # -------- Residual function for least-squares -------------------------
    def _residuals(p):
        A, C0 = p[0], p[1]
        phis = p[2:]
        model = C0 + A * (1 + np.cos(delta_all + phis[idx_all])) / 2
        return (y_all - model) / sigma_all

    # Initial guesses & bounds
    p0 = [np.ptp(y_all) / 2, np.mean(y_all)] + [0.0] * m
    lower = [0.0, 0.0] + [-2 * np.pi] * m
    upper = [np.inf, np.inf] + [2 * np.pi] * m

    res = least_squares(_residuals, p0, bounds=(lower, upper), max_nfev=MAXFEV)
    if not res.success:
        raise RuntimeError(res.message)

    # -------- Covariance & parameter errors --------------------------------
    J = res.jac
    dof = y_all.size - J.shape[1]
    chi2 = np.sum(res.fun**2)
    red_chi2 = chi2 / dof if dof > 0 else float("nan")
    cov = np.linalg.inv(J.T @ J) * chi2 / dof

    A_fit, C0_fit = res.x[:2]
    A_err, C0_err = np.sqrt(np.diag(cov)[:2])

    # -------- Visibility and its uncertainty ------------------------------
    V = A_fit / (A_fit + 2 * C0_fit)
    denom = (A_fit + 2 * C0_fit) ** 2
    V_err = np.sqrt((2 * C0_fit / denom * A_err) ** 2 + (-2 * A_fit / denom * C0_err) ** 2)

    print(f"\nGlobal {label} fit (shared A, C0 across {m} datasets):")
    print(f"  A  = {A_fit:.2f} ± {A_err:.2f}")
    print(f"  C0 = {C0_fit:.2f} ± {C0_err:.2f}")
    print(
        f"  Visibility V = {V:.4f} ± {V_err:.4f} "
        f"[{V - V_err:.4f}, {V + V_err:.4f}]   (reduced χ² = {red_chi2:.2f})"
    )

    return {
        "A": A_fit,
        "A_err": A_err,
        "C0": C0_fit,
        "C0_err": C0_err,
        "V": V,
        "V_err": V_err,
        "chi2red": red_chi2,
    }


# ---------------------------------------------------------------------------
# Joint global cosine fit for Idler and Coincidence data
# ---------------------------------------------------------------------------
def global_joint_cosine_fit(
    datasets: list[dict],
    *,
    ni_key: str = "Ni_corr",
    nc_key: str = "Nc_corr",
    ni_raw_key: str = "Ni",
    nc_raw_key: str = "Nc",
) -> dict[str, float]:
    """
    Perform a *single* hierarchical fit to all idler (N_i) and coincidence
    (N_c) scans, sharing a per-dataset phase φ_k **and** one global phase
    offset φ_ic between the two series::

        N_i = C0_i + A_i · (1 + cos(δ + φ_k)) / 2
        N_c = C0_c + A_c · (1 + cos(δ + φ_k + φ_ic)) / 2

    The fit returns visibilities, phase offset and uncertainties.
    """
    deltas, y, sigma, idx, typ = [], [], [], [], []  # typ: 0 → N_i, 1 → N_c
    for k, ds in enumerate(datasets):
        δ = delta_from_steps(ds["piezo_steps"], ds["steps_per_2pi"])

        # Idler
        deltas.append(δ)
        y.append(ds[ni_key])
        sigma.append(np.sqrt(np.maximum(ds[ni_raw_key], 1)))
        n_k = len(δ)
        idx.append(np.full(n_k, k, dtype=int))
        typ.append(np.zeros(n_k, dtype=int))

        # Coincidence
        deltas.append(δ)
        y.append(ds[nc_key])
        sigma.append(np.sqrt(np.maximum(ds[nc_raw_key], 1)))
        idx.append(np.full(n_k, k, dtype=int))
        typ.append(np.ones(n_k, dtype=int))

    delta_all = np.concatenate(deltas)
    y_all = np.concatenate(y)
    sigma_all = np.concatenate(sigma)
    idx_all = np.concatenate(idx)
    typ_all = np.concatenate(typ)

    m = len(datasets)  # number of φ_k parameters

    def _residuals(p):
        A_i, C0_i, A_c, C0_c, phi_ic = p[:5]
        phis = p[5:]
        phase_k = phis[idx_all]
        model = np.where(
            typ_all == 0,
            C0_i + A_i * (1 + np.cos(delta_all + phase_k)) / 2,
            C0_c + A_c * (1 + np.cos(delta_all + phase_k + phi_ic)) / 2,
        )
        return (y_all - model) / sigma_all

    # Initial guesses
    Ni_all = np.concatenate([ds[ni_key] for ds in datasets])
    Nc_all = np.concatenate([ds[nc_key] for ds in datasets])
    p0 = [
        np.ptp(Ni_all) / 2,
        np.mean(Ni_all),
        np.ptp(Nc_all) / 2,
        np.mean(Nc_all),
        0.0,
    ] + [0.0] * m

    lower = [0.0, 0.0, 0.0, 0.0, -2 * np.pi] + [-2 * np.pi] * m
    upper = [np.inf, np.inf, np.inf, np.inf, 2 * np.pi] + [2 * np.pi] * m

    res = least_squares(_residuals, p0, bounds=(lower, upper), max_nfev=MAXFEV)
    if not res.success:
        raise RuntimeError(res.message)

    J = res.jac
    dof = y_all.size - J.shape[1]
    chi2 = np.sum(res.fun**2)
    red_chi2 = chi2 / dof if dof > 0 else float("nan")
    cov = np.linalg.inv(J.T @ J) * chi2 / dof

    A_i, C0_i, A_c, C0_c, phi_ic = res.x[:5]
    A_i_err, C0_i_err, A_c_err, C0_c_err, phi_ic_err = np.sqrt(np.diag(cov)[:5])

    phi_ic = phi_ic % (2 * np.pi)
    phis = res.x[5:] % (2 * np.pi)
    phis_err = np.sqrt(np.diag(cov)[5:])

    # Visibilities
    V_i = A_i / (A_i + 2 * C0_i)
    V_c = A_c / (A_c + 2 * C0_c)
    denom_i = (A_i + 2 * C0_i) ** 2
    denom_c = (A_c + 2 * C0_c) ** 2
    V_i_err = np.sqrt((2 * C0_i / denom_i * A_i_err) ** 2 + (-2 * A_i / denom_i * C0_i_err) ** 2)
    V_c_err = np.sqrt((2 * C0_c / denom_c * A_c_err) ** 2 + (-2 * A_c / denom_c * C0_c_err) ** 2)

    print("\nGlobal joint fit (N_i & N_c):")
    print(f"  A_i  = {A_i:.2f} ± {A_i_err:.2f}")
    print(f"  C0_i = {C0_i:.2f} ± {C0_i_err:.2f}")
    print(f"  A_c  = {A_c:.2f} ± {A_c_err:.2f}")
    print(f"  C0_c = {C0_c:.2f} ± {C0_c_err:.2f}")
    print(
        f"  φ_ic = {phi_ic:.2f} ± {phi_ic_err:.2f} rad "
        f"({np.degrees(phi_ic):.1f} ± {np.degrees(phi_ic_err):.1f}°)"
    )
    # Phase offset of the first dataset’s idler scan
    print(
        f"  φ_i0 = {phis[0]:.2f} ± {phis_err[0]:.2f} rad "
        f"({np.degrees(phis[0]):.1f} ± {np.degrees(phis_err[0]):.1f}°)"
    )
    print(f"  Vi = {V_i:.4f} ± {V_i_err:.4f}   [{V_i - V_i_err:.4f}, {V_i + V_i_err:.4f}]")
    print(f"  Vc = {V_c:.4f} ± {V_c_err:.4f}   [{V_c - V_c_err:.4f}, {V_c + V_c_err:.4f}]")
    print(f"  reduced χ² = {red_chi2:.2f}")
    # -------- Difference curve N_i - N_c parameters ------------------------
    A_d = np.sqrt(A_i**2 + A_c**2 - 2 * A_i * A_c * np.cos(phi_ic))

    # phase of the difference cosine
    phi_0 = np.arctan2(A_c * np.sin(phi_ic), A_i - A_c * np.cos(phi_ic))

    # 1-σ uncertainty on phi_0 (error propagation)
    sinφ, cosφ = np.sin(phi_ic), np.cos(phi_ic)
    dphi_dAi = -A_c * sinφ / A_d**2
    dphi_dAc = A_i * sinφ / A_d**2
    dphi_dφ = A_c * (A_i * cosφ - A_c) / A_d**2
    g = np.array([dphi_dAi, dphi_dAc, dphi_dφ])
    cov_sub = cov[np.ix_([0, 2, 4], [0, 2, 4])]  # A_i, A_c, phi_ic
    phi_0_err = np.sqrt(g @ cov_sub @ g)

    C_d = C0_i - C0_c + (A_i - A_c) / 2
    C0_d = C_d - A_d / 2
    V_d = A_d / (A_d + 2 * C0_d)

    # --- Uncertainties for A_D, C0_D, V_D ----------------------------------
    # derivatives of A_d
    dAd_dAi = (A_i - A_c * cosφ) / A_d
    dAd_dAc = (A_c - A_i * cosφ) / A_d
    dAd_dφ = (A_i * A_c * sinφ) / A_d

    # derivatives of C0_d
    dC0d_dAi = 0.5 - 0.5 * dAd_dAi
    dC0d_dAc = -0.5 - 0.5 * dAd_dAc
    dC0d_dC0i = 1.0
    dC0d_dC0c = -1.0
    dC0d_dφ = -0.5 * dAd_dφ

    # derivatives of V_d
    denom = A_d + 2 * C0_d
    dVd_dAi = 2 * (C0_d * dAd_dAi - A_d * dC0d_dAi) / denom**2
    dVd_dAc = 2 * (C0_d * dAd_dAc - A_d * dC0d_dAc) / denom**2
    dVd_dC0i = -2 * A_d * dC0d_dC0i / denom**2
    dVd_dC0c = -2 * A_d * dC0d_dC0c / denom**2
    dVd_dφ = 2 * (C0_d * dAd_dφ - A_d * dC0d_dφ) / denom**2

    # gradient vectors in parameter order [A_i, C0_i, A_c, C0_c, phi_ic]
    g_A = np.array([dAd_dAi, 0.0, dAd_dAc, 0.0, dAd_dφ])
    g_C0 = np.array([dC0d_dAi, dC0d_dC0i, dC0d_dAc, dC0d_dC0c, dC0d_dφ])
    g_V = np.array([dVd_dAi, dVd_dC0i, dVd_dAc, dVd_dC0c, dVd_dφ])

    # use only the 5×5 sub-covariance corresponding to [A_i, C0_i, A_c, C0_c, φ_ic]
    cov5 = cov[:5, :5]

    A_d_err = float(np.sqrt(g_A @ cov5 @ g_A))
    C0_d_err = float(np.sqrt(g_C0 @ cov5 @ g_C0))
    V_d_err = float(np.sqrt(g_V @ cov5 @ g_V))

    print("  Difference (N_i - N_c):")
    print(f"    C0_D   = {C0_d:.2f} ± {C0_d_err:.2f}")
    print(f"    A_D    = {A_d:.2f} ± {A_d_err:.2f}")
    print(
        f"    φ0_D = {phi_0:.2f} ± {phi_0_err:.2f} rad "
        f"({np.degrees(phi_0):.1f} ± {np.degrees(phi_0_err):.1f}°)"
    )
    print(f"    V_D    = {V_d:.4f} ± {V_d_err:.4f}")
    print(f"    reduced χ² = {red_chi2:.2f}")

    return {
        "A_i": A_i,
        "A_i_err": A_i_err,
        "C0_i": C0_i,
        "C0_i_err": C0_i_err,
        "V_i": V_i,
        "V_i_err": V_i_err,
        "A_c": A_c,
        "A_c_err": A_c_err,
        "C0_c": C0_c,
        "C0_c_err": C0_c_err,
        "V_c": V_c,
        "V_c_err": V_c_err,
        "phi_ic": phi_ic,
        "phi_ic_err": phi_ic_err,
        "phis": phis,
        "phis_err": phis_err,
        # Difference curve N_i - N_c
        "A_d": A_d,
        "A_d_err": A_d_err,
        "C0_d": C0_d,
        "C0_d_err": C0_d_err,
        "V_d": V_d,
        "V_d_err": V_d_err,
        "phi_0_d": phi_0,
        "phi_0_d_err": phi_0_err,
        "chi2red": red_chi2,
    }


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
    Nc_raw: np.ndarray | None = None,
    Ni_raw: np.ndarray | None = None,
    return_metrics: bool = False,
) -> str | tuple[str, dict]:
    """
    Plot Ns, Ni, and Nc versus phase delay and save the figure.

    Parameters
    ----------
    piezo_steps :
        Array of piezo stage positions (integer steps).
    Ns, Ni, Nc :
        Arrays of signal, idler, and coincidence counts (same length as
        ``piezo_steps``).
    Nc_raw :
        Optional array of *raw* coincidence counts used only for
        Poisson uncertainties (σ = √N).  Defaults to ``Nc``.
    Ni_raw :
        Optional array of *raw* idler counts used only for
        Poisson uncertainties (σ = √N).  Defaults to ``Ni``.
    steps_per_2pi :
        Conversion factor from piezo steps to 2π phase delay.
    output_filename :
        Path where the PDF/PNG will be written.
    label_suffix :
        Optional suffix appended to trace labels, useful when plotting
        multiple datasets on shared axes.
    show :
        If ``True`` also display the figure interactively.
    return_metrics :
        If ``True`` also return a dict containing the fitted visibilities
        and their 1σ uncertainties.

    Returns
    -------
    str | tuple[str, dict]
        If ``return_metrics`` is ``False`` returns just the output
        filename. Otherwise returns ``(output_filename, metrics_dict)``
        where ``metrics_dict`` contains ``V_vis_c``, ``V_err_c``,
        ``V_vis_i``, and ``V_err_i``.
    """

    print()

    # Poisson (√N) uncertainties from raw counts
    if Ni_raw is None:
        Ni_raw = Ni
    if Nc_raw is None:
        Nc_raw = Nc
    Ni_err = np.sqrt(np.maximum(Ni_raw, 1))
    Nc_err = np.sqrt(np.maximum(Nc_raw, 1))

    # Phase delay for x-axis
    delta = delta_from_steps(piezo_steps, steps_per_2pi)

    # ------------------------------------------------------------------
    # Fit coincidence counts with ½(1+cos(δ+φ)) model
    # ------------------------------------------------------------------
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
        maxfev=MAXFEV,
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
        maxfev=MAXFEV,
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
        f"    Visibility Vc = {V_vis_c:.4f} ± {V_err_c:.4f}  [{V_vis_c - V_err_c:.4f},"
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
        f"    Visibility Vi = {V_vis_i:.4f} ± {V_err_i:.4f}  [{V_vis_i - V_err_i:.4f},"
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
    if return_metrics:
        return output_filename, {
            "V_vis_c": V_vis_c,
            "V_err_c": V_err_c,
            "V_vis_i": V_vis_i,
            "V_err_i": V_err_i,
        }
    return output_filename
