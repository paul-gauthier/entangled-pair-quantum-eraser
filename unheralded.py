#!/usr/bin/env python


from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, least_squares

from plot_utils import delta_from_steps
import matplotlib.pyplot as plt

# Piezo-stage positions (common to both data sets)
piezo_steps = np.array([
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
])


eraser_on = '''
N_s	N_i	N_c
4187	1059	24
4166	1024	15
4237	1042	11
3054	1015	8
4251	1048	8
4192	1052	11
4251	1049	11
4191	1066	14
4234	1100	21
4251	1094	23
4239	1095	21
4245	1060	21
4245	1072	18
4247	1023	13
4258	1037	8
4172	1017	4
4227	1053	8
4161	1084	12
4275	1074	16
4211	1134	22
4225	1089	22
4257	1079	22
4240	1083	24
4179	1021	19
'''

eraser_off = '''
N_s	N_i	N_c
3809	1085	21
3842	1111	19
3797	1085	20
3853	1090	19
3819	1031	19
3851	1046	15
3873	1014	18
3820	1079	15
3843	1070	14
3810	1092	19
3802	1107	19
3834	1114	19
3871	1096	18
3788	1102	20
3694	1008	20
3751	1040	17
3776	1014	16
3816	1038	17
3692	1018	17
3721	1040	15
3806	1087	22
3812	1098	18
3812	1115	20
3878	1101	22
'''

# ---------------------------------------------------------------------------
# Global two–component fit of idler / coincidence counts (eraser on/off)
# ---------------------------------------------------------------------------
#
#   N1(δ) = C1 + A1 · ½(1 + cos(δ + φ1))
#   N2(δ) = C2 + A2 · ½(1 + cos(δ + φ2))
#
#   Ni(δ) = (1–e_i)·N1(δ) + e_i·N2(δ)          (single-counts mixture)
#   Nc(δ) = (1–e_c)·N1(δ) + e_c·N2(δ)          (coincidence mixture)
#
# The parameters (C1, A1, C2, A2, e) are fitted separately for the idler (i)
# and coincidence (c) data, while the phases φ1, φ2 are *shared* between all
# four traces:
#
#   Ni_off,  Nc_off,  Ni_on,  Nc_on.
#
# Uncertainties (1 σ) are obtained from the covariance matrix Σ = σ²·(JᵀJ)⁻¹,
# with σ² = 2·cost / dof, where *cost* and *J* come from
# `scipy.optimize.least_squares`.
# ---------------------------------------------------------------------------

from typing import Tuple, List

import numpy as np
from numpy.linalg import pinv

from scipy.optimize import curve_fit, least_squares

from plot_utils import delta_from_steps


# --- helper -----------------------------------------------------------------
def _parse_counts(tsv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (Ns, Ni, Nc) arrays parsed from a TSV block with a header row."""
    rows = [
        [float(x) for x in ln.split()]
        for ln in tsv.strip().splitlines()[1:]  # skip header
        if ln.strip()
    ]
    counts = np.asarray(rows, dtype=float)
    return counts[:, 0], counts[:, 1], counts[:, 2]


def _two_component(
    d: np.ndarray,
    C1: float,
    A1: float,
    C2: float,
    A2: float,
    phi1: float,
    phi2: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (N1(d), N2(d))."""
    N1 = C1 + A1 * (1 + np.cos(d + phi1)) / 2
    N2 = C2 + A2 * (1 + np.cos(d + phi2)) / 2
    return N1, N2


def _global_model(params: np.ndarray, d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    (C1_i, A1_i, C2_i, A2_i, e_i,
     C1_c, A1_c, C2_c, A2_c, e_c,
     phi1, phi2) = params

    N1_i, N2_i = _two_component(d, C1_i, A1_i, C2_i, A2_i, phi1, phi2)
    N1_c, N2_c = _two_component(d, C1_c, A1_c, C2_c, A2_c, phi1, phi2)

    Ni = (1 - e_i) * N1_i + e_i * N2_i
    Nc = (1 - e_c) * N1_c + e_c * N2_c
    return Ni, Nc


def _residuals(params: np.ndarray,
               d: np.ndarray,
               Ni_off: np.ndarray,
               Nc_off: np.ndarray,
               Ni_on: np.ndarray,
               Nc_on: np.ndarray) -> np.ndarray:
    """Weighted residual vector concatenating all four traces."""
    Ni_pred, Nc_pred = _global_model(params, d)

    w_Ni_off = 1.0 / np.sqrt(Ni_off)
    w_Ni_on  = 1.0 / np.sqrt(Ni_on)
    w_Nc_off = 1.0 / np.sqrt(Nc_off)
    w_Nc_on  = 1.0 / np.sqrt(Nc_on)

    res = (Ni_off - Ni_pred) * w_Ni_off
    res = np.r_[res, (Ni_on  - Ni_pred) * w_Ni_on]
    res = np.r_[res, (Nc_off - Nc_pred) * w_Nc_off]
    res = np.r_[res, (Nc_on  - Nc_pred) * w_Nc_on]
    return res


def _initial_guesses(delta: np.ndarray,
                     Ni_off: np.ndarray,
                     Ni_on: np.ndarray,
                     Nc_off: np.ndarray,
                     Nc_on: np.ndarray) -> List[float]:
    """Generate a sensible starting vector for the optimiser."""
    def _fit_single(y: np.ndarray, d: np.ndarray):
        p0 = [np.ptp(y), np.min(y), 0.0]
        popt, _ = curve_fit(
            lambda x, A, C0, phi: C0 + A * (1 + np.cos(x + phi)) / 2,
            d, y, p0=p0)
        A, C0, phi = popt
        return C0, A, phi

    C0_i_off, A_i_off, phi_i_off = _fit_single(Ni_off, delta)
    C0_i_on,  A_i_on,  phi_i_on  = _fit_single(Ni_on,  delta)
    C0_c_off, A_c_off, phi_c_off = _fit_single(Nc_off, delta)
    C0_c_on,  A_c_on,  phi_c_on  = _fit_single(Nc_on,  delta)

    p0 = [
        C0_i_off, A_i_off,                 # C1_i, A1_i
        C0_i_on,  A_i_on,                  # C2_i, A2_i
        0.5,                               # e_i
        C0_c_off, A_c_off,                 # C1_c, A1_c
        C0_c_on,  A_c_on,                  # C2_c, A2_c
        0.5,                               # e_c
        phi_i_off, phi_i_off + np.pi / 2   # phi1, phi2
    ]
    return p0


def run_global_fit(show_params: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Perform the global fit and return (best_params, 1σ-errors)."""
    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------
    _, Ni_on,  Nc_on  = _parse_counts(eraser_on)
    _, Ni_off, Nc_off = _parse_counts(eraser_off)

    delta = delta_from_steps(piezo_steps)  # radians

    # ---------------------------------------------------------------
    # Fit
    # ---------------------------------------------------------------
    p0 = _initial_guesses(delta, Ni_off, Ni_on, Nc_off, Nc_on)

    lb = [0, 0, 0, 0, 0,   0, 0, 0, 0, 0,   -np.pi, -np.pi]
    ub = [np.inf]*10 + [np.pi, np.pi]

    result = least_squares(
        _residuals,
        p0,
        bounds=(lb, ub),
        args=(delta, Ni_off, Nc_off, Ni_on, Nc_on),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )

    # ---------------------------------------------------------------
    # Error estimation
    # ---------------------------------------------------------------
    dof = len(_residuals(p0, delta, Ni_off, Nc_off, Ni_on, Nc_on)) - len(result.x)
    if dof <= 0:
        raise RuntimeError("Non-positive degrees of freedom")

    sigma_sq = 2.0 * result.cost / dof
    try:
        cov = sigma_sq * pinv(result.jac.T @ result.jac)
    except Exception:  # numerical issues
        cov = np.full((len(result.x), len(result.x)), np.nan)

    errors = np.sqrt(np.diag(cov))

    # ---------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------
    if show_params:
        labels = [
            "C1_i", "A1_i", "C2_i", "A2_i", "e_i",
            "C1_c", "A1_c", "C2_c", "A2_c", "e_c",
            "phi1", "phi2",
        ]
        print("\n----- GLOBAL-FIT RESULTS -----")
        for name, val, err in zip(labels, result.x, errors):
            if "phi" in name:
                print(f"{name:6} = {val:+9.4f} ± {err:6.4f} rad  "
                      f"({np.degrees(val):+6.1f}° ± {np.degrees(err):.1f}°)")
            else:
                print(f"{name:6} = {val:10.3f} ± {err:.3f}")
        print(f"\nχ² / dof = {2*result.cost:.1f} / {dof}")

    return result.x, errors


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_global_fit()
