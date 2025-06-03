#!/usr/bin/env python

"""
Determine the eraser fraction e for the “eraser-on” and “eraser-off”
runs by globally fitting a two-component interference model to the idler
count data.  Execute this module directly to print the fitted
parameters.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from plot_utils import delta_from_steps

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
# Data parsing utilities
# ---------------------------------------------------------------------------
def _parse_counts(tsv: str):
    """Return the (Ns, Ni, Nc) arrays parsed from a TSV block with header."""
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


# Extract idler counts for both runs
_, Ni_on, _ = _parse_counts(eraser_on)
_, Ni_off, _ = _parse_counts(eraser_off)

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def _ni1(delta, C1, A1, phi1):
    return C1 + A1 * (1 + np.cos(delta + phi1)) / 2


def _ni2(delta, C2, A2, phi2):
    return C2 + A2 * (1 + np.cos(delta + phi2)) / 2


def _ni_total(delta, C1, A1, phi1, C2, A2, phi2, e):
    """Weighted mixture of the two idler populations."""
    return (1.0 - e) * _ni1(delta, C1, A1, phi1) + e * _ni2(delta, C2, A2, phi2)


def _joint_model(
    delta_concat: np.ndarray,
    C1,
    A1,
    phi1,
    C2,
    A2,
    phi2,
    e_on,
    e_off,
    npts: int,
):
    """Model for concatenated δ arrays of the two runs."""
    delta_on = delta_concat[:npts]
    delta_off = delta_concat[npts:]
    pred_on = _ni_total(delta_on, C1, A1, phi1, C2, A2, phi2, e_on)
    pred_off = _ni_total(delta_off, C1, A1, phi1, C2, A2, phi2, e_off)
    return np.concatenate([pred_on, pred_off])


def fit_eraser_fraction(
    piezo_steps: np.ndarray,
    Ni_on: np.ndarray,
    Ni_off: np.ndarray,
):
    """
    Fit the global model to Ni_on and Ni_off and print best-fit parameters.

    Returns
    -------
    dict
        Mapping of parameter names to fitted values and 1-σ errors.
    """
    npts = len(piezo_steps)
    delta = delta_from_steps(piezo_steps)
    delta_cat = np.concatenate([delta, delta])
    Ni_cat = np.concatenate([Ni_on, Ni_off])
    sigma_cat = np.sqrt(Ni_cat)  # Poisson uncertainties

    # Initial guesses
    p0 = [
        Ni_off.min(), Ni_off.ptp(), 0.0,       # C1, A1, φ1
        Ni_on.min(), Ni_on.ptp(),  0.0,        # C2, A2, φ2
        0.5, 0.0,                              # e_on, e_off
    ]
    bounds = (
        [0, 0, -np.pi,   0, 0, -np.pi,   0, 0],             # lower
        [np.inf, np.inf, np.pi,  np.inf, np.inf, np.pi, 1, 1],  # upper
    )

    popt, pcov = curve_fit(
        lambda dcat, C1, A1, phi1, C2, A2, phi2, e_on, e_off: _joint_model(
            dcat, C1, A1, phi1, C2, A2, phi2, e_on, e_off, npts
        ),
        delta_cat,
        Ni_cat,
        p0=p0,
        sigma=sigma_cat,
        absolute_sigma=True,
        bounds=bounds,
        maxfev=20000,
    )

    perr = np.sqrt(np.diag(pcov))
    names = ("C1", "A1", "phi1", "C2", "A2", "phi2", "e_on", "e_off")
    results = {k: v for k, v in zip(names, popt)}
    results_err = {f"{k}_err": v for k, v in zip(names, perr)}

    print("\nGlobal fit results (idler counts)")
    for k in names:
        val = results[k]
        err = results_err[f"{k}_err"]
        if "phi" in k:
            print(f"{k:5s} = {val:8.4f} ± {err:.4f} rad ({np.degrees(val):.1f}°)")
        else:
            print(f"{k:5s} = {val:8.4f} ± {err:.4f}")
    print(
        f"\nEraser fraction (ON)  e_on  = {results['e_on']:.3f} ± {results_err['e_on_err']:.3f}"
    )
    print(
        f"Eraser fraction (OFF) e_off = {results['e_off']:.3f} ± {results_err['e_off_err']:.3f}"
    )
    return {**results, **results_err}


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fit_eraser_fraction(piezo_steps, Ni_on, Ni_off)
