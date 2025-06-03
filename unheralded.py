#!/usr/bin/env python


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
# Fitting routine to extract the “eraser fraction” e
# ---------------------------------------------------------------------------
from plots import _parse_counts


def fit_eraser_fraction(*, verbose: bool = True):
    """
    Simultaneously fit
        • idler singles, eraser *OFF*   (N_i_off)
        • idler singles, eraser *ON*    (N_i_on)
        • coincidence counts, eraser ON (N_c_on)

    to the composite model

        N_i1(δ) = C1 + A1 · ½(1 + cos(δ + φ1))
        N_i2(δ) = C2 + A2 · ½(1 + cos(δ + φ2))

        N_i_total_off(δ) = (1 − e_off) · N_i1 + e_off · N_i2
        N_i_total_on (δ) = (1 −  e_on) · N_i1 +  e_on · N_i2
        N_c_total_on  (δ) = Cc + Ac · ½(1 + cos(δ + φ2))          (only branch 2 interferes)

    Returns
    -------
    dict
        {
            "e_off": (value, σ),
            "e_on":  (value, σ),
            "popt":  array of best-fit parameters,
            "perr":  1-σ uncertainties,
            "chi2":  χ² value,
            "dof":   degrees of freedom,
        }
    """
    # ------------------------------------------------------------------
    # Parse raw TSV tables
    _, Ni_off, _ = _parse_counts(eraser_off)
    _, Ni_on,  Nc_on = _parse_counts(eraser_on)

    delta = delta_from_steps(piezo_steps)

    # Combined data vectors ------------------------------------------------
    x_delta = np.concatenate([delta, delta, delta])
    y_vals  = np.concatenate([Ni_off, Ni_on, Nc_on])
    y_err   = np.sqrt(y_vals)                       # Poisson σ ≈ √N

    data_id = np.concatenate([
        np.zeros_like(delta, dtype=int),            # 0 → Ni_off
        np.ones_like(delta,  dtype=int),            # 1 → Ni_on
        np.full_like(delta, 2, dtype=int),          # 2 → Nc_on
    ])

    xdata = (x_delta, data_id)

    # ------------------------------------------------------------------
    # Model
    def _model(x, C1, A1, phi1, C2, A2, phi2, e_off, e_on, Cc, Ac):
        d, who = x
        Ni1 = C1 + A1 * (1 + np.cos(d + phi1)) / 2
        Ni2 = C2 + A2 * (1 + np.cos(d + phi2)) / 2

        out = np.empty_like(d)

        mask0 = who == 0
        mask1 = who == 1
        mask2 = who == 2

        out[mask0] = (1 - e_off) * Ni1[mask0] + e_off * Ni2[mask0]
        out[mask1] = (1 -  e_on) * Ni1[mask1] +  e_on * Ni2[mask1]
        out[mask2] = Cc + Ac * (1 + np.cos(d[mask2] + phi2)) / 2  # same φ₂ branch
        return out

    # ------------------------------------------------------------------
    # Initial guesses & bounds
    p0 = [
        np.mean(Ni_off),           # C1
        np.ptp(Ni_off) / 2,        # A1
        0.0,                       # phi1
        np.mean(Ni_on),            # C2
        np.ptp(Ni_on) / 2,         # A2
        0.0,                       # phi2
        0.0,                       # e_off
        0.1,                       # e_on
        np.mean(Nc_on) * 0.1,      # Cc
        np.ptp(Nc_on) / 2,         # Ac
    ]

    lower = [
        0, 0, -2 * np.pi,
        0, 0, -2 * np.pi,
        0, 0,
        0, 0,
    ]
    upper = [
        np.inf, np.inf, 2 * np.pi,
        np.inf, np.inf, 2 * np.pi,
        1, 1,
        np.inf, np.inf,
    ]

    # ------------------------------------------------------------------
    # Fit
    popt, pcov = curve_fit(
        _model,
        xdata,
        y_vals,
        p0=p0,
        bounds=(lower, upper),
        sigma=y_err,
        absolute_sigma=True,
        maxfev=10_000,
    )
    perr = np.sqrt(np.diag(pcov))

    # Goodness-of-fit
    resid = (y_vals - _model(xdata, *popt)) / y_err
    chi2_val = np.sum(resid ** 2)
    dof_val = y_vals.size - popt.size

    # ------------------------------------------------------------------
    # Console output
    if verbose:
        names = ["C1", "A1", "phi1",
                 "C2", "A2", "phi2",
                 "e_off", "e_on",
                 "Cc", "Ac"]
        print("\nFit results (±1σ):")
        for n, v, e in zip(names, popt, perr):
            unit = " rad" if "phi" in n else ""
            print(f"  {n:>5} = {v:8.3f} ± {e:.3f}{unit}")
        print(f"\nχ² / dof = {chi2_val:.1f} / {dof_val} = {chi2_val/dof_val:.2f}")
        print(f"\n⇒ e_off = {popt[6]:.3f} ± {perr[6]:.3f}")
        print(f"⇒ e_on  = {popt[7]:.3f} ± {perr[7]:.3f}\n")

    return {
        "e_off": (popt[6], perr[6]),
        "e_on":  (popt[7], perr[7]),
        "popt": popt,
        "perr": perr,
        "chi2": chi2_val,
        "dof":  dof_val,
    }


if __name__ == "__main__":
    fit_eraser_fraction()
