#!/usr/bin/env python

import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from plot_utils import delta_from_steps, delta_to_nm, nm_to_delta


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

novel = """
N_i
1018
1044
1037
1052
1088
1088
1083
1074
1129
1066
1036
1049
1058
1055
1086
1050
1096
1098
1107
1087
1071
1015
1028
1047
"""

# ---------------------------------------------------------------------------
# Joint fit of the “orphan + eraser” model to the two data sets
# ---------------------------------------------------------------------------
import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from plot_utils import delta_from_steps, delta_to_nm, nm_to_delta


def _parse_counts(tsv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parse a tab-separated block with header  ``N_s  N_i  N_c``.
    Returns (Ns, Ni, Nc) as float arrays.
    """
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ---------------------------------------------------------------------------
# Model for a single scan
# ---------------------------------------------------------------------------
def _Ni(delta, R, f, phi, e):
    return R * (
        0.5 * f
        + 0.5 * (1 - f) * (1 - e)
        + 0.5 * (1 - f) * e * (1 + np.cos(delta + phi))
    )


def _Nc(delta, R, f, phi, phi_c, e):
    return R * (1 - f) * (
        0.5 * (1 - e) + 0.5 * e * (1 + np.cos(delta + phi + phi_c))
    )


def _model(params, delta):
    R, f, phi, phi_c, e_on, e_off = params
    Ni_on_pred = _Ni(delta, R, f, phi, e_on)
    Nc_on_pred = _Nc(delta, R, f, phi, phi_c, e_on)
    Ni_off_pred = _Ni(delta, R, f, phi, e_off)
    Nc_off_pred = _Nc(delta, R, f, phi, phi_c, e_off)
    return np.concatenate([Ni_on_pred, Nc_on_pred, Ni_off_pred, Nc_off_pred])


# ---------------------------------------------------------------------------
# Fit and plot
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Convert piezo steps → phase delay
    delta = delta_from_steps(piezo_steps)

    # Parse count tables
    _, Ni_on, Nc_on = _parse_counts(eraser_on)
    _, Ni_off, Nc_off = _parse_counts(eraser_off)

    # Concatenate data and Poisson σ
    y_data = np.concatenate([Ni_on, Nc_on, Ni_off, Nc_off])
    sigma = np.sqrt(y_data)

    # Residuals for least squares
    def residuals(p):
        return (_model(p, delta) - y_data) / sigma

    # Initial guesses and bounds
    p0 = [1100.0, 0.2, 0.0, 0.0, 0.9, 0.1]
    bounds = ([0.0, 0.0, -math.pi, -math.pi, 0.0, 0.0],
              [np.inf, 1.0,  math.pi,  math.pi, 1.0, 1.0])

    result = least_squares(residuals, p0, bounds=bounds, xtol=1e-12)
    R, f, phi, phi_c, e_on, e_off = result.x
    chi2 = 2 * result.cost
    ndf = y_data.size - len(p0)

    print("\nBest-fit parameters")
    print("-------------------")
    print(f"R         = {R:.2f} counts/s")
    print(f"f         = {f:.4f}")
    print(f"phi       = {np.degrees(phi):.1f}°")
    print(f"phi_c     = {np.degrees(phi_c):.1f}°")
    print(f"e_on      = {e_on:.3f}")
    print(f"e_off     = {e_off:.3f}")
    print(f"chi²/ndf  = {chi2/ndf:.2f}")

    # Fine grid for smooth curves
    delta_fine = np.linspace(delta.min(), delta.max(), 500)
    Ni_on_fit = _Ni(delta_fine, R, f, phi, e_on)
    Nc_on_fit = _Nc(delta_fine, R, f, phi, phi_c, e_on)
    Ni_off_fit = _Ni(delta_fine, R, f, phi, e_off)
    Nc_off_fit = _Nc(delta_fine, R, f, phi, phi_c, e_off)

    # ----------------------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------------------
    plt.rcParams.update({"font.size": 14})
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # --- Idler counts ------------------------------------------------------
    ax1.errorbar(
        delta,
        Ni_on,
        yerr=np.sqrt(Ni_on),
        fmt="s",
        color="tab:green",
        label="Nᵢ (eraser-on)",
        capsize=3,
    )
    ax1.errorbar(
        delta,
        Ni_off,
        yerr=np.sqrt(Ni_off),
        fmt="o",
        color="limegreen",
        label="Nᵢ (eraser-off)",
        capsize=3,
    )
    ax1.plot(delta_fine, Ni_on_fit, color="tab:green", ls="--")
    ax1.plot(delta_fine, Ni_off_fit, color="limegreen", ls="--")
    ax1.set_ylabel("Counts/sec")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.7)

    # --- Coincidence counts ------------------------------------------------
    ax2.errorbar(
        delta,
        Nc_on,
        yerr=np.sqrt(Nc_on),
        fmt="x",
        color="tab:red",
        label="N_c (eraser-on)",
        capsize=3,
    )
    ax2.errorbar(
        delta,
        Nc_off,
        yerr=np.sqrt(Nc_off),
        fmt="+",
        color="orange",
        label="N_c (eraser-off)",
        capsize=3,
    )
    ax2.plot(delta_fine, Nc_on_fit, color="tab:red", ls="--")
    ax2.plot(delta_fine, Nc_off_fit, color="orange", ls="--")
    ax2.set_ylabel("Counts/sec")
    ax2.set_xlabel(r"Phase Delay $\delta$ (rad)")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.7)

    # Nice π ticks
    xticks = np.arange(0, delta.max() + np.pi / 2, np.pi)
    xticklabels = ["0"] + [f"{i}$\\pi$" for i in range(1, len(xticks))]
    xticklabels = [s.replace("1$\\pi$", "$\\pi$") for s in xticklabels]
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xticklabels)

    # Secondary axis in nm
    ax_nm = ax2.secondary_xaxis("top", functions=(delta_to_nm, nm_to_delta))
    ax_nm.set_xlabel("Piezo Displacement (nm)", fontsize=12)

    fig.suptitle("Unheralded idler & coincidence counts with global fit", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("unheralded_fit.pdf")
    plt.close(fig)
    print("Plot saved as unheralded_fit.pdf")
