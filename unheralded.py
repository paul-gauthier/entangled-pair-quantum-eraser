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
# Model fitting and plotting for orphan/eraser model
# ---------------------------------------------------------------------------
import math

# ---------------------------------------------------------------------------
# Helpers to parse the embedded TSV blocks
# ---------------------------------------------------------------------------
def _parse_counts(tsv: str):
    """Parse a TSV block and return NumPy arrays.

    • 3-column tables → (Ns, Ni, Nc)
    • 1-column tables → Ni
    """
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]        # skip header
        if line.strip()
    ]
    arr = np.asarray(rows, dtype=float)
    if arr.shape[1] == 3:
        return arr[:, 0], arr[:, 1], arr[:, 2]
    elif arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError("Unexpected column count")

# ---------------------------------------------------------------------------
# Prepare data
# ---------------------------------------------------------------------------
Ns_on, Ni_on, Nc_on = _parse_counts(eraser_on)
Ns_off, Ni_off, Nc_off = _parse_counts(eraser_off)
Ni_novel = _parse_counts(novel)

delta = delta_from_steps(piezo_steps)

# ---------------------------------------------------------------------------
# Models from orphans.py
# ---------------------------------------------------------------------------
def model_Ni(d, R, f, e, phi):
    return R * (
        0.5 * f
        + 0.5 * (1 - f) * (1 - e)
        + 0.5 * (1 - f) * e * (np.cos(d + phi) + 1)
    )


def model_Nc(d, R, f, e, phi, phi_c):
    return R * (1 - f) * (
        0.5 * (1 - e)
        + 0.5 * e * (np.cos(d + phi + phi_c) + 1)
    )

# ---------------------------------------------------------------------------
# Fit: eraser-on & eraser-off simultaneously  (R, f, phi_c global)
# ---------------------------------------------------------------------------
def _residuals(p):
    R, f, phi_c, e_on, phi_on, e_off, phi_off = p
    return np.concatenate(
        [
            model_Ni(delta, R, f, e_on,  phi_on) - Ni_on,
            model_Nc(delta, R, f, e_on,  phi_on, phi_c) - Nc_on,
            model_Ni(delta, R, f, e_off, phi_off) - Ni_off,
            model_Nc(delta, R, f, e_off, phi_off, phi_c) - Nc_off,
        ]
    )


p0 = [
    np.mean(Ni_on),       # R
    0.1,                  # f
    0.0,                  # phi_c
    0.9, 0.0,             # e_on,  phi_on
    0.1, 0.0,             # e_off, phi_off
]
lb = [0, 0, -2 * np.pi, 0, -2 * np.pi, 0, -2 * np.pi]
ub = [np.inf, 1,  2 * np.pi, 1,  2 * np.pi, 1,  2 * np.pi]

opt = least_squares(_residuals, p0, bounds=(lb, ub))
R_fit, f_fit, phi_c_fit, e_on_fit, phi_on_fit, e_off_fit, phi_off_fit = opt.x

# ---------------------------------------------------------------------------
# Fit: novel Ni only, using global R, f, phi_c
# ---------------------------------------------------------------------------
def _residuals_novel(p):
    e_n, phi_n = p
    return model_Ni(delta, R_fit, f_fit, e_n, phi_n) - Ni_novel


p0_novel = [0.5, 0.0]
lb_n, ub_n = [0, -2 * np.pi], [1, 2 * np.pi]
opt_novel = least_squares(_residuals_novel, p0_novel, bounds=(lb_n, ub_n))
e_novel_fit, phi_novel_fit = opt_novel.x

# ---------------------------------------------------------------------------
# Report results
# ---------------------------------------------------------------------------
deg = np.degrees
print("\nFitted parameters")
print("-----------------")
print(f"Global : R = {R_fit:.2f},  f = {f_fit:.3f},  phi_c = {phi_c_fit:.3f} rad ({deg(phi_c_fit):.1f}°)")
print(f"Eraser-ON : e = {e_on_fit:.3f},  phi = {phi_on_fit:.3f} rad ({deg(phi_on_fit):.1f}°)")
print(f"Eraser-OFF: e = {e_off_fit:.3f}, phi = {phi_off_fit:.3f} rad ({deg(phi_off_fit):.1f}°)")
print(f"Novel     : e = {e_novel_fit:.3f},  phi = {phi_novel_fit:.3f} rad ({deg(phi_novel_fit):.1f}°)")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _plot(name, Ni_obs, Nc_obs, e_fit, phi_fit, filename):
    d_fine = np.linspace(delta.min(), delta.max(), 500)

    n_rows = 2 if Nc_obs is not None else 1
    fig, axes = plt.subplots(n_rows, 1, sharex=True, figsize=(9, 6))
    axes = np.atleast_1d(axes)

    axes[0].errorbar(delta, Ni_obs, yerr=np.sqrt(Ni_obs),
                     fmt="o", label="data $N_i$")
    axes[0].plot(d_fine,
                 model_Ni(d_fine, R_fit, f_fit, e_fit, phi_fit),
                 label="fit")
    axes[0].set_ylabel("$N_i$ (counts/s)")
    axes[0].grid(True, ls=":")
    axes[0].legend()

    if Nc_obs is not None:
        axes[1].errorbar(delta, Nc_obs, yerr=np.sqrt(Nc_obs),
                         fmt="x", color="tab:red", label="data $N_c$")
        axes[1].plot(d_fine,
                     model_Nc(d_fine, R_fit, f_fit, e_fit, phi_fit, phi_c_fit),
                     color="tab:red", label="fit")
        axes[1].set_ylabel("$N_c$ (counts/s)")
        axes[1].grid(True, ls=":")
        axes[1].legend()

    axes[-1].set_xlabel(r"Phase delay $\delta$ (rad)")
    fig.suptitle(name)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


_plot("Eraser-ON",  Ni_on,  Nc_on,  e_on_fit,  phi_on_fit,  "unheralded_eraser_on.pdf")
_plot("Eraser-OFF", Ni_off, Nc_off, e_off_fit, phi_off_fit, "unheralded_eraser_off.pdf")
_plot("Novel (Ni only)", Ni_novel, None, e_novel_fit, phi_novel_fit, "unheralded_novel.pdf")
