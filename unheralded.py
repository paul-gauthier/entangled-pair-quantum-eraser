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
# Fitting model to the “eraser-on” and “eraser-off” data sets
# ---------------------------------------------------------------------------

def _parse_counts(tsv: str):
    """Return (Ns, Ni, Nc) arrays parsed from a TSV block with a header row."""
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


def _counts_model(R, f, e, delta, phi):
    """Return (N_i, N_c) for given parameters."""
    Ni = R / 2 * (1 + (1 - f) * e * np.cos(delta + phi))
    Nc = R / 2 * (1 - f) * (1 + e * np.cos(delta + phi))
    return Ni, Nc


def _residuals(p, delta, Ni_on, Nc_on, Ni_off, Nc_off):
    """Stacked residuals weighted by √N (Poisson)."""
    R, f, phi, e_on, e_off = p
    Ni_pred_on, Nc_pred_on = _counts_model(R, f, e_on, delta, phi)
    Ni_pred_off, Nc_pred_off = _counts_model(R, f, e_off, delta, phi)

    # √N weights – avoid divide-by-zero
    w = lambda x: np.sqrt(np.maximum(x, 1.0))

    res = np.concatenate(
        [
            (Ni_on - Ni_pred_on) / w(Ni_on),
            (Nc_on - Nc_pred_on) / w(Nc_on),
            (Ni_off - Ni_pred_off) / w(Ni_off),
            (Nc_off - Nc_pred_off) / w(Nc_off),
        ]
    )
    return res


def main(*, show: bool = True) -> str:
    # Parse data -------------------------------------------------------------
    Ns_on, Ni_on, Nc_on = _parse_counts(eraser_on)
    Ns_off, Ni_off, Nc_off = _parse_counts(eraser_off)
    delta = delta_from_steps(piezo_steps)

    # Fit -------------------------------------------------------------------
    R0 = 2 * Ni_on.mean()
    f0 = max(0.0, 1 - Nc_on.mean() / Ni_on.mean())
    p0 = np.array([R0, f0, 0.0, 0.8, 0.1])  # R, f, φ, e_on, e_off

    bounds = ([0, 0, -np.pi, 0, 0], [np.inf, 1, np.pi, 1, 1])
    res = least_squares(
        _residuals,
        p0,
        bounds=bounds,
        args=(delta, Ni_on, Nc_on, Ni_off, Nc_off),
    )
    R_fit, f_fit, phi_fit, e_on_fit, e_off_fit = res.x

    print("----- Joint fit results -----")
    print(f"R      = {R_fit:.2f} counts/s")
    print(f"f      = {f_fit:.3f}")
    print(f"φ      = {phi_fit:.2f} rad ({np.degrees(phi_fit):.1f}°)")
    print(f"e_on   = {e_on_fit:.3f}")
    print(f"e_off  = {e_off_fit:.3f}")
    print("------------------------------\n")

    # Predictions -----------------------------------------------------------
    delta_fine = np.linspace(delta.min(), delta.max(), 500)
    Ni_pred_on,  Nc_pred_on  = _counts_model(R_fit, f_fit, e_on_fit,  delta_fine, phi_fit)
    Ni_pred_off, Nc_pred_off = _counts_model(R_fit, f_fit, e_off_fit, delta_fine, phi_fit)

    # Errors
    Ni_err_on,  Nc_err_on  = np.sqrt(Ni_on),  np.sqrt(Nc_on)
    Ni_err_off, Nc_err_off = np.sqrt(Ni_off), np.sqrt(Nc_off)

    # Plot ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, sharex="all", figsize=(12, 8))
    (ax_Ni_on, ax_Ni_off), (ax_Nc_on, ax_Nc_off) = axes
    plt.rcParams.update({"font.size": 14})

    # Idler ON
    ax_Ni_on.errorbar(delta, Ni_on, yerr=Ni_err_on, fmt="s", color="tab:blue",
                      label="Data", capsize=3)
    ax_Ni_on.plot(delta_fine, Ni_pred_on, "--", color="tab:blue", label="Fit")
    ax_Ni_on.set_title("Idler – eraser ON")
    ax_Ni_on.set_ylabel("Counts / s")
    ax_Ni_on.grid(True, linestyle=":", alpha=0.7)
    ax_Ni_on.legend()

    # Idler OFF
    ax_Ni_off.errorbar(delta, Ni_off, yerr=Ni_err_off, fmt="s", color="tab:green",
                       label="Data", capsize=3)
    ax_Ni_off.plot(delta_fine, Ni_pred_off, "--", color="tab:green", label="Fit")
    ax_Ni_off.set_title("Idler – eraser OFF")
    ax_Ni_off.grid(True, linestyle=":", alpha=0.7)
    ax_Ni_off.legend()

    # Coincidence ON
    ax_Nc_on.errorbar(delta, Nc_on, yerr=Nc_err_on, fmt="x", color="tab:red",
                      label="Data", capsize=3)
    ax_Nc_on.plot(delta_fine, Nc_pred_on, "--", color="tab:red", label="Fit")
    ax_Nc_on.set_title("Coincidences – eraser ON")
    ax_Nc_on.set_xlabel("Phase delay δ (rad)")
    ax_Nc_on.set_ylabel("Counts / s")
    ax_Nc_on.grid(True, linestyle=":", alpha=0.7)
    ax_Nc_on.legend()

    # Coincidence OFF
    ax_Nc_off.errorbar(delta, Nc_off, yerr=Nc_err_off, fmt="x", color="tab:purple",
                       label="Data", capsize=3)
    ax_Nc_off.plot(delta_fine, Nc_pred_off, "--", color="tab:purple", label="Fit")
    ax_Nc_off.set_title("Coincidences – eraser OFF")
    ax_Nc_off.set_xlabel("Phase delay δ (rad)")
    ax_Nc_off.grid(True, linestyle=":", alpha=0.7)
    ax_Nc_off.legend()

    # Secondary x-axis (nm) on bottom row
    for ax in axes[-1]:
        ax_nm = ax.secondary_xaxis("top", functions=(delta_to_nm, nm_to_delta))
        ax_nm.set_xlabel("Piezo displacement (nm)")

    fig.tight_layout()
    out_pdf = "unheralded_fit.pdf"
    plt.savefig(out_pdf)
    print(f"Plot saved as {out_pdf}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_pdf


if __name__ == "__main__":
    main(show=True)
