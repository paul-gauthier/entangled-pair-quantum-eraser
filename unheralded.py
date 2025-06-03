#!/usr/bin/env python


from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, least_squares

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
# Analysis to extract the quantum-eraser mixture fractions (e_off, e_on)
# using the four traces {Ni_off, Nc_off, Ni_on, Nc_on}.
# ---------------------------------------------------------------------------
import numpy as np

# ------------------------------------------------------------------ #
# Re-use the simple TSV parser from plots.py (copied here for parity)
# ------------------------------------------------------------------ #
def _parse_counts(tsv: str):
    """Return (Ns, Ni, Nc) arrays parsed from a TSV block with a header row."""
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    counts = np.asarray(rows, dtype=float)
    return counts[:, 0], counts[:, 1], counts[:, 2]


# ------------------------------------------------------------------ #
# Cosine helpers                                                     #
# ------------------------------------------------------------------ #
def _one_cos(delta: np.ndarray, C: float, A: float, phi: float) -> np.ndarray:
    """C + A·½(1+cos(δ+φ))."""
    return C + A * (1.0 + np.cos(delta + phi)) / 2.0


def _ni_total(delta, e, C1, A1, phi1, C2, A2, phi2):
    return (1.0 - e) * _one_cos(delta, C1, A1, phi1) + e * _one_cos(
        delta, C2, A2, phi2
    )


def _nc_total(delta, e, D1, B1, psi1, D2, B2, psi2):
    return (1.0 - e) * _one_cos(delta, D1, B1, psi1) + e * _one_cos(
        delta, D2, B2, psi2
    )


# ------------------------------------------------------------------ #
# Execute the fit when run as a script                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    # Prepare data
    Ns_off, Ni_off, Nc_off = _parse_counts(eraser_off)
    Ns_on,  Ni_on,  Nc_on  = _parse_counts(eraser_on)
    delta = delta_from_steps(piezo_steps)

    # Residuals (weighted by Poisson σ ≈ √N)
    def _residual(p):
        (
            C1, A1, phi1,
            C2, A2, phi2,
            D1, B1, psi1,
            D2, B2, psi2,
            e_off, e_on,
        ) = p

        r = np.concatenate(
            [
                (Ni_off - _ni_total(delta, e_off, C1, A1, phi1, C2, A2, phi2))
                / np.sqrt(Ni_off),
                (Ni_on  - _ni_total(delta, e_on,  C1, A1, phi1, C2, A2, phi2))
                / np.sqrt(Ni_on),
                (Nc_off - _nc_total(delta, e_off, D1, B1, psi1, D2, B2, psi2))
                / np.sqrt(Nc_off),
                (Nc_on  - _nc_total(delta, e_on,  D1, B1, psi1, D2, B2, psi2))
                / np.sqrt(Nc_on),
            ]
        )
        return r

    # Initial guesses
    p0 = [
        np.mean(Ni_off), np.ptp(Ni_off) / 2, 0.0,           # C1 A1 φ1
        np.mean(Ni_off), np.ptp(Ni_off) / 2, np.pi,         # C2 A2 φ2
        np.mean(Nc_off), np.ptp(Nc_off) / 2, 0.0,           # D1 B1 ψ1
        np.mean(Nc_off), np.ptp(Nc_off) / 2, np.pi,         # D2 B2 ψ2
        0.25, 0.75,                                         # e_off e_on
    ]

    # Bounds (e ∈ [0,1])
    lower = [-np.inf] * 12 + [0.0, 0.0]
    upper = [ np.inf] * 12 + [1.0, 1.0]

    # Fit
    res = least_squares(_residual, p0, bounds=(lower, upper), method="trf")

    (
        C1, A1, phi1,
        C2, A2, phi2,
        D1, B1, psi1,
        D2, B2, psi2,
        e_off_fit, e_on_fit,
    ) = res.x

    # Report
    print("\n=== Quantum-eraser mixture analysis ===")
    print(f"Erasure fraction OFF (e_off): {e_off_fit:.4f}")
    print(f"Erasure fraction ON  (e_on) : {e_on_fit:.4f}\n")

    deg = np.degrees
    print("Idler classes:")
    print(f"  class-1  C1={C1:.1f}, A1={A1:.1f}, φ1={phi1:.2f} rad ({deg(phi1):.1f}°)")
    print(f"  class-2  C2={C2:.1f}, A2={A2:.1f}, φ2={phi2:.2f} rad ({deg(phi2):.1f}°)")
    print("Coincidence classes:")
    print(f"  class-1  D1={D1:.1f}, B1={B1:.1f}, ψ1={psi1:.2f} rad ({deg(psi1):.1f}°)")
    print(f"  class-2  D2={D2:.1f}, B2={B2:.1f}, ψ2={psi2:.2f} rad ({deg(psi2):.1f}°)")
    dof = len(_residual(p0)) - res.x.size
    print(f"Reduced χ² = {res.cost / dof:.2f}")
