#!/usr/bin/env python

import numpy as np
from scipy.optimize import curve_fit
from plot_utils import delta_from_steps, _cos_model


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
# Analysis utilities and main routine
# ---------------------------------------------------------------------------
def _parse_counts(tsv: str):
    """
    Return (Ns, Ni, Nc) arrays parsed from a TSV block with a header row.
    """
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    counts = np.asarray(rows, dtype=float)
    return counts[:, 0], counts[:, 1], counts[:, 2]


def _parse_single(tsv: str):
    """
    Return a NumPy array of single-column counts parsed from a TSV block.
    """
    rows = [
        float(line.strip())
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    return np.asarray(rows, dtype=float)


def _fit_visibility(piezo_steps: np.ndarray, counts: np.ndarray):
    """
    Fit counts versus phase delay with a cosine model and return
    (A_fit, C0_fit, phi_fit, visibility).
    """
    delta = delta_from_steps(piezo_steps)
    errs = np.sqrt(counts)
    p0 = [np.ptp(counts), np.min(counts), 0.0]
    A_fit, C0_fit, phi_fit = curve_fit(
        _cos_model,
        delta,
        counts,
        p0=p0,
        sigma=errs,
        absolute_sigma=True,
    )[0]
    if A_fit < 0:  # enforce non-negative modulation depth
        A_fit = -A_fit
        phi_fit += np.pi
        C0_fit -= A_fit
    phi_fit = (phi_fit + np.pi) % (2 * np.pi) - np.pi
    visibility = A_fit / (A_fit + 2 * C0_fit) if (A_fit + 2 * C0_fit) else 0
    return A_fit, C0_fit, phi_fit, visibility


if __name__ == "__main__":
    # Optional: parse calibration runs (not used in calculation but handy to inspect)
    Ns_on, Ni_on, Nc_on = _parse_counts(eraser_on)
    Ns_off, Ni_off, Nc_off = _parse_counts(eraser_off)

    # Parse and analyse the novel single-idler scan
    Ni_novel = _parse_single(novel)
    _, _, _, vis_novel = _fit_visibility(piezo_steps, Ni_novel)
    percent_erased = vis_novel * 100

    print("\n=== Novel single-idler scan analysis ===")
    print(f"Visibility (from N_i) : {vis_novel:.3f}")
    print(f"Estimated idlers with erased which-way info ≈ {percent_erased:.1f}%")
