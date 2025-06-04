#!/usr/bin/env python


from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, least_squares

from plot_utils import delta_from_steps, _cos_model
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
# Learn a visibility model from the labelled “eraser-on/off” runs and
# use it to predict the coincidence-curve visibility of an un-labelled
# idler-singles data set contained in `novel`.
# ---------------------------------------------------------------------------

from typing import Tuple

# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def _parse_Ni(tsv: str) -> np.ndarray:
    """
    Return the idler singles (N_i) column from a TSV block.
    The block may contain either one numeric column (novel run)
    or three numeric columns (N_s, N_i, N_c).
    """
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    return np.asarray([r[1] if len(r) >= 2 else r[0] for r in rows], dtype=float)


def _parse_Nc(tsv: str) -> np.ndarray:
    """Return coincidence counts (N_c) from a 3-column TSV block."""
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    return np.asarray([r[2] for r in rows], dtype=float)


def _fit_vis_and_phase(counts: np.ndarray) -> Tuple[float, float]:
    """
    Fit `counts` with the cosine model and return
    (visibility, fitted phase φ wrapped into (−π, π]).
    """
    delta = delta_from_steps(piezo_steps)
    p0 = [np.ptp(counts), np.min(counts), 0.0]
    A, C0, phi = curve_fit(
        _cos_model,
        delta,
        counts,
        p0=p0,
        sigma=np.sqrt(counts),
        absolute_sigma=True,
    )[0]
    # enforce positive modulation depth
    if A < 0:
        A, phi, C0 = -A, phi + np.pi, C0 - A
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    visibility = A / (A + 2 * C0) if (A + 2 * C0) != 0 else 0.0
    return visibility, phi


# ------------------------------------------------------------------ #
# Training data (“eraser on/off”)                                    #
# ------------------------------------------------------------------ #
Ni_on = _parse_Ni(eraser_on)
Ni_off = _parse_Ni(eraser_off)
Nc_on = _parse_Nc(eraser_on)
Nc_off = _parse_Nc(eraser_off)

Vi_on, phi_on = _fit_vis_and_phase(Ni_on)
Vi_off, phi_off = _fit_vis_and_phase(Ni_off)
Vc_on, _ = _fit_vis_and_phase(Nc_on)
Vc_off, _ = _fit_vis_and_phase(Nc_off)

# Decision boundary and target visibilities
_phi_thresh = (phi_on + phi_off) / 2
_high_vis = Vc_on if phi_on > phi_off else Vc_off
_low_vis = Vc_off if phi_on > phi_off else Vc_on


def predict_Vc_from_Ni(Ni: np.ndarray) -> float:
    """
    Predict the visibility of the coincidence curve that would accompany
    an idler-singles trace `Ni`.
    """
    _, phi = _fit_vis_and_phase(Ni)
    return _high_vis if phi > _phi_thresh else _low_vis


# ------------------------------------------------------------------ #
# Stand-alone usage                                                  #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    Ni_novel = _parse_Ni(novel)
    Vc_pred = predict_Vc_from_Ni(Ni_novel)
    print(f"Predicted coincidence visibility for the novel run: {Vc_pred:.3f}")
