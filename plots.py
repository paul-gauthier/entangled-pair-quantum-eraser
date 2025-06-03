#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

# Data from lab-5.tex table
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
# Parse the tab-separated tables into NumPy arrays
def _parse_counts(tsv: str):
    """Return (Ns, Ni, Nc) arrays parsed from a TSV block with a header row."""
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip header
        if line.strip()
    ]
    counts = np.asarray(rows, dtype=float)
    return counts[:, 0], counts[:, 1], counts[:, 2]


# Pre-compute arrays for both configurations
Ns_on, Ni_on, Nc_on = _parse_counts(eraser_on)
Ns_off, Ni_off, Nc_off = _parse_counts(eraser_off)

# ---------------------------------------------------------------------------
# Delegate all plotting to the reusable utility in `plot_utils.py`
from plot_utils import plot_counts

if __name__ == "__main__":
    # Plot with quantum-eraser ON
    plot_counts(
        piezo_steps,
        Ns_on,
        Ni_on,
        Nc_on,
        output_filename="counts_vs_phase_delay_eraser_on.pdf",
        label_suffix=r"eraser-on",
    )

    # Plot with quantum-eraser OFF
    plot_counts(
        piezo_steps,
        Ns_off,
        Ni_off,
        Nc_off,
        output_filename="counts_vs_phase_delay_eraser_off.pdf",
        label_suffix=r"eraser-off",
    )
