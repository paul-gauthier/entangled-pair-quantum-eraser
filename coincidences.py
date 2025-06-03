#!/usr/bin/env python

import numpy as np
from plot_utils import plot_coincidence_counts_only
import matplotlib.pyplot as plt
data = """
Piezo motor position   Signal blocked   Eraser on      Eraser off
0	1018	24	21
2	1044	15	19
4	1037	11	20
6	1052	8	19
8	1088	8	19
10	1088	11	15
12	1083	11	18
14	1074	14	15
16	1129	21	14
18	1066	23	19
20	1036	21	19
22	1049	21	19
24	1058	18	18
26	1055	13	20
28	1086	8	20
30	1050	4	17
32	1096	8	16
34	1098	12	17
36	1107	16	17
38	1087	22	15
40	1071	22	22
42	1015	22	18
44	1028	24	20
46	1047	19	22
"""

if __name__ == "__main__":
    # Parse the data
    lines = data.strip().splitlines()
    header = lines[0].split('\t')
    data_rows = []
    for line in lines[1:]:
        if line.strip():
            data_rows.append([float(x) for x in line.split('\t')])

    parsed_data = np.array(data_rows)

    piezo_steps = parsed_data[:, 0] # Already step values
    Nc_signal_blocked = parsed_data[:, 1]
    Nc_eraser_on = parsed_data[:, 2]
    Nc_eraser_off = parsed_data[:, 3]

    # Plot for "Signal blocked"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_signal_blocked,
        output_filename="coincidence_counts_signal_blocked.pdf",
        label_suffix="Idler counts, signal-blocked"
    )

    # Plot for "Eraser on"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_eraser_on,
        output_filename="coincidence_counts_eraser_on.pdf",
        label_suffix="Coincidence counts, eraser-on"
    )

    # Plot for "Eraser off"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_eraser_off,
        output_filename="coincidence_counts_eraser_off.pdf",
        label_suffix="Coincidence counts, eraser-off"
    )
