import numpy as np
from plot_utils import plot_coincidence_counts_only
import matplotlib.pyplot as plt
data = """
Piezo motor position   Signal blocked   Eraser on      Eraser off
0	1296	24	21
2	1322	15	19
4	1315	11	20
6	1330	8	19
8	1366	8	19
10	1366	11	15
12	1361	11	18
14	1352	14	15
16	1407	21	14
18	1344	23	19
20	1314	21	19
22	1327	21	19
24	1336	18	18
26	1333	13	20
28	1364	8	20
30	1328	4	17
32	1374	8	16
34	1376	12	17
36	1385	16	17
38	1365	22	15
40	1349	22	22
42	1293	22	18
44	1306	24	20
46	1325	19	22
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
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_signal_blocked,
        output_filename="coincidence_counts_signal_blocked.png",
        label_suffix="Idler counts, signal-blocked"
    )

    # Plot for "Eraser on"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_eraser_on,
        output_filename="coincidence_counts_eraser_on.pdf",
        label_suffix="Coincidence counts, eraser-on"
    )
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_eraser_on,
        output_filename="coincidence_counts_eraser_on.png",
        label_suffix="Coincidence counts, eraser-on"
    )

    # Plot for "Eraser off"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_eraser_off,
        output_filename="coincidence_counts_eraser_off.pdf",
        label_suffix="Coincidence counts, eraser-off"
    )
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_eraser_off,
        output_filename="coincidence_counts_eraser_off.png",
        label_suffix="Coincidence counts, eraser-off"
    )
