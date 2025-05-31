import numpy as np
from plot_utils import plot_coincidence_counts_only
import matplotlib.pyplot as plt
data = """
Piezo motor position	Eraser off	Eraser on
0	21	22
2	16	16
4	23	16
6	20	16
8	22	23
10	20	28
12	22	34
14	22	40
16	20	42
18	18	33
20	20	25
22	15	24
24	19	16
26	20	12
28	20	16
30	20	22
32	20	33
34	19	38
36	18	45
38	22	40
40	19	37
42	21	25
44	20	21
46	20	17
48	19	16
50	19	18
52	22	26
54	20	34
56	22	40
58	25	42
60	20	37
62	25	29
64	21	21
66	18	16
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
    Nc_off = parsed_data[:, 1]
    Nc_on = parsed_data[:, 2]

    # Plot for "Eraser off"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_off,
        output_filename="coincidence_counts_eraser_off.pdf",
        label_suffix="eraser-off"
    )
    # Plot for "Eraser off"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_off,
        output_filename="coincidence_counts_eraser_off.png",
        label_suffix="eraser-off"
    )

    # Plot for "Eraser on"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_on,
        output_filename="coincidence_counts_eraser_on.pdf",
        label_suffix="eraser-on"
    )
    # Plot for "Eraser on"
    plot_coincidence_counts_only(
        piezo_steps,
        Nc_on,
        output_filename="coincidence_counts_eraser_on.png",
        label_suffix="eraser-on"
    )
