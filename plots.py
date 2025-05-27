import matplotlib.pyplot as plt
import numpy as np

# Data from lab-5.tex table
piezo_steps = np.array([
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58
])

Ns_counts = np.array([
    12412, 12400, 11925, 12277, 12462, 12386, 12413, 12081, 12457, 12200,
    12217, 12231, 12302, 11910, 12253, 12015, 12303, 12246, 12352, 12293,
    12406, 12402, 12402, 12411, 11874, 12345, 12144, 12340, 12223, 12484
])

Nc_counts = np.array([
    178, 188, 152, 123, 97, 69, 61, 68, 93, 119, 152, 163, 165, 142, 141,
    79, 66, 56, 58, 85, 137, 177, 182, 164, 142, 127, 84, 59, 54, 73
])

Ni_counts = np.array([
    6493, 6237, 5425, 5107, 4556, 4154, 4136, 4528, 5183, 5631, 6093, 6307,
    6154, 5356, 4975, 4172, 4019, 4111, 4599, 5257, 5911, 6307, 6317, 6039,
    5161, 4786, 4115, 3998, 4174, 4786
])

# Corrected data from lab-5.tex table
Ns_corrected_counts = np.array([
    10398, 10386, 9911, 10263, 10448, 10372, 10399, 10067, 10443, 10186,
    10203, 10217, 10288, 9896, 10239, 10001, 10289, 10232, 10338, 10279,
    10392, 10388, 10388, 10397, 9860, 10331, 10130, 10326, 10209, 10470
])

Ni_corrected_counts = np.array([
    4816, 4560, 3748, 3430, 2879, 2477, 2459, 2851, 3506, 3954,
    4416, 4630, 4477, 3679, 3298, 2495, 2342, 2434, 2922, 3580,
    4234, 4630, 4640, 4362, 3484, 3109, 2438, 2321, 2497, 3109
])

Nc_corrected_counts = np.array([
    176.0, 186.1, 150.4, 121.4, 95.6, 67.7, 59.7, 66.6, 91.4, 117.3,
    150.1, 161.1, 163.1, 140.4, 139.5, 77.7, 64.8, 54.7, 56.6, 83.4,
    135.2, 175.0, 180.0, 162.1, 140.5, 125.5, 82.8, 57.8, 52.7, 71.5
])

# Piezo steps are the same for corrected data
piezo_steps_corrected = piezo_steps

# Data from lab-5.tex table: piezo_counts_shutter_closed
piezo_steps_shutter_closed = np.array([
    0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58
])

Ns_shutter_closed_counts = np.array([
    10258, 10474, 9845, 10256, 9846, 9940, 10353, 10377, 9778, 10385,
    10450, 9932, 10138, 10366, 10336, 10131, 9869, 10301, 10309, 10303,
    10083, 10504, 10432, 10432, 10360, 10241, 10365, 10422, 10052, 10367
])

Ni_shutter_closed_counts = np.array([
    1175, 1243, 1133, 1268, 1110, 1203, 1219, 1274, 1161, 1236,
    1263, 1185, 1218, 1286, 1310, 1281, 1170, 1239, 1335, 1313,
    1262, 1358, 1280, 1258, 1192, 1185, 1210, 1196, 1170, 1274
])

Nc_shutter_closed_counts = np.array([
    50.1, 48.1, 59.2, 53.1, 52.2, 48.1, 56.1, 48.1, 49.2, 54.1,
    55.1, 46.1, 45.1, 50.1, 52.1, 57.1, 55.2, 49.1, 52.1, 55.1,
    56.1, 47.1, 49.1, 41.1, 43.1, 53.1, 49.1, 55.1, 51.1, 39.1
])


# ---------------------------------------------------------------------------
# Delegate all plotting to the reusable utility in `plot_utils.py`
from plot_utils import plot_counts

if __name__ == "__main__":
    plot_counts(
        piezo_steps,
        Ns_counts,
        Ni_counts,
        Nc_counts,
        output_filename="counts_vs_phase_delay_combined.pdf",
    )
    plot_counts(
        piezo_steps_corrected,
        Ns_corrected_counts,
        Ni_corrected_counts,
        Nc_corrected_counts,
        output_filename="counts_vs_phase_delay_corrected_combined.pdf",
        label_suffix=" , Corrected",
    )
    plot_counts(
        piezo_steps_shutter_closed,
        Ns_shutter_closed_counts,
        Ni_shutter_closed_counts,
        Nc_shutter_closed_counts,
        output_filename="counts_vs_phase_delay_shutter_closed_combined.pdf",
        label_suffix=" , Corrected",
    )
