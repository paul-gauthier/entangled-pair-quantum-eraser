import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Comparing Signal and Idler
data = """
Angle	Signal	Idler
-45	4815	782
-22.5	1609	427
0	266	720
22.5	1761	1393
45	4879	1918
67.5	8716	1938
90	10049	1582
112.5	8428	1046
135	5127	621
"""

# Both runs of the idler (3sec acq, 8sec acq)
data = """
Angle	Idler	Idler
-45	805	782
-22.5	411	427
0	715	720
22.5	1263	1393
45	1939	1918
67.5	2020	1938
90	1668	1582
112.5	1110	1046
135	600	621
"""

# idlers with upper mzi arm blocked
data="""
Angler Idler90 Idler0
-45	782	1990
-22.5	427	2244
0	720	1836
22.5	1393	1190
45	1918	605
67.5	1938	406
90	1582	596
112.5	1046	943
135	621	1236
"""

# signal; idlers with lower mzi arm blocked
data="""
Angle	Signal	Idler
-45	4815	1645
-22.5	1609	1057
0	266	252
22.5	1761	756
45	4879	1361
67.5	8716	1860
90	10049	1887
112.5	8428	1534
135	5127	1057
"""

# idlers: upper arm blocked, lower arm blocked (after Sahil tweaked fixed MZI HWP)
data="""
Angle	Idler	Idler
-45	1645	1653
-22.5	1057	803
0	252	508
22.5	756	826
45	1361	1457
67.5	1860	1946
90	1887	1963
112.5	1534	1607
135	1057	1114
"""

def parse_data(data_str):
    """Parse the tab-separated data string."""
    lines = data_str.strip().split('\n')[1:]  # Skip header
    angles, signals, idlers = [], [], []

    for line in lines:
        angle, signal, idler = line.split('\t')
        angles.append(float(angle))
        signals.append(float(signal))
        idlers.append(float(idler))

    return np.array(angles), np.array(signals), np.array(idlers)

def cos_squared_model(theta, A, B, phi):
    """Cosine squared model: A * cos²(theta + phi) + B"""
    return A * np.cos(np.radians(theta + phi))**2 + B

def fit_and_plot():
    """Fit cosine squared curves to signal and idler data."""
    angles, signals, idlers = parse_data(data)

    # Convert angles to radians for fitting
    angles_rad = np.radians(angles)

    # Initial parameter guesses
    # For signal: amplitude, offset, phase shift
    signal_guess = [np.max(signals) - np.min(signals), np.min(signals), 0]
    idler_guess = [np.max(idlers) - np.min(idlers), np.min(idlers), 0]

    # Fit the curves
    signal_params, signal_cov = curve_fit(cos_squared_model, angles, signals, p0=signal_guess)
    idler_params, idler_cov = curve_fit(cos_squared_model, angles, idlers, p0=idler_guess)

    # Print fit parameters
    print("Signal fit parameters:")
    print(f"  Amplitude (A): {signal_params[0]:.1f} ± {np.sqrt(signal_cov[0,0]):.1f}")
    print(f"  Offset (B): {signal_params[1]:.1f} ± {np.sqrt(signal_cov[1,1]):.1f}")
    print(f"  Phase shift (φ): {signal_params[2]:.1f}° ± {np.sqrt(signal_cov[2,2]):.1f}°")
    print()

    print("Idler fit parameters:")
    print(f"  Amplitude (A): {idler_params[0]:.1f} ± {np.sqrt(idler_cov[0,0]):.1f}")
    print(f"  Offset (B): {idler_params[1]:.1f} ± {np.sqrt(idler_cov[1,1]):.1f}")
    print(f"  Phase shift (φ): {idler_params[2]:.1f}° ± {np.sqrt(idler_cov[2,2]):.1f}°")
    print()

    # Generate smooth curves for plotting
    angles_smooth = np.linspace(-45, 135, 200)
    signal_fit = cos_squared_model(angles_smooth, *signal_params)
    idler_fit = cos_squared_model(angles_smooth, *idler_params)

    # Plot the data and fits
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(angles, signals, color='blue', label='Signal LP', alpha=0.7)
    plt.plot(angles_smooth, signal_fit, 'b-', label=f'Fit: {signal_params[0]:.0f}cos²(θ+{signal_params[2]:.1f}°)+{signal_params[1]:.0f}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Signal counts')
    plt.title('Signal vs Linear Polarizer Angle')
    plt.xticks(np.arange(-45, 180, 45))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.scatter(angles, idlers, color='red', label='Idler LP', alpha=0.7)
    plt.plot(angles_smooth, idler_fit, 'r-', label=f'Fit: {idler_params[0]:.0f}cos²(θ+{idler_params[2]:.1f}°)+{idler_params[1]:.0f}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Idler counts')
    plt.title('Idler vs Linear Polarizer Angle')
    plt.xticks(np.arange(-45, 180, 45))
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('polarizer_alignment_fits.png')
    plt.show()

    return signal_params, idler_params

if __name__ == "__main__":
    fit_and_plot()
