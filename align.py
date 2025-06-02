import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
    plt.scatter(angles, signals, color='blue', label='Signal data', alpha=0.7)
    plt.plot(angles_smooth, signal_fit, 'b-', label=f'Fit: {signal_params[0]:.0f}cos²(θ+{signal_params[2]:.1f}°)+{signal_params[1]:.0f}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Signal counts')
    plt.title('Signal vs Linear Polarizer Angle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(angles, idlers, color='red', label='Idler data', alpha=0.7)
    plt.plot(angles_smooth, idler_fit, 'r-', label=f'Fit: {idler_params[0]:.0f}cos²(θ+{idler_params[2]:.1f}°)+{idler_params[1]:.0f}')
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Idler counts')
    plt.title('Idler vs Linear Polarizer Angle')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('polarizer_alignment_fits.pdf')
    plt.show()
    
    return signal_params, idler_params

if __name__ == "__main__":
    fit_and_plot()
