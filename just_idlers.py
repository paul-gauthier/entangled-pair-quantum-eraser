#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from readcsv import parse_photonics_csv
from plot_utils import delta_from_steps

def cos_model(delta, A, phi, C):
    """Model: C + A * cos(delta + phi)"""
    return C + A * np.cos(delta + phi)

def fit_ni_data(datasets):
    """Fit N_i data for each dataset and return fit parameters."""
    fit_results = []
    
    for dataset in datasets:
        if dataset.N_i is None or len(dataset.N_i) == 0:
            fit_results.append(None)
            continue
            
        # Convert piezo steps to phase
        delta = delta_from_steps(dataset.piezo_pos)
        
        # Initial guess
        y_mean = np.mean(dataset.N_i)
        y_amplitude = (np.max(dataset.N_i) - np.min(dataset.N_i)) / 2
        
        try:
            # Fit to model: C + A * cos(delta + phi)
            popt, pcov = curve_fit(
                cos_model, 
                delta, 
                dataset.N_i, 
                p0=[y_amplitude, 0, y_mean],  # [A, phi, C]
                maxfev=5000
            )
            
            # Calculate uncertainties
            perr = np.sqrt(np.diag(pcov))
            
            fit_results.append({
                'A': popt[0], 'A_err': perr[0],
                'phi': popt[1], 'phi_err': perr[1], 
                'C': popt[2], 'C_err': perr[2],
                'popt': popt,
                'pcov': pcov
            })
        except Exception as e:
            print(f"Fit failed for {dataset.name}: {e}")
            fit_results.append(None)
    
    return fit_results

def plot_ni_fits(datasets, fit_results):
    """Plot N_i data and fits for each dataset."""
    # Filter out datasets with no N_i data
    valid_datasets = [(ds, fit) for ds, fit in zip(datasets, fit_results) 
                     if ds.N_i is not None and fit is not None]
    
    if not valid_datasets:
        print("No valid N_i data to plot")
        return
    
    n_datasets = len(valid_datasets)
    fig, axes = plt.subplots(n_datasets, 1, figsize=(10, 4*n_datasets), sharex=True)
    
    if n_datasets == 1:
        axes = [axes]
    
    for i, (dataset, fit_result) in enumerate(valid_datasets):
        ax = axes[i]
        
        # Convert piezo steps to phase
        delta = delta_from_steps(dataset.piezo_pos)
        
        # Plot data
        ax.plot(delta, dataset.N_i, 'o', markersize=4, alpha=0.7, label='Data')
        
        # Plot fit
        delta_fine = np.linspace(delta.min(), delta.max(), 200)
        fit_curve = cos_model(delta_fine, *fit_result['popt'])
        ax.plot(delta_fine, fit_curve, '-', linewidth=2, label='Fit')
        
        # Format fit parameters for display
        A = fit_result['A']
        A_err = fit_result['A_err']
        phi = fit_result['phi']
        phi_err = fit_result['phi_err']
        C = fit_result['C']
        C_err = fit_result['C_err']
        
        # Add fit parameters to plot
        fit_text = (f'$N_i = {C:.1f} + {A:.1f} \\cos(\\delta + {phi:.2f})$\n'
                   f'$A = {A:.1f} \\pm {A_err:.1f}$\n'
                   f'$\\phi = {phi:.3f} \\pm {phi_err:.3f}$ rad\n'
                   f'$C = {C:.1f} \\pm {C_err:.1f}$')
        
        ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_ylabel('$N_i$ (counts)')
        ax.set_title(f'{dataset.name}')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Phase delay $\\delta$ (rad)')
    plt.tight_layout()
    plt.savefig('ni_fits.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Load the CSV data
    datasets = parse_photonics_csv('2025-06-02.csv')
    
    print(f"Loaded {len(datasets)} datasets:")
    for dataset in datasets:
        dataset.print()
    
    # Fit N_i data
    fit_results = fit_ni_data(datasets)
    
    # Print fit results
    print("Fit results:")
    for i, (dataset, fit_result) in enumerate(zip(datasets, fit_results)):
        if fit_result is not None:
            print(f"{dataset.name}:")
            print(f"  A = {fit_result['A']:.3f} ± {fit_result['A_err']:.3f}")
            print(f"  φ = {fit_result['phi']:.3f} ± {fit_result['phi_err']:.3f} rad")
            print(f"  C = {fit_result['C']:.3f} ± {fit_result['C_err']:.3f}")
            print()
    
    # Plot results
    plot_ni_fits(datasets, fit_results)

if __name__ == "__main__":
    main()
