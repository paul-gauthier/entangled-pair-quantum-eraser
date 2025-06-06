
# Entangled pair quantum eraser

This repository contains a lightweight SymPy toolkit that models a
Mach–Zehnder–based entangled pair quantum-eraser experiment.

![fig.png](fig.png)

## Docs

- [Description of the experiment, relevant theory](render/lab-6-entangled.pdf) 
- [Results](render/2025-06-02-visibility.pdf) that show erasing which-way information from the signal photon (by changing the signal linear polarizer angle) controls self-interference of the idler in the MZI.

## SymPy model of experiment as quantum operator

- [Symbolic model of the apparatus](lab6entangled.py) as a quantum operator.
- Includes a function to apply the operator to an initial state vector (|Phi+>, |VV>, etc) with given angle settings (signal LP<sub>s</sub>, idler LP<sub>i</sub>, MZI HWP).
- Computes the symbolic formula for expected coincidence counts versus phase delay, and expected visibility.

## Utilities

- [plot_heatmap.py](plot_heatmap.py) 
  - Draws visibility heat-maps to visualize the sensitivity of
    interference visibility to misalignments of the three experimental
    angles (signal LP<sub>s</sub>, idler LP<sub>i</sub>, MZI HWP).
  - Uses the symbolic model of the experiment, computes visibility from the formula for expected coincidence counts versus phase delay.
  - Plots heatmaps showing variation in visibility as function of angle error.
  - See [visibility_heatmaps_combined.pdf](visibility_heatmaps_combined.pdf)
- [plot_utils.py](plot_utils.py)   
  - Utilities for plotting signal, idler and coincidence counts from experimental data, 
  fitting counts to best-fit curve, computing visibility.
- [plots.py](plots.py) 
  - Parses experimental data and creates plots for signal, idler, coincidence data.
- [coincidences.py](coincidences.py)
  - Parses experimental data and creates plots for coincidence data only.


