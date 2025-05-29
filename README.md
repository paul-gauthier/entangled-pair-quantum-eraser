
# Entangled pair quantum eraser

This repository contains a lightweight SymPy toolkit that models a
Mach–Zehnder–based entangled pair quantum-eraser experiment.

- `lab6entangled.py` 
  - Symbolic model of the apparatus as a quantum operator.
  - Function to apply the operator to an initial state vector (|Phi+>, |VV>, etc) and obtain the symbolic formula for expected coincidence counts versus phase delay.
- `plot_heatmap.py` 
  - Draws visibility heat-maps to visualize the sensitivity of
    interference visibility to misalignments of the three experimental
    angles (signal LP<sub>s</sub>, idler LP<sub>i</sub>, MZI HWP).
  - Uses the symbolic model of the experiment, computes visibility from the formula for expected coincidence counts versus phase delay.
  - Plots heatmaps showing variation in visibility as function of angle error.
  - See `visibility_heatmaps_combined.pdf`
- `plot_utils.py`   
  - Utilities for plotting signal, idler and coincidence counts from experimental data, 
  fitting counts to best-fit curve, computing visibility.
- `plots.py` 
  - Parses experimental data and creates plots. 
- `lab-6-entangled.tex` and `lab-6-entangled.pdf` 
  - Report that explains the theory, derivations and experimental procedure.

![fig.png](fig.png)

