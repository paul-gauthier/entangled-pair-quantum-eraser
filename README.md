# Quantum Eraser Lab 6 – Simulation & Analysis

This repository contains a lightweight **SymPy / NumPy** toolkit that models the
Mach–Zehnder–based quantum-eraser experiment performed in Lab 6 and produces the
figures for the accompanying hand-out.

Main components
---------------

| File | Role |
|------|------|
| `lab6entangled.py` | Symbolic description of the interferometer and helper functions to compute coincidence probabilities & visibility for single- and two-photon states. |
| `plot_heatmap.py`  | Draws visibility heat-maps versus mis-alignment of the three experimental angles (signal LP<sub>s</sub>, idler LP<sub>i</sub>, MZI HWP). |
| `plot_utils.py`    | Unit-conversion helpers (piezo steps ⇄ nm ⇄ phase delay) and a high-level plotting routine for raw counts. |
| `plots.py` | Parses TSV count logs and creates publication-quality plots. |
| `lab-6-entangled.tex` | LaTeX report that explains the theory, derivations and experimental procedure. |

Quick start
-----------

```bash
# 1. Set-up virtual environment & install deps
python3 -m venv venv && source venv/bin/activate
pip install sympy numpy matplotlib latexmk

# 2. Run the symbolic model
python lab6entangled.py

# 3. Generate 3 × 3 visibility heat-maps
python plot_heatmap.py
open visibility_heatmaps_combined.pdf   # macOS preview
```

Compile the hand-out (requires a LaTeX distribution with `pgfplots`):

```bash
latexmk -pdf lab-6-entangled.tex
```

License
-------

MIT License – have fun exploring quantum optics!
