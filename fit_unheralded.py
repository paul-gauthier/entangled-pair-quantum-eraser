#!/usr/bin/env python
"""
Global least-squares fit of the phenomenological model documented in ``orphans.py``
to the eraser-on / eraser-off data contained in ``unheralded.py``.

The model equations (singles and coincidences):

    N_i(δ) = ½ R · [1 + (1-f_before) · e_run · cos(δ + φ_run)]
    N_c(δ) = ½ R · g · [1 + e_run · cos(δ + φ_run + φ_c)]

where  g = (1-f_before)(1-f_after).

We fix  R = 2·⟨N_i⟩  and fit the seven unknowns

    f_before, g, φ_c,  e_on, φ_on,  e_off, φ_off

simultaneously to all four traces.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import least_squares

from plot_utils import delta_from_steps
import unheralded  # provides piezo_steps + the TSV data blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_counts(tsv: str):
    """Return (Ns, Ni, Nc) arrays parsed from a TSV block with a header row."""
    rows = [
        [float(x) for x in line.split()]
        for line in tsv.strip().splitlines()[1:]  # skip the header
        if line.strip()
    ]
    data = np.asarray(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


# Parse the two data sets
Ns_on,  Ni_on,  Nc_on  = _parse_counts(unheralded.eraser_on)
Ns_off, Ni_off, Nc_off = _parse_counts(unheralded.eraser_off)

# Common phase-delay array
delta = delta_from_steps(unheralded.piezo_steps)

# Fix the pair-production rate to twice the grand mean of idler singles
R_FIXED = 2.0 * np.mean(np.concatenate([Ni_on, Ni_off]))


# ---------------------------------------------------------------------------
# Residuals for the simultaneous χ² fit of all four traces
# ---------------------------------------------------------------------------
def _residuals(params: np.ndarray) -> np.ndarray:
    """
    Weighted residuals for least-squares optimisation.

        params = [f_before, g, φ_c,
                  e_on,    φ_on,
                  e_off,   φ_off]
    """
    f_before, g, phi_c, e_on, phi_on, e_off, phi_off = params

    one_minus_fb = 1.0 - f_before

    # Model curves
    Ni_on_model  = 0.5 * R_FIXED * (1 + one_minus_fb * e_on  * np.cos(delta + phi_on))
    Nc_on_model  = 0.5 * R_FIXED * g * (1 + e_on  * np.cos(delta + phi_on + phi_c))
    Ni_off_model = 0.5 * R_FIXED * (1 + one_minus_fb * e_off * np.cos(delta + phi_off))
    Nc_off_model = 0.5 * R_FIXED * g * (1 + e_off * np.cos(delta + phi_off + phi_c))

    # Poisson σ = √N weighting
    res_Ni_on  = (Ni_on  - Ni_on_model)  / np.sqrt(Ni_on  + 1.0)
    res_Nc_on  = (Nc_on  - Nc_on_model)  / np.sqrt(Nc_on  + 1.0)
    res_Ni_off = (Ni_off - Ni_off_model) / np.sqrt(Ni_off + 1.0)
    res_Nc_off = (Nc_off - Nc_off_model) / np.sqrt(Nc_off + 1.0)

    return np.concatenate([res_Ni_on, res_Nc_on, res_Ni_off, res_Nc_off])


# ---------------------------------------------------------------------------
# Fit driver
# ---------------------------------------------------------------------------
def fit_parameters():
    # Initial guesses -------------------------------------------------------
    f_before_0 = 0.90
    g_0        = 0.10
    phi_c_0    = 0.0

    # Visibility-based guesses for e_on / e_off
    def _visibility(y):
        return (np.max(y) - np.min(y)) / (np.max(y) + np.min(y) + 1e-9)

    e_on_0  = _visibility(Nc_on)
    e_off_0 = max(_visibility(Nc_off), 0.02)  # keep away from zero for stability

    phi_on_0  = 0.0
    phi_off_0 = 0.0

    p0 = np.array([f_before_0, g_0, phi_c_0,
                   e_on_0,     phi_on_0,
                   e_off_0,    phi_off_0])

    # Bounds:   f_before, g in [0,1];  e in [0,1];  phases in [-π, π]
    lower = np.array([0.0, 0.0, -math.pi,
                      0.0, -math.pi,
                      0.0, -math.pi])
    upper = np.array([1.0, 1.0,  math.pi,
                      1.0,  math.pi,
                      1.0,  math.pi])

    result = least_squares(
        _residuals,
        p0,
        bounds=(lower, upper),
        method="trf",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )

    # Unpack & report -------------------------------------------------------
    f_before, g, phi_c, e_on, phi_on, e_off, phi_off = result.x
    one_minus_fb = 1.0 - f_before
    f_after = 1.0 - g / max(one_minus_fb, 1e-9)

    print("\n==== Global fit results (idler singles + coincidences, two runs) ====")
    print(f"  R (fixed)        : {R_FIXED:.2f} counts /s")
    print(f"  f_before         : {f_before:.3f}")
    print(f"  f_after          : {f_after:.3f}")
    print(f"  g = (1-f_b)(1-f_a): {g:.3f}")
    print(f"  φ_c              : {phi_c:.3f} rad  ({np.degrees(phi_c):.1f}°)")
    print("  --- eraser ON run ---")
    print(f"    e_on           : {e_on:.3f}")
    print(f"    φ_on           : {phi_on:.3f} rad  ({np.degrees(phi_on):.1f}°)")
    print(f"    Predicted V_i  : {(one_minus_fb * e_on):.3f}")
    print(f"    Predicted V_c  : {e_on:.3f}")
    print("  --- eraser OFF run ---")
    print(f"    e_off          : {e_off:.3f}")
    print(f"    φ_off          : {phi_off:.3f} rad  ({np.degrees(phi_off):.1f}°)")
    print(f"    Predicted V_i  : {(one_minus_fb * e_off):.3f}")
    print(f"    Predicted V_c  : {e_off:.3f}")
    print("  ================================================================\n")

    return result


if __name__ == "__main__":
    fit_parameters()
