#!/usr/bin/env python
"""
Fit the novel idler-singles data from ``unheralded.py`` with the reduced model

    N_i(δ) = ½ · R_FIXED · [1 + (1 − f_before) · e_novel · cos(δ + φ_novel)]

The pair-production rate R_FIXED and the pre-eraser loss fraction f_before are
taken from the global four-trace fit implemented in ``fit_unheralded.py``.
Only two parameters are free:

    e_novel   – effective eraser efficiency for this data set (0 ≤ e ≤ 1)
    φ_novel   – phase offset (wrapped to (−π, π])

The script prints the best-fit values and saves a diagnostic plot
``ni_novel_fit.pdf`` showing the data and fitted curve.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from plot_utils import delta_from_steps, delta_to_nm, nm_to_delta
import fit_unheralded                       # provides R_FIXED and f_before via fit
import unheralded                           # provides piezo_steps & novel TSV


# ---------------------------------------------------------------------------
# Retrieve R_FIXED and f_before from the global four-trace fit
# ---------------------------------------------------------------------------
_global_result = fit_unheralded.fit_parameters()  # run / reuse existing global fit
R_FIXED: float = fit_unheralded.R_FIXED
novel_f_before_scale = 0.5
f_before: float = float(_global_result.x[0]) * novel_f_before_scale
one_minus_fb: float = 1.0 - f_before

# ---------------------------------------------------------------------------
# Parse the novel idler-singles counts
# ---------------------------------------------------------------------------
Ni_novel = np.array(
    [
        float(line.strip())
        for line in unheralded.novel.strip().splitlines()[1:]  # skip header
        if line.strip()
    ],
    dtype=float,
)
delta = delta_from_steps(unheralded.piezo_steps)

if len(Ni_novel) != len(delta):
    raise RuntimeError(
        f"Length mismatch: {len(Ni_novel)=} vs {len(delta)=}. "
        "piezo_steps and novel data must be equal length."
    )


def _ni_model(d: np.ndarray | float, e: float, phi: float) -> np.ndarray | float:
    """Reduced idler-singles model with fixed R and f_before."""
    return 0.5 * R_FIXED * (1.0 + one_minus_fb * e * np.cos(d + phi))


# ---------------------------------------------------------------------------
# Two-parameter χ² fit (Poisson σ = √N weighting)
# ---------------------------------------------------------------------------
p0 = [0.5, 0.0]  # initial guesses
popt, pcov = curve_fit(
    _ni_model,
    delta,
    Ni_novel,
    p0=p0,
    sigma=np.sqrt(Ni_novel) + 1.0,  # avoid zero σ
    absolute_sigma=True,
)
e_novel, phi_novel = popt
e_err, phi_err = np.sqrt(np.diag(pcov))

# Wrap phase into (−π, π]
phi_novel = (phi_novel + np.pi) % (2 * np.pi) - np.pi

# Predicted visibility
vis_pred = one_minus_fb * e_novel

print("\n==== Fit to novel idler-singles data ====")
print(f"  R_FIXED        : {R_FIXED:.2f} counts/s (fixed)")
print(f"  f_before       : {f_before:.3f} (fixed)")
print(f"  e_novel        : {e_novel:.3f} ± {e_err:.3f}")
print(
    f"  φ_novel        : {phi_novel:.3f} ± {phi_err:.3f} rad  "
    f"({np.degrees(phi_novel):.1f}°)"
)
print(f"  Predicted V_i  : {vis_pred:.3f}")
print("=========================================\n")

# ---------------------------------------------------------------------------
# Diagnostic plot
# ---------------------------------------------------------------------------
delta_fine = np.linspace(delta.min(), delta.max(), 500)
Ni_fit = _ni_model(delta_fine, e_novel, phi_novel)

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(9, 5))

ax.errorbar(
    delta,
    Ni_novel,
    yerr=np.sqrt(Ni_novel),
    fmt="o",
    color="tab:green",
    label="Novel $N_i$ data",
    capsize=3,
)
ax.plot(delta_fine, Ni_fit, "--", color="tab:green", label="Best fit")

ax.set_xlabel(r"Phase delay $\delta$ (rad)")
ax.set_ylabel(r"Counts / s")
ax.grid(True, linestyle=":", alpha=0.6)

# Secondary x-axis: piezo displacement (nm)
ax_nm = ax.secondary_xaxis("top", functions=(delta_to_nm, nm_to_delta))
ax_nm.set_xlabel("Piezo displacement (nm)")

ax.legend(loc="upper right")
fig.tight_layout()

outfile = "ni_novel_fit.pdf"
plt.savefig(outfile)
plt.close(fig)
print(f"Saved diagnostic plot to {outfile}")


if __name__ == "__main__":
    # All work is already done at import time; nothing more to execute.
    pass
