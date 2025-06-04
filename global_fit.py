#!/usr/bin/env python3
"""
Global fitting utilities for photonics datasets.
"""
from __future__ import annotations

from typing import List, Tuple
import numpy as np
from scipy.optimize import least_squares


def _photonics_model_counts(
    positions: np.ndarray,
    period_steps: float,
    R: float,
    *,
    f_before: float,
    f_after: float,
    phi_c: float,
    e: float,
    phi: float,
    f_eraser: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict N_i and N_c for a set of piezo positions."""
    delta = 2 * np.pi * positions / period_steps

    N_i_pred = R * (
        0.5 * f_before
        + 0.5 * (1 - f_before) * f_eraser
        + 0.5 * (1 - f_before) * (1 - f_eraser) * (1 - e)
        + 0.5 * (1 - f_before * f_eraser) * e * (np.cos(delta + phi) + 1)
    )

    N_c_pred = (
        R
        * (1 - f_before)
        * (1 - f_eraser)
        * (1 - f_after)
        * (
            0.5 * (1 - e)
            + 0.5 * e * (np.cos(delta + phi + phi_c) + 1)
        )
    )
    return N_i_pred, N_c_pred


def fit_global_model(
    datasets,
    period_steps: float,
    *,
    fixed_phi: bool = False,
):
    """Jointly fit all datasets, returning the least_squares result."""
    n = len(datasets)

    # initial guess
    x0 = [0.1, 0.1, 0.0]                       # f_before, f_after, phi_c
    if fixed_phi:
        x0.append(0.0)                         # global phi
    x0.extend([0.5] * n)                       # e_i
    if not fixed_phi:
        x0.extend([0.0] * n)                   # phi_i
    x0 = np.asarray(x0)

    # bounds
    lo = [0.0, 0.0, -np.pi]
    hi = [1.0, 1.0, np.pi]
    if fixed_phi:
        lo.append(-np.pi)
        hi.append(np.pi)
    lo.extend([0.0] * n)
    hi.extend([1.0] * n)
    if not fixed_phi:
        lo.extend([-np.pi] * n)
        hi.extend([np.pi] * n)
    lo, hi = np.asarray(lo), np.asarray(hi)

    # pre-compute observations
    obs = []
    for ds in datasets:
        pos = np.array(sorted(ds.piezo_data.keys()))
        Ni = np.array([ds.piezo_data[p]["N_i"] - ds.dark_counts.get("N_i", 0) for p in pos])
        Nc = np.array([ds.piezo_data[p]["N_c"] - ds.dark_counts.get("N_c", 0) for p in pos])
        R = 2.0 * np.mean(Ni)
        f_eraser = float(ds.metadata.get("f_eraser", 0.5))
        obs.append(dict(pos=pos, Ni=Ni, Nc=Nc, R=R, f_eraser=f_eraser))

    def residuals(x):
        idx = 0
        f_before = x[idx]; idx += 1
        f_after = x[idx]; idx += 1
        phi_c = x[idx]; idx += 1

        if fixed_phi:
            phi_g = x[idx]; idx += 1

        e_vals = x[idx:idx + n]
        idx += n

        if fixed_phi:
            phi_vals = [phi_g] * n
        else:
            phi_vals = x[idx:idx + n]

        res = []
        for i, o in enumerate(obs):
            Ni_pred, Nc_pred = _photonics_model_counts(
                o["pos"],
                period_steps,
                o["R"],
                f_before=f_before,
                f_after=f_after,
                phi_c=phi_c,
                e=e_vals[i],
                phi=phi_vals[i],
                f_eraser=o["f_eraser"],
            )
            res.append((Ni_pred - o["Ni"]) / np.sqrt(np.maximum(o["Ni"], 1)))
            res.append((Nc_pred - o["Nc"]) / np.sqrt(np.maximum(o["Nc"], 1)))
        return np.concatenate(res)

    return least_squares(
        residuals,
        x0,
        bounds=(lo, hi),
        method="trf",
        verbose=1,
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=10000,
    )
