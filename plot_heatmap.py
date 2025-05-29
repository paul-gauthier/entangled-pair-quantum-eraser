import math
import numpy as np
import matplotlib.pyplot as plt
from sympy import Matrix # For type hint initial_state: Matrix

# This creates a circular dependency:
# lab6entangled.py imports plot_visibility_heatmap_xy from this file.
# This file imports demo_pair, phi_plus_state, psi_vv from lab6entangled.py.
# Python can handle this if imports are at the top level and definitions follow.
from lab6entangled import demo_pair, phi_plus_state, psi_vv


# ------------------------------------------------------------------------
# Generic 2-parameter visibility heat-map
# ------------------------------------------------------------------------

def plot_visibility_heatmap_xy(
    fig,
    ax,
    *,
    initial_state: Matrix,
    base_mzi_hwp_angle,
    base_idler_lp_angle,
    base_signal_lp_angle,
    x_param: str,
    y_param: str,
    eps_range_deg: np.ndarray | None = None,
):
    """
    Draw a visibility heat-map as a function of two independent ε variations
    applied to any pair of parameters chosen from {'signal', 'idler', 'hwp'}.

    Parameters
    ----------
    fig, ax : matplotlib objects
        Target figure/axes.
    initial_state : sympy.Matrix
        Input joint state |ψ⟩.
    base_mzi_hwp_angle, base_idler_lp_angle, base_signal_lp_angle : float
        Nominal angles (radians) about which epsilons are applied.
    x_param, y_param : str
        Parameter controlled by the x- / y-axis – one of
        {'signal', 'idler', 'hwp'} and different from each other.
    eps_range_deg : 1-D array, optional
        Range of ε values (degrees).  Default = −5…+5 in 1° steps.
    """

    # jitter so we don't analytically drop the influence of these angles
    jitter = 1/100000
    # Create copies to modify with jitter, to avoid altering the originals if they are passed by reference elsewhere
    current_mzi_hwp_angle = base_mzi_hwp_angle + jitter
    current_idler_lp_angle = base_idler_lp_angle + jitter
    current_signal_lp_angle = base_signal_lp_angle + jitter

    if x_param == y_param:
        raise ValueError("x_param and y_param must be distinct")

    if eps_range_deg is None:
        eps_range_deg = np.linspace(-5, 5, 11)

    visibility_values = np.zeros((len(eps_range_deg), len(eps_range_deg)))

    base_angles_for_plot_labels = { # Store original base angles for labels
        "signal": base_signal_lp_angle,
        "idler":  base_idler_lp_angle,
        "hwp":    base_mzi_hwp_angle,
    }

    current_base_angles = { # These include jitter
        "signal": current_signal_lp_angle,
        "idler":  current_idler_lp_angle,
        "hwp":    current_mzi_hwp_angle,
    }

    for i, y_eps_deg in enumerate(eps_range_deg):
        for j, x_eps_deg in enumerate(eps_range_deg):
            angles = current_base_angles.copy()
            angles[x_param] += math.radians(x_eps_deg)
            angles[y_param] += math.radians(y_eps_deg)

            _, vis_temp = demo_pair(
                initial_state=initial_state,
                mzi_hwp_angle=angles["hwp"],
                idler_lp_angle=angles["idler"],
                signal_lp_angle=angles["signal"],
            )
            visibility_values[i, j] = float(vis_temp.evalf())

    mean_visibility = np.mean(visibility_values)
    VISIBILITY_PLOT_RANGE = 0.35 # This was local in the original function
    vmin = mean_visibility - VISIBILITY_PLOT_RANGE / 2
    vmax = mean_visibility + VISIBILITY_PLOT_RANGE / 2

    if vmin < 0:
        vmax = VISIBILITY_PLOT_RANGE
        vmin = 0
    if vmax > 1:
        vmin = 1 - VISIBILITY_PLOT_RANGE
        vmax = 1

    im = ax.imshow(visibility_values, origin="lower", extent=[eps_range_deg.min(), eps_range_deg.max()] * 2, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    title_fontsize, label_fontsize, tick_fontsize = 24, 20, 18
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Visibility", fontsize=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)

    nice = {"signal": "LP$_s$", "idler": "LP$_i$", "hwp": "MZI HWP"}
    x_base_deg = math.degrees(base_angles_for_plot_labels[x_param]) # Use original for labels
    y_base_deg = math.degrees(base_angles_for_plot_labels[y_param]) # Use original for labels
    ax.set_xlabel(f'{nice[x_param]} {x_base_deg:.0f}+ε (deg)', fontsize=label_fontsize)
    ax.set_ylabel(f'{nice[y_param]} {y_base_deg:.0f}+ε (deg)', fontsize=label_fontsize)

    state_str = "Unknown State"
    if initial_state.equals(phi_plus_state): state_str = "Φ⁺"
    elif initial_state.equals(psi_vv): state_str = "ψ$_{VV}$"

    # Use original base angles for title consistency
    signal_deg_label = math.degrees(base_angles_for_plot_labels["signal"])
    hwp_deg_label = math.degrees(base_angles_for_plot_labels["hwp"])
    idler_deg_label = math.degrees(base_angles_for_plot_labels["idler"])

    signal_str = f"Signal {signal_deg_label:.0f}°{'+ε' if x_param == 'signal' or y_param == 'signal' else ''}"
    hwp_str = f"HWP {hwp_deg_label:.0f}°{'+ε' if x_param == 'hwp' or y_param == 'hwp' else ''}"
    idler_str = f"Idler {idler_deg_label:.0f}°{'+ε' if x_param == 'idler' or y_param == 'idler' else ''}"
    dynamic_config_str = f"{signal_str}, {hwp_str}, {idler_str}"
    final_title = f"{state_str}: {dynamic_config_str}"
    ax.set_title(final_title, fontsize=title_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax.grid(True, linestyle="--", alpha=0.6)

