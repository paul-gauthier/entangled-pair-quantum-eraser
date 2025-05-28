#!/usr/bin/env python

from sys import exit

from sympy import *
from sympy.physics.quantum import TensorProduct
import numpy as np
import matplotlib.pyplot as plt
import math

from show import show
from dump import dump

TP = TensorProduct

##############################################################
# Basis states

psi_x = psi_b = Matrix([1, 0])
psi_y = psi_t = Matrix([0, 1])

H = Matrix([1, 0])
V = Matrix([0, 1])

#show(psi_b)
#show(psi_t)
#show(H)
#show(V)

# Q1

psi_b_H = TP(psi_b, H)
psi_b_V = TP(psi_b, V)
psi_t_H = TP(psi_t, H)
psi_t_V = TP(psi_t, V)
#show(psi_b_H)
#show(psi_b_V)
#show(psi_t_H)
#show(psi_t_V)

##############################################################
# Beamsplitter
B_hat = Matrix([
    [1, I],
    [I, 1],
]) / sqrt(2)
#show(B_hat, sqrt(2))

I22 = eye(2,2)
#show(I22)

B_hat_prime = TP(B_hat, I22)
#show(B_hat_prime, sqrt(2))

##############################################################
# Mirror
M_hat = Matrix([
    [0, 1],
    [1, 0],
])

#show(M_hat)

M_hat_prime = TP(M_hat, I22)
#show(M_hat_prime)


##############################################################
# Phase delay (mirror on piezo stage)
delta = symbols("delta", real=True)

A_hat = Matrix([
    [1, 0],
    [0, exp(I*delta)],
])

#show(A_hat)

A_hat_prime = TP(A_hat, I22)
#show(A_hat_prime)


##############################################################
# HWP with adjustable angle in upper arm, HWP @ 0 degrees in lower arm
# Q5
vartheta = symbols("vartheta", real=True)

W_hat_prime = Matrix([
    [cos(2*vartheta), sin(2*vartheta), 0, 0],
    [sin(2*vartheta), -cos(2*vartheta), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

##############################################################
# Compose an MZI with:
# - Adjustable HWP in upper arm
# - Fixed HWP @ 0deg in lower arm
# - Adjustable phase delay in lower arm
#
#show(W_hat_prime)
#show(M_hat_prime * B_hat_prime, 2/sqrt(2))
#show(A_hat_prime * M_hat_prime * B_hat_prime, 2/sqrt(2))
#show(W_hat_prime * A_hat_prime * M_hat_prime * B_hat_prime, 2/sqrt(2))

Z_hat_prime = B_hat_prime * W_hat_prime * A_hat_prime * M_hat_prime * B_hat_prime

#show(Z_hat_prime, 2)

psi_b_D = (psi_b_H + psi_b_V) / sqrt(2)
psi_t_D = (psi_t_H + psi_t_V) / sqrt(2)

##############################################################
# A linear polarizer set at angle theta

theta = symbols("theta", real=True)

# This is for V=90, H=0
P_hat_standard = Matrix([
    [cos(theta)**2, cos(theta)*sin(theta)],
    [cos(theta)*sin(theta), sin(theta)**2]
])

# Convert to V=0, H=90 which we use in our lab
P_hat = P_hat_standard.subs(theta, theta + pi/2)

Diag_hat = P_hat.subs(theta, pi/4)
H_out = Diag_hat * H
V_out = Diag_hat * V

D = (H+V)/sqrt(2)
#show(D, sqrt(2))
#show(D/sqrt(2), 2)

# We should lost half the photons
assert H_out.norm()**2 == H.norm()**2/2
assert V_out.norm()**2 == V.norm()**2/2

#dump(H_out.norm()**2, H.norm()**2)

##############################################################
# An adjustable polarizer for the x/horizontal exit of the MZI
P_hat_prime = TP(psi_x * psi_x.T, P_hat) + TP(psi_y * psi_y.T, I22)

#show(P_hat)
#show(P_hat_prime)

#show(P_hat_prime.subs(theta, pi/4))

#show(P_hat_prime.subs(theta, pi/4) * psi_b_H, 2)
#show(P_hat_prime.subs(theta, pi/4) * psi_b_V, 2)
#show(psi_b_D, sqrt(2))
#show(psi_b_D/sqrt(2), 2)

#show(P_hat_prime.subs(theta, pi/4) * psi_t_H)
#show(P_hat_prime.subs(theta, pi/4) * psi_t_V)

##############################################################
# An operator for the MZI with adjustable polarizer for the x/horizontal exit
E_hat_prime = P_hat_prime * Z_hat_prime
#show(E_hat_prime, 2)

##############################################################
# An instance of the MZI+LP at specific settings
#
# HWP_u at vartheta=45, theta=LP_i at 90
E_hat_prime_45_90 = E_hat_prime.subs(vartheta, pi/4).subs(theta, pi/2)
#show(E_hat_prime_45_90, 4)

def test_E_hat_prime_45_90():
    # Send in D light
    psi_eraser = (E_hat_prime_45_90 * psi_b_D)
    show(psi_eraser, 2)

    psi_x_H_final = psi_b_H.dot(psi_eraser)
    show(psi_x_H_final, 4)
    prob_psi_x_H_final = abs(psi_x_H_final)**2
    prob_psi_x_H_final = simplify(prob_psi_x_H_final.rewrite(cos))
    show(prob_psi_x_H_final)
    min_final = minimum(prob_psi_x_H_final, delta)
    max_final = maximum(prob_psi_x_H_final, delta)
    show(min_final)
    show(max_final)
    dump(min_final.evalf(3))
    dump(max_final.evalf(3))
    dump(max_final.evalf(3)-min_final.evalf(3))


    # Projector onto the 'b' spatial state, identity in polarization space
    # psi_b is Matrix([1, 0])
    # I22 is eye(2,2)
    projector_b_spatial = TP(psi_b * psi_b.T, I22)
    show(projector_b_spatial) # Optional: to see the projector matrix

    # Probability = <psi_eraser | P_b_spatial | psi_eraser>
    # .H gives the Hermitian conjugate (bra)
    # The result of the product is a 1x1 matrix, so access its element [0,0]
    prob_b_spatial_qm = (psi_eraser.H * projector_b_spatial * psi_eraser)[0,0]
    show(prob_b_spatial_qm) # To display the symbolic probability
    prob_b_spatial_qm_simplified = simplify(prob_b_spatial_qm.rewrite(cos))
    show(prob_b_spatial_qm_simplified) # To display the simplified symbolic probability

    min_prob_b = minimum(prob_b_spatial_qm_simplified, delta)
    max_prob_b = maximum(prob_b_spatial_qm_simplified, delta)
    show(min_prob_b)
    show(max_prob_b)
    dump(min_prob_b.evalf(3))
    dump(max_prob_b.evalf(3))
    dump((max_prob_b - min_prob_b).evalf(3))

#test_E_hat_prime_45_90()

##############################################################
# Extension: two-photon (signal & idler) processing
# -----------------------------------------------
# Each photon spans a 4-D Hilbert space (2 spatial ⊗ 2 polarisation).
# The composite space (signal ⊗ idler) is therefore 16-D.

I44 = eye(4)

# Signal: variable linear polariser P̂′(θ) – acts on signal only
P_hat_prime_signal = TP(P_hat_prime, I44)


def process_signal_idler(initial_state, theta_val, E_hat_prime_current):
    """
    Apply  P_hat_prime(theta)  to the signal photon followed by
    E_hat_prime_current  to the idler photon.

    Parameters
    ----------
    initial_state : sympy.Matrix (16×1)
        Joint state |ψ⟩ of signal ⊗ idler.
    theta_val : sympy expression or float
        Polariser angle θ for the signal (radians).
    E_hat_prime_current : sympy.Matrix
        Operator to apply to the idler photon.

    Returns
    -------
    sympy.Matrix (16×1)
        Final (unnormalised) state after the two-stage process.
    """
    # Substitute θ in the signal polariser
    P_signal = P_hat_prime_signal.subs(theta, theta_val)

    # Create the idler operator
    E_hat_prime_current_idler = TP(I44, E_hat_prime_current)

    # Total operator  (E_idler) ⋅ (P_signal)
    total_op = E_hat_prime_current_idler * P_signal

    return total_op * initial_state


def coincident_probability(initial_state, theta_val, E_hat_prime_current):
    """
    Compute the probability for a coincident detection in which

      • the signal photon is detected (no restriction on path or polarisation), and
      • the idler photon exits the b-path (any polarisation).

    Parameters
    ----------
    initial_state : sympy.Matrix (16×1)
        Joint input state |ψ⟩ of  signal ⊗ idler  (before any optics).
    theta_val : sympy expression or float
        Signal-polariser angle θ (radians).
    E_hat_prime_current : sympy.Matrix
        Operator to apply to the idler photon.

    Returns
    -------
          probability – real detection probability  |amplitude|².
    """
    # State after the two-stage optical process
    psi_out = process_signal_idler(initial_state, theta_val, E_hat_prime_current)

    # Projector onto the idler b-path (identity in polarisation)
    projector_b_spatial = TP(psi_b * psi_b.T, I22)
    proj_joint = TP(I44, projector_b_spatial)

    # Detection probability  ⟨ψ_out| P |ψ_out⟩
    probability = simplify((psi_out.H * proj_joint * psi_out)[0, 0])

    return probability


def demo_pair(initial_state, mzi_hwp_angle, idler_lp_angle, signal_lp_angle):
    """
    Compute the coincident-detection probability for the given parameters
    and the corresponding interference visibility.

    Parameters
    ----------
    initial_state :
        16-component joint state |ψ⟩ of signal ⊗ idler.
    mzi_hwp_angle :
        HWP angle (ϑ) in the upper arm of the MZI.
    idler_lp_angle :
        Linear-polariser angle (θ) in the idler arm.
    signal_lp_angle :
        Linear-polariser angle (θ) in the signal arm.

    Returns
    -------
    tuple
        (probability expression, visibility expression)
    """

    # HWP_u at vartheta = mzi_hwp_angle, LP_i at theta = idler_lp_angle
    E_hat_prime_current = E_hat_prime.subs(vartheta, mzi_hwp_angle).subs(theta, idler_lp_angle)

    # Probability for coincident detection
    prob = coincident_probability(initial_state, signal_lp_angle, E_hat_prime_current)

    # Simplify and express in terms of cos(δ) before finding extrema
    prob_simplified = simplify(prob.rewrite(cos))

    # Extremal values with respect to the phase delay δ
    min_prob = minimum(prob_simplified, delta)
    max_prob = maximum(prob_simplified, delta)

    # Visibility  V = (max − min) / (max + min)
    visibility = simplify((max_prob - min_prob) / (max_prob + min_prob))

    print("#" * 80)
    dump(mzi_hwp_angle, idler_lp_angle, signal_lp_angle)
    show(prob_simplified)  # symbolic probability
    dump(min_prob.evalf(5), max_prob.evalf(5), visibility.evalf(5))

    return prob_simplified, visibility


def plot_visibility_heatmap(fig, ax, initial_state, mzi_hwp_angle, idler_lp_angle_base_rad, signal_lp_angle_base_rad, plot_title_str):
    """
    Generates and saves a heatmap of visibility.

    Visibility is computed by varying idler and signal linear polarizer angles
    by an epsilon amount around their respective base angles.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure object to draw on.
    ax : matplotlib.axes.Axes
        The axes object to draw on.
    initial_state : sympy.Matrix
        The initial quantum state for the demo_pair calculation.
    mzi_hwp_angle : float or sympy expression
        The MZI HWP angle (vartheta).
    idler_lp_angle_base_rad : float or sympy expression
        The base angle for the idler linear polarizer (in radians).
    signal_lp_angle_base_rad : float or sympy expression
        The base angle for the signal linear polarizer (in radians).
    plot_title_str : str
        The title for the heatmap plot.
    """
    print(f"Generating visibility heatmap for: {plot_title_str}")
    epsilon_degrees_range = np.linspace(-5, 5, 11)  # -5 to +5 degrees in 1 degree steps
    visibility_values = np.zeros((len(epsilon_degrees_range), len(epsilon_degrees_range)))

    for i, idler_eps_deg in enumerate(epsilon_degrees_range):
        for j, signal_eps_deg in enumerate(epsilon_degrees_range):
            idler_eps_rad = math.radians(idler_eps_deg)
            signal_eps_rad = math.radians(signal_eps_deg)

            current_idler_lp_angle = idler_lp_angle_base_rad + idler_eps_rad
            current_mzi_hwp_angle = mzi_hwp_angle + idler_eps_rad/2 # move hwp with idler lp
            current_signal_lp_angle = signal_lp_angle_base_rad + signal_eps_rad

            _prob_temp, vis_temp = demo_pair(
                initial_state=initial_state,
                mzi_hwp_angle=current_mzi_hwp_angle,
                idler_lp_angle=current_idler_lp_angle,
                signal_lp_angle=current_signal_lp_angle,
            )
            visibility_values[i, j] = vis_temp.evalf()

    print(f"Finished generating data for: {plot_title_str}")

    im = ax.imshow(visibility_values, origin='lower',
               extent=[epsilon_degrees_range.min(), epsilon_degrees_range.max(),
                       epsilon_degrees_range.min(), epsilon_degrees_range.max()],
               aspect='auto', cmap='viridis')
    fig.colorbar(im, ax=ax, label='Visibility')
    ax.set_xlabel(f'Signal LP Epsilon (degrees from {math.degrees(signal_lp_angle_base_rad):.0f}°)')
    ax.set_ylabel(f'Idler LP Epsilon (degrees from {math.degrees(idler_lp_angle_base_rad):.0f}°)')
    ax.set_title(plot_title_str)
    ax.grid(True, linestyle='--', alpha=0.6)

# ------------------------------------------------------------------------
# Generic 2-parameter visibility heat-map
# ------------------------------------------------------------------------
def plot_visibility_heatmap_xy(
    fig,
    ax,
    *,
    initial_state,
    base_mzi_hwp_angle,
    base_idler_lp_angle,
    base_signal_lp_angle,
    x_param: str,
    y_param: str,
    plot_title: str,
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
    plot_title : str
        Title for the axes.
    eps_range_deg : 1-D array, optional
        Range of ε values (degrees).  Default = −5…+5 in 1° steps.
    """
    if x_param == y_param:
        raise ValueError("x_param and y_param must be distinct")

    if eps_range_deg is None:
        eps_range_deg = np.linspace(-5, 5, 11)

    visibility_values = np.zeros((len(eps_range_deg), len(eps_range_deg)))

    base_angles = {
        "signal": base_signal_lp_angle,
        "idler":  base_idler_lp_angle,
        "hwp":    base_mzi_hwp_angle,
    }

    for i, y_eps_deg in enumerate(eps_range_deg):
        for j, x_eps_deg in enumerate(eps_range_deg):
            # Start from the nominal settings
            angles = base_angles.copy()

            # Apply ε deviations
            angles[x_param] += math.radians(x_eps_deg)
            angles[y_param] += math.radians(y_eps_deg)

            # Evaluate visibility
            _, vis_temp = demo_pair(
                initial_state=initial_state,
                mzi_hwp_angle=angles["hwp"],
                idler_lp_angle=angles["idler"],
                signal_lp_angle=angles["signal"],
            )
            visibility_values[i, j] = float(vis_temp.evalf())

    im = ax.imshow(
        visibility_values,
        origin="lower",
        extent=[eps_range_deg.min(), eps_range_deg.max()] * 2,
        aspect="auto",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="Visibility")

    nice = {"signal": "Signal LP", "idler": "Idler LP", "hwp": "MZI HWP"}

    x_base_deg = math.degrees(base_angles[x_param])
    y_base_deg = math.degrees(base_angles[y_param])

    ax.set_xlabel(f'{nice[x_param]} {x_base_deg:.0f}+ε (deg)')
    ax.set_ylabel(f'{nice[y_param]} {y_base_deg:.0f}+ε (deg)')

    # Construct title: Prefix from input + dynamic part with base angles
    # Assumes plot_title is like "User Prefix : Old Dynamic Part"
    signal_deg = math.degrees(base_signal_lp_angle)
    hwp_deg = math.degrees(base_mzi_hwp_angle)
    idler_deg = math.degrees(base_idler_lp_angle)

    signal_str = f"Signal {signal_deg:.0f}°{'+e' if x_param == 'signal' or y_param == 'signal' else ''}"
    hwp_str = f"HWP {hwp_deg:.0f}°{'+e' if x_param == 'hwp' or y_param == 'hwp' else ''}"
    idler_str = f"Idler {idler_deg:.0f}°{'+e' if x_param == 'idler' or y_param == 'idler' else ''}"
    dynamic_title_part = f"{signal_str}, {hwp_str}, {idler_str}"
    final_title = dynamic_title_part
    ax.set_title(final_title)
    ax.grid(True, linestyle="--", alpha=0.6)

# Build phi+ state
psi_hh = TP(psi_b_H, psi_b_H)
psi_vv = TP(psi_b_V, psi_b_V)
phi_plus_state = (psi_hh + psi_vv) / sqrt(2)   # 16×1 column state
assert phi_plus_state.norm() == 1

##############################################################


# Proper settings, eraser on
prob, visibility = demo_pair(
    phi_plus_state,
    mzi_hwp_angle=pi/4,  # swap H/V in the upper arm
    idler_lp_angle=pi/2, # 90 degree = H
    signal_lp_angle=pi/4,   # 45 = pi/4 = Eraser on
)
# Proper settings, eraser off
demo_pair(
    phi_plus_state,
    mzi_hwp_angle=pi/4,  # swap H/V in the upper arm
    idler_lp_angle=pi/2, # 90 degree = H
    signal_lp_angle=0,   # 0 = Eraser off
)
# Proper settings, eraser off
demo_pair(
    phi_plus_state,
    mzi_hwp_angle=pi/4,    # swap H/V in the upper arm
    idler_lp_angle=pi/2,   # 90 degree = H
    signal_lp_angle=pi/2,  # 90 = Eraser off
)

expected_prob = (1 - cos(delta)) / 8
assert prob.equals(expected_prob), f"Probability {prob} != expected {expected_prob}"

fig, axes = plt.subplots(3, 3, figsize=(24, 21))  # 3×3 grid: rows = base states, cols = param pairs

# -----------------------------------------------------------------------------
# Row 0 – Φ⁺, eraser ON (Signal 45°, Idler 90°)
# -----------------------------------------------------------------------------
plot_visibility_heatmap_xy(
    fig, axes[0, 0],
    initial_state=phi_plus_state,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=pi/4,
    x_param="signal", y_param="idler",
    plot_title='Φ⁺, eraser-on : Idler ε  vs  Signal ε',
)

plot_visibility_heatmap_xy(
    fig, axes[0, 1],
    initial_state=phi_plus_state,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=pi/4,
    x_param="signal", y_param="hwp",
    plot_title='Φ⁺, eraser-on : HWP ε  vs  Signal ε',
)

plot_visibility_heatmap_xy(
    fig, axes[0, 2],
    initial_state=phi_plus_state,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=pi/4,
    x_param="idler", y_param="hwp",
    plot_title='Φ⁺, eraser-on : HWP ε  vs  Idler ε',
)

# -----------------------------------------------------------------------------
# Row 1 – Φ⁺, eraser OFF (Signal 0°, Idler 90°)
# -----------------------------------------------------------------------------
plot_visibility_heatmap_xy(
    fig, axes[1, 0],
    initial_state=phi_plus_state,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=0,
    x_param="signal", y_param="idler",
    plot_title='Φ⁺, eraser-off : Idler ε  vs  Signal ε',
)

plot_visibility_heatmap_xy(
    fig, axes[1, 1],
    initial_state=phi_plus_state,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=0,
    x_param="signal", y_param="hwp",
    plot_title='Φ⁺, eraser-off : HWP ε  vs  Signal ε',
)

plot_visibility_heatmap_xy(
    fig, axes[1, 2],
    initial_state=phi_plus_state,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=0,
    x_param="idler", y_param="hwp",
    plot_title='Φ⁺, eraser-off : HWP ε  vs  Idler ε',
)

# -----------------------------------------------------------------------------
# Row 2 – |ψ_VV⟩ input (Signal 90°, Idler 90°)
# -----------------------------------------------------------------------------
plot_visibility_heatmap_xy(
    fig, axes[2, 0],
    initial_state=psi_vv,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=pi/2,
    x_param="signal", y_param="idler",
    plot_title='ψ_VV : Idler ε  vs  Signal ε',
)

plot_visibility_heatmap_xy(
    fig, axes[2, 1],
    initial_state=psi_vv,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=pi/2,
    x_param="signal", y_param="hwp",
    plot_title='ψ_VV : HWP ε  vs  Signal ε',
)

plot_visibility_heatmap_xy(
    fig, axes[2, 2],
    initial_state=psi_vv,
    base_mzi_hwp_angle=pi/4,
    base_idler_lp_angle=pi/2,
    base_signal_lp_angle=pi/2,
    x_param="idler", y_param="hwp",
    plot_title='ψ_VV : HWP ε  vs  Idler ε',
)

plt.tight_layout()
output_filename_combined = "visibility_heatmaps_combined.png"
plt.savefig(output_filename_combined, dpi=300)
print(f"Saved combined heatmap to {output_filename_combined}")


exit()


##############################################################
#
# This matches the collected data.
# Both interfere, with N_c_eraser_on = 1/2 N_c_eraser_off
#

# Misconfigured on Friday, "eraser on"
demo_pair(
    psi_vv, # Pump HWP was set to 45 => Pump @ 90/H => 0/V signals&idlers
    mzi_hwp_angle=pi/4,   # swap H/V in the upper arm
    idler_lp_angle=pi/4,  # This was set to 45deg instead of 90 deg
    signal_lp_angle=pi/4, # Eraser on
)


# Misconfigured on Friday, "eraser off"
prob, visibility = demo_pair(
    psi_vv, # Pump HWP was set to 45 => Pump @ 90/H => 0/V signals&idlers
    mzi_hwp_angle=pi/4,   # swap H/V in the upper arm
    idler_lp_angle=pi/4,  # This was set to 45deg instead of 90 deg
    signal_lp_angle=0, # Eraser off
)
