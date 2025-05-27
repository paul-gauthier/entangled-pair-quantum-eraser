#!/usr/bin/env python

from sys import exit

from sympy import *
from sympy.physics.quantum import TensorProduct
import numpy as np
import matplotlib.pyplot as plt

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

# Idler: fixed “eraser” stage (Ê′₄₅,₉₀) – acts on idler only
E_hat_prime_45_90_idler = TP(I44, E_hat_prime_45_90)

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

    # HWP_u at vartheta=, LP_i at theta=
    E_hat_prime_current = E_hat_prime.subs(vartheta, mzi_hwp_angle).subs(theta, idler_lp_angle)

    # Probability for coincident detection
    prob = coincident_probability(initial_state, signal_lp_angle, E_hat_prime_current)

    print("#"*80)
    dump(mzi_hwp_angle, idler_lp_angle, signal_lp_angle)
    #prob = simplify(prob.rewrite(sin))
    show(prob)  # symbolic probability

    return prob


# Build phi+ state
psi_hh = TP(psi_b_H, psi_b_H)
psi_vv = TP(psi_b_V, psi_b_V)
phi_plus_state = (psi_hh + psi_vv) / sqrt(2)   # 16×1 column state


##############################################################
# Proper settings, eraser on
prob = demo_pair(
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
prob = demo_pair(
    psi_vv, # Pump HWP was set to 45 => Pump @ 90/H => 0/V signals&idlers
    mzi_hwp_angle=pi/4,   # swap H/V in the upper arm
    idler_lp_angle=pi/4,  # This was set to 45deg instead of 90 deg
    signal_lp_angle=0, # Eraser off
)
