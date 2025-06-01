#!/usr/bin/env python

from sys import exit

from sympy import *
from sympy.physics.quantum import TensorProduct
import numpy as np
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

# show(psi_b)
# show(psi_t)
# show(H)
# show(V)

psi_b_H = TP(psi_b, H)
psi_b_V = TP(psi_b, V)
psi_t_H = TP(psi_t, H)
psi_t_V = TP(psi_t, V)
# show(psi_b_H)
# show(psi_b_V)
# show(psi_t_H)
# show(psi_t_V)

##############################################################
# Beamsplitter
B_hat = Matrix(
    [
        [1, I],
        [I, 1],
    ]
) / sqrt(2)
# show(B_hat, sqrt(2))

I22 = eye(2, 2)
# show(I22)

B_hat_prime = TP(B_hat, I22)
# show(B_hat_prime, sqrt(2))

##############################################################
# Mirror
M_hat = Matrix(
    [
        [0, 1],
        [1, 0],
    ]
)

# show(M_hat)

M_hat_prime = TP(M_hat, I22)
# show(M_hat_prime)


##############################################################
# Phase delay (mirror on piezo stage)
delta = symbols("delta", real=True)

A_hat = Matrix(
    [
        [1, 0],
        [0, exp(I * delta)],
    ]
)

# show(A_hat)

A_hat_prime = TP(A_hat, I22)
# show(A_hat_prime)


##############################################################
# HWP with adjustable angle in upper arm, HWP @ 0 degrees in lower arm

vartheta = symbols("vartheta", real=True)

W_hat_prime = Matrix(
    [
        [cos(2 * vartheta), sin(2 * vartheta), 0, 0],
        [sin(2 * vartheta), -cos(2 * vartheta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1],
    ]
)

##############################################################
# Compose an MZI with:
# - Adjustable HWP in upper arm
# - Fixed HWP @ 0deg in lower arm
# - Adjustable phase delay in lower arm
#
# show(W_hat_prime)
# show(M_hat_prime * B_hat_prime, 2/sqrt(2))
# show(A_hat_prime * M_hat_prime * B_hat_prime, 2/sqrt(2))
# show(W_hat_prime * A_hat_prime * M_hat_prime * B_hat_prime, 2/sqrt(2))

Z_hat_prime = B_hat_prime * W_hat_prime * A_hat_prime * M_hat_prime * B_hat_prime

# show(Z_hat_prime, 2)

psi_b_D = (psi_b_H + psi_b_V) / sqrt(2)
psi_b_A = (psi_b_H - psi_b_V) / sqrt(2)
psi_t_D = (psi_t_H + psi_t_V) / sqrt(2)

##############################################################
# A linear polarizer set at angle theta

theta = symbols("theta", real=True)

# This is for V=90, H=0
P_hat_standard = Matrix(
    [
        [cos(theta) ** 2, cos(theta) * sin(theta)],
        [cos(theta) * sin(theta), sin(theta) ** 2],
    ]
)

# Convert to V=0, H=90 which we use in our lab
P_hat = P_hat_standard.subs(theta, theta + pi / 2)

Diag_hat = P_hat.subs(theta, pi / 4)
H_out = Diag_hat * H
V_out = Diag_hat * V

D = (H + V) / sqrt(2)
# show(D, sqrt(2))
# show(D/sqrt(2), 2)

# We should lost half the photons
assert H_out.norm() ** 2 == H.norm() ** 2 / 2
assert V_out.norm() ** 2 == V.norm() ** 2 / 2

# dump(H_out.norm()**2, H.norm()**2)

##############################################################
# An adjustable polarizer for the x/horizontal exit of the MZI
P_hat_prime = TP(psi_x * psi_x.T, P_hat) + TP(psi_y * psi_y.T, I22)

# show(P_hat)
# show(P_hat_prime)

# show(P_hat_prime.subs(theta, pi/4))

# show(P_hat_prime.subs(theta, pi/4) * psi_b_H, 2)
# show(P_hat_prime.subs(theta, pi/4) * psi_b_V, 2)
# show(psi_b_D, sqrt(2))
# show(psi_b_D/sqrt(2), 2)

# show(P_hat_prime.subs(theta, pi/4) * psi_t_H)
# show(P_hat_prime.subs(theta, pi/4) * psi_t_V)

##############################################################
# An operator for the MZI with adjustable polarizer for the x/horizontal exit
E_hat_prime = P_hat_prime * Z_hat_prime
# show(E_hat_prime, 2)

##############################################################
# An instance of the MZI+LP at specific settings
#
# HWP_u at vartheta=45, theta=LP_i at 90
E_hat_prime_45_90 = E_hat_prime.subs(vartheta, pi / 4).subs(theta, pi / 2)
# show(E_hat_prime_45_90, 4)


def test_E_hat_prime_45_90():
    # Send in D light
    psi_eraser = E_hat_prime_45_90 * psi_b_D
    show(psi_eraser, 2)

    psi_x_H_final = psi_b_H.dot(psi_eraser)
    show(psi_x_H_final, 4)
    prob_psi_x_H_final = abs(psi_x_H_final) ** 2
    prob_psi_x_H_final = simplify(prob_psi_x_H_final.rewrite(cos))
    show(prob_psi_x_H_final)
    min_final = minimum(prob_psi_x_H_final, delta)
    max_final = maximum(prob_psi_x_H_final, delta)
    show(min_final)
    show(max_final)
    dump(min_final.evalf(3))
    dump(max_final.evalf(3))
    dump(max_final.evalf(3) - min_final.evalf(3))

    # Projector onto the 'b' spatial state, identity in polarization space
    # psi_b is Matrix([1, 0])
    # I22 is eye(2,2)
    projector_b_spatial = TP(psi_b * psi_b.T, I22)
    show(projector_b_spatial)  # Optional: to see the projector matrix

    # Probability = <psi_eraser | P_b_spatial | psi_eraser>
    # .H gives the Hermitian conjugate (bra)
    # The result of the product is a 1x1 matrix, so access its element [0,0]
    prob_b_spatial_qm = (psi_eraser.H * projector_b_spatial * psi_eraser)[0, 0]
    show(prob_b_spatial_qm)  # To display the symbolic probability
    prob_b_spatial_qm_simplified = simplify(prob_b_spatial_qm.rewrite(cos))
    show(prob_b_spatial_qm_simplified)  # To display the simplified symbolic probability

    min_prob_b = minimum(prob_b_spatial_qm_simplified, delta)
    max_prob_b = maximum(prob_b_spatial_qm_simplified, delta)
    show(min_prob_b)
    show(max_prob_b)
    dump(min_prob_b.evalf(3))
    dump(max_prob_b.evalf(3))
    dump((max_prob_b - min_prob_b).evalf(3))


# test_E_hat_prime_45_90()

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

    # Apply a −π/8 rotation to both photons’ polarisation (H,V basis)
    rot_angle = -pi / 8
    R_pol = Matrix([[cos(rot_angle), -sin(rot_angle)],
                    [sin(rot_angle),  cos(rot_angle)]])
    rot_op_signal = TP(I22, R_pol)   # 4×4 acting on signal photon
    rot_op_idler = TP(I22, R_pol)    # 4×4 acting on idler photon
    rotation_operator = TP(rot_op_signal, rot_op_idler)  # 16×16
    initial_state = rotation_operator * initial_state

    # HWP_u at vartheta = mzi_hwp_angle, LP_i at theta = idler_lp_angle
    E_hat_prime_current = E_hat_prime.subs(vartheta, mzi_hwp_angle).subs(
        theta, idler_lp_angle
    )

    # Probability for coincident detection
    prob = coincident_probability(initial_state, signal_lp_angle, E_hat_prime_current)

    # Simplify and express in terms of cos(δ) before finding extrema
    prob_simplified = simplify(prob.rewrite(cos))

    # Extremal values with respect to the phase delay δ
    min_prob = minimum(prob_simplified.evalf(), delta)
    max_prob = maximum(prob_simplified.evalf(), delta)

    # Visibility  V = (max − min) / (max + min)
    visibility = simplify((max_prob - min_prob) / (max_prob + min_prob))

    print("#" * 80)
    dump(mzi_hwp_angle, idler_lp_angle, signal_lp_angle)
    show(prob_simplified)  # symbolic probability
    dump(min_prob.evalf(5), max_prob.evalf(5), visibility.evalf(5))

    return prob_simplified, visibility


# Build phi+ state
psi_hh = TP(psi_b_H, psi_b_H)
psi_vv = TP(psi_b_V, psi_b_V)
phi_plus_state = (psi_hh + psi_vv) / sqrt(2)  # 16×1 column state
assert phi_plus_state.norm() == 1


hwp_ep = 0 # pi/8 seems to reproduce all the experimental data
lpi_ep = hwp_ep


def model_nominal_setup():
    ##############################################################
    # Model the entangled pair quantum eraser in the nominal eraser on & off conditions

    print("=" * 100)
    print()
    print("model_nominal_setup")

    # Common parameters
    initial_state = phi_plus_state
    mzi_hwp_angle = pi / 4 + hwp_ep  # swap H/V in the upper arm
    idler_lp_angle = pi / 2 + lpi_ep  # 90 degree = H

    # Proper settings, eraser on at 45
    prob, visibility = demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=pi / 4,  # 45 = pi/4 = Eraser on
    )
    # Proper settings, eraser off at 0
    demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=0,  # 0 = Eraser off
    )

    expected_prob = (1 - cos(delta)) / 8
    #assert prob.equals(expected_prob), f"Probability {prob} != expected {expected_prob}"

def model_2025_05_29_lab_session():
    ##############################################################
    # Model the entangled pair quantum eraser in the +/- pi/8 eraser on & off conditions

    print("=" * 100)
    print()
    print("model_2025_05_29_lab_session")

    # Common parameters
    initial_state = phi_plus_state
    mzi_hwp_angle = pi / 4 + hwp_ep  # swap H/V in the upper arm
    idler_lp_angle = pi / 2 + lpi_ep  # 90 degree = H

    # Proper settings, eraser on at 45
    prob, visibility = demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=pi/8,  # Eraser on
    )
    # Proper settings, eraser off at 0
    demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=-pi/8,  # Eraser off
    )

    expected_prob = (1 - cos(delta)) / 8
    #assert prob.equals(expected_prob), f"Probability {prob} != expected {expected_prob}"


def model_2025_05_23_lab_session():

    ##############################################################
    #
    # This matches the collected data from Friday's lab session where the
    # wrong Pump HWP and MZI/Idler LP settings were input into the
    # apparatus.
    #
    # Both interfere, with N_c_eraser_on = 1/2 N_c_eraser_off
    #

    # Common parameters
    initial_state = psi_vv  # Pump HWP was set to 45 => Pump @ 90/H => 0/V signals&idlers
    mzi_hwp_angle = pi / 4  # swap H/V in the upper arm
    idler_lp_angle = pi / 4  # This was set to 45deg instead of 90 deg

    # Misconfigured on Friday, "eraser on"
    demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=pi / 4,  # Eraser on
    )

    # Misconfigured on Friday, "eraser off"
    prob, visibility = demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=0,  # Eraser off
    )


def model_rotated_pairs():

    # Model what happens if pairs are rotated by -22.5 at their source.

    # This does NOT match the data collected on 2025-05-29, where
    # Signal LP @ -22.5 got V=0   on 2025-05-29, model says V=0.7 <= wrong
    # Signal LP @ +22.5 got V=0.5 on 2025-05-29, model says V=0.7 <= wrong

    # Nor the 2025-05-23 results
    # Signal LP angle @ 45 got V>0.5  on 2025-05-23 with that setting, model says V=1
    # Signal LP angle @  0 got V=0.48 on 2025-05-23 with that setting, model says V=0 <= wrong

    # The model is immune to rotating the pairs' polarizations. Regardless of pair rotation:
    # LP @ 45 model always gives V=1
    # LP @  0 model always gives V=0

    # ------------------------------------------------------------------
    # Rotated −π/8 (−22.5°) φ⁺ initial state
    #   • Original V (0°) → −22.5°
    #   • Original H (90°) →  67.5°
    # ------------------------------------------------------------------
    alpha = -pi / 8  # rotation angle

    # Single-photon polarisation basis rotated by −π/8
    H_rot_basis = cos(alpha) * H + sin(alpha) * V  # |H⟩ → 67.5°
    V_rot_basis = -sin(alpha) * H + cos(alpha) * V  # |V⟩ → −22.5°

    # Tensor-product basis vectors (b-path, rotated polarisation)
    psi_b_H_rot = TP(psi_b, H_rot_basis)
    psi_b_V_rot = TP(psi_b, V_rot_basis)

    # φ⁺ state in the rotated basis
    phi_plus_rotated_neg_pi_8 = (
        TP(psi_b_H_rot, psi_b_H_rot) + TP(psi_b_V_rot, psi_b_V_rot)
    ) / sqrt(2)

    assert simplify(phi_plus_rotated_neg_pi_8.norm()) - 1.0 < 1e-9, simplify(
        phi_plus_rotated_neg_pi_8.norm()
    )

    deg_45 = pi / 4
    deg_90 = pi / 2
    deg_22_5 = pi / 8
    assert deg_45.evalf() == math.radians(45)
    assert deg_90.evalf() == math.radians(90)
    assert deg_22_5.evalf() == math.radians(22.5)

    # Common parameters
    initial_state = phi_plus_rotated_neg_pi_8
    mzi_hwp_angle = deg_45
    idler_lp_angle = deg_90

    # Proper settings, eraser on at 45
    prob, visibility = demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=-deg_45,
    )
    # Proper settings, eraser off at 0
    demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=0,
    )


###


def model_lab6():

    print("=" * 100)
    print()
    print("model_lab6")

    psi_hh = TP(psi_b_H, psi_b_H)

    # Common parameters
    initial_state = psi_hh
    mzi_hwp_angle = pi / 4 + hwp_ep # swap H/V in the upper arm

    # Proper settings, eraser on at 45
    prob, visibility = demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=pi/4 + lpi_ep,   # 45 = pi/4 = Eraser on
        signal_lp_angle=pi / 2,
    )

    # Proper settings, eraser off at 90
    demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=pi/2 + lpi_ep,   # 90 = pi/2 = Eraser off
        signal_lp_angle=pi / 2,
    )

def model_unbalanced_pairs(percent_HH=80):
    """
    Model entangled pairs with unequal amplitudes: alpha|HH> + beta|VV>
    where alpha and beta are calculated based on the percentage of |HH> component
    to ensure the state is normalized.

    Parameters
    ----------
    percent_HH : float
        Percentage of the |HH> component (0 to 100)
    """

    print("=" * 100)
    print()
    print("model_unbalanced_pairs", percent_HH)

    # Calculate alpha and beta to ensure normalization: |alpha|^2 + |beta|^2 = 1
    # where |alpha|^2 = percent_HH/100
    alpha = sqrt(percent_HH / 100)
    beta = sqrt(1 - alpha**2)  # = sqrt((100-percent_HH)/100)

    # Build unbalanced state
    psi_hh = TP(psi_b_H, psi_b_H)
    psi_vv = TP(psi_b_V, psi_b_V)
    unbalanced_state = alpha * psi_hh + beta * psi_vv

    # Verify normalization
    assert (
        abs(unbalanced_state.norm() - 1) < 1e-9
    ), f"State not normalized: {unbalanced_state.norm()}"

    print("#" * 80)
    print(f"Testing unbalanced state with {percent_HH}% |HH>")
    print(f"alpha={alpha:.4f}, beta={beta:.4f}")
    print("#" * 80)

    # Common parameters
    initial_state = unbalanced_state
    mzi_hwp_angle = pi / 4 + hwp_ep # swap H/V in the upper arm
    idler_lp_angle = pi / 2 + lpi_ep # 90 degree = H

    # Proper settings, eraser on at 45
    prob, visibility = demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=pi / 4,  # 45 = pi/4 = Eraser on
    )

    # Proper settings, eraser off at 90
    demo_pair(
        initial_state,
        mzi_hwp_angle=mzi_hwp_angle,
        idler_lp_angle=idler_lp_angle,
        signal_lp_angle=pi / 2,  # 90 = Eraser off
    )


def model_mixed_idler_signals_V():
    """
    Density-matrix implementation of the mixed-idler case.

    The idler photon starts in the completely mixed polarisation state

        ρ_i = ½ (|H⟩⟨H| + |V⟩⟨V|) ,

    while the signal photon is prepared in |V⟩.  The optics are applied via

        ρ_out = U ρ_in U†

    and the coincidence probability is

        P = Tr[ Π ρ_out ]

    where Π projects the idler onto the b-path.
    """

    print("=" * 100)
    print()
    print("model_mixed_idler_signals_V")


    # ──────────────────────────────────────────────────────────
    # Input density matrix  ρ_in = |V⟩⟨V|_s ⊗ ½ I₂,i
    psi_s_V = psi_b_V
    ketbra_s_V = psi_s_V * psi_s_V.T                      # |V⟩⟨V|_s  (4×4)

    psi_i_H = psi_b_H
    psi_i_V = psi_b_V
    rho_i = (psi_i_H * psi_i_H.T + psi_i_V * psi_i_V.T) / 2  # ½ I₂  (4×4)

    rho_in = TP(ketbra_s_V, rho_i)                        # 16×16

    # ──────────────────────────────────────────────────────────
    # Optical settings (nominal quantum-eraser configuration)
    mzi_hwp_angle = pi / 4 + hwp_ep      # HWP_u swaps H/V
    idler_lp_angle = pi / 2 + lpi_ep     # LP_i at 90° (H)
    signal_lp_angle = 0         # Signal LP transmits V (0°)

    # Operators acting on the two photons
    P_signal = P_hat_prime_signal.subs(theta, signal_lp_angle)
    E_hat_idler = TP(
        I44,
        E_hat_prime.subs(vartheta, mzi_hwp_angle).subs(theta, idler_lp_angle),
    )

    total_op = E_hat_idler * P_signal                      # U = (E ⊗ I)·(P ⊗ I)

    # Propagate density matrix
    rho_out = simplify(total_op * rho_in * total_op.H)

    # ──────────────────────────────────────────────────────────
    # Coincidence projector  Π = I_s ⊗ |b⟩⟨b|_i
    projector_b_spatial = TP(psi_b * psi_b.T, I22)
    coincidence_proj = TP(I44, projector_b_spatial)

    # Detection probability  P = Tr[ Π ρ_out ]
    prob = simplify((coincidence_proj * rho_out).trace().rewrite(cos))

    # Visibility
    min_prob = minimum(prob, delta)
    max_prob = maximum(prob, delta)
    visibility = simplify((max_prob - min_prob) / (max_prob + min_prob))

    show(prob)
    dump(min_prob.evalf(5), max_prob.evalf(5), visibility.evalf(5))

    return prob, visibility


# model_rotated_pairs()
model_nominal_setup()
model_2025_05_29_lab_session()
model_mixed_idler_signals_V()
model_lab6()
#model_unbalanced_pairs(100)  # Try with 80% |HH> component
