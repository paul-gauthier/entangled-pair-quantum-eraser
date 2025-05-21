#!/usr/bin/env python

from sys import exit

from sympy import *
from sympy.physics.quantum import TensorProduct
import numpy as np
import matplotlib.pyplot as plt

from show import show
from dump import dump

TP = TensorProduct

psi_x = psi_b = Matrix([1, 0])
psi_y = psi_t = Matrix([0, 1])

H = Matrix([1, 0])
V = Matrix([0, 1])

show(psi_b)
show(psi_t)
show(H)
show(V)

# Q1

psi_b_H = TP(psi_b, H)
psi_b_V = TP(psi_b, V)
psi_t_H = TP(psi_t, H)
psi_t_V = TP(psi_t, V)
show(psi_b_H)
show(psi_b_V)
show(psi_t_H)
show(psi_t_V)



I22 = eye(2,2)
show(I22)

psi_b_D = (psi_b_H + psi_b_V) / sqrt(2)
psi_t_D = (psi_t_H + psi_t_V) / sqrt(2)


def add_exit_LP():
    ###############################
    # A linear polarizer set at angle theta
    theta = symbols("theta", real=True)

    P_hat = Matrix([
        [cos(theta)**2, cos(theta)*sin(theta)],
        [cos(theta)*sin(theta), sin(theta)**2]
    ])

    Diag_hat = P_hat.subs(theta, pi/4)
    H_out = show(Diag_hat * H, 2)
    V_out = show(Diag_hat * V, 2)

    D = (H+V)/sqrt(2)
    show(D, sqrt(2))
    show(D/sqrt(2), 2)

    # We should lost half the photons
    assert H_out.norm()**2 == H.norm()**2/2
    assert V_out.norm()**2 == V.norm()**2/2

    dump(H_out.norm()**2, H.norm()**2)

    P_hat_prime = TP(psi_x * psi_x.T, P_hat) + TP(psi_y * psi_y.T, I22)

    P_hat_prime_diag = P_hat_prime.subs(theta, pi/4)
    show(P_hat)
    show(P_hat_prime)
    show(P_hat_prime_diag)

    psi_b_H_prime = show(P_hat_prime_diag * psi_b_H, 2)
    psi_b_V_prime = show(P_hat_prime_diag * psi_b_V, 2)
    psi_b_D_prime = show(P_hat_prime_diag * psi_b_D, sqrt(2))
    exit()

    dump(psi_b_H_prime.norm()**2)
    dump(psi_b_V_prime.norm()**2)
    dump(psi_b_D_prime.norm()**2)

add_exit_LP()


def diag_projection_from_lab():
    ###############################
    # A linear polarizer set at angle theta


    E_hat_xy = Matrix([
        [1/sqrt(2), 1/sqrt(2), 0, 0],
        [1/sqrt(2), 1/sqrt(2), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    show(E_hat_xy)

    psi_b_H_prime = show(E_hat_xy * psi_b_H, sqrt(2))
    psi_b_V_prime = show(E_hat_xy * psi_b_V, sqrt(2))
    psi_b_D_prime = show(E_hat_xy * psi_b_D)

    show(psi_b_H)
    show(psi_b_V)
    show(psi_b_D, sqrt(2))
    show(sqrt(2)*psi_b_D)

    dump(psi_b_H_prime.norm()**2)
    dump(psi_b_V_prime.norm()**2)
    dump(psi_b_D_prime.norm()**2)

diag_projection_from_lab()
