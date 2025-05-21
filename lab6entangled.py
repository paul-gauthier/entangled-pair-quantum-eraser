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


B_hat = Matrix([
    [1, I],
    [I, 1],
]) / sqrt(2)
#show(B_hat, sqrt(2))

I22 = eye(2,2)
#show(I22)

B_hat_prime = TP(B_hat, I22)
#show(B_hat_prime, sqrt(2))

M_hat = Matrix([
    [0, 1],
    [1, 0],
])

#show(M_hat)

M_hat_prime = TP(M_hat, I22)
#show(M_hat_prime)


delta = symbols("delta", real=True)

A_hat = Matrix([
    [1, 0],
    [0, exp(I*delta)],
])

#show(A_hat)

A_hat_prime = TP(A_hat, I22)
#show(A_hat_prime)



# Q5
vartheta = symbols("vartheta", real=True)

W_hat_prime = Matrix([
    [cos(2*vartheta), sin(2*vartheta), 0, 0],
    [sin(2*vartheta), -cos(2*vartheta), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

#show(W_hat_prime)
#show(M_hat_prime * B_hat_prime, 2/sqrt(2))
#show(A_hat_prime * M_hat_prime * B_hat_prime, 2/sqrt(2))
#show(W_hat_prime * A_hat_prime * M_hat_prime * B_hat_prime, 2/sqrt(2))

Z_hat_prime = B_hat_prime * W_hat_prime * A_hat_prime * M_hat_prime * B_hat_prime

show(Z_hat_prime, 2)



psi_b_D = (psi_b_H + psi_b_V) / sqrt(2)
psi_t_D = (psi_t_H + psi_t_V) / sqrt(2)

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
#show(D, sqrt(2))
#show(D/sqrt(2), 2)

# We should lost half the photons
assert H_out.norm()**2 == H.norm()**2/2
assert V_out.norm()**2 == V.norm()**2/2

#dump(H_out.norm()**2, H.norm()**2)

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


E_hat_prime = P_hat_prime * Z_hat_prime
show(E_hat_prime, 2)

E_hat_prime_45_45 = E_hat_prime.subs(vartheta, pi/4).subs(theta, pi/4)
show(E_hat_prime_45_45, 4)

psi_eraser = (E_hat_prime_45_45 * psi_b_H)
show(psi_eraser, 4)

psi_x_D_final = psi_b_D.dot(psi_eraser)
show(psi_x_D_final, 4)
prob_psi_x_D_final = abs(psi_x_D_final)**2
prob_psi_x_D_final = simplify(prob_psi_x_D_final.rewrite(cos))
show(prob_psi_x_D_final)
min_final = minimum(prob_psi_x_D_final, delta)
max_final = maximum(prob_psi_x_D_final, delta)
show(min_final)
show(max_final)
dump(min_final.evalf(3))
dump(max_final.evalf(3))
dump(max_final.evalf(3)-min_final.evalf(3))

assert prob_psi_x_D_final.equals(q11alt_ans)

psi_t_H_final = abs(psi_t_H.dot(psi_eraser))**2
show(psi_t_H_final, 4)

psi_t_V_final = abs(psi_t_V.dot(psi_eraser))**2
show(psi_t_V_final, 4)

# Projector onto the 't' spatial state, identity in polarization space
# psi_t is Matrix([0, 1])
# I22 is eye(2,2)
projector_t_spatial = TP(psi_t * psi_t.T, I22)
show(projector_t_spatial) # Optional: to see the projector matrix

# Probability = <psi_eraser | P_t_spatial | psi_eraser>
# .H gives the Hermitian conjugate (bra)
# The result of the product is a 1x1 matrix, so access its element [0,0]
prob_t_spatial_qm = (psi_eraser.H * projector_t_spatial * psi_eraser)[0,0]
show(prob_t_spatial_qm) # To display the symbolic probability

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

# Simplify the complementary probability expression for plotting
prob_t_spatial_qm_simplified = simplify(prob_t_spatial_qm.rewrite(cos))

min_prob_b = minimum(prob_b_spatial_qm_simplified, delta)
max_prob_b = maximum(prob_b_spatial_qm_simplified, delta)
show(min_prob_b)
show(max_prob_b)
dump(min_prob_b.evalf(3))
dump(max_prob_b.evalf(3))
dump((max_prob_b - min_prob_b).evalf(3))
