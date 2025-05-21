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


B_hat = Matrix([
    [1, I],
    [I, 1],
]) / sqrt(2)
show(B_hat, sqrt(2))

I22 = eye(2,2)
show(I22)

B_hat_prime = TP(B_hat, I22)
show(B_hat_prime, sqrt(2))


# Q2

def q2():
    a = show(B_hat_prime * psi_b_V, sqrt(2))

    # \frac{1}{\sqrt{2}} \left[\begin{matrix}0\\1\\0\\i\end{matrix}\right]

    b = show(1/sqrt(2) * (psi_b_V + I*psi_t_V), sqrt(2))

    assert a.equals(b)

#q2()

# Q3

M_hat = Matrix([
    [0, 1],
    [1, 0],
])

show(M_hat)
exit()
show(I22)

M_hat_prime = TP(M_hat, I22)
show(M_hat_prime)



# Q4

delta = symbols("delta", real=True)

A_hat = Matrix([
    [1, 0],
    [0, exp(I*delta)],
])

show(A_hat)

A_hat_prime = TP(A_hat, I22)
show(A_hat_prime)



# Q5
vartheta = symbols("vartheta", real=True)

W_hat_prime = Matrix([
    [cos(2*vartheta), sin(2*vartheta), 0, 0],
    [sin(2*vartheta), -cos(2*vartheta), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, -1]
])

show(W_hat_prime)


show(M_hat_prime * B_hat_prime, 2/sqrt(2))
show(A_hat_prime * M_hat_prime * B_hat_prime, 2/sqrt(2))
show(W_hat_prime * A_hat_prime * M_hat_prime * B_hat_prime, 2/sqrt(2))

Z_hat_prime = B_hat_prime * W_hat_prime * A_hat_prime * M_hat_prime * B_hat_prime

show(Z_hat_prime, 2)

# Q6

def q6():
    # For θ = π/4, from the question
    Z_hat_prime_pi_4 = Matrix([
        [I*exp(I*delta), I, -exp(I*delta), 1],
        [I, -I*exp(I*delta), 1, exp(I*delta)],
        [exp(I*delta), -1, I*exp(I*delta), I],
        [-1, -exp(I*delta), I, -I*exp(I*delta)]
    ]) / 2

    assert Z_hat_prime.subs(vartheta, pi/4).equals(Z_hat_prime_pi_4)

    show(Z_hat_prime.subs(vartheta, pi/4), 2)
    show(Z_hat_prime_pi_4, 2)

#q6()

# Q7

def q7():
    Z_hat_prime_vartheta0 = Z_hat_prime.subs(vartheta, 0)

    psi_b_V_prime = Z_hat_prime_vartheta0 * psi_b_V
    show(psi_b_V_prime, 2)

    psi_b_V_component = simplify(psi_b_V.dot(psi_b_V_prime))
    show(psi_b_V_component)


    P_b_V_abs_sq = abs(psi_b_V_component)**2
    show(P_b_V_abs_sq)

    P_b_V_complex_conj = psi_b_V_component * conjugate(psi_b_V_component)
    show(P_b_V_complex_conj)

    assert P_b_V_abs_sq.equals(P_b_V_complex_conj)


    P_b_V = P_b_V_abs_sq.rewrite(cos)
    show(P_b_V, 2)
    #P_b_V = trigsimp(P_b_V, method="fu")
    #show(P_b_V, 2)

    return P_b_V


#P_b_V_from_q7 = q7()


# Q8

def q8():
    psi_b_V_prime = Z_hat_prime * psi_b_V
    show(psi_b_V_prime, 2)

#q8()


# Q9

def q9():
    psi_b_V_component = psi_b_V.dot(Z_hat_prime * psi_b_V)
    show(psi_b_V_component)
    P_b_V = abs(psi_b_V_component)**2
    #P_x = psi_x_component * conjugate(psi_x_component)
    show(P_b_V, 4)
    P_b_V = P_b_V.rewrite(cos)
    show(P_b_V, 4)

    #P_b_V = trigsimp(P_b_V, method="fu")
    #show(P_b_V)

    return P_b_V

#P_b_V_from_q9 = q9()

# Q10

def q10():
    show(P_b_V_from_q9.subs(vartheta, pi/4))
    assert P_b_V_from_q9.subs(vartheta, 0).equals(P_b_V_from_q7)
#q10()


# Q11

psi_b_D = (psi_b_H + psi_b_V) / sqrt(2)
psi_t_D = (psi_t_H + psi_t_V) / sqrt(2)

def q11():
    print("#"*60)
    print("#"*60)

    show(psi_b_D, sqrt(2))

    psi_b_V_component = psi_b_V.dot(Z_hat_prime * psi_b_D)
    show(psi_b_V_component, 2*sqrt(2))

    ans = psi_b_V_component.subs(vartheta, pi/4)
    show(ans, 4)
    ans = abs(ans)**2
    show(ans, 4)

    return ans

q11_ans = q11()

# Did Q11 really want abs(<psi_b_D|Z_hat|psi_b_V>)**2 ?
def q11_alt():

    psi_b_D_component = psi_b_D.dot(Z_hat_prime * psi_b_H)
    show(psi_b_D_component, 2*sqrt(2))

    ans = psi_b_D_component.subs(vartheta, pi/4)
    show(ans, 4)
    ans = abs(ans)**2
    show(ans, 4)
    return ans

    show(q11_ans, 4)
    assert ans.equals(q11_ans)

    psi_t_D_component = psi_t_D.dot(Z_hat_prime * psi_b_H)
    show(psi_t_D_component, 2*sqrt(2))

    ans = psi_t_D_component.subs(vartheta, pi/4)
    show(ans, 4)
    ans = abs(ans)**2
    show(ans, 4)

q11alt_ans = q11_alt()


def horizontal_output_prob_without_LP():
    Z_hat_prime_45 = Z_hat_prime.subs(vartheta, pi/4)
    show(Z_hat_prime_45, 4)

    psi_Z_hat_prime_45 = show(Z_hat_prime_45 * psi_b_V)

    projector_b_spatial = TP(psi_b * psi_b.T, I22)
    #show(projector_b_spatial) # Optional: to see the projector matrix

    prob_b_spatial_qm = (psi_Z_hat_prime_45.H * projector_b_spatial * psi_Z_hat_prime_45)[0,0]
    show(prob_b_spatial_qm) # To display the symbolic probability
    prob_b_spatial_qm_simplified = simplify(prob_b_spatial_qm.rewrite(cos))
    show(prob_b_spatial_qm_simplified) # To display the simplified symbolic probability



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

    show(P_hat)
    show(P_hat_prime)

    show(P_hat_prime.subs(theta, pi/4))
    #exit()

    show(P_hat_prime.subs(theta, pi/4) * psi_b_H, 2)
    show(P_hat_prime.subs(theta, pi/4) * psi_b_V, 2)
    show(psi_b_D, sqrt(2))
    show(psi_b_D/sqrt(2), 2)

    show(P_hat_prime.subs(theta, pi/4) * psi_t_H)
    show(P_hat_prime.subs(theta, pi/4) * psi_t_V)


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

    return E_hat_prime, theta

E_hat_prime, theta = add_exit_LP()





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

    show(E_hat_xy * psi_b_H, 2)
    show(E_hat_xy * psi_b_V, 2)
    show(psi_b_D, sqrt(2))
    show(psi_b_D/sqrt(2), 2)

    show(E_hat_xy * psi_t_H)
    show(E_hat_xy * psi_t_V)

    psi_eraser = (E_hat_xy * psi_b_V)
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

    assert prob_psi_x_D_final.equals(q11_ans)

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

#diag_projection_from_lab()
