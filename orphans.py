
# Consider a model for the N_i and N_c counts from an experiment:

# `f` the fraction of "orphan" idlers, which reach the idler detector but whose paired signal
# photon did not reach its detector. The which way info on the signal has decohered into
# the environment. The idler can not self-interfere.
#
# f is a fixed attribute of the apparatus.

# `e` is the fraction of idlers which are being erased in the experiment. This
# depends on the setting of the signal LP, but also on imperfections in the apparatus.
# In real experimental results, e will never be 1 or 0. Hopefully, when the eraser is on
# e will approach 1 and when it's off, it will approach 0. But pragmatically, it's not
# likely to be especially close to these extremes.
#
# e varies depending on specific experimental settings like the signal LP angle.

# `delta` is the phase difference in the MZI arms introduced by the piezo mirror stage.
#
# Each experiment sweeps a range of piezo mirror positions to collect counts at varying delta.

# `phi` is the offset to align delta with the phase of the other arm
#
# phi is different for each experiment, because thermal drift can alter the relative
# path lengths of the MZI arms.

# `phi_c` is the extra phase seen only by coincidences
#
# phi_c is a fixed attribute of the apparatus.

# `R` is the rate of entangled pair production.
#
# R is a fixed constant that doesn't vary between experiments.

N__i = R * (
    # Orphans can't interfere, half go out each MZI port
    + 1/2 * f

    # Non-erased pairs can't interfere either, half go out each MZI port
    + 1/2 * (1 - f) * (1 - e)

    # Non-orphaned, erased idlers will oscillate between all and none going out each MZI port
    + 1/2 * (1 - f) * e * (cos(delta + phi) + 1)
)

N_c = R * (1 - f) * ( # Only non-orphans get coincidences
    # Non-erased pairs can't interfere, half go out each MZI port
    + 1/2 * (1 - e)

    # erased pairs oscillate between all and none going out each MZI port
    + 1/2 * e * (cos(delta + phi + phi_c) + 1)

)
