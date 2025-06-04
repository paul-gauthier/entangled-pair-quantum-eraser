
# Consider a model for the N_i and N_c counts from an experiment:

# `f_before` the fraction of signal photons that are lost before they reach the eraser.
# The which way info on the signal has decohered into the environment.
# The idler can not self-interfere.
#
# f_before is a fixed attribute of the apparatus.

# `f_eraser` represents the fraction of signals which are absorbed/lost during
# the erasing operation. Their which way info has decohered into the environment.
# Their idler partner can not self-interfere.
#
# Erasing is performed with an LP at a specific angle.
# Typically the angles are 0, 45 or 90, all of which will absorb half the
# arriving Phi+ signals and transmit the other half. So f_eraser will usually
# be 0.5 in this case, but can be set to other constants for other experimental
# setups.
#
# `f_eraser` is a constant provided for each experiment, not fitted from data.

# `f_after` is the fraction of signal photons lost after the eraser before reaching
# the signal detector.
#
# f_after is a fixed attribute of the apparatus.

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
# phi can either be fixed across experiments or vary.
#
# Fixed when the data collection was interleaved so that all the datasets had their
# piezo 0 data collected, then all datasets had their piezo 2 data collected, etc.
#
# Varies if we are analyzing a series of experiments where data was collected at
# different times. Thermal variation can change the relative length of the MZI arms
# over time.

# `phi_c` is the extra phase seen only by coincidences
#
# phi_c is a fixed attribute of the apparatus.

# `R` is the rate of entangled pair production.
#
# R is a fixed constant that doesn't vary between experiments.

N__i = R * (
    # Signals lost before the eraser can't interfere, half go out each MZI port
    + 1/2 * f_before

    # Signals lost at the eraser can't interfere, half go out each MZI port
    + 1/2 * (1 - f_before) * f_eraser

    # Non-erased pairs can't interfere either, half go out each MZI port
    + 1/2 * (1 - f_before) * (1 - f_eraser) * (1 - e)

    # Signals that reached the eraser and were erased
    # ... their idlers will oscillate between all and none going out each MZI port
    + 1/2 * (1 - f_before * f_eraser) * e * (cos(delta + phi) + 1)
)

# signals that pass the eraser and reach the detector
N_c = R * (1 - f_before) * (1 - f_eraser) * (1 - f_after) * (

    # Non-erased pairs can't interfere, half go out each MZI port
    + 1/2 * (1 - e)

    # erased pairs oscillate between all and none going out each MZI port
    + 1/2 * e * (cos(delta + phi + phi_c) + 1)
)

# When fitting data to these models, we can use these approaches:
#
# • R cancels out of every visibility and cosine amplitude ratio, so
# it can only be inferred from the absolute level of the idler
# singles.  That is fine, but its value is irrelevant to everything
# else; you could fix R = 2 × mean(N_i) and make the fit more stable.
#
