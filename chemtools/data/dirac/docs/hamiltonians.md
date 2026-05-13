orphan  

# Selecting a two-component Hamiltonian other than X2C

There are several two-component Hamiltonians besides the X2C Hamiltonian
(see `HAMILTONIAN_.X2C`) implemented in DIRAC arising from the
decoupling transformation of the one-electron DIRAC Hamiltonian.
Likewise spin-orbit interaction terms can be left out and calculations
can be performed in the spin-free mode (i.e. in boson symmetry).

Besides having spin-orbit effects from the decoupling transformation,
there is the external AMFI spin-orbit operator which can contribute
either with the one-electron spin-orbit operator (accurate to the first
order in V, very poor) or with the more valuable mean-field
contribution, u_so(1) (also in the first order), which is a reasonable
approximation of the 'Gaunt' term at the four-component level.

The best two-component Hamiltonian is X2C+MFSSO (similar to BSS+MFSSO or
IOTC+MFSSO) where one-electron scalar and spin-orbit effects are up to
infinite order, and AMFI MFSSO contributions (mean-field spin-same
orbit) provide a 'screening' of one-electron spin-orbit terms.

## Examples

Both scalar and spin-orbit relativistic effects up to infinite order:

    .BSS
     099

Scalar relativistic effects of type "from the beginning" up to the
infinite order, no spin-orbit interaction:

    .SPINFREE
    .BSS
     109

Scalar relativistic effects "from the end" up to the infinite order:

    .SPINFREE
    .BSS
     009

Traditional scalar relativistic "from the beginning" second-order
Douglas-Kroll-Hess Hamiltonian:

    .SPINFREE
    .BSS
     102

Scalar relativistic "from the end" second-order Douglas-Kroll-Hess
Hamiltonian:

    .SPINFREE
    .BSS
     002

Douglas-Kroll-Hess Hamiltonian with first-order spin-orbit and
second-order scalar relativistic effects:

    .BSS
     012

Douglas-Kroll-Hess Hamiltonian with second-order spin-orbit and
second-order scalar relativistic effects:

    .BSS
     022

In the two-component variational scheme is possible to combine external
AMFI spin-orbit terms, Ilias2001, together with BSS integrals - see the
`\*AMFI` section. Note that AMFI provides only one-center atomic
integrals.

Infinite order scalar and spin-orbit terms, (one-electron) mean-field
spin-same-orbit (MFSSO2) from AMFI:

    .BSS
     2999

This is in fact the 'maximum' of two-component relativity, resembling
Dirac-Coulomb Hamiltonian.

Second-order Douglas-Kroll-Hess spin-free from 'the beginning' with
first order spin-orbit (SO1) term plus mean-field spin-same-orbit
(MFSSO) from AMFI:

    .BSS
     2112

SO1 may come either from BSS-transformation or from AMFI. In the latter
case it is only for one-center and for point nucleus.

Infinite order scalar and spin-orbit terms, (one-electron) mean-field
spin-same and spin-other-orbit (MFSO2) from AMFI:

    .BSS
     3999

This mimics the Dirac-Coulomb-Gaunt Hamiltonian, as the spin-other-orbit
(SOO) term comes from the Gaunt interaction term.

DIRAC allows to switch off AMFI spin-orbit contributions from various
centers. See the keyword `AMFI_.NOAMFC`.

Infinite order scalar terms "from the beginning", (one-electron) spin
orbit (SO1) and the mean-field spin-same-orbit (MFSSO2) terms
exclusively from AMFI:

    .BSS
     4109

Second order (DKH) scalar terms "from the beginning", (one-electron)
spin orbit (SO1) and the mean-field spin-same-orbit (MFSSO2) terms
exclusively from AMFI:

    .BSS
     4102

Infinite order scalar terms "from the end", (one-electron) spin orbit
(SO1) and the mean-field spin-same-orbit and spin-other-orbit (MFSO2)
terms from AMFI:

    .BSS
     5009

Infinite order scalar terms "from the end", and the (one-electron) spin
orbit term (SO1) from AMFI:

    .BSS
     6009

AMFI one-electron spin-orbit terms - SO1 - are currently for the point
nucleus.

For the BSS value 'axyz' of y=0, DIRAC employs spin-free picture change
transformation of property operators, although the system is not in the
boson (spin-free) symmetry for a\>1.

Rough first order, DKH1 (not recommended for practical calculations):

    .BSS
     111

For comparison purposes between BSS-SO1 and AMFI-SO1 atomic one-center
integrals (point nucleus only) use:

    .BSS
     001
