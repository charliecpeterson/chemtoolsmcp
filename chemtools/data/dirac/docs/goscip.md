orphan  

# GOSCIP

Let us remind first that DIRAC uses the so-called averaged SCF scheme,
which means that all the configurations belonging to the reference space
receive the same weight.

In the case when one (or few) of the states from the reference space is
(or are) more important then others, than the large open-shell space can
lead to dis-balanced treatment and therefore to higher SCF energies than
those obtained with a smaller reference space.

The "better" SCF configuration is that one giving the lowest total
energy. Upon suitable averaged open-shell SCF occupation one can perform
complete open-shell CI (COSCI) calculatons, one example is given below.

## Sample inputs

Calculate the spin-orbit components of the 2P ground state of the
fluorine atom using a COSCI wavefunction. This will represent the
"uncorrelated" result since the CI will merely give the energies of the
individual determinants in this simple case.

The same calculation can also be done automatically by specifying
`WAVE_FUNCTION_.RESOLVE` in the `**WAVE FUNCTION` section of the input.

The full calculation is given as input example of the *cosci_energy*
test:

    &GOSCIP NELEC=5, IPRNT=1 &END
    &POPANA THRESH=1.0D-4, DEGEN=1.0D-12, SELPOP=50.0 &END

**NOTE: In the developer's version the first line reads as**

    &GOSCIP NELACT=5, IPRNT=1 &END
    &POPANA THRESH=1.0D-4, DEGEN=1.0D-12, SELPOP=50.0 &END
