orphan  

# DIRRCI:Sample inputs

Calculate the spin-orbit components of the 2P ground state of the
fluorine atom using a CISD wavefunction.

The full calculation is given as input example 5.cisd.energy REF? in the
test directory:

    &RASORB  NELEC=7, NRAS1=1,1, NRAS2=2*0,3,3, MAXH1=2, MAXE3=2  &END
    &CIROOT IREPNA=' 1Eu', NROOTS=3 &END
    &DIRECT  MAXITER=15 &END
