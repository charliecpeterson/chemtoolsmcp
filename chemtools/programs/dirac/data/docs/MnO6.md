orphan  

# The $`MnO_6`$ system

The $`MnO_6`$ compound is good example of the oxide crystal structure.

In the following we show how one can get converged molecular spinors (or
molecular orbitals) of its ground state using single determinant SCF
method.

We take into account two parameters for controling the SCF convergence -
the `SCF_.OPENFACTOR` and the `SCF_.OLEVEL` keywords.

Here the noAcAc abbreviation ("no active-active") means value of
OPENFAC=0, what is no active-active interation among open-shell spinors.
The wAcAc ("with active-active") means OPENFAC=1, that is full
active-active interation (or full open-shell contribution to the Fock
matrix).

For simplicity, we employ the two-component Hamiltonian, see the keyword
`HAMILTONIAN_.X2C`.

## The way to the convergence

With the aim to achieve the SCF convergence we investigated these three
ways:

1\. Starting from noAcAc we get the SCF convergence. We save the DFCOEF
file with molecular spinors. Input files to download are
`MnO6.x2c.bare_noacac.inp  <../../../../test/tutorial_MnO6/MnO6.x2c.bare_noacac.inp>`
and
`MnO6.crystal.mol   <../../../../test/tutorial_MnO6/MnO6.crystal.mol>`:
:

    pam --inp=MnO6.x2c.bare_noacac.inp  --mol=MnO6.crystal.mol --get "DFCOEF=DFCOEF.MnO6.noAcAc"

2\. Using AcAc with molecular spinors (the DFCOEF file) from the
previous SCF run the $`MnO_6`$ system does not converge with the default
DIIS accelerator. The new input file to download is
`MnO6.x2c.wacac.inp <../../../../test/tutorial_MnO6/MnO6.x2c.wacac.inp>`:
:

    pam --inp=MnO6.x2c.wacac.inp  --mol=MnO6.crystal.mol --put "DFCOEF.MnO6.noAcAc=DFCOEF"

3\. Using the AcAc interaction with the parameter of OLEVEL=0.2, and,
with the DFCOEF file from the point 1, we finally obtain the
convergence. The total SCF energy of the system is *-1617.5839185788545
a.u.* (see the corresponding output file,
`MnO6.x2c.wacac_olvlshift_MnO6.crystal.out <../../../../test/tutorial_MnO6/result/MnO6.x2c.wacac_olvlshift_MnO6.crystal.out>`)
The new DIRAC input file to download is
`MnO6.x2c.wacac.inp <../../../../test/tutorial_MnO6/MnO6.x2c.wacac_olvlshift.inp>`:
:

    pam --inp=MnO6.x2c.wacac_olvlshift.inp  --mol=MnO6.crystal.mol --put "DFCOEF.MnO6.noAcAc=DFCOEF" --get "DFCOEF=DFCOEF.MnO6.wAcAc"

This tutorial shows that it is important to try all possible means to
achieve the SCF convergence if the default setting is not working.
