orphan  

# Molecular mean-field X2C

The molecular-mean-field X2C Hamiltonian (X2Cmmf) is **available** for
the following methods:

- Coupled Cluster (CC): module RELCCSD
- multi-reference (MR) CC: Fock-space CC (including the intermediate
  Hamiltonian IHFSCC version): module RELCCSD

The remaining post-SCF modules are **not supported** yet:

- KR-MCSCF
- KR-CI
- TD-DFT

The X2Cmmf Hamiltonian scheme can be used for the Dirac-Coulomb as well
as the Dirac-Coulomb-Gaunt Hamiltonian (add .GAUNT to \*\*HAMILTONIAN).

## One-step X2Cmmf

In order to perform a two-component (MR)CC after a four-component SCF
calculation add the keyword .X2Cmmf to the `**HAMILTONIAN` keyword
section:

    **HAMILTONIAN
    .X2Cmmf

After the SCF calculation Dirac will carry out the transformation to
two-component prior to the 4-index transformation of the two-electron
integrals and the CC run.

## Two-step X2Cmmf

Besides the one-step X2Cmmf protocol as described above one may also
proceed in seperate steps. This may be useful if one would like to use
less/more MPI processes in the SCF+decoupling step than in the
4-index/CC calculation. The first step (SCF+decoupling) simply requires
the .X2Cmmf keyword under the \*\*HAMILTONIAN input deck (same as
above):

    **HAMILTONIAN
    .X2Cmmf

Save the files DFCOEF, AOMOMAT and X2CMAT, for example using the *pam*
script:

    ./pam ... --outcmo --get="AOMOMAT X2CMAT"

To restart now any further calculation, for example the 4-index
transformation (MOLTRA) and/or the (MR)CC run copy the above files to
your scratch directory:

    ./pam ... --incmo --put="AOMOMAT X2CMAT"

and set the keyword combination as indicated below under the
\*\*HAMILTONIAN input deck:

    **HAMILTONIAN
    .X2C
    *X2C
    .mmf-restart

In order to run a (MR)CC calculation add also the following keyword(s)
to the namelist RELCCSSD input:

    *CCSORT 
    .USEOE
    .NORECMP
    END

or the new RELCCSD deck input:

    *CCSORT
    .NORECMP

Dirac will automatically read the two-component coefficients from the
file DFCOEF and proceed in two-component mode for any wave function
analysis or post-HF step required for a (MR)CC calculation.
