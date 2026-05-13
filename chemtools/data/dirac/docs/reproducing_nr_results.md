orphan  

# How to reproduce nonrelativistic results

## DIRAC nonrelativistic Hamiltonians

To approach the nonrelativistic limit you can either switch to the
4-component Lévy-Leblond Hamiltonian:

    **DIRAC
    [...]
    **HAMILTONIAN
    .LEVY-LEBLOND

or the 2-component nonrelativistic (Schrodinger) Hamiltonian:

    **DIRAC
    [...]
    **HAMILTONIAN
    .NONREL

## Changing speed of light

Another option to reproduce nonrelativistic data with the Dirac
relativistic Hamiltonian is to increase the speed of light (here to 2000
a.u.):

    **DIRAC
    [...]
    **GENERAL
    .CVALUE
    2000.0

## Adjusting the nucleus model

Most nonrelativistic programs employ a point nucleus model. Point
nucleus is default for `.NONREL` and Gaussian nucleus is default for
`` `.LEVY-LEBLOND ``. You can always force point nuclei in DIRAC:

    **DIRAC
    [...]
    **INTEGRALS
    .NUCMOD
     1

## Basis sets

With the above mentioned nonrelativistic Hamiltonians the user can also
employ variety of standard nonrelativistic basis sets present in the
"basis_dalton" directory.

Altogether, to reproduce results of most nonrelativistic packages the
user should set nonrelativistic Hamiltonian, switch to point nucleus if
necessary for the compatibility, and choose some nonrelativistic basis
set.

Recommended method of choice is the closed shell Hartree-Fock SCF, which
can be followed by the CCSD(T) correlation method. In this case you are
to check the occupied shells, and active space, respectively.

# An example using DALTON

Apart from specifying a nonrelativistic Hamiltonian and a point charge
nuclear model in the DIRAC input, the user should be aware of the fact
that DIRAC uses Cartesian Gaussian functions, while DALTON uses
Spherical Gaussian functions.

One route is to force DALTON to use Cartesian Gaussian functions (refer
to the DALTON manual) and the following DIRAC input file:

    **DIRAC
    [...]
    **GENERAL
    .SPHTRA
     0 0
    **INTEGRALS
    .NUCMOD
    1
    **HAMILTONIAN
    .LEVY-LEBLOND
    *END OF

A second route is to use DALTON defaults regarding basis functions and
the following DIRAC input file:

    **DIRAC
    [...]
    **INTEGRALS
    .NUCMOD
    1
    **HAMILTONIAN
    .LEVY-LEBLOND
    *END OF

The user should make sure that the geometry is specified in atomic units
as the conversion factor from Angstrom to AU may be different in the two
programs.
