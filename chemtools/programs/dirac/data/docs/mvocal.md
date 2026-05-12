orphan  

<div class="index">

\*MVOCAL

</div>

# \*MVOCAL

## Introduction

Modified virtual orbitals.

The virtual canonical Hartree-Fock orbitals are not the optimal choice
for correlated calculations in which the virtual space is truncated; the
lower orbitals tend to be diffuse since they see the *N* + 1 electron
system. One may exploit the freedom of rotation amongst the virtual
orbitals to obtain orbitals better suited for correlated calculations.
The default option in DIRAC when `WAVE_FUNCTION_.MVO` has been specified
is to generate the modified virtual orbitals by construction of a Fock
operator for the system with some electrons removed, as first proposed
by Bauschlicher Bauschlicher1980. The orbital string specified with
`MVOCAL_.VECMVO` specifies which occupied orbitals to include in the
construction of the modified Hartree-Fock potential for the virtual
orbitals.

*Note* that the resulting modified virtual orbitals are no longer
canonical and can *not* be used in MP2 calculations; they can be used in
the various CI and CC modules, and as start orbitals for `*KRMCSCF`.

## Keywords

<div class="index">

.PRINT

</div>

### .PRINT

Print level.

*Default:*

    .PRINT
     0

<div class="index">

.VECMVO

</div>

### .VECMVO

Read `orbital_strings` of orbitals to be included in the Hartree-Fock
potential when constructing the cationic Fock operator. Default is the
doubly-occupied molecular orbitals from the SCF.

<div class="index">

.INTFLG

</div>

### .INTFLG

Specify what two-electron integrals to include during the construction
of the modified virtual orbitals (default: `HAMILTONIAN_.INTFLG` under
`**HAMILTONIAN`).
