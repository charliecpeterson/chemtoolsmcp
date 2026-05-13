orphan  

<div class="index">

\*MP2 NO

</div>

# \*MP2 NO

## Introduction

MP2 natural orbitals module by S. Knecht and H. J. Aa. Jensen.

The MP2 calculation will produce the MP2 energy and the natural orbitals
for the density matrix through second order. The primary purpose of this
option is to generate good starting orbitals for CI or MCSCF but also CC
wave functions, but it may of course also be used to obtain the MP2
energy, perhaps with frozen core orbitals. For valence correlation
calculations it is recommended that the core orbitals are excluded via
the `MP2_NO_.ACTIVE` in order to obtain the appropriate correlating
orbitals as start for an MCSCF calculation. As the commonly used basis
sets do not contain correlating orbitals for the core orbitals and as
the core correlation energy therefore becomes arbitrary, a thoughtfully
chosen `MP2_NO_.ACTIVE` option can also be of benefit in MP2 energy
calculations. See also Ref. Jensen1988 for more details.

*Note*: the module works at present only for closed-shell Hartree-Fock
reference wave functions.

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

.ACTIVE

</div>

### .ACTIVE

Read `orbital_strings` of orbitals to be included in the MP2 calculation
when constructing the MP2 density and natural orbitals.

<div class="index">

.SEL NO

</div>

### .SEL NO

Select a minimal set of natural orbitals given as a string analog to the
`orbital_strings` defined as active orbitals in subsequent correlation
calculations. The numbers are: max. occupation, min occupation, safety
tolerance (here +-10%).

*Default*:

    .SEL NO
     NO-occ 1.99 0.001 0.1

<div class="index">

.SCHEME

</div>

### .SCHEME

Sets the integral transformation scheme for the `**MOLTRA` part.
*Default*: default scheme of MOLTRA (scheme 6):

    .SCHEME
     6

<div class="index">

.MULPOP

</div>

### .MULPOP

Perform a Mulliken population analysis (see `MP2_NO_.MULPOP`) of the MP2
natural orbitals. This can be useful for a comparison with a Mulliken
population analysis of the Hartree-Fock orbitals.

<div class="index">

.INTFLG

</div>

### .INTFLG

Specify what two-electron integrals to include during the construction
of the MP2 natural orbitals (default: `HAMILTONIAN_.INTFLG` under
`**HAMILTONIAN`).
