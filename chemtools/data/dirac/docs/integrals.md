orphan  

<div class="index">

\*\*INTEGRALS

</div>

# \*\*INTEGRALS

## General directives

<div class="index">

.NUCMOD

</div>

### .NUCMOD

Specify nuclear model.

Point nucleus:

    .NUCMOD
     1

Gaussian charge distribution (default):

    .NUCMOD
     2

The point nucleus model is useful to compare Lévy-Leblond type
calculations with regular nonrelativistic calculations done with another
code, e.g. [Dalton](http://www.daltonprogram.org). The two methods
should give precisely the same energies.

For the Gaussian charge distribution the default exponents are in
accordance with values proposed by Visscher and Dyall Visscher1997b.

<div class="index">

.SELECT

</div>

### .SELECT

Restrict range of nuclei in one-electron integrals involving single
atomic centers, for example electric field gradients.

Example: Restrict range to four nuclei (first number), the nuclei 1, 3,
7, and 8:

    .SELECT
     4
     1
     3
     7
     8

<div class="index">

.MAGCOR

</div>

### .MAGCOR

Print the symmetrized nuclear magnetic moments. This corresponds to
taking symmetry combinations of rotations, not coordinates, at each
nuclear center. The numbering is used in labels of various magnetic
integrals.

<div class="index">

.PRINT

</div>

### .PRINT

General print level in the integral module. Default:

    .PRINT
     1

<div class="index">

\*ONEINT

</div>

# \*ONEINT

## One-electron integrals

This subsection gives directives for the generation of one-electron
integrals. Based on input in the other section, the program will
determine what integrals to calculate.

<div class="index">

.PRINT

</div>

### .PRINT

Print level in one-electron integral routines. By default print level is
taken from `**INTEGRALS`.

<div class="index">

\*READIN

</div>

# \*READIN

## The mol file

This subsection allows changes of defaults in the reading of the mol
file.

<div class="index">

.UNCONTRACT

</div>

### .UNCONTRACT

Decontract basis sets specified as contracted (when being read from the
library, for example). In the case of using non-relativistically
contracted basis set the decontraction is necessary for heavy elements.
Two-component quasirelativistic Hamiltonians (like X2C) work only with
decontracted basis set. By default only the small component is
decontracted.

<div class="index">

.PRINT

</div>

### .PRINT

Print level in the reading of the mol file. By default print level is
taken from \*\*INTEGRALS.

<div class="index">

.MAXPRI

</div>

### .MAXPRI

Maximum number of primitive functions in a given block in basis file.
Default:

    .MAXPRI
     35

<div class="index">

\*TWOINT

</div>

# \*TWOINT

## Two-electron integrals

This subsection gives directives for the generation of two-electron
integrals. It also gives directives for the construction of Fock
matrices, such as screening.

<div class="index">

.SCREEN

</div>

### .SCREEN

Screening threshold for integral direct calculations of Fock matrices
Saue1997. Default:

    .SCREEN
     1.0D-12

Note that the screening threshold may influence the convergence. In
general, the screening threshold must be about three orders of magnitude
smaller than the desired norm of the electronic gradient at convergence.

Choosing a negative value for the screening turns the integral screening
completely off.

<div class="index">

.ICEDIF

</div>

### .ICEDIF

Separate screening of Coulomb and exchange contributions Saue1997.
Useful for fine regulation of the convergence process. Default: Coulomb
and exchange on (1 = on; off = 0):

    .ICEDIF
     1 1

<div class="index">

.THRFAC

</div>

### .THRFAC

Adjust the integral thresholds for SL and SS integrals. For conventional
integral calculations only integrals above the threshold given in the
mol file are written to disk. The thresholds for the SL and SS integrals
are divided by the factors given here. Default:

    .THRFAC
     1.0 1.0

<div class="index">

.AOFOCK

</div>

### .AOFOCK

Do direct Fock matrix construction in non-symmetry-adapted basis (AO
basis).

The direct Fock matrix construction is performed in AO basis using the
skeleton matrix approach. This may give better screening, and does give
more tasks for better parallelization, but AOFOCK is more memory
intensive than SOFOCK. Default is AOFOCK if 25 or more MPI nodes.

<div class="index">

.SOFOCK

</div>

### .SOFOCK

Do direct Fock matrix construction in symmetry-adapted basis (SO basis).
This is the default setting.

AOFOCK may give better screening, and does give more tasks for better
parallelization, but AOFOCK is more memory intensive than SOFOCK.
Default is SOFOCK if at most 24 MPI nodes.

<div class="index">

.PRINT

</div>

### .PRINT

Set the print level in two-electron integral routines for the
calculation of a particular shell quadruplet. The print level is changed
only for the given shell quadruplet. A zero matches all shells, thus:

    .PRINT
     4 0 0 0 0

or just:

    .PRINT
     4

sets the print level to 4 for all shell quadruplets.

Use with care to avoid massive output! At print level 15 the individual
integrals are printed.

<div class="index">

.TIME

</div>

### .TIME

Give detailed timing for integral calculation.
