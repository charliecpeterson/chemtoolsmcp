orphan  

<div class="index">

\*PROJECTION

</div>

# \*PROJECTION

The current solution may be projected down onto another set of
coefficients generated from the basis, e.g. one may project molecular
solutions down onto atomic solutions in order to evaluate atomic
contributions. The coefficients of the fragments are read simultaneously
and the overlap between them is taken into account Faegri2001 .

The projection analysis can be thought of as a Mulliken analysis based
on the atomic (fragment) orbitals; it is therefore very much less
sensitive to the basis set.

Normally, the fragments are calculated in the full molecular basis. This
is done by zeroing charges of the remaining atoms of the molecule in the
basis set input and adjusting occupation in the menu file. It is,
however, possible to calculate the fragments in the own basis (a subset
of the full molecular basis) using `PROJECTION_.OWNBAS`. The advantage
is faster calculations and conservation of atomic symmetry for atomic
fragments. To avoid working with symmetry-combinations of atomic centers
for the fragments, it may sometimes be advantageous to dump molecular
coefficients in *C*<sub>1</sub> symmetry using `GENERAL_.ACMOUT` under
`**GENERAL` and do the analysis without symmetry. When
`PROJECTION_.OWNBAS` is used, one then needs to calculate each atomic
type only once in its own basis in order to do the complete analysis.

<div class="index">

.VECPRJ

</div>

## .VECPRJ

For each fermion irrep, give an `orbital_strings` of orbitals to
analyze.

*Default:* Analyze the occupied electronic solutions.

<div class="index">

.VECREF

</div>

## .VECREF

First give number of fragments to project onto, and then for each
fragment give filename of MO coefficients and the number of
symmetry-independent nuclei in this fragment and for each fermion irrep,
give an `orbital_strings` of reference orbitals.

*Example:*

    .VECREF
    4
    AFH1XX
    1
    1
    AFH2XX
    1
    1
    AFX1XX
    1
    1..43
    AFX2XX
    1
    1..43

<div class="index">

.OWNBAS

</div>

## .OWNBAS

Calculate fragments in their own basis.

This keyword must be used with some care as the list of fragments is
assumed to be identical to that of symmetry independent centers.

<div class="index">

.C1READ

</div>

## .C1READ

Read C1 coefficients for fragments.

<div class="index">

.ATOMS

</div>

## .ATOMS

Analyze molecular orbitals in terms of atomic fragments. For each atomic
type give filename of AO coefficients and an orbital string (see
`orbital_strings`) of atomic orbitals to include. DIRAC assumes that the
atomic orbitals are calculated in their proper atomic basis *without*
symmetry, that is, using the `GENERAL_.ACMOUT` keyword and the resulting
DFACMO file.

*Example:*

    .ATOMS
    AFHXXX
    1
    AFXXXX
    1..43

<div class="index">

.POLREF

</div>

## .POLREF

If the polarization contribution is too big, the projection analysis
looses its meaning. By activating this keyword Intrinsic Atomic Orbitals
(IAOs), as [formulated](http://doi.org/10.1021/ct400687b) by Gerald
Knizia, are generated, thus eliminating completely the polarization
contribution.

<div class="index">

.PROTHR

</div>

## .PROTHR

Set threshold for absolute value of projection coefficients to be
printed.

*Default:*

    .PROTHR
     0.01

<div class="index">

.WGPOP

</div>

## .WGPOP

Split overlap densities according to weight of contributions.

<div class="index">

.PRINT

</div>

## .PRINT

Print level.

*Default:*

    .PRINT
     0
