orphan  

<div class="index">

\*ESR

</div>

# \*ESR

This section gives directives for the calculation of ESR/EPR parameters,
g-tensors and Hyperfine Coupling tensors (HFC).

## General control statements

<div class="index">

.PRINT

</div>

### .PRINT

Print level.

*Default:*

    .PRINT
     0

## Definition of the multiplicity and CI

<div class="index">

.MULTIP

</div>

### .MULTIP

    .MULTIP
     2

*Default:* Doublet for an odd number of electrons, triplet for an even
number of electrons.

<div class="index">

.CI ROOTS

</div>

### .CI ROOTS

    .CI ROOTS
     5

*Default:* The multiplicity.

<div class="index">

.ESR CI

</div>

### .ESR CI

Genreal specification of the ESR CI space with GAS, generalized active
spaces. For easier description they are divided in three groups:
RAS1-GAS spaces, defining max number of holes in the doubly occupied
orbitals in the reference wave function; RAS2-GAS, one "CAS" space with
the open shells from the reference wave functions; RAS3-GAS spaces,
defining max number of electrons in the empty (virtual) orbitals in the
refenece wave function.

*Default:* "RESOLVE", that is, CI in the open shell orbitals from the
reference wave function.

*Example:* S, SD, and SDT from the occupied orbitals, S and SD to the
empty orbitals. The RAS2-GAS space is defined implicitly by the
reference wave function, typically and average-of-configurations (AOC)
Hartree-Fock. The example is for a case with inversion symmetry, if no
inversion symmetry there is only one set orbitals in each GAS space.

    .ESR CI
     3 2        ! 3 RAS1-GAS spaces and 2 RAS3-GAS spaces
     1          ! max 1 hole in first RAS1-GAS space
     2 2        ! 2 g and 2 u orbitals in RAS1-GAS
     2          ! max 2 holes in second RAS1-GAS space
     4 2        ! 4 g and 2 u orbitals in RAS1-GAS
     3          ! max 3 holes in third and last RAS1-GAS space
     4 1        ! 4 g and 1 u orbitals in RAS1-GAS
     2          ! max 2 electrons in first RAS3-GAS space
     4 1        ! 4 g and 1 u orbitals in RAS3-GAS
     1          ! max 1 electron in second and last RAS3-GAS space
     4 1        ! 4 g and 1 u orbitals in RAS3-GAS

<div class="index">

.THRCI

</div>

### .THRCI

    .THRCI
     1.d-5

*Default:* 1.d-4 for ppt accuracy. Can also be modified with .G10PPM
keyword.

<div class="index">

.G10PPM

</div>

### .G10PPM

Converge CI for 10 ppm accuracy in g-tensor values, default is ppt
accuracy.

<div class="index">

.STATE

</div>

### .STATE

The first CI root belonging to the multiplet of quasi-degenerate states.
For example, if the ground state is a non-degenerate singlet state
(non-relativistically a spin singlet) followed by the triplet state you
want to study, then you should specify state no. 2 here.

*Default:*

    .STATE
     1

<div class="index">

.KR-CON

</div>

### .KR-CON

Use Kramers conjugation for odd-number of electrons. For a doublet state
this means the program only needs to converge one CI state instead of
two CI states - reduces the CI time with a factor 2.

## Definition of additional properties

The g-tensor and HFC-tensors for all selectred nuclei are always
calculated.

<div class="index">

.SELECT

</div>

### .SELECT

Select which nuclei to calculate HFC for. Default is *all*, which can be
quite time-consumig if the molecule has many atoms. (You cannot use
`INTEGRALS_.SELECT` under `**INTEGRALS` if you use this keyword, they do
the same thing. It is included here for convenience.) To select three
nuclei, no. 3, 7, and 8:

    .SELECT
     3
     3 7 8
