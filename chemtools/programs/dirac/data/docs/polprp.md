orphan  

# Kramers restricted Polarization Propagator in the ADC framework

See Pernpointner2014 for the initial POLPRP implementation. The
transition moment implementation and parallelization is described in
Pernpointner2017.

<div class="index">

\*\*POLPRP

</div>

# \*\*POLPRP

Specification of the overall calculation. Davidson diagonalizer (see
DAVIDSON section) is automatically invoked.

<div class="index">

.STATES

</div>

## .STATES

Specifies the symmetries of the excited final states to be calculated.
If you leave out this keyword all symmetries are checked for the
occurrence of possible final states and those are calculated in the
respective perturbational order. If the keyword is present, states have
to be specified according to the following input example

*Input example:*

    .STATES
    4
    1,3,17,19

This determines calculation for all eigenstates in the symmetries
(irreducible representations) 1,3,17 and 19. Keep in mind that the
numbering scheme for the double group ireps is a bit unconventional and
depends on the type of Hamiltonian used.

<div class="index">

.DOEXTE

</div>

## .DOEXTE

Presence of this keyword triggers an extended ADC(2) calculation in the
sense that first-order contributions are included in the 2p2h
(satellite) block. This often helps to localize states having a sizeable
amount of double excitation character because those will alter their
energy considerably when calculated in the extended scheme.

<div class="index">

.WRITET

</div>

## .WRITET

Determines the threshold for writing matrix elements into the ADC matrix
(default = 0.0). Can save considerable numerical effort with negligible
loss of accuracy. Thresholds of 1.0E-06 are in order but it also depends
a bit on the system. Value follows the keyword.

<div class="index">

.NODIAG

</div>

## .NODIAG

Diagonalization is skipped if keyword is present and only the
eigenvectors of the particle-hole block are written to disk. This is
mainly for test purposes since getting full eigenvectors is required for
the calculation of transition moments.

<div class="index">

.DOTRMO

</div>

## .DOTRMO

Transition moments are calculated if keyword is present, otherwise no
TMs are calculated.

<div class="index">

.PRINT

</div>

## .PRINT

Printlevel (default = 0) number follows the keyword. More output is
produced if print level \> 0.

<div class="index">

\*\*DAVIDSON

</div>

# \*\*DAVIDSON

Specifications for the Davidson diagonalizer. Holds for POLPRP and for
other propagators if activated. POLPRP relies on Davidson exclusively
and does not address other diagonalizers such as Lanczos.

<div class="index">

.DVROOTS

</div>

## .DVROOTS

Number of desired roots. Remember that Davidson is an iterative
diagonalizer that will not be able to generate thousands of roots.
Normally one asks for 10 to 50 roots at the spectral boundary that will
be well converged. This is a typical situation occurring for the valence
excitation spectra.

<div class="index">

.DVMAXSP

</div>

## .DVMAXSP

This is the maximum size of the Krylov space generated internally and
where the full Hamiltonian is projected on, in other words, this is the
dimension of your trial space. If the dimension is reached during the
iterations one macroiteration cycle is finalized and the subspace will
be collapsed to the number of roots. The resulting new set of vectors is
already a much better approximation to the true eigenvectors entering
the next macrocycle.

<div class="index">

.DVMAXIT

</div>

## .DVMAXIT

Number of macrocycles you want to allow. For large problems requiring a
lot of memory, DVMAXSP can be reduced leading to more macrocycles. If
convergence is still unsatisfactory one slowly increases DVMAXSP.

<div class="index">

.DVCONV

</div>

## .DVCONV

Requested convergence of the eigenvalues. Usually a convergence of
1.0E-06 is absolutely sufficient.

<div class="index">

.DVREORT

</div>

## .DVREORT

Force reorthogonalization of ADCEVEC.XX start vectors initially and in
each macrocycle. This may be useful if eigenvectors of a previous
incomplete run that are not perfectly orthogonal shall be reused. This
is unnecessary in a regular run since the trial spece during a Davidson
run is kept strictly orthogonal and serves as the construction space for
the true eigenvectors of the Hamilton matrix.

<div class="index">

.SKIPCC

</div>

## .SKIPCC

With this keyword you can restart the calculation at a very advanced
stage after the sorting step of the two-electron integrals. Therefore
you skip SCF, MOLTRA and integral sorting (OOOO, VOOO, ...). This is if
a POLPRP run only needs to be executed. Works in serial and parallel.
The only files you should provide are the ftXX.YY files containing the
sorted integral batches, MDPROP (needed for transition moments) and
MRCONEE among the .inp and .mol file. In parallel run this works only if
you have an underlying shared file system, so each node can access their
own ftXX.nodenumber file.

The default values are

::  
DVROOTS 5 DVMAXSP 200 DVMAXIT 5 DVCONV 1.0E-05 DVREORT = .false. SKIPCC
= .false.
