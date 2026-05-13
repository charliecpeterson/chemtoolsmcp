orphan  

<div class="index">

\*\*DIRAC

</div>

# \*\*DIRAC

By default DIRAC only activates the generation of the basis set (e.g.
kinetic balance conditions for small component functions) and the
one-electron modules. Two-electron integrals over the atomic basis
functions will be calculated when needed by the job modules given below.

We recommend that you use the [Dalton](http://www.daltonprogram.org)
program package if you want to e.g. save the two-electron integrals to
disk for another purpose, this is not possible with DIRAC.

<div class="index">

.WAVE FUNCTION

</div>

## .WAVE FUNCTION

Activates the wave function module(s). This activates the reading of the
`**WAVE FUNCTION` section, where the desired wave function type(s) must
be specified. If you read in converged MO coefficients from the
CHECKPOINT file and you want to skip the SCF step and proceed directly
to properties or the post-SCF step, then a trick to do this is to
comment out (or remove) this keyword.

<div class="index">

.ANALYZE

</div>

## .ANALYZE

Activates the Hartree--Fock or Kohn--Sham analysis module. This
activates the reading of the `**ANALYZE` section.

<div class="index">

.PROPERTIES

</div>

## .PROPERTIES

Activates the property module (which will call the integral module for
property integrals). This activates the reading of the `**PROPERTIES`
section.

<div class="index">

.OPTIMIZE

</div>

## .OPTIMIZE

Activates the geometry optimization. This activates the reading of the
`*OPTIMIZE` subsection.

<div class="index">

.4INDEX

</div>

## .4INDEX

Explicitly activates the transformation of integrals to molecular
orbital basis. This activates the reading of the `**MOLTRA` section.

These transformed integrals are currently only used by the `**RELCC`,
`RELADC`, `**POLPRP`, `DIRRCI`, and `*LUCITA` modules, and if one of
these three modules are requested under `**WAVE FUNCTION`, then this
flag is automatically activated unless .NO4INDEX is specified in this
input module.

By default, molecular orbitals with orbital energy between -10 and +20
hartree are included, this can be modified in the `**MOLTRA` section.

<div class="index">

.NO4INDEX

</div>

## .NO4INDEX

Do not automatically activate integral transformation to molecular
orbital basis if any of `**RELCC`, `RELADC`, `**POLPRP`, `DIRRCI`, and
`*LUCITA` modules are requested under `**WAVE FUNCTION`.

This keyword is utilized when repeating correlated CC or CI calculations
(with different parameters for instance) based on saved files after the
integral transformation.

<div class="index">

.TITLE

</div>

## .TITLE

Title line (max. 50 characters). Example:

    .TITLE
     my first DIRAC calculation

<div class="index">

.INPTEST

</div>

## .INPTEST

Input test - no job modules are called, only verification of DIRAC input
files. It is often useful to start a new set of calculations with an
input test in order to check that input file processing is correct
before submitting your (long-term run) job.

<div class="index">

.ONLY INTEGRALS

</div>

## .ONLY INTEGRALS

Stop after the calculation of the one-electron integrals for the
Hamiltonian and the one-electron integrals specified under
`**INTEGRALS`. The integrals are written to disk.

<div class="index">

.XMLOUT

</div>

## .XMLOUT

Create the output xml-file containing selected data (in development).
