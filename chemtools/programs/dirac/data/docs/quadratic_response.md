orphan  

<div class="index">

\*QUADRATIC RESPONSE

</div>

# \*QUADRATIC RESPONSE

This section gives directives for the calculation of quadratic response
functions Saue2002a.

## General control statements

<div class="index">

.PRINT

</div>

### .PRINT

Print level.

*Default:*

    .PRINT
     0

## Definition of the quadratic response function

<div class="index">

.DIPLEN

</div>

### .DIPLEN

Specification of dipole operators for A, B, and C (see
`one_electron_operators` for details).

<div class="index">

.A OPERATOR

</div>

### .A OPERATOR

Specification of the A operator (see `one_electron_operators` for
details).

<div class="index">

.B OPERATOR

</div>

### .B OPERATOR

Specification of the B operator (see `one_electron_operators` for
details).

<div class="index">

.C OPERATOR

</div>

### .C OPERATOR

Specification of the C operator (see `one_electron_operators` for
details).

<div class="index">

.B FREQ

</div>

### .B FREQ

Specify frequencies of operator B.

*Example:* 3 different frequencies.

    .B FREQ
     3
     0.001
     0.002
     0.01

*Default:* Static case.

    .B FREQ
     1
     0.0

<div class="index">

.C FREQ

</div>

### .C FREQ

Specify frequencies of operator C (see `QUADRATIC_RESPONSE_.B FREQ`).

<div class="index">

.ALLCMB

</div>

### .ALLCMB

Evaluate all nonzero quadratic response functions and thereby
disregarding analysis of overall permutational symmetry.

*Default:* Evaluate only unique, nonzero, response functions.

## Excited state properties

**This page describes unreleased functionality. The keywords may not be
available in your version of DIRAC.**

First order properties of excited states can be computed from the
quadratic response function.

<div class="index">

.EXCPRP

</div>

### .EXCPRP

Give the number of "left" and "right" states in each boson symmetry.

*Example*:

    .EXCPRP
    5 5 5 5
    0 0 0 0

Compute the excited state expectation values \|\langle
i\|\hat{A}\|i\rangle[\|, where i goes from 1 to 5 in each symmetry (four
symmetries in this case). The zeros can be substituted with positive
integers to generate transition state moments \\\\langle
i\\\\hat{A}\\j\\rangle\|](##SUBST##|,
where i goes from 1 to 5 in each symmetry (four symmetries in this case). The
zeros can be substituted with positive integers to generate transition state moments \|\\langle i\|\\hat{A}\|j\\rangle|).

## Control variational parameters

<div class="index">

.SKIPEE

</div>

### .SKIPEE

Exclude all rotations between occupied positive-energy and virtual
positive-energy orbitals.

<div class="index">

.SKIPEP

</div>

### .SKIPEP

Exclude all rotations between occupied positive-energy and virtual
negative-energy orbitals.

## Control reduced equations

<div class="index">

.MAXITR

</div>

### .MAXITR

Maximum number of iterations.

*Default:*

    .MAXITR
     30

<div class="index">

.MAXRED

</div>

### .MAXRED

Maximum dimension of matrix in reduced system.

*Default:*

    .MAXRED
     100

<div class="index">

.THRESH

</div>

### .THRESH

Threshold for convergence of reduced system.

*Default:*

    .THRESH
     1.0D-5

## Control integral contributions

The user is encouraged to experiment with these options since they may
have an important effect on run time.

<div class="index">

.INTFLG

</div>

### .INTFLG

Specify what two-electron integrals to include (default:
`HAMILTONIAN_.INTFLG` under `**HAMILTONIAN`).

<div class="index">

.CNVINT

</div>

### .CNVINT

Set threshold for convergence before adding SL and SS integrals to
SCF-iterations.

*2 (real) Arguments:*

    .CNVINT
     CNVXQR(1) CNVXQR(2)

*Default:* Very large numbers.

<div class="index">

.ITRINT

</div>

### .ITRINT

Set the number of iterations before adding SL and SS integrals to
SCF-iterations.

*Default:*

    .ITRINT
     1 1

## Control trial vectors

<div class="index">

.XQRNRM

</div>

### .XQRNRM

Normalize trial vectors. Using normalized trial vectors will reduce
efficiency of screening.

*Default:* Use un-normalized vectors.

## Advanced/debug flags

<div class="index">

.NOPREC

</div>

### .NOPREC

No preconditioning of initial trial vectors.

*Default:* Preconditioning of trial vectors.

<div class="index">

.RESFAC

</div>

### .RESFAC

New trial vector will be generated only for variational parameter
classes whose residual has a norm that is larger than a fraction
1/RESFAC of the maximum norm.

*Default:*

    .RESFAC
     1000.0
