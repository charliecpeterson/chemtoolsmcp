orphan  

<div class="index">

\*LINEAR RESPONSE

</div>

# \*LINEAR RESPONSE

Linear Response module written by T. Saue and H. J. Aa. Jensen Saue2003.

## General control statements

<div class="index">

.PRINT

</div>

### .PRINT

Print level.

*Default:*

    .PRINT
     0

## Definition of the linear response function

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

.OPERATORS

</div>

### .OPERATORS

Specification of both the A and B operators (see
`one_electron_operators` for details).

<div class="index">

.EPOLE

</div>

### .EPOLE

Specification of electric Cartesian multipole operators of order L .
Specify order.

*Example:* Electric dipole operators

::  
.EPOLE 1

<div class="index">

.MPOLE

</div>

### .MPOLE

Specification of magnetic Cartesian multipole operators of order L .
Specify order. *Example:* Magnetic dipole operators

::  
.MPOLE 1

<div class="index">

.TRIAB

</div>

### .TRIAB

Enforce triangularity of response function.

``` math
\langle\langle A; B \rangle\rangle_\omega = \langle\langle B; A \rangle\rangle_\omega
```

Only one function is calculated.

*Default:* Deactivated.

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

.IMAGIN

</div>

### .IMAGIN

Employ imaginary frequencies.

<div class="index">

.ALLCMB

</div>

### .ALLCMB

Form all possible combinations even if imaginary.

*Default:* Deactivated.

<div class="index">

.UNCOUP

</div>

### .UNCOUP

Uncoupled calculation.

## Control variational parameters

<div class="index">

.OCCUP

</div>

### .OCCUP

For each fermion ircop give an `orbital_strings` of inactive orbitals to
include in the linear response calculation.

<div class="index">

.VIRTUA

</div>

### .VIRTUA

For each fermion ircop give an `orbital_strings` of virtual orbitals to
include in the linear response calculation.

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
     200

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

.REAXVC

</div>

### .REAXVC

Read solution vectors from file XVCFIL

    .REAXVC
    XVCFIL

*Default:* No restart on solution vectors. The file has to have six
characters. Make sure there is no blank character in front of the file
name.

For a restart on solution vectors it is useful to set

    .REAXVC
    XVCFIL
    .ITRINT
     0 0

otherwise LS-integrals (and SS-integrals) are switched on later and one
may first iterate away and then back to a possibly converged response
vector.

Often you have a converged SCF wave function along with a response
vector. In this case make sure that

    **DIRAC
    #.WAVE FUNCTION

is commented out. Make then also sure that you use the CHECKPOINT file
which has been obtained in the *same* calculation as the response vector
file. Otherwise you may observe more response solver iterations than
necessary.

<div class="index">

.XLRNRM

</div>

### .XLRNRM

Normalize trial vectors. Using normalized trial vectors will reduce
efficiency of screening. CLARIFY!

*Default:* Use un-normalized vectors.

## Analysis

<div class="index">

.ANALYZ

</div>

### .ANALYZ

The linear response function is obtained by contracting the property
gradient $`\boldsymbol{E}_{A}^{[1]}`$ associated with property $`A`$
with the solution vector $`\boldsymbol{X}_{B}`$ associated with property
$`B`$:

``` math
\langle\langle\hat{A};\hat{B}\rangle\rangle_{\omega_{b}}=\boldsymbol{E}_{A}^{[1]\dagger}\boldsymbol{X}_{B}
```

The indices of the two vectors run over orbital rotation indices, that
is, one occupied and one virtual index. This analysis shows the most
important contributions from orbital pairs to the dot product as well as
the most important occupied and virtual orbitals. More information about
these orbitals can then be gleaned from Mulliken population analysis.

<div class="index">

.ANATHR

</div>

### .ANATHR

This keyword adjusts the number of terms shown in the above analysis.
The threshold is in terms of percentage value to the total value of the
linear response function.

*Default:* $`2.0`$.

## Advanced/debug flags

<div class="index">

.E2CHEK

</div>

### .E2CHEK

Generate a complete set of trial vector which implicitly allows the
explicit construction of the electronic Hessian. Only to be used for
small systems !

<div class="index">

.ONLYSF

</div>

### .ONLYSF

Only call FMOLI in sigmavector routine: only generate one-index
transformed Fock matrix Saue2003.

<div class="index">

.ONLYSG

</div>

### .ONLYSG

Only call FMOLI in sigmavector routine: 2-electron Fock matrices using
one-index transformed densities Saue2003.

<div class="index">

.STERNHEIM

</div>

### .STERNHEIM

Set diagonal elements of orbital part of Hessian equal to

``` math
-2 m c^2
```

for rotations between occupied positive-energy and virtual
negative-energy orbitals.

*Default:* Deactivated.

<div class="index">

.STERNC

</div>

### .STERNC

(Sternheim complement) allows to separate basis set incompleteness from
the replacement of an inner sum over negative-energy orbitals only by
the full sum. In order to benefit from this functionality (only for
specialists !), you should run with print level 2 under properties.

Then you can do a sequence of calculations: 1) .SKIPEP 2) .STERNH 3)
.STERNC The diamagnetic contribution of 1) is the non-relativistic
expectation value, whereas 2) is the Sternheim approximation, that is
replacing orbital energy differences with

``` math
-2 m c^2
```

With no basis set incompleteness the sum of the diamagnetic contribution
2) and the paramagnetic contribution 3) should equal the diamagnetic
contribution of 1).

*Default:* Deactivated.

<div class="index">

.COMPRESSION

</div>

### .COMPRESSION

Reduce number of orbital variation parameters by checking corresponding
elements of gradient vector against a threshold. This may reduce memory.

*Default:* No compression.

    .COMPRESSION
     0.0

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
