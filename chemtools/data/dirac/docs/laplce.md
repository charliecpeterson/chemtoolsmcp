orphan  

<div class="index">

\*LAPLCE

</div>

# \*LAPLCE

## Introduction

Laplace transformation module by B. Helmich-Paris.

The Laplace transform of orbital energies denominators can evaluated
numerically by quadrature procedures. This is an interesting
reformulation for perturbation-theory based approaches like Green's
functions methods, Møller-Plesset perturbation theory etc. Here the
minimax algorithm is employed, which is very fast and gives also a good
accuracy for vanishing HOMO-LUMO gaps. See also Ref. Takatsuka2008 and
HelmichParis2016 for more details.

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

.NUMPTS

</div>

### .NUMPTS

A fixed number of quadrature points given by the user. If not given the
number of quadrature points to reach a given accuracy TOLLAP is
estimated.

*Default*:

    .NUMPTS
     0

<div class="index">

.MXITER

</div>

### .MXITER

Maximum number iterations in the Newton-Maehly algorithm that searches
for roots and extrema of the numerical error distribution.

*Default:*

    .MXITER
      100

<div class="index">

.STEPMX

</div>

### .STEPMX

Maximum step length of a line search part of the Newton-Maehly algorithm
when far away from quadratic convergence region.

*Default*:

    .STEPMX
      0.3

<div class="index">

.TOLRNG

</div>

### .TOLRNG

Tolerance threshold for the Newton-Maehly procedure that determines the
extremum points. You should not touch that.

    .TOLRNG
     1.D-10

<div class="index">

.TOLPAR

</div>

### .TOLPAR

Tolerance threshold for the Newton procedure that computes the Laplace
parameters at each extremum point. You should not touch that.

    .TOLPAR
     1.D-15

<div class="index">

.XPNTS

</div>

### .XPNTS

You can start the iterative quadrature algorithm with some guess for the
exponents. This only works if you are close to the final exponents. Also
you have to provide the weights (WGHTS) and the number of quadrature
points (NUMPTS).

<div class="index">

.WGHTS

</div>

### .WGHTS

You can start the iterative quadrature algorithm with some guess for the
weights. This only works if you are close to the final weights. Also you
have to provide the exponents (XPNTS) and the number of quadrature
points (NUMPTS).
