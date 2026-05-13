orphan  

<div class="index">

\*\*GRID

</div>

# \*\*GRID

The numerical integration scheme uses the Becke partitioning Becke1988a
of the molecular volume into atomic ones, for which the quadrature is
performed in spherical coordinates. Radial integration is carried out
using the scheme proposed by Lindh *et al.* Lindh2001, while the angular
integration is handled by a set of highly accurate Lebedev grids. Note
that the Becke partitioning scheme generally uses Slater-Bragg radii for
atomic size adjustments. For the heavier elements, with their more
variable oxidation states, this can lead to errors. If the user invokes
the `GRID_.ATSIZE` keyword, DIRAC tries to deduce relative atomic sizes
from available densities, if it can find coefficients.

The **radial integration** employs an exponential grid

``` math
r_k=e^{kh};\quad w_k = r_k;\quad S=\sum_{k=-\infty}^{\infty}=w_kf\left(r_k\right)
```

and is specified in terms of step size $`h`$, the inner point $`r_1`$
and the outermost point $`r_{k_H}`$, chosen to provide the required
relative precision $`R`$ (equal to the discretization error $`R_D`$.

<div class="index">

.RADINT

</div>

## .RADINT

Specify the maximum error in the radial integration.

*Default:*

    .RADINT
     1.0D-13

<div class="index">

.ANGINT

</div>

## .ANGINT

Specify the precision of the Lebedev angular grid. The angular
integration of spherical harmonics will be exact to the L-value given by
the user. By default the grid will be pruned. The highest implemented
value is 64.

*Default:*

    .ANGINT
     41

<div class="index">

.NOPRUN

</div>

## .NOPRUN

Turn off the pruning of the angular grid.

<div class="index">

.ANGMIN

</div>

## .ANGMIN

Specify the minimum precision of the Lebedev angular grid after pruning.

    .ANGMIN
     LEBMIN

Close to the nucleus, the precision of the angular grid will be less
than L-value given by `GRID_.ANGINT`, but it will not be less than the
integer LEBMIN. Note that giving a LEBMIN value greater than or equal to
the L-value given by `GRID_.ANGINT` is equivalent to turning the pruning
off.

*Default:*

    .ANGMIN
     15

<div class="index">

.ATSIZE

</div>

## .ATSIZE

Generate new estimates for atomic size ratios for use in the Becke
partitioning scheme. Relative sizes for a pair of atoms A and B are
calculated from their contribution to the large component density along
the line connecting A and B.

<div class="index">

.IMPORT

</div>

## .IMPORT

    .IMPORT
    numerical_grid

Import previously exported numerical grid.

For debugging you can also create your own grid file. The grid file is
formatted - in free format. A grid file for three points could look like
this:

    3
    0.1  0.1 0.1    1.0
    0.01 0.2 0.4    0.9
    9.9  9.9 9.0    0.8
    -1

The first integer is the number of points, then come the x-, y-, and
z-coordinate, and the weight for each point. The last line is a negative
integer. Works also in parallel.

You can export the DIRAC grid simply by copying back the file
"numerical_grid" from the scratch directory.

<div class="index">

.NOZIP

</div>

## .NOZIP

DIRAC will try to remove redundant grid points when symmetry is present.
Each reflection plane reduces the number of grid points by a factor of
two. With this keyword you can turn this default symmetry grid
compression off.

<div class="index">

.4CGRID

</div>

## .4CGRID

Include the small component basis in the generation of the DFT grid even
if you run a 1- or 2-component calculation.

This can be useful if you want to compare to a 4-component calculation
using the identical integration grid.

By default the small component is ignored in the grid generation when
running 1- or 2-component DFT calculations.

<div class="index">

.DEBUG

</div>

## .DEBUG

Very poor grid - corresponds to

    .RADINT
     1.0D-3
    .ANGINT
     10

<div class="index">

.COARSE

</div>

## .COARSE

Coarse grid - corresponds to

    .RADINT
     1.0D-11
    .ANGINT
     35

<div class="index">

.ULTRAFINE

</div>

## .ULTRAFINE

A better grid than default - corresponds to

    .RADINT
     2.0D-15
    .ANGINT
     64

<div class="index">

.INTCHK

</div>

## .INTCHK

Test the performance of the grid by computing the overlap matrix
numerically and analyzing the errors. In addition, the error matrix can
be printed.

*Default (no test):*

    .INTCHK
     0

*Error analysis:*

    .INTCHK
     1

*Print error matrix:*

    .INTCHK
     2
