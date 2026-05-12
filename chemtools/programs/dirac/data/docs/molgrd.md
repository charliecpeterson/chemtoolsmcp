orphan  

<div class="index">

\*MOLGRD

</div>

# \*MOLGRD

The section gives directives for control of a single point evaluation of
the molecular gradient activated with `PROPERTIES_.MOLGRD` under
`**PROPERTIES`.

The molecular gradient module is usually called from within the geometry
optimization module.

**Advanced options**

<div class="index">

.TRICK

</div>

## .TRICK

If activated the calculation of the SL and SS two-electron integral
contributions to the gradient is skipped if their contribution is
estimated to be small. An "empirical" estimate of the norms of the LS
and SS two-electron gradients based on the norm of the LL two-electron
gradient. If deactivated all contributions to gradient are calculated
and if activated then only "necessary" contributions are calculated.

*Default:* Deactivated.

<div class="index">

.INTFLG

</div>

## .INTFLG

Specify what two-electron integrals to include (default:
`HAMILTONIAN_.INTFLG` under `**HAMILTONIAN`).

<div class="index">

.SCREEN

</div>

## .SCREEN

Screening threshold.

*Default:*

    .SCREEN
     1.0D-17

Which means that screening is turned off by default. This is due to an
enormous demand of memory.

**Programmers options**

<div class="index">

.PRINT

</div>

## .PRINT

Print level.

A print level of less than 3 gives only the total gradient. Print levels
from 3+ gives individual contributions to the gradient (kinetic energy
gradient, nuclear attraction gradient etc.). Print levels of 5+ print
some matrices, and 10+ give massive output!

*Default:* Print level `GENERAL_.PRINT` from `**GENERAL`.

<div class="index">

.NUMGRA

</div>

## .NUMGRA

Calculate the molecular gradient using numerical differentiation.

*Default:* Analytical evaluation if implemented.
