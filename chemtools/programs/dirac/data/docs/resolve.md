orphan  

<div class="index">

\*RESOLVE

</div>

# \*RESOLVE

## Introduction

Resolve open-shell states.

The individual electronic states of an average-of-configurations
calculation are resolved by a full CI in the small open-shell
manifold(s) using the `GOSCIP` module.

Note that even if this is a small CI diagonalization one needs in
principle to generate *all* AO integrals for the initial 4-index
transformation. The cost of calculation can be greatly reduced invoking
integral screening (controlled by `RESOLVE_.SCREEN`) in the 4-index
transformation, possibly also by leaving out at least SS integrals
(using `RESOLVE_.INTFLG`).

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

.INTFLG

</div>

### .INTFLG

Specify what two-electron integrals to include (default:
`HAMILTONIAN_.INTFLG` under `**HAMILTONIAN`).

<div class="index">

.SCREEN

</div>

### .SCREEN

Screening threshold in the 4-index transformation.

A negative number deactivates screening.

*Default:*

    .SCREEN
     1.0D-14

<div class="index">

.SCHEME

</div>

### .SCHEME

Choose integral transformation algorithm (PLEASE CLARIFY!)

*Default:*

    .SCHEME
     4
