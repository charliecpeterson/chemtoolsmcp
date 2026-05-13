orphan  

<div class="index">

\*PCM

</div>

# \*PCM

Polarizable continuum model (PCM) directives.

General information on the PCM can be found in Tomasi2005, whereas
details concerning the Dirac implementation are found in DiRemigio2015.

> [!WARNING]
> Calculations with the polarizable continuum model **cannot** exploit
> molecular point group symmetry and/or 2-component Hamiltonians.

## **General**

<div class="index">

.SKIPSS

</div>

### .SKIPSS

Performs a PCM-SCF calculation with the PCM-SCC approximation. The
small-small block of the electrostatic potential is neglected, in the
spirit of the SCC approximation described in Visscher1997a. It is not
employed by default.

> [!WARNING]
> Currently not working for response calculations.

<div class="index">

.PRINT

</div>

### .PRINT

Print level in the PCM subroutines. Default:

    .PRINT
     0

## **Advanced/Debug**

<div class="index">

.SEPARA

</div>

### .SEPARA

Performs PCM-SCF separating the nuclear and electronic electrostatic
potential and apparent surface charge. Results are unaffected. *For
debug purpose only*

<div class="index">

.DOSPF

</div>

### .DOSPF

Remove spin-orbit dependent part from PCM potential. *For debug purpose
only*

<div class="index">

.SKIPOI

</div>

### .SKIPOI

Skips the calculation of the one-index transformed apparent surface
charge in the linear response function. *For debug purpose only*
