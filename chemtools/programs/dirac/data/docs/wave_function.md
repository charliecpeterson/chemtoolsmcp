orphan  

<div class="index">

\*\*WAVE FUNCTION

</div>

# \*\*WAVE FUNCTION

## **Get the wave function**

This section allows the specification of which wave function module(s)
to activate. By default no modules are activated. To activate any of
these modules you must also specify `DIRAC_.WAVE FUNCTION` under
`**DIRAC`, otherwise this input is not read.

Note that the order below specifies the order in which the different
modules are called if you ask for more than one.

<div class="index">

.SCF

</div>

### .SCF

Activates the Hartree-Fock/Kohn-Sham module.

Specification of the SCF module can be given in the `*SCF` subsection.

If `HAMILTONIAN_.DFT` has been specified under `**HAMILTONIAN`, then a
Kohn-Sham calculation will be performed, otherwise a Hartree-Fock
calculation will be performed.

<div class="index">

.RESOLVE

</div>

### .RESOLVE

Resolve open-shell states: do a small CI calculation to get the
individual energies of the states present in an
average-of-configurations open-shell Hartree-Fock calculation (see
`*RESOLVE`).

<div class="index">

.COSCI

</div>

### .COSCI

Activates advanced COSCI method, see `*COSCI`.

<div class="index">

.MP2

</div>

### .MP2

Activates `*MP2CAL`.

<div class="index">

.MVO

</div>

### .MVO

Calculate modified virtual orbitals (see `*MVOCAL`). Default after
open-shell SCF is modified virtual orbitals based on the closed-shell
molecular orbitals. There is no default for closed-shell SCF.

<div class="index">

.MP2 NO

</div>

### .MP2 NO

Activates the `*MP2 NO` module to calculate MP2 natural orbitals.

<div class="index">

.RELCCSD

</div>

### .RELCCSD

Activates the `**RELCC` (and the `**MOLTRA` module to get 4-index
transformed integrals).

By default, molecular orbitals with orbital energy between -10 and +20
hartree (a.u.) are included, this can be modified in the `**MOLTRA`
section.

<div class="index">

.RELADC

</div>

### .RELADC

Activates the `RELADC` and calculates the single and double ionization
spectra by the (A)lgebraic (D)iagrammatic (C)onstruction ADC. Also
activates the `**MOLTRA` module to get 4-index transformed integrals.

<div class="index">

.POLPRP

</div>

### .POLPRP

Activates the `**POLPRP` module for calculation of the excitation
spectrum by the strict or extended second order (A)lgebraic
(D)iagrammatic (C)onstruction ADC. Also activates the `**MOLTRA` module
to get 4-index transformed integrals.

<div class="index">

.DIRRCI

</div>

### .DIRRCI

Activates the MOLFDIR CI module (and also the `**MOLTRA` module to get
4-index transformed integrals).

Specification of input for the MOLFDIR CI module is given in the
`DIRRCI` and `GOSCIP` sections.

By default, molecular orbitals with orbital energy between -10 and +20
hartree (a.u.) are included, this can be modified in the `**MOLTRA`
section.

<div class="index">

.LUCITA

</div>

### .LUCITA

Activates the `*LUCITA` (and the `**MOLTRA` module to get 4-index
transformed integrals).

By default, molecular orbitals with orbital energy between -10 and +20
hartree (a.u.) are included, this can be modified in the `**MOLTRA`
section.

<div class="index">

.EXACC

</div>

### .EXACC

Activates the `**EXACC` module, the new coupled cluster implementation
based on the ExaTensor library.

<div class="index">

.CASPT2

</div>

### .CASPT2

Activates the `*CASPT2` module. complete active space second-order
perturbation theory calculation.

## **Pre-SCF orbital manipulations**

<div class="index">

.REORDER MO

</div>

### .REORDER MO

Interchange initial molecular orbitals prior to the SCF-calculation. The
start orbitals from DFCOEF are read and reordered.

For each fermion irrep give the new order of orbitals.

*Example:*

    .REORDER MO'S
    1..8,10,9

<div class="index">

.ORBROT

</div>

### .ORBROT

Jacobi rotations between pairs of orbitals.

On the line following the keyword, give first the rotation angle, then
on the following line(s) for each fermion irrep, give an
`orbital_strings` of orbitals to rotate.

## **Post-SCF orbital manipulations**

<div class="index">

.POST SCF REORDER MO

</div>

### .POST SCF REORDER MO

Interchange converged molecular orbitals. The orbitals from DFCOEF are
read and reordered just before exiting the SCF subroutine.

For each fermion irrep give the new order of orbitals.

*Example:*

    .POST DHF REORDER MO'S
    1..8,10,9

<div class="index">

.PHCOEF

</div>

### .PHCOEF

Phase adjustment of coefficients DFCOEF: make the largest element of a
given orbital real and positive.

<div class="index">

.KRCI

</div>

### .KRCI

Activates the `*KRCI` module for the calculation of ground and excited
states at the relativistic CI level.

<div class="index">

.KRMCSCF

</div>

### .KRMCSCF

Activates the `*KRMCSCF` module for the optimization of ground and
excited states (in other than the ground state symmetry) at the
relativistic MCSCF level.

<div class="index">

.LAPLCE

</div>

### .LAPLCE

Activates the `*LAPLCE` module to compute weights for Laplace
transformation of orbital energy denominators with the algorithm of
Helmich-Paris. No subsequent calculations, only output of the Laplace
points and weights.
