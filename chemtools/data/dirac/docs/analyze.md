orphan  

<div class="index">

\*\*ANALYZE

</div>

# \*\*ANALYZE

Wave function analyzing tool.

This section is aimed to analyze the final Hartree-Fock wave function by
using one or more analysis modules. By default none of them is
activated.

<div class="index">

.PRIVEC

</div>

## .PRIVEC

Print vectors. Activates the `*PRIVEC` subsection.

<div class="index">

.MULPOP

</div>

## .MULPOP

Perform Mulliken population analysis Mulliken1955. Continues to
`*MULPOP` subsection.

<div class="index">

.PROJECTION

</div>

## .PROJECTION

Perform projection analysis Faegri2001 Activates the `*PROJECTION`
subsection.

<div class="index">

.DENSITY

</div>

## .DENSITY

Write density to a formatted file in Gaussian cube format. Activates the
`*DENSITY` subsection.

<div class="index">

.LOCALIZATION

</div>

## .LOCALIZATION

Localize orbitals using the Pipek-Mezey criterion Dubillard2006 .

Continues to the activated `*LOCALIZATION` subsection. Note that in the
present implementation this only works in $`C_1`$ symmetry.
