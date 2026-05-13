orphan  

<div class="index">

\*NMR

</div>

# \*NMR

This section gives directives for the calculation of NMR parameters.

If common gauge origin (CGO) is used, i.e. :LONDON not specified, then
the user can define the gauge origin used for the external magnetic
field with `HAMILTONIAN_.GAUGEORIGIN` or `HAMILTONIAN_.GO ANG` under
`**HAMILTONIAN`. If `NMR_.USECM` is specified then center-of-mass is
used as gauge origin. Default is (0, 0, 0).

**Advanced options**

<div class="index">

.LONDON

</div>

## .LONDON

Activate calculations of magnetic properties (NMR shielding constants
and magnetizabilities) with London atomic orbitals.

*Default:* Use conventional atomic orbitals.

<div class="index">

.USECM

</div>

## .USECM

Use the center of mass as the gauge origin.

<div class="index">

.USEBC

</div>

## .USEBC

Use the nuclei charge barycenter as the gauge origin.

<div class="index">

.INTFLG

</div>

## .INTFLG

Specify what two-electron integrals to include in the two-electron
London contributions to the magnetic field property gradient (default:
`HAMILTONIAN_.INTFLG` under `**HAMILTONIAN`).

<div class="index">

.NOTWO

</div>

## .NOTWO

Do not calculate the two-electron London contributions for the magnetic
field property gradient when London atomic orbitals are used.

<div class="index">

.NOONEI

</div>

## .NOONEI

Do not calculate the {H(0),T(B)} reorthonormalization terms for the
magnetic field property gradient when London atomic orbitals are used.

<div class="index">

.NOORTH

</div>

## .NOORTH

Do not calculate the {T(B),h(mK)} reorthonormalization contributions for
the expectation value term when London atomic orbitals are used.

<div class="index">

.SYMCON

</div>

## .SYMCON

Employ the symmetric connection for reorthonormalization terms when
using London atomic orbitals.

*Default:* Use the natural connection.

<div class="index">

.EXPPED

</div>

## .EXPPED

Keyword used in the DFT calculations only. The contributions to the
density perturbed by an external magnetic field in LAO basis ("direct"
LAO term and "reorthonormalization" term) are exported on files,
<span class="title-ref">pertden_direct_lao.FINAL</span> and
<span class="title-ref">pertden_reorth_lao.FINAL</span>.
