orphan  

<div class="index">

\*AMFI

</div>

# \*AMFI

> [!NOTE]
> We highly recommend to use the `\*XAMFI` module rather than the
> present \*AMFI module as highlighted by the numerical examples
> provided in Knecht2022.

Specification of the AMFI, RELSCF directives.

The AMFI program of B.Schimmelpfennig serves for calculating spin-orbit
contributions to various quasirelativistic two-component Hamiltonians.
Its important part is the independent scalar atomic code - RELSCF (named
as AT34), which is providing atomic orbital coefficients for the AMFI
mean-field summation.

The user may control not only the extent of the output, but also provide
here some parameters for the RELSCF code. Note that closed/open-shell
HF-SCF code gives both the nonrelativistic and the spin-free
second-order Douglass-Krol-Hess SCF energies, which can be compared with
DIRAC results.

RELSCF is using only decontrated basis sets restricted to occupied
shells only; for example, if you set some spdf-basis for a DIRAC
BSS+AMFI run of Magnesium (12 electrons), RELSCF will use, for the sake
of computational effectivity, this basis set with only s- and p-
exponents.

<div class="index">

.PRNT_A

</div>

## .PRNT_A

Print level for AMFI calculations.

*Default:*

    .PRNT_A
     0

<div class="index">

.PRNT_S

</div>

## .PRNT_S

Print level for the RELSCF part.

*Default:*

    .PRNT_S
     0

<div class="index">

.MXITER

</div>

## .MXITER

Maximum number of iterations in RELSCF.

*Default:*

    .MXITER
     50

<div class="index">

.AMFICH

</div>

## .AMFICH

Set up an artificial charge of the calculated system which has effect
only on the AMFI mean-field summation(s). This does not change the
electronic occupation for the DIRAC SCF procedure.

The reason is that AMFI-attached scalar relativistic Hartree-Fock SCF
module might not be converging for certain high-spin open-shell atoms
(like Pt,Yb), what is anyway expected for a single-determinant code. A
small positive charge (+1, +2) usually helps to get converged molecular
orbitals of given atom with a negligible change in results due to AMFI
mean-field contribution. The AMFI mean-field contributions are large for
core shells, but small for valence orbitals.

In the case of polyatomic systems the overall AMFI-charge is
proportionally distributed over individual atoms and rounded to the
closest integer value.

Example:

    .AMFICH
    +1

The user can check the mean-field occupation/charge of each handled atom
in the AMFI/RELSCF output, together with the RELSCF convergence status.

<div class="index">

.NOAMFC

</div>

## .NOAMFC

Discard selected centers from AMFI calculations. Related the
`HAMILTONIAN_.BSS` keyword, whose numerical value *axyz* must have y=0
(i.e. 'spin-free' from the DIRAC BSS side, while AMFI provides SO terms
of first order in potential).

First line specifies how many atoms are neglected, the second line gives
centers to be omitted. Number for each center of calculated system is
determined in the DIRAC output file.

Infinite order scalar terms "from the end", and the (one-electron) spin
orbit term (SO1) from AMFI. Centers 1 and 3 are neglected.

    .BSS
    5009
    .NOAMFC
    2
    1  3

<div class="index">

.ONLSCF

</div>

## .ONLSCF

Stop DIRAC calculation after RELSCF program has finished. The
(converged) AO coefficients are written to the file *RELSCF_COEF* and
can be used to restart from.

This may be useful in combination with the keyword `AMFI_.AMFICH`,
*i.e.* the user may first run the RELSCF step for an atom with an
artificial charge.

The obtained coefficients can then be used to start the RELSCF step for
the neutral atom for which otherwise one would not obtain convergence.

Input example:

    .ONLSCF
