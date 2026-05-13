orphan  

# Magnetizabilities with London Atomic Orbitals

## Introduction

In this tutorial we will look at the calculation of (static)
magnetizabilities with DIRAC. For more details about the method, see
Ilias2013.

The component $`\zeta_{\alpha\beta}`$ of the static $`3\times 3`$
magnetizability tensor describes connects component $`\alpha`$ of
first-order induced magnetic dipole to component $`\beta`$ of the
external inducing homogeneous magnetic field

``` math
m^{(1)}_\alpha = \sum_\beta \zeta_{\alpha\beta} B_{\beta} = \frac{1}{2}\int\left(\mathbf{r}_G\times\mathbf{j}^{B_\beta}\right)_{\alpha}d\tau
```

The first-order induced magnetic dipole is proportional to the direct
product of the position vector $`\textbf{r}_G`$ with repect to some
arbitrary (gauge) origin $`G`$ and the first-order induced current
density $`\mathbf{j}^{\mathbf{B}}`$. Although the static induced
magnetic dipole is formally independent of the gauge-origin this does
not hold true in the finite basis approximation, where excruciating slow
convergence of the magnetizability with respect to basis set is
typically observed. London Atomic Orbitals (LAOs), also known as Gauge
Including Atomic Orbitals (GIAOs), remove any reference to the arbitrary
gauge origin by shifting the gauge origin to the centers of individual
basis functions and effectively cure both problems.

## Example: $`NF_3`$

We shall illustrate these features using the nitrogen trifluoride
molecule.

![alternate text](NF3.jpg)

The molecular input file <span class="title-ref">NF3.mol</span> uses the
experimental geometry and automatic symmetry detection.

<div class="literalinclude">

NF3.mol

</div>

Upon input one of the nitrogen atom is at the origin with one of the N-F
bonds aligned with the z-axis. DIRAC detects the full symmetry
$`C_{3v}`$. but uses the Abelian subgroup $`C_s`$.

<div class="literalinclude">

NF3sym.txt

</div>

The molecule is furthere centered at the center of mass with the
$`C_3`$-axis aligned with the z-axis; this is seen from the xyz-file
given in the output

<div class="literalinclude">

NF3geo.txt

</div>

### Generating the HF wave function

We first run a Hartree-Fock calculation to generate orbitals and
corresponding energies using <span class="title-ref">scf.inp</span>

<div class="literalinclude">

scf.inp

</div>

and the command:

    pam --inp=scf --mol=NF3 --outcmo

You may notice in the output that DIRAC will center and rotate the
molecule. It detects the full $`C_{3v}`$ symmetry, but will run the
calculation in the lower $`C_s`$ symmetry, with reflection in the *xy*
-plane, with the $`C_3`$ rotation around the *x* -axis.

### Magnetizabilities: first attempt

We first calculate the magnetizability using the input file
<span class="title-ref">cgo.inp</span>

<div class="literalinclude">

cgo.inp

</div>

where the (common) gauge origin has been set to the center of mass using
the `NMR_.USECM` keyword. Notice that we use `GENERAL_.RKBIMP` (?
`PROPERTIES_.RKBIMP` ) to convert our molecular coefficients from
restricted to unrestricted kinetic balance (RKB $`\rightarrow`$ UKB),
the former employed for the generation of orbitals, and the latter
employed for the response calculations, indicated by
`HAMILTONIAN_.URKBAL`. This corresponds to the use of [simple magnetic
balance](../simple_magnetic_balance/tutorial.html). The calculation is
run using:

    pam --inp=cgo --mol=NF3 --incmo

and gives the total magnetizability tensor

<div class="literalinclude">

cgo_NF3_total

</div>

here reported in atomic units $`e^2a_0^2/m_e`$, corresponding to 7.89104
$`\cdot 10^{-29}`$ J/T.

### Magnetizabilities using LAOs

We now activate LAOs using the input file
<span class="title-ref">lao.inp</span>

<div class="literalinclude">

lao.inp

</div>

which gives the total magnetizability tensor

<div class="literalinclude">

lao_NF3_total

</div>

which is markedly different from the CGO. The question now is: Which
result is 'best' ? From microwave spectroscopy (see Stone1969) the
magnetizability anisotropy, defined as

``` math
\zeta_{ani} = \zeta_{\perp} - \zeta_\parallel
```

has been found to be -0.63 $`(\pm 0.32)\ e^2a_0^2/m_e`$. With CGO and
LAOs we obtain +0.4421 and -0.6457 $`e^2a_0^2/m_e`$, respectively,
clearly favoring the LAO calculation. However, it very often happens
that one gets the right answer for the wrong reason, so let us
investigate the gauge-origin independence of the result as well as basis
set convergence.

### Gauge-origin dependence

To investigate gauge-origin dependence we do a CGO and LAO calculation
with the gauge origin placed along the $`C_3`$ axis:

<div class="literalinclude">

shift.inp

</div>

Please note that when shifting the gauge origin in this manner you
should limit the gauge origin to symmetry-independent points, that is,
you should stay on symmetry elements like the *xz* mirror plane in this
case. The CGO calculation now gives

<div class="literalinclude">

cgo2_NF3_total

</div>

We see that the parallel component $`\zeta_\parallel`$ is unchanged,
whereas the perpendicular component $`\zeta_\perp`$ is dramatically
different. With LAOs we get

<div class="literalinclude">

lao2_NF3_total

</div>

where the numerical differences with respect to the original calculation
is below the convergence threshold of the linear response calculation.
We can therefore see that the use of LAOs removes the gauge dependence
in the finite basis approximation.

### Basis-set convergence

The basis set convergence is illustrated by the following table, taken
from Ilias2013, showing CGO(LAO) magnetizabilities (in $`e^2a_0^2/m_e`$)
for a wide range of basis sets.

<table style="width:97%;">
<colgroup>
<col style="width: 10%" />
<col style="width: 25%" />
<col style="width: 21%" />
<col style="width: 20%" />
<col style="width: 20%" />
</colgroup>
<tbody>
<tr>
<td><blockquote>
<p>Basis</p>
</blockquote></td>
<td><span class="math inline"><em>ζ</em><sub>∥</sub></span></td>
<td><span class="math inline"><em>ζ</em><sub>⟂</sub></span></td>
<td><span
class="math inline"><em>ζ</em><sub><em>i</em><em>s</em><em>o</em></sub></span></td>
<td><span
class="math inline"><em>ζ</em><sub><em>a</em><em>n</em><em>i</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p>DZ</p>
</blockquote></td>
<td>-14.83 (-4.27)</td>
<td>-9.78 (-4.87)</td>
<td>-11.47 (-4.67)</td>
<td>+5.04 (-0.60)</td>
</tr>
<tr>
<td><blockquote>
<p>TZ</p>
</blockquote></td>
<td><blockquote>
<p>-7.88 (-4.36)</p>
</blockquote></td>
<td>-6.52 (-4.96)</td>
<td><blockquote>
<p>-6.97 (-4.76)</p>
</blockquote></td>
<td>+1.37 (-0.60)</td>
</tr>
<tr>
<td><blockquote>
<p>QZ</p>
</blockquote></td>
<td><blockquote>
<p>-5.65 (-4.43)</p>
</blockquote></td>
<td>-5.54 (-5.04)</td>
<td><blockquote>
<p>-5.58 (-4.83)</p>
</blockquote></td>
<td>+0.11 (-0.61)</td>
</tr>
<tr>
<td><blockquote>
<p>aug-DZ</p>
</blockquote></td>
<td><blockquote>
<p>-6.75 (-4.57)</p>
</blockquote></td>
<td>-6.31 (-5.22)</td>
<td><blockquote>
<p>-6.46 (-5.00)</p>
</blockquote></td>
<td>+0.44 (-0.65)</td>
</tr>
<tr>
<td><blockquote>
<p>aug-TZ</p>
</blockquote></td>
<td><blockquote>
<p>-5.01 (-4.64)</p>
</blockquote></td>
<td>-5.42 (-5.24)</td>
<td><blockquote>
<p>-5.28 (-5.04)</p>
</blockquote></td>
<td>-0.41 (-0.60)</td>
</tr>
<tr>
<td><blockquote>
<p>aug-QZ</p>
</blockquote></td>
<td><blockquote>
<p>-4.70 (-4.64)</p>
</blockquote></td>
<td>-5.26 (-5.24)</td>
<td><blockquote>
<p>-5.08 (-5.04)</p>
</blockquote></td>
<td>-0.56 (-0.60)</td>
</tr>
<tr>
<td>d-aug-DZ</td>
<td><blockquote>
<p>-6.37 (-4.57)</p>
</blockquote></td>
<td>-6.18 (-5.22)</td>
<td><blockquote>
<p>-6.24 (-5.00)</p>
</blockquote></td>
<td>+0.19 (-0.65)</td>
</tr>
<tr>
<td>d-aug-TZ</td>
<td><blockquote>
<p>-5.01 (-4.69)</p>
</blockquote></td>
<td>-5.45 (-5.27)</td>
<td><blockquote>
<p>-5.31 (-5.08)</p>
</blockquote></td>
<td>-0.44 (-0.58)</td>
</tr>
<tr>
<td>d-aug-QZ</td>
<td><blockquote>
<p>-4.76 (-4.68)</p>
</blockquote></td>
<td>-5.31 (-5.28)</td>
<td><blockquote>
<p>-5.13 (-5.08)</p>
</blockquote></td>
<td>-0.55 (-0.60)</td>
</tr>
</tbody>
</table>

The basis set convergence is seen to be dramatically different, again
clearly in favour of LAOs.
