orphan  

# Introduction

In order to run a DIRAC calculation you need an input file of arbitrary
name, e.g. `dirac.inp` which will be setup in the following and which
contains the computational directives. In addition, a second file is
needed providing the molecular structure. DIRAC can handle two formats
but for convenience we here focus on the standard **XYZ** molecular
input file, named for example `molecule.xyz`. The run-command inside
your cluster run script looks then as follows:

    $ pam --noarch --inp=dirac.inp --mol=molecule.xyz

where the DIRAC output is re-directed to the file `dirac_molecule.out`.
Make sure that you rename the output file with a proper name if you want
to take a look at what you did back home.

More information on running DIRAC calculations can be found in the
`FirstSteps` section of this documentation.

If not stated otherwise we will always use a Gaussian nuclear model in
DIRAC.

> [!NOTE]
> It is important that you understand the inputs. Before running
> calculations, carefully go through the input with reference to the
> manual that you find on the [DIRAC pages](http://www.diracprogram.org)

> [!NOTE]
> You are expected to make a report presenting your results, also
> including some theoretical background as well as computational
> details.

# Atomic calculations

> [!NOTE]
> **Exercise A** In these exercises we look at relativity in atomic
> systems. Three columns of the periodic table will be explored: 1. From
> group 2: Mg, Sr and Ra 2. From group 12: Zn, Cd and Hg 3. From group
> 18: Ne, Kr and Rn
>
> The exercises are: 1. Perform 4-component HF-calculations on your
> series of three atoms. We focus on the *outermost* $`p`$ orbitals: i)
> How does spin-orbit splitting vary as a function of nuclear charge Z ?
> ii) How do spin-orbit interaction affect orbital sizes
> $`<r^2>^{1/2}`of spin-orbit partners down the series ?
> 2. Perform 4-component *spin-free* and *non-relativistic* HF-calculations on the heaviest atom of your series in order to investigate the effects of spin-orbit coupling and scalar relativistic effects. i) Make a table showing orbital energies and orbital sizes of all orbitals for each of the three Hamiltonians. ii) What orbitals contract (stabilize) and expand (destabilize) ? iii) Plot the densities of the *outermost* :math:`p`$
> orbitals with the three Hamiltonians. 3. Perform 2-component
> relativistic HF-calculations on the heaviest atom of your series using
> the following Hamiltonians: i) X2C ii) second-order Douglas-Kroll-Hess
> iii) ZORA and iv) scaled ZORA. Make a table of energies of all
> orbitals obtained with these Hamiltonians comparing with those
> obtained with the 4-component Dirac-Coulomb Hamiltonian. Helpful
> information for these exercises is given below !

We shall start gently by looking at atoms. It will be useful to recall
the form and notation for atomic orbitals. In the non-relativistic case
atomic orbitals have the general form

``` math
\psi_{n\ell m}\left(\boldsymbol{r}\right)=R_{n \ell}\left(r\right)Y_{\ell m}\left(\theta,\phi\right)
```

where appears the principal quantum number $`n`$, the azimuthal quantum
number $`\ell`$ (orbital angular momentum) and the magnetic quantum
number $`m`$. Orbitals are indicated using the letter code
$`s,p,d,f,\ldots`$ for orbital angular momentum $`\ell=0,1,2,3,\ldots`$
. Complete specification requires multiplying these spatial functions
with spin $`\alpha`$ or $`\beta`$ functions

In the 2-component relativistic case, the orbitals have the general form

``` math
\psi_{n\kappa m_{j}}\left(\boldsymbol{r}\right)=R_{n\kappa}\left(r\right)\chi_{\kappa,m_{j}}\left(\theta,\phi\right)
```

where $`\chi_{\kappa,m_{j}}`$ are *2-component* angular functions. The
quantum number $`\kappa`$ requires some explanation. Spin-orbit
interaction leads to a coupling of orbital and spin angular momentum to
form total angular momentum
$`\mathbf{j} = \boldsymbol{\ell} + \mathbf{s}`$. The associated quantum
number $`j`$ can take the values $`j=\ell+1/2`$ and $`j=|\ell-1/2|`$.
However, just giving $`j`$ does not specify an orbital uniquely, since
it can for instance not distinguish $`s_{1/2}`$ and $`p_{1/2}`$ orbitals
(both have $`j=1/2`$). This is possible with the quantum number
$`\kappa`$ according to the following table:

<table style="width:96%;">
<colgroup>
<col style="width: 13%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
<col style="width: 11%" />
</colgroup>
<tbody>
<tr>
<td></td>
<td><span class="math inline"><em>s</em><sub>1/2</sub></span></td>
<td><span class="math inline"><em>p</em><sub>1/2</sub></span></td>
<td><span class="math inline"><em>p</em><sub>3/2</sub></span></td>
<td><span class="math inline"><em>d</em><sub>3/2</sub></span></td>
<td><span class="math inline"><em>d</em><sub>5/2</sub></span></td>
<td><span class="math inline"><em>f</em><sub>5/2</sub></span></td>
<td><span class="math inline"><em>f</em><sub>7/2</sub></span></td>
</tr>
<tr>
<td><span class="math inline"><em>κ</em></span></td>
<td><blockquote>
<dl>
<dt><code>-1</code></dt>
<dd>
&#10;</dd>
</dl>
</blockquote></td>
<td><blockquote>
<p>+1</p>
</blockquote></td>
<td><blockquote>
<dl>
<dt><code>-2</code></dt>
<dd>
&#10;</dd>
</dl>
</blockquote></td>
<td><blockquote>
<p>+2</p>
</blockquote></td>
<td><blockquote>
<dl>
<dt><code>-3</code></dt>
<dd>
&#10;</dd>
</dl>
</blockquote></td>
<td><blockquote>
<p>+3</p>
</blockquote></td>
<td><blockquote>
<dl>
<dt><code>-4</code></dt>
<dd>
&#10;</dd>
</dl>
</blockquote></td>
</tr>
</tbody>
</table>

4-component atomic orbitals have separate radial and angular parts for
both the large and small components

``` math
\begin{aligned}
\psi_{n\kappa m_{j}}\left(\boldsymbol{r}\right)=\left[\begin{array}{c}\psi^{L}\\\psi^{S}\end{array}\right]
=\left[\begin{array}{c}R^{L}\left(r\right)\chi_{\kappa,m_{j}}\left(\theta,\phi\right)\\{\color{red}i}R^{S}\left(r\right)\chi_{-\kappa,m_{j}}\left(\theta,\phi\right)\end{array}\right]
\end{aligned}
```

with the notation for orbitals dictated by the quantum number of the
large component.

With these introductory remarks in mind, we are ready to do our first
calculation.

We will run a Hartree-Fock calculation on the neon atom. We shall use
the input file <span class="title-ref">Ne.inp</span>
(`download <Ne.inp>`):

<div class="literalinclude">

Ne.inp

</div>

and the xyz-file <span class="title-ref">Ne.xyz</span>
(`download <Ne.xyz>`):

<div class="literalinclude">

Ne.xyz

</div>

Neon has the ground-state configuration $`1s^2s^22p^6`$. Note that under
the keyword (`SCF_.CLOSED SHELL`) we specify the occupation of *gerade*
and *ungerade* orbitals separately, that is:

    .CLOSED SHELL
    4 6

For this particular calculation we have chosen to use the dyall.2zp,
which is a double zeta basis set for SCF calculations:

    *BASIS
    .DEFAULT
    dyall.2zp

The minimal commmand for running this calculations is:

    pam --inp=Ne.inp --mol=Ne.xyz

Looking in the output, we find information about the SCF-cycles and the
final energy

<div class="literalinclude">

Ne.scf

</div>

The entry *SS Coulombic correction* refers to a default approximation in
which the calculations of the expensive $`(SS|SS)`$ two-electron
integrals are replaced by a simple energy correction see Visscher1997a
for further details. It is strictly zero for atoms.

The final energy is followed by a list of orbital eigenvalues. In the
present calculation we have activated Mulliken population analysis and
find further information about the occupied orbitals in the output

<div class="literalinclude">

Ne.pop

</div>

Note again the separation on *gerade* and *ungerade* orbitals. We shall
focus on the valence $`p`$ orbitals of neon and the heavier homologues.
In the present calculation we note that the $`2p_{1/2}`$ and
$`2p_{3/2}`$ orbitals have energies -0.851172 $`E_h`$ and -0.846688
$`E_h`$, respectively, corresponding to a spin-orbit splitting of
0.004484 $`E_h`$, or 984.2 $`\mbox{cm}^{-1}`$ (the converstion factor is
2.19474625E+5).

We have also chosen to have get some information about orbital sizes. We
do not have direct access to radial expectation values $`<r>`$, but can
calculate the rms $`<r^2>^{1/2}`$ by calculating
$`<r^2>=<x^2>+<y^2>+<z^2>`$. This is specified in the input as:

    *EXPECTATION VALUE
    .ORBANA
    .OPERATOR
    'XXSECMOM'
    .OPERATOR
    'YYSECMOM'
    .OPERATOR
    'ZZSECMOM'

(see `one_electron_operators` for syntax). We have also used the keyword
<span class="title-ref">.ORBANA</span> to ask for contributions from
individual orbitals. In the output we therefore for instance find

<div class="literalinclude">

Ne.xx

</div>

One column gives the matrix elements
$`\langle i|x^|i\rangle> for occupied orbitals :math:`$varphi_i\`,
whereas the final column gives the matrix element multiplied with the
occupation number of the orbital.

We are interested in the rms of the valence p orbitals. We make the
following table

<table style="width:96%;">
<colgroup>
<col style="width: 7%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 15%" />
<col style="width: 21%" />
</colgroup>
<tbody>
<tr>
<td>Orb.</td>
<td><blockquote>
<p><span class="math inline"> &lt; <em>x</em><sup>2</sup>&gt;</span></p>
</blockquote></td>
<td><blockquote>
<p><span class="math inline"> &lt; <em>y</em><sup>2</sup>&gt;</span></p>
</blockquote></td>
<td><blockquote>
<p><span class="math inline"> &lt; <em>z</em><sup>2</sup>&gt;</span></p>
</blockquote></td>
<td><span
class="math inline"> &lt; <em>r</em><sup>2</sup>&gt;</span></td>
<td><span
class="math inline"> &lt; <em>r</em><sup>2</sup>&gt;<sup>1/2</sup></span></td>
</tr>
<tr>
<td>E1u 1</td>
<td>0.405856904615</td>
<td>0.405856904615</td>
<td>0.405856904615</td>
<td><blockquote>
<p>1.218</p>
</blockquote></td>
<td><blockquote>
<p>1.103</p>
</blockquote></td>
</tr>
<tr>
<td>E1u 2</td>
<td>0.489542686491</td>
<td>0.489542686491</td>
<td>0.244771343246</td>
<td><blockquote>
<p>1.223</p>
</blockquote></td>
<td><blockquote>
<p>1.106</p>
</blockquote></td>
</tr>
<tr>
<td>E1u 3</td>
<td>0.326361790990</td>
<td>0.326361790990</td>
<td>0.571133134233</td>
<td><blockquote>
<p>1.223</p>
</blockquote></td>
<td><blockquote>
<p>1.106</p>
</blockquote></td>
</tr>
</tbody>
</table>

From the Mulliken analysis output we see that the three listed orbitals
correspond to $`2p_{1/2,1/2}`$, $`2p_{3/2,1/2}`$, and $`2p_{3/2,1/2}`$,
respectively, where the second subscript corresponds to the value of
$`|m_j|`$. We see that the $`2p_{3/2}`$ orbitals all have the same size
($`<r^2>^{1/2}=1.106 a_0`$) and are slightly bigger than the spin-orbit
partner $`2p_{1/2}`$ ($`<r^2>^{1/2}=1.103 a_0`$).

It may also be of interest to have a look at the shape of the atomic
orbitals. Here things are somewhat more complicated than in the
non-relativistic case since we have seen that the orbitals are complex
4-component vector functions. What *is* possible is to plot orbital
densities, see `orbital_densities` for instructions.

So far we have done 4-component relativistic calculations. It is
possible to successively turn off spin-orbit interaction and scalar
relativistic effects using the keywords `HAMILTONIAN_.SPINFREE` and
`HAMILTONIAN_.LEVY-LEBLOND`. The latter calculation is based on the
Lévy-Leblond Hamiltonian, which is a 4-component *non-relativistic*
Hamiltonian.

It is also possible to explore various approximate 2-component
Hamiltonians. We shall be interested in the following ones:

> 1.  The eXact 2-Component Hamiltonian (X2X). Specification:
>
>         **HAMILTONIAN
>         .X2C
>
> 2.  The ZORA Hamiltonian. Specification:
>
>         **HAMILTONIAN
>         .ZORA
>         0  0
>
> 3.  The scaled ZORA Hamiltonian. Specification:
>
>         **HAMILTONIAN
>         .ZORA
>         0  1
>
> 4.  The second-order Douglas-Kroll-Hess Hamiltonian Specification:
>
>         **HAMILTONIAN
>         .DKH2

# Molecular calculations

> [!NOTE]
> **Exercise M** : In these exercises we look at relativistic effects on
> the spectroscopic constants of noble metal monohydrides. Experimental
> data can be found at [these NIST
> pages](https://webbook.nist.gov/chemistry/form-ser/) by giving the
> chemical formula and selecting *Constants of diatomic molecules*. Note
> that we are looking at spectroscopic constants for the ground state,
> normally denoted *X* and found at the bottom of the table. Three
> different DFT functionals will be explored: 1. B3LYP 2. PBE 3. PBE0
>
> The exercises are: 1. Perform 4-component DFT-calculations on the
> monohydrides of the noble metals Cu, Ag and Au in order to investigate
> the effects of spin-orbit coupling and scalar relativity (again using
> the keywords `HAMILTONIAN_.SPINFREE` and `HAMILTONIAN_.LEVY-LEBLOND`)
> on equilibrium distances $`r_e`$, harmonic frequencies $`\omega_e`$
> and anharmonic constants $`\omega_e x_e`$. 2. For AuH, explore the
> non-relativistic limit by recalculating spectroscopic constants
> setting $`c=4000`$ a.u. 3. For CuH, explore the *ultra-relativistic*
> limit by recalculating spectroscopic constants setting $`c=20`$ a.u.
> Helpful information for these exercises is given below !

As a sample calculation, let us look at CuH. These are simple diatomic
species, so we shall simply obtain spectroscopic constants by
calculating the energy at a selection of interatomic distances about the
expected minimum. We can then calculate the equilibrium distance
$`r_e`$, harmonic frequency $`\omega_e`$ and anharmonic constant
$`\omega_e x_e`$ using the utility program `twofit`. The manual pages
provide theoretical background. The executable twofit.x is in the same
directory as the DIRAC executable.

We will run a DFT calculation on CuH, here using the B3LYP functional.
We shall use the input file <span class="title-ref">CuH.inp</span>
(`download <CuH.inp>`):

<div class="literalinclude">

CuH.inp

</div>

and the xyz-file <span class="title-ref">CuH.xyz</span>
(`download <CuH.xyz>`):

<div class="literalinclude">

CuH.xyz

</div>

The minimal commmand for running this calculations is:

    pam --inp=CuH.inp --mol=CuH.xyz

Looking in the output, we find information about the SCF-cycles and the
final energy

<div class="literalinclude">

CuH.scf

</div>

We may now see that the entry *SS Coulombic correction* is non-zero,
albeit very small. In this particular case the energy correction is
calculated as

``` math
E^{SS}_{inter}=\frac{q^S_{Cu}q^S_H}{R_{Cu-H}}
```

where $`q^S`$ is the contribution of the small components to the atomic
charge.

In passing we note that the output contains an xyz-file

<div class="literalinclude">

CuHmod.xyz

</div>

which is modified compared to the input file. This is because DIRAC
checked the original xyz-input and found linear symmetry

<div class="literalinclude">

CuH.sym

</div>

DIRAC uses *linear supersymmetry* in these calculations; this means that
the underlying integrals are adapted to the $`C_{2v}`$ subgroup, but the
Fock matrix has the full block structure due to linear symmetry. In the
same manner, the atomic calculations above uses atomic supersymmetry.

We are now ready to scan the potential surface of CuH in the vicinity of
the expected minimum. A pedestrian way is to run the calculation point
by point, modfifying the input xyz-file for each new geometry. However,
it is also possible to automatize the scan, e.g. (using bash)

<div class="literalinclude">

CuH.script

</div>

Total energies can then be extracted for instance using:

    grep 'Total energy                             :' CuH_*out > CuH.pot

After editing the file <span class="title-ref">CuH.pot</span>
(`download <CuH.pot>`) reads:

<div class="literalinclude">

CuH.pot

</div>

A polynomial fit of order 6, using `twofit` (output
`here <CuH.pot.twofit>`), gives the spectroscopic constants
$`r_e = 1.4604`$ Å, $`\omega_e=1958.2\mbox{ cm}^{-1}`$ and
$`\omega_e x_e=34.6\mbox{ cm}^{-1}`$. From the NIST page we find
$`r_e = 1.46263`$ Å, $`\omega_e=1941.26\mbox{ cm}^{-1}`$ and
$`\omega_e x_e=37.51\mbox{ cm}^{-1}`$, so our results are not bad !

We have seen that relativistic effects are dictated by the Lorentz
factor

``` math
\gamma = \frac{1}{\sqrt{1-\frac{v^2}{c^2}}}
```

where $`v`$ is the speed of the particle and $`c`$ is the speed of
light. We have also seen that for hydrogen-like atoms the speed of the
$`1s`$ orbital is $`Z`$ (nuclear charge) in atomic units, to be compared
with the speed of light which is around 137 in the same units. The
non-relativistic limit can be attained by setting the speed of light to
infinity. This is not possible on a computer, so we settle for a large,
but finite value. It is also possible to investigate the
*ultra-relativistic* case by lowering the speed of light. In DIRAC the
speed of light can be adjusted using the `GENERAL_.CVALUE` keyword.

> [!NOTE]
> When extracting spectroscopic constants using `twofit` it is important
> that the minimum $`r_e`$ is in the interval of selected points,
> ideally somewhere in the middle, so you may have to add further points
> to your scan to achieve this. Relativistic effects are important, so
> when searching for an unknown minimum, you may first make a rather
> coarse grid of points (spacing 0.1 Å) and use `twofit` to guide you in
> the right direction, where you can put a finer grid (spacing 0.01 Å).
