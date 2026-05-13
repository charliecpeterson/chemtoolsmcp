orphan  

# 1s core-ionization of N2 by TD-DFT

## Introduction

We want to study the excitation of a 1s electron of the $`N_2`$ molecule
to an empty orbital. More precisely we shall look at the excitation of
an electron from the bonding $`1s\sigma_g`$ or anti-bonding
$`1s\sigma_u`$ -orbitals to the vacant $`2p\pi_g`$ or $`2p\sigma_u`$
orbitals (see MO-diagram below).

<img src="N2_MOdiagram.jpg" class="align-center" alt="N2_MOdiagram" />

Note that this diagram does not take spin-orbit into account, but we
shall consider this interaction later on. Let us first consider the
possible final states. One electron leaves from one of four
spin-orbitals and enters one of six spin-orbitals. This gives 24
determinants which translates into the following states:

| Configuration                 | States             |
|-------------------------------|--------------------|
| $`1s\sigma_g^{-1}2p\pi_g`$    | $`^{1,3}\Pi_g`$    |
| $`1s\sigma_g^{-1}2p\sigma_u`$ | $`^{1,3}\Sigma_u`$ |
| $`1s\sigma_u^{-1}2p\pi_g`$    | $`^{1,3}\Pi_u`$    |
| $`1s\sigma_u^{-1}2p\sigma_u`$ | $`^{1,3}\Sigma_g`$ |

## Spin-orbit free calculation

### Preparing the input files

We employ the following molecular input file
<span class="title-ref">N2.mol</span>

<div class="literalinclude">

N2.mol

</div>

Here we do not provide any symmetry information, meaning that we ask
DIRAC to detect it. DIRAC will find that the full group is
$`D_{\infty h}`$. With spin-orbit coupling DIRAC will then activate
linear supersymmetry, but in the spin-orbit free case it will simply use
the highest Abelian single point group, that is $`D_{2h}`$. For the
final states DIRAC will employ the *total* symmetry, that is the
combined spin and spatial symmetry. Here we shall keep in mind that the
singlet spin function is totally symmetric ($`A_g`$), whereas the
triplet spin functions transform as rotations. We know the triplet
functions as:

``` math
T_{-1} = \alpha_1\alpha_2;\quad T_0=\frac{1}{\sqrt{2}}\left(\alpha_1\beta_2-\beta_1\alpha_2\right);\quad T_{+1}=\beta_1\beta_2
```

but for our purposes it will be more convenient to form the
combinations:

``` math
T_x=\frac{1}{\sqrt{2}}\left(T_{-1}-T_{+1}\right);\quad T_y=\frac{i}{\sqrt{2}}\left(T_{-1}+T_{+1}\right);\quad T_z=T_0
```

which transform as rotations $`R_x\left(B_{3g}\right)`$ ,
$`R_y\left(B_{2g}\right)`$ and $`R_z\left(B_{1g}\right)`$ ,
respectively.

We can now set up the following correlation of states:

<table style="width:99%;">
<colgroup>
<col style="width: 15%" />
<col style="width: 20%" />
<col style="width: 15%" />
<col style="width: 47%" />
</colgroup>
<thead>
<tr>
<th>State</th>
<th>Spin</th>
<th>Spatial</th>
<th>Spin <span class="math inline">⊗</span> Spatial</th>
</tr>
</thead>
<tbody>
<tr>
<td><blockquote>
<p><span
class="math inline"><sup>1</sup><em>Σ</em><sub><em>g</em></sub></span></p>
</blockquote></td>
<td><span
class="math inline"><em>A</em><sub><em>g</em></sub></span></td>
<td><span
class="math inline"><em>A</em><sub><em>g</em></sub></span></td>
<td><span
class="math inline"><em>A</em><sub><em>g</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p><span
class="math inline"><sup>1</sup><em>Σ</em><sub><em>u</em></sub></span></p>
</blockquote></td>
<td><span
class="math inline"><em>A</em><sub><em>g</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>1<em>u</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>1<em>u</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p><span
class="math inline"><sup>1</sup><em>Π</em><sub><em>x</em>, <em>y</em>; <em>g</em></sub></span></p>
</blockquote></td>
<td><span
class="math inline"><em>A</em><sub><em>g</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>3<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>3<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p><span
class="math inline"><sup>1</sup><em>Π</em><sub><em>x</em>, <em>y</em>; <em>u</em></sub></span></p>
</blockquote></td>
<td><span
class="math inline"><em>A</em><sub><em>g</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>3<em>u</em></sub>, <em>B</em><sub>2<em>u</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>3<em>u</em></sub>, <em>B</em><sub>2<em>u</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p><span
class="math inline"><sup>3</sup><em>Σ</em><sub><em>g</em></sub></span></p>
</blockquote></td>
<td><span
class="math inline"><em>B</em><sub>3<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub>, <em>B</em><sub>1<em>g</em></sub></span></td>
<td><span
class="math inline"><em>A</em><sub><em>g</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>3<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub>, <em>B</em><sub>1<em>g</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p><span
class="math inline"><sup>3</sup><em>Σ</em><sub><em>u</em></sub></span></p>
</blockquote></td>
<td><span
class="math inline"><em>B</em><sub>3<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub>, <em>B</em><sub>1<em>g</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>1<em>u</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>2<em>u</em></sub>, <em>B</em><sub>3<em>u</em></sub>, <em>A</em><sub><em>u</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p><span
class="math inline"><sup>3</sup><em>Π</em><sub><em>x</em>, <em>y</em>; <em>g</em></sub></span></p>
</blockquote></td>
<td><span
class="math inline"><em>B</em><sub>3<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub>, <em>B</em><sub>1<em>g</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>3<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub></span></td>
<td><span
class="math inline">(<em>A</em><sub><em>g</em></sub>, <em>B</em><sub>1<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub>), (<em>B</em><sub>1<em>g</em></sub>, <em>A</em><sub><em>g</em></sub>, <em>B</em><sub>3<em>g</em></sub>)</span></td>
</tr>
<tr>
<td><blockquote>
<p><span
class="math inline"><sup>3</sup><em>Π</em><sub><em>x</em>, <em>y</em>; <em>u</em></sub></span></p>
</blockquote></td>
<td><span
class="math inline"><em>B</em><sub>3<em>g</em></sub>, <em>B</em><sub>2<em>g</em></sub>, <em>B</em><sub>1<em>g</em></sub></span></td>
<td><span
class="math inline"><em>B</em><sub>3<em>u</em></sub>, <em>B</em><sub>2<em>u</em></sub></span></td>
<td><span
class="math inline">(<em>A</em><sub><em>u</em></sub>, <em>B</em><sub>1<em>u</em></sub>, <em>B</em><sub>2<em>u</em></sub>), (<em>B</em><sub>1<em>u</em></sub>, <em>A</em><sub><em>u</em></sub>, <em>B</em><sub>3<em>u</em></sub>)</span></td>
</tr>
</tbody>
</table>

Counting total symmetries we find the 24 microstates are evenly
distributed amongst the eight irreps of $`D_{2h}`$:

| Irrep | Core-ionized state |  |
|----|----|----|
| $`A_g`$ | $`^1\Sigma_g, ^{3(x)}\Pi_{x;g}, ^{3(y)}\Pi_{y;g}`$ | $`x^2, y^2, z^2`$ |
| $`B_{3u}`$ | $`^{1}\Pi_{x;u}, ^{3(y)}\Sigma_u, ^{3(z)}\Pi_{y;u}`$ | $`x`$ |
| $`B_{2u}`$ | $`^{1}\Pi_{y;u}, ^{3(x)}\Sigma_u, ^{3(z)}\Pi_{x;u}`$ | $`y`$ |
| $`B_{1g}`$ | $`^{3(z)}\Sigma_g, ^{3(y)}\Pi_{x;g}, ^{3(x)}\Pi_{y;g}`$ | $`xy`$ |
| $`B_{1u}`$ | $`^1\Sigma_u, ^{3(y)}\Pi_{x;u}, ^{3(x)}\Pi_{y;u}`$ | $`z`$ |
| $`B_{2g}`$ | $`^{1}\Pi_{y;g}, ^{3(y)}\Sigma_g, ^{3(z)}\Pi_{x;g}`$ | $`xz`$ |
| $`B_{3g}`$ | $`^{1}\Pi_{x;g}, ^{3(x)}\Sigma_g, ^{3(z)}\Pi_{y;g}`$ | $`yz`$ |
| $`A_{u}`$ | $`^{3(z)}\Sigma_u, ^{3(x)}\Pi_{x;u}, ^{3(y)}\Pi_{y;u}`$ | $`xyz`$ |

From these considerations we now set up the following menu file for our
calculation

<div class="literalinclude">

N2spf.inp

</div>

In the `*SCF` section we give the electron occupation of $`N_2`$: 6 and
8 electrons in *gerade* and *ungerade* orbitals, respectively. We also
ask for a Mulliken population analysis (`ANALYZE_.MULPOP`) for the
occupied orbitals and the orbitals involved in the core excitation.

Let us now look at how we set up the calculation of excitation energies
under <span class="title-ref">\*EXCITATION ENERGIES</span>. We have seen
that there are three excitations per boson irrep. Note that the
numbering of irreps follow what you for instance find in the $`D_{2h}`$
direct product table in the output:

    |   | Ag   B3u  B2u  B1g  B1u  B2g  B3g  Au 
    -----+----------------------------------------
    Ag  | Ag   B3u  B2u  B1g  B1u  B2g  B3g  Au 
    B3u | B3u  Ag   B1g  B2u  B2g  B1u  Au   B3g
    B2u | B2u  B1g  Ag   B3u  B3g  Au   B1u  B2g
    B1g | B1g  B2u  B3u  Ag   Au   B3g  B2g  B1u
    B1u | B1u  B2g  B3g  Au   Ag   B3u  B2u  B1g
    B2g | B2g  B1u  Au   B3g  B3u  Ag   B1g  B2u
    B3g | B3g  Au   B1u  B2g  B2u  B1g  Ag   B3u
    Au  | Au   B3g  B2g  B1u  B1g  B2u  B3u  Ag 

Note also that we skip excitations in $`B_{2u}`$ and $`B_{2g}`$, since
they are related by symmetry to the excitations of $`B_{3u}`$ and
$`B_{3g}`$, respectively.

If nothing further is specified the excitation energies are calculated
by a "bottoms-up" approach and so we will get valence excitations only,
since the core-excitations are much higher in energy. We therefore
restrict the excitations to the occupied $`1s\sigma_g`$ and
$`1s\sigma_u`$ orbitals.

We furthermore ask for transition moments to be calculated with respect
to the component of the dipole moment operator. These will be non-zero
only for excitations in irreps $`B_{3u}, B_{2u}`$ and $`B_{1u}`$.
Finally we ask for analysis of what orbitals contribute to the various
excitations. For this the Mulliken population analysis may come in handy
as reference.

### Looking at the output

After running the calculation, let us now look at the output. The
following excitation energies were calculated

<div class="literalinclude">

N2spf_exc.txt

</div>

DIRAC assumes that excitation energies that are within $`10^{-9}\ E_h`$
of each other come from the same degenerate state. This threshold is
somewhat arbitrary and we shall see that DIRAc is not always correct.

There is sufficient symmetry in the calculation (symmetry distinct
rotation) to allow DIRAC to pinpoint the symmetry of the core-ionized
state and we therefore find the following distribution

<div class="literalinclude">

N2spf_exc2.txt

</div>

The first and second block refers to singlet and triplet states,
respectively. Based on the discussion in the preceeding section we see
that levels 1, 2 and 3 all come from a $`^3\Pi_u`$ which in $`D_{2h}`$
splits into $`^3B_{2u}`$ and $`^3B_{3u}`$. After careful inspection we
can set up the following table

<table style="width:65%;">
<colgroup>
<col style="width: 11%" />
<col style="width: 25%" />
<col style="width: 29%" />
</colgroup>
<thead>
<tr>
<th>Level</th>
<th>eigenvalue (eV)</th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0.000</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Σ</em><sub><em>g</em></sub></span></td>
</tr>
<tr>
<td>1,2,3</td>
<td><blockquote>
<p>388.780</p>
</blockquote></td>
<td><span
class="math inline"><sup>3</sup><em>Π</em><sub><em>u</em></sub></span></td>
</tr>
<tr>
<td>4,5,6</td>
<td><blockquote>
<p>388.831</p>
</blockquote></td>
<td><span
class="math inline"><sup>3</sup><em>Π</em><sub><em>g</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p>7</p>
</blockquote></td>
<td><blockquote>
<p>389.916</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Π</em><sub><em>u</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p>8</p>
</blockquote></td>
<td><blockquote>
<p>389.936</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Π</em><sub><em>g</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p>9,10</p>
</blockquote></td>
<td><blockquote>
<p>399.834</p>
</blockquote></td>
<td><span
class="math inline"><sup>3</sup><em>Σ</em><sub><em>g</em></sub></span></td>
</tr>
<tr>
<td>11,12</td>
<td><blockquote>
<p>399.876</p>
</blockquote></td>
<td><span
class="math inline"><sup>3</sup><em>Σ</em><sub><em>u</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p>13</p>
</blockquote></td>
<td><blockquote>
<p>400.519</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Σ</em><sub><em>g</em></sub></span></td>
</tr>
<tr>
<td><blockquote>
<p>14</p>
</blockquote></td>
<td><blockquote>
<p>400.568</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Σ</em><sub><em>u</em></sub></span></td>
</tr>
</tbody>
</table>

Looking further down in the output we find dominant inactive and virtual
orbitals. Restricting attention to $`B_{3u}`$ total symmetry we find
that the first excited state $`^3\Pi_u`$, at 388.78 eV, is dominated by
the excitation <span class="title-ref">1(i:E1u) ---\> 4(v:E1g)</span>,
which, as can be inferred from the Mulliken population analysis,
corresponds to $`1s\sigma_u \rightarrow 2p\pi_{y;g}`$. The second
excited state $`^1\Pi_u`$, at 389.91 eV, corresponds to
<span class="title-ref">1(i:E1u) ---\> 5(v:E1g)</span>
($`1s\sigma_u \rightarrow 2p\pi_{x;g}`$), whereas the third excited
state $`^3\Sigma_u`$, at 399.88 eV, is dominated by
<span class="title-ref">1(i:E1g) ---\> 5(v:E1u)</span>
($`1s\sigma_g \rightarrow 2p\sigma_u`$).

Within the electric dipole approximation only singlet states get
oscillator strengths. In the output we find

<div class="literalinclude">

N2spf_osc.txt

</div>

showing intensity to the $`^1\Pi_u`$ and $`^1\Sigma_u`$ states.

## Including spin-orbit

Spin-orbit is included by simply commenting out the keyword
`HAMILTONIAN_.SPINFREE` in the input above:

    **HAMILTONIAN
    !.SPINFREE

This leads to the following states:

<div class="literalinclude">

N2so_exc.txt

</div>

with the following distribution on linear symmetries:

<div class="literalinclude">

N2so_exc2.txt

</div>

Comparing with the preceeding section we see the following spin-orbit
decomposition of the :math:Lambda-S\` states:

<table style="width:99%;">
<colgroup>
<col style="width: 5%" />
<col style="width: 12%" />
<col style="width: 15%" />
<col style="width: 65%" />
</colgroup>
<thead>
<tr>
<th>Level</th>
<th>eigenvalue (eV)</th>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td><blockquote>
<p>0</p>
</blockquote></td>
<td><blockquote>
<p>0.000</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Σ</em><sub><em>g</em></sub><sup>+</sup></span></td>
<td><span class="math inline">0<sub><em>g</em></sub><sup>+</sup></span>
(0.000)</td>
</tr>
<tr>
<td>1,2,3</td>
<td><blockquote>
<p>388.780</p>
</blockquote></td>
<td><span
class="math inline"><sup>3</sup><em>Π</em><sub><em>u</em></sub></span></td>
<td><span class="math inline">0<sub><em>u</em></sub><sup>+</sup></span>
(388.771), <span
class="math inline">0<sub><em>u</em></sub><sup>−</sup></span> (388.771),
<span class="math inline">1<sub><em>u</em></sub></span> (388.780), <span
class="math inline">2<sub><em>u</em></sub></span> (388.788)</td>
</tr>
<tr>
<td>4,5,6</td>
<td><blockquote>
<p>388.831</p>
</blockquote></td>
<td><span
class="math inline"><sup>3</sup><em>Π</em><sub><em>g</em></sub></span></td>
<td><span class="math inline">0<sub><em>g</em></sub><sup>+</sup></span>
(388.821), <span
class="math inline">0<sub><em>g</em></sub><sup>−</sup></span> (388.822),
<span class="math inline">1<sub><em>g</em></sub></span> (388.830), <span
class="math inline">2<sub><em>g</em></sub></span> (388.839)</td>
</tr>
<tr>
<td><blockquote>
<p>7</p>
</blockquote></td>
<td><blockquote>
<p>389.916</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Π</em><sub><em>u</em></sub></span></td>
<td><span class="math inline">1<sub><em>u</em></sub></span>
(389.916)</td>
</tr>
<tr>
<td><blockquote>
<p>8</p>
</blockquote></td>
<td><blockquote>
<p>389.936</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Π</em><sub><em>g</em></sub></span></td>
<td><span class="math inline">1<sub><em>g</em></sub></span>
(389.936)</td>
</tr>
<tr>
<td><blockquote>
<p>9,10</p>
</blockquote></td>
<td><blockquote>
<p>399.834</p>
</blockquote></td>
<td><span
class="math inline"><sup>3</sup><em>Σ</em><sub><em>g</em></sub><sup>+</sup></span></td>
<td><span class="math inline">0<sub><em>g</em></sub><sup>−</sup></span>
(399.834), <span class="math inline">1<sub><em>g</em></sub></span>
(399.834)</td>
</tr>
<tr>
<td>11,12</td>
<td><blockquote>
<p>399.876</p>
</blockquote></td>
<td><span
class="math inline"><sup>3</sup><em>Σ</em><sub><em>u</em></sub><sup>+</sup></span></td>
<td><span class="math inline">0<sub><em>u</em></sub><sup>−</sup></span>
(399.876), <span class="math inline">1<sub><em>u</em></sub></span>
(399.876)</td>
</tr>
<tr>
<td><blockquote>
<p>13</p>
</blockquote></td>
<td><blockquote>
<p>400.519</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Σ</em><sub><em>g</em></sub><sup>+</sup></span></td>
<td><span class="math inline">0<sub><em>g</em></sub><sup>+</sup></span>
(400.519)</td>
</tr>
<tr>
<td><blockquote>
<p>14</p>
</blockquote></td>
<td><blockquote>
<p>400.568</p>
</blockquote></td>
<td><span
class="math inline"><sup>1</sup><em>Σ</em><sub><em>u</em></sub><sup>+</sup></span></td>
<td><span class="math inline">0<sub><em>u</em></sub><sup>+</sup></span>
(400.568)</td>
</tr>
</tbody>
</table>

Note that the energies are given relative to the lowest level, that is,
the ground state and that it is somewhat stabilized by spin-orbit
coupling.

The effect of spin-orbit coupling shows up in the oscillator strengths:

<div class="literalinclude">

N2so_osc.txt

</div>

What we see is the $`0_u^+`$ and $`1_u`$ components of the $`^3\Pi_u`$
state stealing intensity from the singlet states. This change is not
very spectacular since the nitrogen molecule is composed of light atoms
for which relativistic effects are not very strong. We can mimic a more
strongly relativistic system by reducing the speed of light to e.g. 20
a.u.:

    **GENERAL                                                                                           
    .CVALUE                                                                                             
    20.0D0

We now see

<div class="literalinclude">

N2so20_osc.txt

</div>

## Simulating the core-excitation spectrum using complex response

[Gas Phase Core Excitation
Database](http://unicorn.mcmaster.ca/corex/cedb-title.html)
