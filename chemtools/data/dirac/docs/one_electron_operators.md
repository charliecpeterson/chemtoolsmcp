orphan  

# One-electron operators

## Syntax for specifying one-electron operators

In DIRAC, a general one-electron 4-component operator is generated from
linear combinations of operators having the basic form

``` math
\hat{\boldsymbol{P}} = f \, \boldsymbol{M} \, \hat{\Omega},
```

where $`f`$ is a scalar factor, $`\hat{\Omega}`$ is a scalar operator,
and $`\boldsymbol{M}`$ is one of the following $`4 \times 4`$
time-reversal symmetric matrices:

``` math
& \boldsymbol{I}, \, \boldsymbol{\beta}, \, \boldsymbol{\gamma}^5, \, \boldsymbol{\beta}\boldsymbol{\gamma}^5,
```
``` math
& i\boldsymbol{\alpha}_x, \, i\boldsymbol{\alpha}_y, \, i\boldsymbol{\alpha}_z, \, i\boldsymbol{\beta}\boldsymbol{\alpha}_x, \, i\boldsymbol{\beta}\boldsymbol{\alpha}_y, \, i\boldsymbol{\beta}\boldsymbol{\alpha}_z,
```
``` math
& i\boldsymbol{\Sigma}_x, \, i\boldsymbol{\Sigma}_y, \, i\boldsymbol{\Sigma}_z, \, i\boldsymbol{\beta}\boldsymbol{\Sigma}_x, \, i\boldsymbol{\beta}\boldsymbol{\Sigma}_y, \, i\boldsymbol{\beta}\boldsymbol{\Sigma}_z.
```

One thing to notice is that $`\boldsymbol{I}`$, $`\boldsymbol{\beta}`$,
$`\boldsymbol{\gamma}^5`$, and
$`\boldsymbol{\beta}\boldsymbol{\gamma}^5`$ are time-reversal symmetric
operators, and they are implemented with such symmetry in DIRAC. On the
other hand, $`\boldsymbol{\alpha}_k`$, $`\boldsymbol{\Sigma}_k`$,
$`\boldsymbol{\beta\alpha}_k`$, and $`\boldsymbol{\beta\Sigma}_k`$ (with
$`k=x,y`$, and $`z`$) are time-reversal antisymmetric operators. This
information is saved by the program, but these matrices (i.e.,
$`\boldsymbol{\alpha}_k`$, $`\boldsymbol{\Sigma}_k`$,
$`\boldsymbol{\beta\alpha}_k`$, and $`\boldsymbol{\beta\Sigma}_k`$) are
multiplied by the imaginary unit $`i=\sqrt{-1}`$ in order to make them
time-reversal symmetric and hence fit into the quaternion symmetry
scheme of DIRAC (see Saue1999 and Salek2005 for more information).

Another important aspect to note is that DIRAC does not use the
$`4 \times 4`$ $`\boldsymbol{M}`$ matrices in the standard
representation. These matrices are reordered following a criterion
described in Section [Quaternion algebra and time-reversal
symmetry](#quaternion-algebra-and-time-reversal-symmetry). Further
details about the implemented operators are provided in that Section,
where the connection between the Dirac matrices and the quaternion
algebra is given.

### Operator specification

Operators are specified by the keyword `HAMILTONIAN_.OPERATOR` with the
following arguments:

    .OPERATOR
     'user defined operator name'
     operator type keyword
     one-electron scalar operator label for each component
     CMULT
     FACTORS
     factor for each component
     COMFACTOR
     common factor for all components

Note that the arguments following the keyword `HAMILTONIAN_.OPERATOR`
must start with a blank.

The arguments are optional, except for the one-electron scalar operator
label. If no operator type keyword is given, DIRAC assumes the operator
type to be $`\boldsymbol{I}`$, the diagonal $`4 \times 4`$ matrix.
Component factors under the keyword `FACTORS`, as well as the common
factor under the keyword `COMFACTOR`, are all equal to one if not
specified.

The keyword `CMULT` assures multiplication of the individual factors (or
common factor) by the speed of light in vacuum, $`c`$. This option has
the further advantage that `CMULT` follows any user-specified
modification of the speed of light, as provided by `GENERAL_.CVALUE`.

### Operator type keywords

There are 23 basic operator types implemented in DIRAC. They are listed
in the following Table, containing their keywords, operator forms,
associated number of factors (NF), and time-reversal symmetry (TRS)
without imaginary phase:

| **Keyword** | **Operator form** | **NF** | **TRS (without i)** |
|----|----|----|----|
| DIAGONAL | $`f \boldsymbol{I} \hat{\Omega}`$ | 1 | Symmetric |
| BETA | $`f \boldsymbol{\beta} \hat{\Omega}`$ | 1 | Symmetric |
| GAMMA5 | $`f \boldsymbol{\gamma}^5 \hat{\Omega}`$ | 1 | Symmetric |
| BETAGAMM | $`f \boldsymbol{\beta} \boldsymbol{\gamma}^5 \hat{\Omega}`$ | 1 | Symmetric |
| XALPHA | $`f i\boldsymbol{\alpha}_x \hat{\Omega}`$ | 1 | Antisymmetric |
| YALPHA | $`f i\boldsymbol{\alpha}_y \hat{\Omega}`$ | 1 | Antisymmetric |
| ZALPHA | $`f i\boldsymbol{\alpha}_z \hat{\Omega}`$ | 1 | Antisymmetric |
| XAVECTOR | $`f_1 i\boldsymbol{\alpha}_y \hat{\Omega}_z - f_2 i\boldsymbol{\alpha}_z \hat{\Omega}_y`$ | 2 | Antisymmetric |
| YAVECTOR | $`f_1 i\boldsymbol{\alpha}_z \hat{\Omega}_x - f_2 i\boldsymbol{\alpha}_x \hat{\Omega}_z`$ | 2 | Antisymmetric |
| ZAVECTOR | $`f_1 i\boldsymbol{\alpha}_x \hat{\Omega}_y - f_2 i\boldsymbol{\alpha}_y \hat{\Omega}_x`$ | 2 | Antisymmetric |
| ALPHADOT | $`f_1 i\boldsymbol{\alpha}_x \hat{\Omega}_x + f_2 i\boldsymbol{\alpha}_y \hat{\Omega}_y + f_3 i\boldsymbol{\alpha}_z \hat{\Omega}_z`$ | 3 | Antisymmetric |
| XBETAALP | $`f i\boldsymbol{\beta} \boldsymbol{\alpha}_x \hat{\Omega}`$ | 1 | Antisymmetric |
| YBETAALP | $`f i\boldsymbol{\beta} \boldsymbol{\alpha}_y \hat{\Omega}`$ | 1 | Antisymmetric |
| ZBETAALP | $`f i\boldsymbol{\beta} \boldsymbol{\alpha}_z \hat{\Omega}`$ | 1 | Antisymmetric |
| XSIGMA | $`f i\boldsymbol{\Sigma}_x \hat{\Omega}`$ | 1 | Antisymmetric |
| YSIGMA | $`f i\boldsymbol{\Sigma}_y \hat{\Omega}`$ | 1 | Antisymmetric |
| ZSIGMA | $`f i\boldsymbol{\Sigma}_z \hat{\Omega}`$ | 1 | Antisymmetric |
| SIGMADOT | $`f_1 i\boldsymbol{\Sigma}_x \hat{\Omega}_x + f_2 i\boldsymbol{\Sigma}_y \hat{\Omega}_y + f_3 i\boldsymbol{\Sigma}_z \hat{\Omega}_z`$ | 1 | Antisymmetric |
| XBETASIG | $`f i\boldsymbol{\beta} \boldsymbol{\Sigma}_x \hat{\Omega}`$ | 1 | Antisymmetric |
| YBETASIG | $`f i\boldsymbol{\beta} \boldsymbol{\Sigma}_y \hat{\Omega}`$ | 1 | Antisymmetric |
| ZBETASIG | $`f i\boldsymbol{\beta} \boldsymbol{\Sigma}_z \hat{\Omega}`$ | 1 | Antisymmetric |
| BEALDOT | $`f_1 i\boldsymbol{\beta}\boldsymbol{\alpha}_x \hat{\Omega}_x + f_2 i\boldsymbol{\beta}\boldsymbol{\alpha}_y \hat{\Omega}_y + f_3 i\boldsymbol{\beta}\boldsymbol{\alpha}_z \hat{\Omega}_z`$ | 3 | Antisymmetric |
| iBETAGAM | $`f \boldsymbol{\beta} \boldsymbol{\gamma}^5 \hat{\Omega}`$ | 1 | Antisymmetric |

<span class="title-ref">Note for advanced users:</span> iBETAGAM
corresponds to the time-reversal symmetric operator
$`f \boldsymbol{\beta} \boldsymbol{\gamma}^5 \hat{\Omega}`$, but has
been assigned a time-reversal antisymmetry. This operator can be used,
for example, for calculations of the molecular enhancement factors of
the electron electric dipole moment (eEDM). For further details, see
Section [Quaternion algebra and time-reversal
symmetry](#quaternion-algebra-and-time-reversal-symmetry).

### One-electron scalar operator labels

<table style="width:98%;">
<colgroup>
<col style="width: 7%" />
<col style="width: 19%" />
<col style="width: 7%" />
<col style="width: 9%" />
<col style="width: 53%" />
</colgroup>
<thead>
<tr>
<th><strong>Operator</strong> <strong>label</strong></th>
<th><strong>Integral description</strong></th>
<th><strong>Symmetry</strong></th>
<th><strong>Components</strong></th>
<th><strong>Operators</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td>KINENER</td>
<td>NR kinetic energy</td>
<td><blockquote>
<p>S</p>
</blockquote></td>
<td>KINENER</td>
<td><span class="math inline">$\hat{\Omega}_1 =
-\frac{1}{2}\hat{\nabla}^2$</span></td>
</tr>
<tr>
<td>MOLFIELD</td>
<td>Nuclear attraction</td>
<td><blockquote>
<p>S</p>
</blockquote></td>
<td>MOLFIELD</td>
<td><span
class="math inline"><em>Ω̂</em><sub>1</sub> = ∑<sub><em>K</em></sub><em>Z</em><sub><em>K</em></sub>|<strong>r</strong> − <strong>R</strong><sub><em>K</em></sub>|<sup>−1</sup></span></td>
</tr>
<tr>
<td>OVERLAP</td>
<td>Overlap</td>
<td><blockquote>
<p>S</p>
</blockquote></td>
<td>OVERLAP</td>
<td><span class="math inline"><em>Ω̂</em><sub>1</sub> = 1</span></td>
</tr>
<tr>
<td>FERMI C</td>
<td>Fermi contact</td>
<td><blockquote>
<p>S</p>
</blockquote></td>
<td>FC NAMab</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{4 \pi g_{e}}{3}
\delta(\boldsymbol{r}-\boldsymbol{R}_K)$</span></td>
</tr>
<tr>
<td>PVC</td>
<td>Gaussian nuclear density normalized to 1, <span
class="math inline"><em>ρ</em><sub><em>K</em></sub><sup><em>G</em></sup>(<strong>r</strong>)</span></td>
<td><blockquote>
<p>S</p>
</blockquote></td>
<td>PVCNAMab</td>
<td><dl>
<dt><span
class="math inline"><em>Ω̂</em><sub>1</sub> = <em>ρ</em><sub><em>K</em></sub><sup><em>G</em></sup>(<strong>r</strong>)</span></dt>
<dd>
<p><span
class="math inline"> = (<em>ζ</em><sub><em>K</em></sub>/<em>π</em>)<sup>3/2</sup>exp (−<em>ζ</em><sub><em>K</em></sub>|<strong>r</strong> − <strong>R</strong><sub><em>K</em></sub>|<sup>2</sup>)</span></p>
</dd>
</dl></td>
</tr>
<tr>
<td rowspan="3">GRAD FC</td>
<td rowspan="3">Gradient of the Dirac delta distribution</td>
<td rowspan="3"><blockquote>
<p>S</p>
</blockquote></td>
<td>XGFNAMab</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{\partial}{\partial
x} \delta(\boldsymbol{r}-\boldsymbol{R}_K)$</span></td>
</tr>
<tr>
<td>YGFNAMab</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{\partial}{\partial
y} \delta(\boldsymbol{r}-\boldsymbol{R}_K)$</span></td>
</tr>
<tr>
<td>ZGFNAMab</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{\partial}{\partial
z} \delta(\boldsymbol{r}-\boldsymbol{R}_K)$</span></td>
</tr>
<tr>
<td rowspan="3">DRHO</td>
<td rowspan="3">Gradient of the Gaussian nuclear density <span
class="math inline"><em>ρ</em><sub><em>K</em></sub><sup><em>G</em></sup>(<strong>r</strong>)</span></td>
<td rowspan="3"><blockquote>
<p>S</p>
</blockquote></td>
<td>DRHO lmn</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{\partial}{\partial
x} \rho^G_K(\boldsymbol{r})$</span></td>
</tr>
<tr>
<td>DRHO opq</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{\partial}{\partial
y} \rho^G_K(\boldsymbol{r})$</span></td>
</tr>
<tr>
<td>DRHO rst</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{\partial}{\partial
z} \rho^G_K(\boldsymbol{r})$</span></td>
</tr>
<tr>
<td rowspan="3">ANGMOM</td>
<td rowspan="3">Orbital angular momentum around CM</td>
<td rowspan="3"><blockquote>
<p>A</p>
</blockquote></td>
<td>XANGMOM</td>
<td><span class="math inline">$\hat{\Omega}_1 =
-[(\boldsymbol{r}-\boldsymbol{R}_{CM})\times\hat{\nabla}]_x$</span></td>
</tr>
<tr>
<td>YANGMOM</td>
<td><span class="math inline">$\hat{\Omega}_2 =
-[(\boldsymbol{r}-\boldsymbol{R}_{CM})\times\hat{\nabla}]_y$</span></td>
</tr>
<tr>
<td>ZANGMOM</td>
<td><span class="math inline">$\hat{\Omega}_3 =
-[(\boldsymbol{r}-\boldsymbol{R}_{CM})\times\hat{\nabla}]_z$</span></td>
</tr>
<tr>
<td rowspan="3">DIPLEN</td>
<td rowspan="3">Dipole length</td>
<td rowspan="3"><blockquote>
<p>S</p>
</blockquote></td>
<td>XDIPLEN</td>
<td><span
class="math inline"><em>Ω̂</em><sub>1</sub> = <em>x</em></span></td>
</tr>
<tr>
<td>YDIPLEN</td>
<td><span
class="math inline"><em>Ω̂</em><sub>2</sub> = <em>y</em></span></td>
</tr>
<tr>
<td>ZDIPLEN</td>
<td><span
class="math inline"><em>Ω̂</em><sub>3</sub> = <em>z</em></span></td>
</tr>
<tr>
<td rowspan="3">DIPVEL</td>
<td rowspan="3">Dipole velocity</td>
<td rowspan="3"><blockquote>
<p>A</p>
</blockquote></td>
<td>XDIPVEL</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{\partial}{\partial
x}$</span></td>
</tr>
<tr>
<td>YDIPVEL</td>
<td><span class="math inline">$\hat{\Omega}_2 = \frac{\partial}{\partial
y}$</span></td>
</tr>
<tr>
<td>ZDIPVEL</td>
<td><span class="math inline">$\hat{\Omega}_3 = \frac{\partial}{\partial
z}$</span></td>
</tr>
<tr>
<td rowspan="6">QUADRUP</td>
<td rowspan="6">Quadrupole moment</td>
<td rowspan="6"><blockquote>
<p>S</p>
</blockquote></td>
<td>XXQUADRU</td>
<td><span class="math inline">$\hat{\Omega}_1 = \frac{1}{4}
(x^2-r^2)$</span></td>
</tr>
<tr>
<td>XYQUADRU</td>
<td><span class="math inline">$\hat{\Omega}_2 = \frac{1}{4}
xy$</span></td>
</tr>
<tr>
<td>XZQUADRU</td>
<td><span class="math inline">$\hat{\Omega}_3 = \frac{1}{4}
xz$</span></td>
</tr>
<tr>
<td>YYQUADRU</td>
<td><span class="math inline">$\hat{\Omega}_4 = \frac{1}{4}
(y^2-r^2)$</span></td>
</tr>
<tr>
<td>YZQUADRU</td>
<td><span class="math inline">$\hat{\Omega}_5 = \frac{1}{4}
yz$</span></td>
</tr>
<tr>
<td>ZZQUADRU</td>
<td><span class="math inline">$\hat{\Omega}_6 = \frac{1}{4}
(z^2-r^2)$</span></td>
</tr>
<tr>
<td rowspan="6">SECMOM</td>
<td rowspan="6">Second moment</td>
<td rowspan="6"><blockquote>
<p>S</p>
</blockquote></td>
<td>XXSECMOM</td>
<td><span
class="math inline"><em>Ω̂</em><sub>1</sub> = −<em>x</em><sup>2</sup></span></td>
</tr>
<tr>
<td>XYSECMOM</td>
<td><span
class="math inline"><em>Ω̂</em><sub>2</sub> = −<em>x</em><em>y</em></span></td>
</tr>
<tr>
<td>XZSECMOM</td>
<td><span
class="math inline"><em>Ω̂</em><sub>3</sub> = −<em>x</em><em>z</em></span></td>
</tr>
<tr>
<td>YYSECMOM</td>
<td><span
class="math inline"><em>Ω̂</em><sub>4</sub> = −<em>y</em><sup>2</sup></span></td>
</tr>
<tr>
<td>YZSECMOM</td>
<td><span
class="math inline"><em>Ω̂</em><sub>5</sub> = −<em>y</em><em>z</em></span></td>
</tr>
<tr>
<td>ZZSECMOM</td>
<td><span
class="math inline"><em>Ω̂</em><sub>6</sub> = −<em>z</em><sup>2</sup></span></td>
</tr>
<tr>
<td rowspan="6">THETA</td>
<td rowspan="6">Traceless quadrupole</td>
<td rowspan="6"><blockquote>
<p>S</p>
</blockquote></td>
<td>XXTHETA</td>
<td><span class="math inline">$\hat{\Omega}_1 = - \frac{3}{2}
(x^2-\frac{1}{3}r^2)$</span></td>
</tr>
<tr>
<td>XYTHETA</td>
<td><span class="math inline">$\hat{\Omega}_2 = - \frac{3}{2}
xy$</span></td>
</tr>
<tr>
<td>XZTHETA</td>
<td><span class="math inline">$\hat{\Omega}_3 = - \frac{3}{2}
xz$</span></td>
</tr>
<tr>
<td>YYTHETA</td>
<td><span class="math inline">$\hat{\Omega}_4 = - \frac{3}{2}
(y^2-\frac{1}{3}r^2)$</span></td>
</tr>
<tr>
<td>YZTHETA</td>
<td><span class="math inline">$\hat{\Omega}_5 = - \frac{3}{2}
yz$</span></td>
</tr>
<tr>
<td>ZZTHETA</td>
<td><span class="math inline">$\hat{\Omega}_6 = - \frac{3}{2}
(z^2-\frac{1}{3}r^2)$</span></td>
</tr>
<tr>
<td rowspan="3">SEFGMG</td>
<td rowspan="3">Magnetic term of Flambaum-Ginges self-energy
potential</td>
<td rowspan="3"><blockquote>
<p>A</p>
</blockquote></td>
<td>XSEFGMG</td>
<td></td>
</tr>
<tr>
<td>YSEFGMG</td>
<td></td>
</tr>
<tr>
<td>ZSEFGMG</td>
<td></td>
</tr>
</tbody>
</table>

where $`\boldsymbol{R}_{CM}`$ and $`\boldsymbol{R}_K`$ are the position
vectors of the nuclear center of mass and the nucleus $`K`$ with respect
to the origin of coordinates, respectively. In addition, the operators
$`x`$, $`y`$, and $`z`$ are the Cartesian components of the electron
position vector operator with respect to the origin of coordinates.

Note that, in the table above, $`g_e`$ is the electronic g-factor (i.e.,
$`g_e = 2`$), 'NAM' are the first three letters of the name of atom
$`K`$ as given in the .mol file, 'ab' denotes the two-digit index of the
symmetry-adapted nucleus $`K`$, and 'lmn', 'opq', and 'rst' denote
three-digit indices identifying, respectively, the $`x`$, $`y`$, and
$`z`$ Cartesian components of the given operator associated with nucleus
$`K`$ in the symmetry-independent (non-symmetry-adapted) representation.

For the normalized Gaussian nuclear density distribution, the default
exponents ($`\zeta_K`$) are the values proposed by Visscher and Dyall in
Visscher1997b.

The matrix representations of the operators shown in the previous Table
are either symmetric (S) or antisymmetric (A). This information is
provided in the third column of the Table.

Other one-electron operator labels are:

| **Keyword** | **Description** |
|----|----|
| ANGLON | Angular momentum around the nuclei |
| CARMOM | Cartesian moments (symmetric) $`(l + 1)(l + 2)/2`$ components ($`l = i + j + k`$) |
| CM1 | First order magnetic field derivatives of electric field |
| CM2 | Second order magnetic field derivatives of electric field |
| DARWIN | Darwin-type integrals |
| DIASUS | Angular London orbital contribution to diamagnetic susceptibility |
| DSO | Diamagnetic spin-orbit integrals |
| DSUSCGO | Diamagnetic susceptibility with common gauge origin |
| DSUSLH | Angular London orbital contribution to diamagnetic susceptibility |
| DSUSNOL | Diamagnetic susceptibility without London contribution |
| ELFGRDC | Electric field gradient at the individual nuclei, cartesian |
| ELFGRDS | Electric field gradient at the individual nuclei, spherical |
| EXPIKR | Cosine and sine |
| HBDO | Half B-differentiated overlap matrix |
| HDO | Half-derivative overlap integrals |
| HDOBR | Ket-differentiation of HDO-integrals with respect to magnetic field |
| LONMOM | London orbital contribution to angular momentum |
| MAGMOM | One-electron contributions to magnetic moment |
| MASSVEL | Mass velocity |
| NEFIELD | Electric field at the individual nuclei |
| NSTCGO | Nuclear shielding with common gauge origin |
| NUCPOT | Potential energy of the interaction of electrons with individual nuclei, divided by the nuclear charge |
| NUCSHI | Nuclear shielding tensor |
| NUCSLO | London orbital contribution to nuclear shielding tensor |
| NUCSNLO | Nuclear shielding integrals without London orbital contribution |
| PSO | Paramagnetic spin-orbit |
| S1MAG | Second order contribution from overlap matrix to magnetic properties |
| S1MAGL | Bra-differentiation of overlap matrix with respect to magnetic field |
| S1MAGR | Ket-differentiation of overlap matrix with respect to magnetic field |
| SDFC | Spin-dipole + Fermi contact |
| SOLVENT | Electronic solvent integrals |
| SPHMOM | Spherical moments (real combinations), symmetric, ($`2l + 1`$) components ($`m = 0, -1, +1, ..., -l, +l`$) |
| SPIN-DI | Spin-dipole integrals |
| SPNORB | Spatial spin-orbit |
| SQHDO | Half-derivative overlap integrals not to be antisymmetrized |
| SQHDOR | Half-derivative overlap integrals not to be anti-symmetrized |
| SQOVLAP | Second order derivatives overlap integrals |
| UEHLING | Uehling potential |
| SEFGLF | Low-frequency term of Flambaum-Ginges self-energy potential |
| SEFGHF | High-frequency term of Flambaum-Ginges self-energy potential |
| SEPZ | Pyykko-Zhao self-energy model potential |

## Examples of using various operators

We give here several concrete examples on how to construct operators for
various properties.

### Kinetic part of the Dirac Hamiltonian

The kinetic part of the electronic Dirac Hamiltonian is given (in a.u.)
by

``` math
\hat{H}^K & = c\vec{\boldsymbol{\alpha}}\cdot\hat{\vec{p}} = c\left(\boldsymbol{\alpha}_x\hat{p}_x+\boldsymbol{\alpha}_y\hat{p}_y+\boldsymbol{\alpha}_z\hat{p}_z\right)
```
``` math
& = -c \left[ i\boldsymbol{\alpha}_x \frac{\partial}{\partial x} + i\boldsymbol{\alpha}_y \frac{\partial}{\partial y} + i\boldsymbol{\alpha}_z \frac{\partial}{\partial z} \right].
```

This operator may be specified by:

    .OPERATOR
     'Kin energy'
     ALPHADOT
     XDIPVEL
     YDIPVEL
     ZDIPVEL
     COMFACTOR
     -137.03599926085

where 137.03599926085 is the CODATA22 value for $`c`$.

The speed of light $`c`$ is an important parameter in relativistic
theory, but its explicit value in atomic units is not necessarily
remembered. A simpler way to specify the kinetic energy operator is
therefore:

    .OPERATOR
     'Kin energy'
     ALPHADOT
     XDIPVEL
     YDIPVEL
     ZDIPVEL
     CMULT
     COMFACTOR
     -1

### Magnetic interactions

Another example is the operator that describes the electromagnetic
interaction between electrons and a uniform external magnetic field:
$`\hat{H}^B=-\frac{c}{2}\vec{B}\cdot\vec{\boldsymbol{\alpha}}\times\hat{\vec{r}}`$.

The operator to use in this case is
$`-\frac{c}{2}\vec{\boldsymbol{\alpha}}\times\vec{r}`$, and in DIRAC
each component is calculated individually. For example, the
$`x`$-component of this operator is
$`-\frac{c}{2}\left(\boldsymbol{\alpha}_y z - \boldsymbol{\alpha}_z y\right)`$,
and can be requested using:

    .OPERATOR
     'B_x'
     XAVECTOR
     ZDIPLEN
     YDIPLEN
     CMULT
     COMFACTOR
     -0.5

The program will assume all operators to be Hermitian. The operator
`XAVECTOR` is defined as
$`f_1 i\boldsymbol{\alpha}_y \hat{\Omega}_z-f_2 i\boldsymbol{\alpha}_z \hat{\Omega}_y`$
(see [Operator type keywords](#operator-type-keywords)), and in this
particular case the scalar operators are $`\hat{\Omega}_z=z`$ and
$`\hat{\Omega}_y=y`$ (see `ZDIPLEN` and `YDIPLEN` in [One-electron
scalar operator labels](#one-electron-scalar-operator-labels)).
Additionally, $`f_1=f_2=-\frac{c}{2}`$. Then, DIRAC will assume the
operator to be
$`-\frac{c}{2}\left(i\boldsymbol{\alpha}_y z - i\boldsymbol{\alpha}_z y\right)`$.

The imaginary phase $`i`$ makes the operator time-reversal symmetric.

### Diagonal operators

If no other argument is given, the program assumes the operator to be
diagonal and expects the operator name to be the component label, for
instance:

    .OPERATOR
     OVERLAP

### Finite-field perturbation: Electric dipole moment

Another example is a finite-perturbation calculation where the perturbed
Hamiltonian operator arising from the interaction between the electrons
and a uniform external electric field oriented along the $`z`$-axis
($`\hat{H}'=-q E_z \hat{z}=e E_z \hat{z}`$) is added to the unperturbed
Hamiltonian $`\hat{H}_0`$ defined in `**HAMILTONIAN` (default:
Dirac-Coulomb).

If we take $`e E_z = 0.01`$ (in a.u.), the $`\hat{z}`$ dipole length
operator will be used to build the total Hamiltonian
$`\hat{H} = \hat{H}_0 + 0.01 \, \hat{z}`$ by adding the perturbed
Hamiltonian $`\hat{H}'`$ to the Dirac-Coulomb Hamiltonian $`\hat{H}_0`$
using:

    **HAMILTONIAN
    .OPERATOR
     ZDIPLEN
     COMFACTOR
     0.01

<span class="title-ref">Note:</span> Don't forget to decrease the
symmetry of your system.

### Fermi-contact integrals

Here is an example where the Fermi-contact (FC) integrals for a certain
nucleus $`K`$ are added to the Hamiltonian in a finite-field
calculation.

Let's assume we are looking at a PbX dimer (order in the .mol file: 1.
Pb, 2. X) and we want to add (under `**HAMILTONIAN`) the FC interaction
for the Pb nucleus to the unperturbed Hamiltonian $`\hat{H}_0`$ as a
perturbation with a given field strength of $`10^{-9}`$ a.u..

The FC integrals are called using the label `'FC NAMab'` (see
[One-electron scalar operator
labels](#one-electron-scalar-operator-labels)), and the total perturbed
Hamiltonian $`\hat{H}=\hat{H}_0+\hat{H}'`$ (where, in a.u.,
$`\hat{H}'=10^{-9} \, \frac{4\pi}{3} \, g_e \, \delta(\boldsymbol{r}-\boldsymbol{R}_\text{Pb})`$)
can be requested using:

    **HAMILTONIAN
    .OPERATOR
     'Density at Pb nucleus'
     DIAGONAL
     'FC Pb 01'
     FACTORS
     0.000000001

**Important note:** The values obtained after the fit of the
finite-field energies need to be scaled by
$`\frac{3}{4 \pi g_{e}} = \frac{1}{8.3872954891254192}`$ in order to get
the raw density $`\delta(\boldsymbol{r}-\boldsymbol{R}_\text{Pb})`$.

The next example shows how to calculate the electronic density at the
different nuclei in the molecule, as the expectation values
$`\langle -\delta(\boldsymbol{r}-\boldsymbol{R}_K) \rangle`$ (for each
nucleus $`K`$ in the system) for a Dirac-Coulomb Hartree-Fock wave
function including a decomposition of the molecular orbital
contributions to the density:

    **DIRAC
    .WAVE FUNCTION
    .PROPERTIES
    **HAMILTONIAN
    **WAVE FUNCTION
    .SCF
    **PROPERTIES
    .RHONUC
    *EXPECTATION
    .ORBANA
    *END OF

For further details, see `PROPERTIES_.RHONUC`.

### Cartesian moment expectation value

In the following example we calculate the cartesian multipole moment
expectation value $`\langle x^1 y^2 z^3 \rangle`$ for a Levy-Leblond
Hartree-Fock wave function:

    **DIRAC
    .WAVE FUNCTION
    .PROPERTIES
    **HAMILTONIAN
    .LEVY-LEBLOND
    **WAVE FUNCTION
    .SCF
    **PROPERTIES
    *EXPECTATION
    .OPERATOR
     CM010203
    *END OF

## Quaternion algebra and time-reversal symmetry

The sixteen $`4 \times 4`$ matrices listed in Section [Syntax for
specifying one-electron
operators](#syntax-for-specifying-one-electron-operators), i.e.,
$`\boldsymbol{I}`$, $`\boldsymbol{\beta}`$, $`\boldsymbol{\gamma}^5`$,
$`\boldsymbol{\beta}\boldsymbol{\gamma}^5`$, $`i\boldsymbol{\alpha}_x`$,
$`i\boldsymbol{\alpha}_y`$, $`i\boldsymbol{\alpha}_z`$,
$`i\boldsymbol{\beta}\boldsymbol{\alpha}_x`$,
$`i\boldsymbol{\beta}\boldsymbol{\alpha}_y`$,
$`i\boldsymbol{\beta}\boldsymbol{\alpha}_z`$,
$`i\boldsymbol{\Sigma}_x`$, $`i\boldsymbol{\Sigma}_y`$,
$`i\boldsymbol{\Sigma}_z`$,
$`i\boldsymbol{\beta}\boldsymbol{\Sigma}_x`$,
$`i\boldsymbol{\beta}\boldsymbol{\Sigma}_y`$, and
$`i\boldsymbol{\beta}\boldsymbol{\Sigma}_z`$, are usually expressed in
Molecular Physics using the standard representation where, for example,

``` math
\begin{aligned}
\tilde{\boldsymbol{\beta}} =
\begin{pmatrix}
  1 & 0 & 0  &  0 \\
  0 & 1 & 0  &  0 \\
  0 & 0 & -1 &  0 \\
  0 & 0 & 0  & -1
\end{pmatrix}, \;
\tilde{\boldsymbol{\gamma}}^{5} =
\begin{pmatrix}
  0 & 0 & 1 & 0 \\
  0 & 0 & 0 & 1 \\
  1 & 0 & 0 & 0 \\
  0 & 1 & 0 & 0
\end{pmatrix}, \;
i\tilde{\boldsymbol{\alpha}}_z =
\begin{pmatrix}
  0  & 0  & i  &  0 \\
  0  & 0  & 0  & -i \\
  i  & 0  & 0  &  0 \\
  0  & -i & 0  &  0
\end{pmatrix}.
\end{aligned}
```

Here and in what follows, we use the notation $`\tilde{\boldsymbol{U}}`$
to refer to any of these sixteen matrices when they are expressed in the
standard representation.

It can be seen that all these matrices can be written as the Kronecker
product of two $`2 \times 2`$ matrices $`\boldsymbol{A}`$ and
$`\boldsymbol{B}`$, such that
$`\tilde{\boldsymbol{U}} = \boldsymbol{A} \otimes \boldsymbol{B}`$. In
particular, $`\boldsymbol{A}`$ is one of the real matrices
$`\boldsymbol{I}`$, $`\boldsymbol{\sigma}_x`$,
$`i \boldsymbol{\sigma}_y`$, or $`\boldsymbol{\sigma}_z`$, whereas
$`\boldsymbol{B}`$ is $`\boldsymbol{I}`$, $`i \boldsymbol{\sigma}_x`$,
$`i \boldsymbol{\sigma}_y`$, or $`i \boldsymbol{\sigma}_z`$. Here,
$`\boldsymbol{\sigma}_x`$, $`\boldsymbol{\sigma}_y`$, and
$`\boldsymbol{\sigma}_z`$ are the Pauli matrices, and $`i`$ is the
imaginary unit.

The individual matrix elements of $`\tilde{\boldsymbol{U}}`$ are given
as

``` math
\tilde{U}_{pq} = A_{ij} \, B_{kl} \, ,
```

where $`p=2(i-1)+k`$ and $`q=2(j-1)+l`$.

In the DIRAC code, the structure of the Dirac equation with respect to
time-reversal symmetry is displayed by using reordered Dirac 4-spinors,
where the components are grouped on spin labels ($`\alpha`$, $`\beta`$)
rather than large (upper) and small (lower) components ($`L`$, $`S`$),

``` math
\begin{aligned}
\begin{pmatrix}
\psi^{L\alpha} \\ \psi^{L\beta} \\ \psi^{S\alpha} \\ \psi^{S\beta}
\end{pmatrix}
\rightarrow
\begin{pmatrix}
\psi^{L\alpha} \\ \psi^{S\alpha} \\ \psi^{L\beta} \\ \psi^{S\beta}
\end{pmatrix}.
\end{aligned}
```

The same reordering applies to all the matrices
$`\tilde{\boldsymbol{U}}`$, which are then transformed accordingly. In
the original ordering (i.e., in the standard representation), these
matrices are given as

``` math
\begin{aligned}
\tilde{\boldsymbol{U}} =
\begin{pmatrix}
\tilde{U}_{L\alpha,L\alpha} & \tilde{U}_{L\alpha,L\beta} & \tilde{U}_{L\alpha,S\alpha} & \tilde{U}_{L\alpha,S\beta} \\
\tilde{U}_{L\beta,L\alpha}  & \tilde{U}_{L\beta,L\beta}  & \tilde{U}_{L\beta,S\alpha}  & \tilde{U}_{L\beta,S\beta}  \\
\tilde{U}_{S\alpha,L\alpha} & \tilde{U}_{S\alpha,L\beta} & \tilde{U}_{S\alpha,S\alpha} & \tilde{U}_{S\alpha,S\beta} \\
\tilde{U}_{S\beta,L\alpha}  & \tilde{U}_{S\beta,L\beta}  & \tilde{U}_{S\beta,S\alpha}  & \tilde{U}_{S\beta,S\beta}
\end{pmatrix}
= \boldsymbol{A} \otimes \boldsymbol{B}.
\end{aligned}
```

However, after reordering one clearly has the transformed matrices

``` math
\begin{aligned}
\boldsymbol{U} =
\begin{pmatrix}
\tilde{U}_{L\alpha,L\alpha} & \tilde{U}_{L\alpha,S\alpha} & \tilde{U}_{L\alpha,L\beta} & \tilde{U}_{L\alpha,S\beta} \\
\tilde{U}_{S\alpha,L\alpha} & \tilde{U}_{S\alpha,S\alpha} & \tilde{U}_{S\alpha,L\beta} & \tilde{U}_{S\alpha,S\beta} \\
\tilde{U}_{L\beta,L\alpha}  & \tilde{U}_{L\beta,S\alpha}  & \tilde{U}_{L\beta,L\beta}  & \tilde{U}_{L\beta,S\beta}  \\
\tilde{U}_{S\beta,L\alpha}  & \tilde{U}_{S\beta,S\alpha}  & \tilde{U}_{S\beta,L\beta}  & \tilde{U}_{S\beta,S\beta}
\end{pmatrix},
\end{aligned}
```

which satisfy

``` math
\boldsymbol{U} = \boldsymbol{P} \, \tilde{\boldsymbol{U}} \, \boldsymbol{P}^{-1} =
\boldsymbol{P} \, ( \boldsymbol{A} \otimes \boldsymbol{B} ) \, \boldsymbol{P}^{-1} =
\boldsymbol{B} \otimes \boldsymbol{A},
```

where the $`4 \times 4`$ Kronecker permutation matrix $`\boldsymbol{P}`$
is a special permutation matrix that swaps the order of the Kronecker
product of two $`2 \times 2`$ matrices, and is given by

``` math
\begin{aligned}
\boldsymbol{P} =
\begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}.
\end{aligned}
```

\[Note that applying $`\boldsymbol{P}`$ on the left permutes the rows,
while applying $`\boldsymbol{P}^{-1}`$ on the right permutes the
columns, ensuring the correct transformation from
$`\boldsymbol{A} \otimes \boldsymbol{B}`$ to
$`\boldsymbol{B} \otimes \boldsymbol{A}`$.\]

In other words, $`U_{PQ} = B_{kl} \, A_{ij}`$, with $`P=2(k-1)+i`$ and
$`Q=2(l-1)+j`$. This reordering guarantees that all the matrices
$`\boldsymbol{U}`$ are time-reversal symmetric.

Additionally, following the arguments given in Ref. Saue1999, it is
possible to make the following mapping involving the quaternion units:

``` math
&\boldsymbol{I}          &\rightarrow & \qquad  1
```
``` math
&i \boldsymbol{\sigma}_z &\rightarrow & \qquad  \check{\imath}
```
``` math
&i \boldsymbol{\sigma}_y &\rightarrow & \qquad  \check{\jmath}
```
``` math
&i \boldsymbol{\sigma}_x &\rightarrow & \qquad  \check{k}
```

or, in a general way, the $`2 \times 2`$ matrices $`\boldsymbol{B}`$ can
be mapped as $`\boldsymbol{B} \rightarrow b`$, where $`b`$ is a
quaternion unit (i.e., $`1`$, $`\check{\imath}`$, $`\check{\jmath}`$, or
$`\check{k}`$) arising from the relationships given above.

Thus, any linear combination of the 16 reordered matrices
$`\boldsymbol{U}`$ can be mapped to a linear combination of
$`2 \times 2`$ matrices $`\boldsymbol{u}`$, each of which contains a
quaternion unit.

To make this discussion clearer, in the following we discuss two
examples. In first place, we transform the $`4 \times 4`$ matrix
$`\tilde{\boldsymbol{\beta}}`$. We start by writting this matrix as the
following Kronecker product of two $`2 \times 2`$ matrices:
$`\tilde{\boldsymbol{\beta}}=\boldsymbol{\sigma}_z \otimes \boldsymbol{I}`$.
In the reordered representation, this matrix will be expressed as
$`\boldsymbol{\beta}=\boldsymbol{I} \otimes \boldsymbol{\sigma}_z`$.
Finally, the corresponding $`\boldsymbol{B}`$ matrix (in this case,
$`\boldsymbol{I}`$) is transformed to a quaternion unit (in the current
example, $`1`$). In this way, the $`4 \times 4`$ reordered matrix
$`\boldsymbol{\beta}`$ can be written in quaternion form as the
$`2 \times 2`$ matrix $`\boldsymbol{\sigma}_z`$. In other words:

``` math
\begin{aligned}
\tilde{\boldsymbol{\beta}} =
\begin{pmatrix}
  1 & 0 & 0  &  0 \\
  0 & 1 & 0  &  0 \\
  0 & 0 & -1 &  0 \\
  0 & 0 & 0  & -1
\end{pmatrix}
\;\rightarrow\;
\boldsymbol{\beta} =
\boldsymbol{P}\,\tilde{\boldsymbol{\beta}}\,\boldsymbol{P}^{-1} =
\begin{pmatrix}
  1 & 0  & 0 &  0 \\
  0 & -1 & 0 &  0 \\
  0 & 0  & 1 &  0 \\
  0 & 0  & 0 & -1
\end{pmatrix}
\;\rightarrow\;
\boldsymbol{\beta}^\text{quat} =
\boldsymbol{\sigma}_z.
\end{aligned}
```

As a second example, we take $`i\tilde{\boldsymbol{\alpha}}_z`$. This
$`4 \times 4`$ matrix, in its original ordering (according to the
standard representation), can be written as the Kronecker product
$`i\tilde{\boldsymbol{\alpha}}_z=\boldsymbol{\sigma}_x \otimes i \boldsymbol{\sigma}_z`$.
This means that in the reordered representation it will be given as
$`i\boldsymbol{\alpha}_z = i \boldsymbol{\sigma}_z \otimes \boldsymbol{\sigma}_x`$.
In this particular case, the corresponding matrix $`\boldsymbol{B}`$ is
$`i \boldsymbol{\sigma}_z`$, so that the quaternion unit $`b`$ is
$`\check{\imath}`$. Therefore, the quaternion form of this reordered
matrix will be
$`i\boldsymbol{\alpha}_z^\text{quat} = \check{\imath} \boldsymbol{\sigma}_x`$.
Equivalently:

``` math
\begin{aligned}
i\tilde{\boldsymbol{\alpha}}_z =
\begin{pmatrix}
  0 & 0  & i  &  0  \\
  0 & 0  & 0  &  -i \\
  i & 0  & 0  &  0  \\
  0 & -i & 0  &  0
\end{pmatrix}
\;\rightarrow\;
i\boldsymbol{\alpha}_z =
\boldsymbol{P}\,i\tilde{\boldsymbol{\alpha}}_z\,\boldsymbol{P}^{-1} =
\begin{pmatrix}
  0 & i  & 0  &  0  \\
  i & 0  & 0  &  0  \\
  0 & 0  & 0  &  -i \\
  0 & 0  & -i &  0
\end{pmatrix}
\;\rightarrow\;
i\boldsymbol{\alpha}_z^\text{quat} =
\check{\imath} \boldsymbol{\sigma}_x.
\end{aligned}
```

The relationships between matrices $`\tilde{\boldsymbol{U}}`$ (i.e., the
sixteen $`4 \times 4`$ matrices in original ordering, which corresponds
to the standard representation) and the quaternion form of their
reordered equivalent $`2 \times 2`$ matrices $`\boldsymbol{u}`$ is given
in the following Table. They follow the transformation rule
$`\tilde{\boldsymbol{U}} = \boldsymbol{A} \otimes \boldsymbol{B}`$
$`\rightarrow`$
$`\boldsymbol{U} = \boldsymbol{P} \, \tilde{\boldsymbol{U}} \, \boldsymbol{P}^{-1} = \boldsymbol{P} \, (\boldsymbol{A} \otimes \boldsymbol{B}) \, \boldsymbol{P}^{-1} = \boldsymbol{B} \otimes \boldsymbol{A}`$
$`\rightarrow`$ $`\boldsymbol{u} = b \boldsymbol{A}`$.

| $`\tilde{\boldsymbol{U}}`$ | $`\boldsymbol{A} \otimes \boldsymbol{B}`$ | $`\boldsymbol{u}`$ |
|----|----|----|
| $`\tilde{\boldsymbol{I}}`$ | $`\boldsymbol{I} \otimes \boldsymbol{I}`$ | $`\boldsymbol{I}`$ |
| $`i\tilde{\boldsymbol{\Sigma}}_z`$ | $`\boldsymbol{I} \otimes i \boldsymbol{\sigma}_z`$ | $`\check{\imath}\, \boldsymbol{I}`$ |
| $`i\tilde{\boldsymbol{\Sigma}}_y`$ | $`\boldsymbol{I} \otimes i \boldsymbol{\sigma}_y`$ | $`\check{\jmath}\, \boldsymbol{I}`$ |
| $`i\tilde{\boldsymbol{\Sigma}}_x`$ | $`\boldsymbol{I} \otimes i \boldsymbol{\sigma}_x`$ | $`\check{k}\, \boldsymbol{I}`$ |
| $`\tilde{\boldsymbol{\beta}}`$ | $`\boldsymbol{\sigma}_z \otimes \boldsymbol{I}`$ | $`\boldsymbol{\sigma}_z`$ |
| $`i\tilde{\boldsymbol{\beta}}\tilde{\boldsymbol{\Sigma}}_z`$ | $`\boldsymbol{\sigma}_z \otimes i \boldsymbol{\sigma}_z`$ | $`\check{\imath}\, \boldsymbol{\sigma}_z`$ |
| $`i\tilde{\boldsymbol{\beta}}\tilde{\boldsymbol{\Sigma}}_y`$ | $`\boldsymbol{\sigma}_z \otimes i \boldsymbol{\sigma}_y`$ | $`\check{\jmath}\, \boldsymbol{\sigma}_z`$ |
| $`i\tilde{\boldsymbol{\beta}}\tilde{\boldsymbol{\Sigma}}_x`$ | $`\boldsymbol{\sigma}_z \otimes i \boldsymbol{\sigma}_x`$ | $`\check{k}\, \boldsymbol{\sigma}_z`$ |
| $`\tilde{\boldsymbol{\beta}}\tilde{\boldsymbol{\gamma}}^5`$ | $`(i\boldsymbol{\sigma}_y) \otimes \boldsymbol{I}`$ | $`(i\boldsymbol{\sigma}_y)`$ |
| $`i\tilde{\boldsymbol{\beta}}\tilde{\boldsymbol{\alpha}}_z`$ | $`(i\boldsymbol{\sigma}_y) \otimes i \boldsymbol{\sigma}_z`$ | $`\check{\imath}\, (i\boldsymbol{\sigma}_y)`$ |
| $`i\tilde{\boldsymbol{\beta}}\tilde{\boldsymbol{\alpha}}_y`$ | $`(i\boldsymbol{\sigma}_y) \otimes i \boldsymbol{\sigma}_y`$ | $`\check{\jmath}\, (i\boldsymbol{\sigma}_y)`$ |
| $`i\tilde{\boldsymbol{\beta}}\tilde{\boldsymbol{\alpha}}_x`$ | $`(i\boldsymbol{\sigma}_y) \otimes i \boldsymbol{\sigma}_x`$ | $`\check{k}\, (i\boldsymbol{\sigma}_y)`$ |
| $`\tilde{\boldsymbol{\gamma}}^5`$ | $`\boldsymbol{\sigma}_x \otimes \boldsymbol{I}`$ | $`\boldsymbol{\sigma}_x`$ |
| $`i\tilde{\boldsymbol{\alpha}}_z`$ | $`\boldsymbol{\sigma}_x \otimes i \boldsymbol{\sigma}_z`$ | $`\check{\imath}\, \boldsymbol{\sigma}_x`$ |
| $`i\tilde{\boldsymbol{\alpha}}_y`$ | $`\boldsymbol{\sigma}_x \otimes i \boldsymbol{\sigma}_y`$ | $`\check{\jmath}\, \boldsymbol{\sigma}_x`$ |
| $`i\tilde{\boldsymbol{\alpha}}_x`$ | $`\boldsymbol{\sigma}_x \otimes i \boldsymbol{\sigma}_x`$ | $`\check{k}\, \boldsymbol{\sigma}_x`$ |

It is important to note that although the time-reversal antisymmetric
operators $`\boldsymbol{\alpha}_k`$, $`\boldsymbol{\Sigma}_k`$,
$`\boldsymbol{\beta\alpha}_k`$, and $`\boldsymbol{\beta\Sigma}_k`$ (with
$`k=x,y`$, and $`z`$) become time-reversal symmetric once they are
multiplied by the imaginary unit $`i`$ and written in quaternion form as
described above, DIRAC saves the information that these operators were
originally T-reversal antisymmetric (before adding the imaginary phase).

For further details on the quaternion symmetry scheme used in DIRAC, see
Ref. Saue1999.

<span class="title-ref">Note:</span> For complete consistency, the names
of the operators related to $`i\boldsymbol{\alpha}_k`$,
$`i\boldsymbol{\Sigma}_k`$, $`i\boldsymbol{\beta\alpha}_k`$, and
$`i\boldsymbol{\beta\Sigma}_k`$ ($`k=x,y,z`$) could have been `XiALPHA`,
`YiALPHA`, `ZiALPHA`, `XiAVECTOR`, `YiAVECTOR`, `ZiAVECTOR`,
`iALPHADOT`, `XiBETAALP`, `YiBETAALP`, `ZiBETAALP`, `XiSIGMA`,
`YiSIGMA`, `ZiSIGMA`, `XiBETASIG`, `YiBETASIG`, `ZiBETASIG`,
`iSIGMADOT`, and `iBEALDOT` (see [Operator type
keywords](#operator-type-keywords)). However, due to historical reasons,
this convention is not used in DIRAC.

<span class="title-ref">Note for advanced users:</span> The operator
iBETAGAM is implemented in the code just as
$`\beta\boldsymbol{\gamma}^5`$, but its time-reversal symmetry without
imaginary phase is set as odd.
