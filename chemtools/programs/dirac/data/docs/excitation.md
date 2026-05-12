orphan  

<div class="index">

\*EXCITATION ENERGIES

</div>

# \*EXCITATION ENERGIES

Calculate excitation energies using time dependent Hartree-Fock or DFT.
The excitation energies are found as the lowest generalized eigenvalues
of the electronic Hessian. DIRAC supports TDDFT kernels from all ground
state functionals included in the code. Currently the iterative
eigenvalue solver may fail to converge more than about twenty roots per
symmetry.

## Define excitations and transition moments

<div class="index">

.EXCITA

</div>

### .EXCITA

    .EXCITA
    SYM N

Number of excitation energies N calculated in boson symmetry no. SYM.
This keyword can be repeated if you want excitation energies in more
than one boson symmetry.

<div class="index">

.OPERATOR

</div>

### .OPERATOR

Specification of a transition moment operator (see
`one_electron_operators` for details). This keyword can be given
multiple times to add more operators.

<div class="index">

.EPOLE

</div>

### .EPOLE

Specification of electric Cartesian multipole operators of order $`n`$

``` math
\hat{Q}_{j_{1}\ldots j_{n}}^{\left[n\right]}=-er_{1}r_{2}\ldots r_{j_{n}}
```

for the calculation of transition moments (note that they contribute to
one order less in the wave vector). Specify order.

*Example:* Electric dipole operators:

    .EPOLE
    1

<div class="index">

.MPOLE

</div>

### .MPOLE

Specification of magnetic Cartesian multipole operators of order $`n`$

``` math
\hat{m}_{j_{1}\ldots j_{n-1};j_{n}}^{\left[n\right]}=\frac{n}{n+1}r_{j_{1}}r_{j_{2}}\ldots r_{j_{n-1}}(\boldsymbol{r}\times\hat{\mathbf{j}})_{j_{n}};\quad\hat{\mathbf{j}}=-ec\boldsymbol{\alpha}
```

for the calculation of transition moments (note that they contribute to
the same order in the wave vector). Specify order.

*Example:* Magnetic dipole operators:

    .MPOLE
    1

<div class="index">

.VPOLE

</div>

### .VPOLE

Specification of electric Cartesian multipole operators of order $`n`$
in the velocity representation

``` math
\hat{\mathcal{Q}}^{[n]}_{j_1\ldots j_{n-1};j_n} = \frac{ie}{\omega}r_{j_1}\ldots r_{j_{n-2}}(c\alpha_{j_n} r_{j_{n-1}} + nc\alpha_{j_{n-1}}r_{j_n})
```

*Example:* Velocity representation electric dipole operator:

    .VPOLE
    1

<div class="index">

.ANALYZE

</div>

### .ANALYZE

Analyze solution vectors and show the most important excitations at the
orbital level.

<div class="index">

.INTENS

</div>

### .INTENS

Invoke calculation of oscillator strengths to order k in the wave
vector. Default order is zero, which corresponds to the widely used
electric-dipole approximation. By default results are given both in the
length and the velocity representation, but a selection can be made
using keywords `EXCITATION_ENERGIES_.NOLENR` and
`EXCITATION_ENERGIES_.NOVELR`. Only even orders contribute. For further
details, see List_JCP2020

*Example:* :

    .INTENS
    2

<div class="index">

.BED

</div>

### .BED

Invoke calculation of oscillator strengths using the full operator
coupling the molecule to an electromagnetic plane wave. The oscillator
strength is then given by

``` math
f_{n\leftarrow 0}=\frac{2\omega}{\hbar e^{2}}\left|\langle n|T\left(\omega\right)|0\rangle\right|^2
```

where appears the effective interaction operator

``` math
T\left(\omega\right)=\frac{ec}{\omega}\left(\boldsymbol{\alpha}\cdot\boldsymbol{\epsilon}\right)e^{+i\left(\mathbf{k}\cdot\mathbf{r}\right)}.
```

In the above expression $`\mathbf{k}`$ refers to the wave vector of
length $`k=\omega/c`$ and direction $`\mathbf{m}`$, whereas the
polarization of the electric component is specificed by
$`\boldsymbol{\epsilon}`$.

Since experiment is typically carried out in an isotropic medium,
rotational average is performed. Rather than rotate the molecule we
shall rotate the experimental apparatus. To rotate we use the unit
vectors of the spherical coordinates

``` math
\begin{aligned}
\begin{array}{lcl} \mathbf{e}_{r} & = & \mathbf{e}_{x}\sin\theta\cos\phi+\mathbf{e}_{y}\sin\theta\sin\phi+\mathbf{e}_{z}\cos\theta\\
\mathbf{e}_{\theta} & = & \mathbf{e}_{x}\cos\theta\cos\phi+\mathbf{e}_{y}\cos\theta\sin\phi-\mathbf{e}_{z}\sin\theta\\ \mathbf{e}_{\phi} & = & -\mathbf{e}_{x}\sin\phi+\mathbf{e}_{y}\cos\phi \end{array}
\end{aligned}
```

We now choose to align the wave vector with the $`\mathbf{e}_r`$ unit
vector, that is

``` math
\mathbf{k} = k\mathbf{e}_r
```

This means that the polarization vector $`\boldsymbol{\epsilon}`$ is in
the plane spanned by the unit vectors $`\mathbf{e}_{\theta}`$ and
$`\mathbf{e}_{\phi}`$. We therefore set

``` math
\boldsymbol{\epsilon} = \cos\chi \mathbf{e}_{\theta}+\sin\chi\mathbf{e}_{\phi}
```

We see the solid angle $`\left(\theta,\phi\right)`$ gives all possible
directions of the wave vector $`\mathbf{k}`$, wheras the angle $`\chi`$
provides all possible orientations of the polarization vector
$`\boldsymbol{\epsilon}`$ in the plane perpendicular to $`\mathbf{k}`$.

The general expression for the rotational average will be

``` math
\left\langle f\left(\boldsymbol{r}\right)\right\rangle _{\theta,\phi,\chi}=\frac{1}{8\pi^{2}}\int_{0}^{2\pi}\int_{0}^{2\pi}\int_{0}^{\pi}f\left(\boldsymbol{r}\right)\sin\theta d\theta d\phi d\chi.
```

In our case we have

``` math
\left\langle f_{n\leftarrow 0}\right\rangle _{\theta,\phi,\chi} = \frac{2\omega}{\hbar e^{2}}\left\langle \epsilon_{\alpha}\epsilon_{\beta}\langle n|\frac{ec}{\omega}\alpha_{\alpha}e^{+i\left(\boldsymbol{k}\cdot\boldsymbol{r}\right)}|0\rangle\langle n|\frac{ec}{\omega}\alpha_{\beta}e^{+i\left(\boldsymbol{k}\cdot\boldsymbol{r}\right)}|0\rangle^{\ast}\right\rangle _{\theta,\phi,\chi},
```

which simplifies to

``` math
\left\langle f_{n\leftarrow 0}\right\rangle _{\theta,\phi,\chi} = \frac{2\omega}{\hbar e^{2}}\left\langle \left\langle\epsilon_{\alpha}\epsilon_{\beta}\right\rangle _{\chi}\langle n|\frac{ec}{\omega}\alpha_{\alpha}e^{+i\left(\boldsymbol{k}\cdot\boldsymbol{r}\right)}|0\rangle\langle n|\frac{ec}{\omega}\alpha_{\beta}e^{+i\left(\boldsymbol{k}\cdot\boldsymbol{r}\right)}|0\rangle^{\ast}\right\rangle _{\theta,\phi},
```

since only the polarization vectors depend on the angle $`\chi`$. The
average over the angle $`\chi`$ can be expressed compactly in terms of
the wave unit vector $`\mathbf{m}`$ ($`\mathbf{k}=k\mathbf{m}`$)

``` math
\left\langle\epsilon_{\alpha}\epsilon_{\beta}\right\rangle _{\chi}=\frac{1}{2}\left(\delta_{\alpha\beta}-m_{\alpha}m_{\beta}\right),
```

whereas the average over angles $`\theta`$ and $`\phi`$ is handled by
[Lebedev quadrature](https://en.wikipedia.org/wiki/Lebedev_quadrature) .

A full account is given in List_JCP2020

<div class="index">

.BEDECD

</div>

### .BEDECD

Invoke calculation of the differential oscillator strengths using the
full interaction operator. This quantity is calculated as the difference
between the oscillator strengths corresponding to left- and right-handed
circularly polarized light. The differential oscillator strenght is
given by

``` math
\Delta f= f_{\text{L}}-f_{\text{R}}=-i\frac{2m_e\omega}{\hbar e^2}\mathbf{e}_{k}\cdot\left(\mathbf{T}\times\mathbf{T}^{\ast}\right),
```

where the transition moments are written in vector form,
$`\mathbf{T}=\langle f|\frac{e}{\omega} c\boldsymbol{\alpha}e^{i\mathbf{k}\cdot\mathbf{r}}|i\rangle`$,
which allows the compact cross-product form in previous equation.

Isotropic averaging of the differential oscillator strength is handled
in an analogous manner as `EXCITATION_ENERGIES_.BED`, with the main
difference being that in the current case, only two angles are required.
The redundancy of the angle $`\chi`$ is a manifestation of axial
symmetry. The rotational average can be written as

``` math
\left\langle \Delta f_{n\leftarrow 0}\right\rangle _{\theta,\phi} = -i\frac{2m_e\omega}{\hbar e^2}\left\langle n|\mathbf{e}_{k}\cdot\left(\mathbf{T}\times\mathbf{T}^{\ast}\right)\right\rangle _{\theta,\phi}.
```

For a full account see vanHorn2021probing.

<div class="index">

.ANGPLOT

</div>

### .ANGPLOT

Print the anisotropic differential oscillator strength (see.
`EXCITATION_ENERGIES_.BEDECD`) for every point on the Lebedev grid.
These points can be used to plot the differential oscillator strength as
a function of the incident angle of the radiation. A full account is
given in vanHorn2021probing.

<div class="index">

.ORIENT

</div>

### .ORIENT

Specify fixed experimental configuration (no rotational average). The
orientation of the wave and polarization vector is given by
specification of the angles $`\theta`$, $`\phi`$ and $`\chi`$, see the
`EXCITATION_ENERGIES_.BED` keyword for more details. For instance, to
specify that the wave vector is along the $`z`$ -axis and the
polarization vector along the $`x`$ - axis, we set

    .ORIENT
    0.0 0.0 0.0

<div class="index">

.NROTAV

</div>

### .NROTAV

As described under the .BED keyword, [Lebedev
quadrature](https://en.wikipedia.org/wiki/Lebedev_quadrature) is
employed for rotational average. This quadrature over the solid angles
can integrate a spherical harmonic to high accuracy with a maximum
angular momentum $`L_\mbox{max}`$. The default value of $`L_\mbox{max}`$
is presently 5, but can be reset with this keyword.

<div class="index">

.BEDCON

</div>

### .BEDCON

Specification of contributions of the full light-matter interaction of
order $`n`$ in the wave vector

``` math
\hat{T}_{\mathrm{full}}^{\left[n\right]}(\omega)=\frac{k^{n}}{n!}\frac{d^{n}}{dk^{n}}\left[\frac{e}{\omega}
\left(c\boldsymbol{\alpha}\cdot\boldsymbol{\epsilon}\right)e^{+i\left(\boldsymbol{k}\cdot\boldsymbol{r}\right)}\right]_{k=0}
=\frac{e}{\omega}\frac{i^{n}}{n!}\left(c\boldsymbol{\alpha}\cdot\boldsymbol{\epsilon}\right)\left(\boldsymbol{k}\cdot\boldsymbol{r}\right)^{n}
```

<div class="index">

.NOLENR

</div>

### .NOLENR

Deactivate length representation.

<div class="index">

.NOVELR

</div>

### .NOVELR

Deactivate velocity representation.

## Control variational parameters

<div class="index">

.OCCUP

</div>

### .OCCUP

For each fermion ircop give an `orbital_strings` of inactive orbitals
from which excitations are allowed. By default excitations from all
occupied orbitals are included in the generalized eigenvalue problem.

Example: :

    .OCCUP
    1..3
    7,8

This would include excitations from gerade orbitals 1,2,3, and ungerade
orbitals 7 and 8.

<div class="index">

.VIRTUA

</div>

### .VIRTUA

For each fermion ircop give an `orbital_strings` of virtual orbitals to
which excitations are allowed. By default excitations to all virtal
orbitals are included in the generalized eigenvalue problem.

<div class="index">

.SKIPEE

</div>

### .SKIPEE

Exclude all rotations between occupied positive-energy and virtual
positive-energy orbitals.

<div class="index">

.SKIPEP

</div>

### .SKIPEP

Exclude all rotations between occupied positive-energy and virtual
negative-energy orbitals.

## Control reduced equations

<div class="index">

.MAXITR

</div>

### .MAXITR

Maximum number of iterations.

*Default:* :

    .MAXITR
     30

<div class="index">

.MAXRED

</div>

### .MAXRED

Maximum dimension of matrix in reduced system.

*Default:* :

    .MAXRED
     200

<div class="index">

.THRESH

</div>

### .THRESH

Threshold for convergence of reduced system.

*Default:* :

    .THRESH
     1.0D-5

## Control integral contributions

The user is encouraged to experiment with these options since they may
have an important effect on run time.

<div class="index">

.INTFLG

</div>

### .INTFLG

Specify what two-electron integrals to include (default:
`HAMILTONIAN_.INTFLG` under `**HAMILTONIAN`).

<div class="index">

.CNVINT

</div>

### .CNVINT

Set threshold for convergence before adding SL and SS integrals to
SCF-iterations.

*2 (real) Arguments:* :

    .CNVINT
     CNVXQR(1) CNVXQR(2)

*Default:* Very large numbers.

<div class="index">

.ITRINT

</div>

### .ITRINT

Set the number of iterations before adding SL and SS integrals to
SCF-iterations.

*Default:* :

    .ITRINT
     1 1

## Advanced/debug flags

<div class="index">

.E2CHEK

</div>

### .E2CHEK

Generate a complete set of trial vector which implicitly allows the
explicit construction of the electronic Hessian. Only to be used for
small systems !

<div class="index">

.ONLYSF

</div>

### .ONLYSF

Only call FMOLI in sigmavector routine: only generate one-index
transformed Fock matrix Saue2003.

<div class="index">

.ONLYSG

</div>

### .ONLYSG

Only call FMOLI in sigmavector routine: 2-electron Fock matrices using
one-index transformed densities Saue2003.

<div class="index">

.GNOISE

</div>

### .GNOISE

To test the robustness of property gradients to numerical noise
artificial noise is added to the MO-coefficients. More precisely, the
user activates noise and provides a "noise level". Then, for each
element of the coefficient array, a pseudo-random number in the interval
(-1,+1\] is selected, multiplied with the "noise level" and added to the
element.

*Example:* :

    .GNOISE
     1.0D-10
