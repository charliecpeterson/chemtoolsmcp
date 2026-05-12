orphan  

<div class="index">

\*\*PROPERTIES

</div>

# \*\*PROPERTIES

This section allows for the evaluation of a large number of molecular
properties. Available properties include:

- Expectation values (e.g. dipole moment and electric field gradients).
- Linear response properties (e.g. polarizability and NMR parameters).
- Quadratic response properties (e.g. hyperpolarizabilities).
- Quasi-degenerate CI perturbation theory for ESR parameters (g-tensor
  and hyperfine coupling tensors).

For convenience some common properties can be specified directly in this
section, which means that the user in principle does not need to know
how they are calculated. Note, however, that response functions are by
default static, but frequencies can be added in the relevant
subsections.

Properties which are not predefined must be specified in detail in the
relevant input section (see `one_electron_operators`).

By default no properties are calculated.

## General control statements

<div class="index">

.PRINT

</div>

### .PRINT

Print level.

*Default:*

    .PRINT
     0

<div class="index">

.ABUNDANCIES

</div>

### .ABUNDANCIES

For properties that make reference to isotopes, give threshold level (in
% abundance) for isotopes to print.

*Default:*

    .ABUNDANCIES
     1.0

<div class="index">

.RKBIMP

</div>

### .RKBIMP

Import coefficients calculated with restricted kinetic balance (RKB) in
a calculation using unrestricted kinetic balance (UKB). This option is a
simple way to generated restricted magnetic balance for the calculation
of NMR shieldings. This option works in the general SO case, but not in
the spinfree case since spinfree calculations are not possible with UKB.

<div class="index">

.NOPCTR

</div>

### .NOPCTR

In two-component infinite-order relativistic calculations (with
`HAMILTONIAN_.X2C`) take only LL block of four-component property
operators to avoid the picture change transformation. Experimental
option, use with care.

<div class="index">

.RDCCDM

</div>

### .RDCCDM

Activates the reading of the file CCDENS obtained from a previous CC
calculation with either the `WAVE_FUNCTION_.RELCCSD` or with
`WAVE_FUNCTION_.EXACC` modules. It is not necessary to use this keyword
in runs in which the correlated and property modules are both activated.

CCDENS is not saved by pam, so unless the scratch directory from the
previous calculation is kept (see --keep_scratch in pam), you should
retrieve it after a correlated calculation, e.g. :

    pam --get=CCDENS ...

and then copy it back for the property calculation e.g. :

    pam --put=CCDENS ...

## Predefined electric properties

<div class="index">

.DIPOLE

</div>

### .DIPOLE

Evaluate the electronic electric dipole moment

Expectation values:
$`\langle\hat{\mu}_\alpha\rangle=-e\langle r_\alpha\rangle`$

*Note:* for charged molecules the total electric dipole moment will
depend on the gauge origin. It is possible to set the nuclei charge
barycenter as the gauge origin so that the computed (total) dipole
magnitude is representative for the electronic system as the nuclei
electric dipole moment is constrained to vanish. Check for USEBC under
`*NMR`.

<div class="index">

.QUADRUPOLE

</div>

### .QUADRUPOLE

Evaluate the electronic traceless electric quadrupole moment

Expectation values:
$`\langle\Theta_{\alpha\beta}\rangle=-e\frac {3}{2}\langle r_\alpha r_\beta-\frac{1}{3}\delta_{\alpha\beta}r^2\rangle`$)

<div class="index">

.EFG

</div>

### .EFG

Evaluate electric field gradients at nuclear positions, see
Visscher_JCP1998 .

Electronic contribution to center $`K`$ (expectation values) :

``` math
\phi^{[2]el}_{\alpha\beta}(\mathbf{R}_K)=\frac{-e}{4\pi\varepsilon_0}\left<\frac{3r_{K;\alpha}r_{K;\beta}-\delta_{\alpha\beta}r_K^2}{r_K^5}\right>
```

Nuclear contributions to center $`K`$ :

``` math
\phi^{[2]nuc}_{\alpha\beta}(\mathbf{R}_K)=\sum_{A\ne K}\frac{Z_Ae}{4\pi\varepsilon_0}\left[\frac{3r_{KA;\alpha}r_{KA;\beta}-\delta_{\alpha\beta}r_{KA}^2}{r_{KA}^5}\right]
```

Results are also reported with respect to a principal axis system for
each center.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.NQCC

</div>

### .NQCC

Evaluate nuclear quadrupole coupling constants (NQCC) (expectation
values). The NQCC is formally defined as

``` math
\frac{e^2qQ}{h}
```

where $`Q`$ is the electric quadrupole moment of the nucleus and
$`q=\phi^{[2]}_{zz}/e`$ (in the principal axis system) is the field
gradient. The NQCC may be extracted from experiment, whereas electronic
structure calculations may provide the field gradient $`q`$. The two
quantities are related as

``` math
\mbox{NQCC [in MHz] } = 234.9647\ \times\ Q\mbox{ [in b] }\ \times\ q\mbox{ [in atomic units }E_h/ea_0^2\mbox{ ]}
```

The calculations proceed similar to `PROPERTIES_.EFG`. The total
electric field gradients for each center are transformed to a principal
axis system for which

``` math
|\phi^{[2]}_{zz}|\ge|\phi^{[2]}_{yy}|\ge|\phi^{[2]}_{xx}|
```

DIRAC reports the more general expressions

``` math
\mbox{NQCC}_{\alpha\alpha}\mbox{ [in MHz] } = 234.9647\ \times\ Q\mbox{ [in b] }\ \times\ \phi^{[2]}_{\alpha\alpha}/e \mbox{ [in atomic units }E_h/ea_0^2\mbox{ ]}
```

The asymmetry factor is defined as

``` math
\eta = \frac{\phi^{[2]}_{xx}-\phi^{[2]}_{yy}}{\phi^{[2]}_{zz}}
```

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.POLARIZABILITY

</div>

### .POLARIZABILITY

Evaluate the electronic dipole polarizability tensor, see Saue2003 (HF)
and Salek2005 (DFT).

Linear response function:
$`\quad\alpha_{\alpha\beta}(-\omega;\omega)=\langle\langle\hat{\mu}_{\alpha};\hat{\mu}_{\beta}\rangle\rangle_{\omega}`$

<div class="index">

.FIRST ORDER HYPERPOLARIZABILITY

</div>

### .FIRST ORDER HYPERPOLARIZABILITY

Evaluate static electronic dipole first-order hyperpolarizability
tensor, see Norman_JCP2004 (HF) and Henriksson:2008 (DFT).

Quadratic response function:
$`\quad\beta_{\alpha\beta\gamma}(-\omega_\sigma;\omega_1,\omega_2)=\langle\langle\hat{\mu}_{\alpha};\hat{\mu}_{\beta},\hat{\mu}_{\gamma}\rangle\rangle_{\omega_1,\omega_2}`$

Results are also given for the static electronic dipole polarizability.

<div class="index">

.TWO-PHOTON

</div>

### .TWO-PHOTON

Evaluate two-photon absorption cross sections Henriksson:2005, obtained
as a first-order residue of the first-order hyperpolarizability. Give
the number of desired states in each boson symmetry. Cannot be specified
in combination with other quadratic response calculations.

*Example:* Point group with four boson irreps, (e.g. $`C_{2v}`$)

    .TWO-PHOTON
     5 5 5 0

## Predefined magnetic properties

<div class="index">

.NMR

</div>

### .NMR

Evaluate nuclear magnetic shieldings and indirect spin-spin couplings
(linear response functions), see Visscher_jcc1999 and Ilias2009.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

See below for advice on calculation of diamagnetic terms.

<div class="index">

.SHIELDING

</div>

### .SHIELDING

Evaluate nuclear magnetic shieldings (linear response), see
Visscher_jcc1999 and Ilias2009 .

Elements of the shielding tensor for center $`K`$ are given by

``` math
\sigma_{K;\mu\nu}=\frac{\partial^2}{\partial m_{K;\mu}\partial B_{0;\nu}}\langle\langle\hat{h}^{hfs}_{K};\hat{h}^Z\rangle\rangle_0
```

where appears the relativistic hyperfine operator

``` math
\hat{h}^{hfs}_{K}=-\sum_i\mathbf{m}_K\cdot\hat{\mathbf{B}}^{el}_{K}(i);\quad \hat{\mathbf{B}}^{el}_{K}(i)=-\frac{1}{4\pi\varepsilon_0 c^2}\frac{\mathbf{r}_{iK}\times ec\boldsymbol{\boldsymbol{\alpha}}}{r_{iK}^3},
```

expressed in terms of the nuclear magnetic dipole $`\mathbf{m}_K`$ and
the operator $`\hat{\mathbf{B}}^{el}_{K}`$ giving the magnetic field due
to the electrons at the nuclear position, and the relativistic Zeeman
operator

``` math
\hat{h}^Z=-\hat{\mathbf{m}}_e^{[1]}\cdot\mathbf{B}_0;
\quad\hat{\mathbf{m}}^{[1]}_e=-\sum_i\frac{e}{2}(\mathbf{r}_{iG}\times c\boldsymbol{\alpha}(i)),
```

expressed in terms of the operator $`\hat{\mathbf{m}}_e^{[1]}`$
associated with the magnetic dipole moment of the electrons and the
external magnetic field $`\mathbf{B}_0`$. Note reference to the gauge
origin $`G`$.

Note that `PROPERTIES_.PRINT` 2 gives the full tensor and longer output.
The `PROPERTIES_.PRINT` 4 gives the raw values in symmetry coordinates
as well.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.MAGNET

</div>

### .MAGNET

Evaluate the (static) magnetizablity tensor Ilias2013

Lineare response function

``` math
\xi_{K;\alpha\beta}= - \frac{\partial^2}{\partial B_{\alpha}\partial B_{\beta}}\langle\langle\hat{h}^Z;\hat{h}^Z\rangle\rangle_0
```

where appears the relativistic Zeeman operator

``` math
\hat{h}^Z=-\hat{\mathbf{m}}_e^{[1]}\cdot\mathbf{B}_0;
\quad\hat{\mathbf{m}}^{[1]}_e=-\sum_i\frac{e}{2}(\mathbf{r}_{iG}\times c\boldsymbol{\alpha}(i)),
```

expressed in terms of the operator $`\hat{\mathbf{m}}_e^{[1]}`$
associated with the magnetic dipole moment of the electrons and the
external magnetic field $`\mathbf{B}_0`$. Note reference to the gauge
origin $`G`$.

<div class="index">

.ROTG

</div>

### .ROTG

Evaluate rotational g-tensors: linear response and nuclear
contributions, see Aucar_JCP2014.

Elements of the rotational g-tensor are given by

``` math
g_{\mu\nu} = g^{nuc}_{\mu\nu} + g^{elec}_{\mu\nu}
```

with

``` math
g^{elec}_{\mu\nu}= - \frac{2 m_p}{e} \frac{\partial^2}{\partial L_{\mu}\partial B_{0;\nu}}\langle\langle\hat{h}^{BO};\hat{h}^Z\rangle\rangle_0
```

where appears the first order correction to the Born-Oppenheimer (BO)
approximation

``` math
\hat{h}^{BO}=-\boldsymbol{\boldsymbol{\omega}}\cdot\hat{\mathbf{J}}_e;
\quad\boldsymbol{\boldsymbol{\omega}}=\mathbf{L}\cdot\mathbf{I}^{-1},
```

expressed in terms of the angular velocity
$`\boldsymbol{\boldsymbol{\omega}}`$ associated with the total angular
momentum of the electrons
$`\hat{\mathbf{J}}_e=\hat{\mathbf{L}}_e+\hat{\mathbf{S}}_e`$, and the
relativistic Zeeman operator

``` math
\hat{h}^Z=-\hat{\mathbf{m}}_e^{[1]}\cdot\mathbf{B}_0;
\quad\hat{\mathbf{m}}^{[1]}_e=-\sum_i\frac{e}{2}(\mathbf{r}_{iG}\times c\boldsymbol{\alpha}(i)),
```

expressed in terms of the operator $`\hat{\mathbf{m}}_e^{[1]}`$
associated with the magnetic dipole moment of the electrons and the
external magnetic field $`\mathbf{B}_0`$. Note reference to the gauge
origin $`G`$.

Results are dimensionless.

The total g-tensor, as well as its linear response and nuclear
contributions are always given separately.

Using `PROPERTIES_.PRINT` 1, the paramagnetic (e-e) and diamagnetic
(e-p) parts of the linear response contributions are given separately,
together with results for the $`\mathbf{L}`$ and $`\mathbf{S}`$ parts of
the linear response.

<div class="index">

.SPIN-SPIN COUPLING

</div>

### .SPIN-SPIN COUPLING

Evaluate indirect spin-spin couplings Visscher_jcc1999 . The indirect
spin-spin tensor $`J_{KL}`$ associated with nuclei $`K`$ and $`L`$ may
be expressed as

``` math
J_{KL}=\frac{\hslash^2}{h}\gamma_{K}\gamma_{L}K_{KL}
```

where appears gyromagnetic ratios $`\gamma_K`$. The elements of the
reduced tensor $`K_{KL}`$ are expressed in terms of linear response
functions as

``` math
K_{KL:\mu\nu} = \frac{\partial^{2}}{\partial m_{K;\mu}\partial m_{L;\nu}}\langle \langle \hat{h}_{K}^{hfs}; \hat{h}_{L}^{hfs}\rangle\rangle_{0}
```

where appears the relativistic hyperfine operator

``` math
\hat{h}^{hfs}_{K}=-\sum_i\mathbf{m}_K\cdot\hat{\mathbf{B}}^{el}_{K}(i);\quad \hat{\mathbf{B}}^{el}_{K}(i)=-\frac{1}{4\pi\varepsilon_0 c^2}\frac{\mathbf{r}_{iK}\times ec\boldsymbol{\boldsymbol{\alpha}}}{r_{iK}^3},
```

expressed in terms of the nuclear magnetic dipole $`\mathbf{m}_K`$ and
the operator $`\hat{\mathbf{B}}^{el}_{K}`$ giving the magnetic field due
to the electrons at the nuclear position.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

The default is to calculate the diamagnetic term via occupied positive
energy to virtual negative energy orbital rotations (also called
electron-positron rotations), see Aucar1999 for the theory. The quality
of this is very basis set dependent. It is generally more accurate to
use the non-relativistic expectation value expression for the
diamagnetic term, activated with keyword .DSO in this section. You must
also add `LINEAR_RESPONSE_.SKIPEP` under `*LINEAR RESPONSE` to exclude
the diamagnetic term from the linear response calculation.

<div class="index">

.DSO

</div>

### .DSO

Evaluate the diamagnetic contribution to indirect spin-spin couplings as
an expectation value of the non-relativistic DSO operator.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.NSTDIAMAGNETIC

</div>

### .NSTDIAMAGNETIC

Evaluate the diamagnetic contribution to nuclear magnetic shielding
tensor as an expectation value of the non-relativistic operator.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.ESR

</div>

### .ESR

Evaluate ESR parameters -- g-tensors and hyperfine coupling tensors --
using first-order quasi-degenerate perturbation theory based on
configuration interaction. No default, it is required to specify the
actual calculation under `*ESR`.

## Mixed electric and magnetic properties

<div class="index">

.OPTROT

</div>

### .OPTROT

Calculate optical rotation. The most common experimental setup uses
light with a frequency corresponding to the sodium D-line (589.29 nm).
The optical rotation is reported as the number of degrees of rotation of
the plane of polarization per mole of sample for a sample cell of length
1~dm, and at a temperature of $`25^\circ{\rm C}`$

``` math
\left[\alpha\right]_D^{25} = -288\cdot 10^{-30}\frac{\pi^2\mathcal{N}a_0^4\omega}{3M}\sum_{\alpha}G^{\prime}_{\alpha\alpha };\quad G^{\prime}_{\alpha\beta}(-\omega;\omega)= -\mbox{Im}\langle \langle \hat{\mu}_{\alpha};\hat{m}_{\beta}\rangle \rangle _\omega
```

where $`M`$ is the molecular mass in g $`\mbox{mol}^{-1}`$ and
$`\mathcal{N}`$ is the number density.

<div class="index">

.VERDET

</div>

### .VERDET

Evaluate Verdet constants Ekstrom2005 for a dynamic electric field
corresponding to Ruby laser wavelength of 694 nm and a static magnetic
field along the propagation direction of the light beam (in this case,
the default frequencies of the quadratic response function thus become
ω<sub>\*B\*</sub> = 0.0656 and ω<sub>\*C\*</sub> = 0.0).

The Verdet constant is given in terms of quadratic response functions

``` math
V(\omega)=\omega C\epsilon_{\alpha\beta\gamma}\mbox{Im}\langle\langle\hat{\mu}_{\alpha};\hat{\mu}_{\beta},\hat{m}_{\gamma}\rangle\rangle_{\omega,0}
```

where $`C=eN/(24c_0\epsilon_0m_e`$ and $`N`$ is the number density of
the gas. A Verdet calculation cannot be specified in combination with
other quadratic response calculations.

The frequencies can be changed using `QUADRATIC_RESPONSE_.B FREQ` in
`*QUADRATIC RESPONSE`.

## Other predefined properties

<div class="index">

.MOLGRD

</div>

### .MOLGRD

Evaluate the molecular gradient, i.e.

``` math
\frac{\partial E}{\partial \mathbf{X}_A}
```

where $`\mathbf{X}_{A}`$ are the coordinates of the nuclei. This is an
expectation value of one- and two-electron operators. Normally the
molecular gradient evaluation is not invoked explicitly with this
keyword but rather implicitly in the geometry optimization module.

<div class="index">

.PVC

</div>

### .PVC

Calculate matrix elements over the nuclear spin-independent
parity-violating operator, e.g. calculate energy differences between
enantiomers, see Laerdahl1999 and Bast2011.

The parity-violating energy is calculated as the expectation value

``` math
E_\text{PV} = \sum_A \langle H_\text{PV}^A \rangle;\quad H_\text{PV}^A= \frac{G_\text{F}}{2 \sqrt{2}} Q_\text{w}^A \sum_i \gamma_5 (i) \rho^A (\mathbf{r}_i)
```

where $`\rho^A`$ is the nuclear charge density of nucleus $`A`$
normalized to unity, and $`G_F`$ is the Fermi coupling constant. The
weak nuclear charge

``` math
Q_\text{w}^A = Z^AC_V^\text{p}+N^AC_V^\text{n}=Z^A (1 - 4 \sin^2 \theta_\text{W}) - N^A
```

is given in terms of both the number of protons and neutrons ---$`Z^A`$
and $`N^A`$--- in the nucleus $`A`$ and the Weinberg angle
$`\theta_\text{W}`$, which describes the rotation of $`B^0`$ and $`W^0`$
bosons by spontaneous symmetry breaking to form photons and $`Z^0`$
bosons.

Using `GENERAL_.CODATA`, $`G_F`$ can be chosen according to the data
reported in different CODATA sets. In the particular case in which
`.PDG94` is used under `GENERAL_.CODATA`,
$`G_F=2.22255\times 10^{-14}E_ha_0^3`$. In addition, DIRAC uses
$`\sin^2 \theta_\text{W} = 0.2319`$ when `.PDG94` is employed under
`GENERAL_.CODATA`. In other cases, the value reported in the latest
available CODATA set is used.

<div class="index">

.PVCSHI

</div>

### .PVCSHI

Calculate parity-violating contribution to the NMR shielding tensor, see
Barra1988 and Bast:2006. Elements of the parity-violating contribution
to the shielding tensor for center $`K`$ are given by

``` math
\sigma^{PV}_{K;\mu\nu}=\frac{\partial^2}{\partial m_{K;\mu}\partial B_{0;\nu}}\langle\langle \hat{h}^\text{PV2}_K;\hat{h}^Z\rangle\rangle_0
```

where the nuclear spin-dependent parity-violating operator is given as

``` math
\hat{h}^\text{PV2}_K = -\frac{G_\text{F}(1-4\sin^2\theta_\text{W})}{\sqrt{2}\,c}\sum_{i}\frac{1}{\hslash \gamma_K}c\boldsymbol{\alpha}\cdot\mathbf{m}_K\rho_K(\mathbf{r}_i)
```

where $`G_F`$ is the Fermi coupling constant, $`\theta_\text{W}`$ is the
Weinberg angle, $`\rho^A`$ is the nuclear charge density of nucleus
$`A`$ normalized to unity, $`\gamma_K`$ is the gyromagnetic ratio, and
$`\mathbf{m}_K=\gamma_K\mathbf{I}_K`$ is the nuclear magnetic dipole
moment.

The relativistic Zeeman operator

``` math
\hat{h}^Z=-\hat{\mathbf{m}}_e^{[1]}\cdot\mathbf{B}_0;
\quad\hat{\mathbf{m}}^{[1]}_e=-\sum_i\frac{e}{2}(\mathbf{r}_{iG}\times c\boldsymbol{\alpha}(i)),
```

is expressed in terms of the operator $`\hat{\mathbf{m}}_e^{[1]}`$
associated with the magnetic dipole moment of the electrons and the
external magnetic field $`\mathbf{B}_0`$. Note reference to the gauge
origin $`G`$.

Results are given in ppm.

Using `GENERAL_.CODATA`, $`G_F`$ can be chosen according to the data
reported in different CODATA sets. In the particular case in which
`.PDG94` is used under `GENERAL_.CODATA`,
$`G_F=2.22255\times 10^{-14}E_ha_0^3`$. In addition, DIRAC uses
$`\sin^2 \theta_\text{W} = 0.2319`$ when `.PDG94` is employed under
`GENERAL_.CODATA`. In other cases, the value reported in the latest
available CODATA set is used.

Using `PROPERTIES_.PRINT` 1, the paramagnetic-like (e-e) and
diamagnetic-like (e-p) parts of the linear response are given
separately.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.PVCSR

</div>

### .PVCSR

Calculate parity-violating contribution to the nuclear spin-rotation
tensor, see Aucar_JCP2021. Elements of the parity-violating contribution
to the nuclear spin-rotation tensor for center $`K`$ are given by

``` math
M^{PV}_{K;\mu\nu} = - \frac{\hslash^2}{h} \frac{\partial^2}{\partial I_{K;\mu}\partial L_{\nu}}\langle\langle\hat{h}^{PV2}_{K};\hat{h}^{BO}\rangle\rangle_0
```

where the nuclear spin-dependent parity-violating operator is given as

``` math
\hat{h}^{PV2}_{K} = -\frac{G_\text{F}(1-4\sin^2\theta_\text{W})}{\sqrt{2}\,c}\sum_{i}\frac{1}{\hslash}c\boldsymbol{\alpha}\cdot\mathbf{I}_K\rho_K(\mathbf{r}_i)
```

where $`G_F`$ is the Fermi coupling constant, $`\theta_\text{W}`$ is the
Weinberg angle, $`\rho^A`$ is the nuclear charge density of nucleus
$`A`$ normalized to unity, and $`I_K`$ is the nuclear spin of nucleus
$`K`$.

The first order correction to the Born-Oppenheimer (BO) approximation

``` math
\hat{h}^{BO}=-\boldsymbol{\boldsymbol{\omega}}\cdot\hat{\mathbf{J}}_e;
\quad\boldsymbol{\boldsymbol{\omega}}=\mathbf{L}\cdot\mathbf{I}^{-1},
```

is expressed in terms of the angular velocity
$`\boldsymbol{\boldsymbol{\omega}}`$ associated with the total angular
momentum of the electrons
$`\hat{\mathbf{J}}_e=\hat{\mathbf{L}}_e+\hat{\mathbf{S}}_e`$. Note that
the origin of the orbital angular momentum is the molecular center of
mass.

Results are given in Hz.

Using `GENERAL_.CODATA`, $`G_F`$ can be chosen according to the data
reported in different CODATA sets. In the particular case in which
`.PDG94` is used under `GENERAL_.CODATA`,
$`G_F=2.22255\times 10^{-14}E_ha_0^3`$. In addition, DIRAC uses
$`\sin^2 \theta_\text{W} = 0.2319`$ when `.PDG94` is employed under
`GENERAL_.CODATA`. In other cases, the value reported in the latest
available CODATA set is used.

Using `PROPERTIES_.PRINT` 1, the paramagnetic-like (e-e) and
diamagnetic-like (e-p) parts of the linear response are given
separately, together with results for the $`\mathbf{L}`$ and
$`\mathbf{S}`$ parts of the linear response.

For more details, the user is welcome to see the Tutorial Section.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.PVCEFG

</div>

### .PVCEFG

Calculate parity-violating contribution to the electric field gradient
tensor, see Aucar2025_PVEFG. Elements of the parity-violating
contribution to the EFG tensor for center $`K`$ are given by

``` math
q^{PV}_{ij} (\mathbf{R}_K) =  \langle\langle \hat q_{ij} (\mathbf{R}_K);     \hat{H}^{PV}_K \rangle\rangle_0
```

where the contribution from the K nucleus to the nuclear
spin-independent parity-violating operator

``` math
\hat{H}^{PV}_K =\frac{G_F}{2\sqrt{2}}\sum_{i,K}Q_{w,K}\gamma_i^5\rho_K(\mathbf{r}_i)
```

is expressed in terms of the Fermi coupling constant
$`G_F=2.222516\times 10^{-14}E_ha_0^3`$ (this value corresponds to
CODATA 2022, but it can change if you use other CODATA set), the weak
nuclear charge $`Q_{w,K}`$ and the normalized nuclear charge density
$`\rho_K`$ (in units of the inverse of cube distances). Results are
given in a.u.

Using `PROPERTIES_.PRINT` 1, the paramagnetic-like (e-e) and
diamagnetic-like (e-p) parts of the linear response are given
separately.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.RHONUC

</div>

### .RHONUC

Calculate electronic density at the nuclear positions, also known as the
contact density (see Knecht2011 and Almoukhalalati:2016b).

It is formally the expectation value

``` math
\rho_e^K = -e\langle\delta^3(\mathbf{r}-\mathbf{R}_K)\rangle.
```

An important observation: In view of picture change effects, see
Knecht2011.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.EFFDEN

</div>

### .EFFDEN

Calculate effective electronic density associated with nuclei, see
Knecht2011. This quantity appears in expressions for the Mössbauer
isomer shift. Starting from the electrostatic electron-nucleus
interaction

``` math
E^{el}\left(R\right)=\int  \rho_e (\mathbf{r})\phi_n(\mathbf{r};R){\rm d}^3 \mathbf{r} ,
```

we consider the change in the electrostatic energy upon a change of
nuclear radius. If we ignore any change in the electronic density
$`\rho_e`$, we may express this as

``` math
\Delta E_{\gamma} = \left.\frac{\partial E^{el}}{\partial R}\right|_{R=R_0}\Delta R = \left[\int \rho_e (\mathbf{r})\frac{\partial\phi_n(\mathbf{r})}{\partial R}{\rm d}^3 \mathbf{r}\right]_{R=R_0} \Delta R = \bar\rho_e\int \left[\frac{\partial\phi_n(\mathbf{r})}{\partial R}{\rm d}^3 \mathbf{r}\right]_{R=R_0} \Delta R
```

where the effective density $`\bar\rho_e`$ is introduced in the last
step. It is often approximated by the contact density, see
`PROPERTIES_.RHONUC` , but this is discouraged since it may introduce
errors on the order of 10%.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.

<div class="index">

.SPIN-ROTATION

</div>

### .SPIN-ROTATION

Evaluate nuclear spin-rotation constants: linear response, expectation
value and nuclear contributions, see Aucar_JCP2012 and AucarChap2019.

Elements of the nuclear spin-rotation tensor for center $`K`$ in a
molecule in equilibrium are given by

``` math
M_{K;\mu\nu} = M^{nuc}_{K;\mu\nu} + M^{elec}_{K;\mu\nu}
```

with

``` math
M^{elec}_{K;\mu\nu} = - \frac{\hslash^2}{h} \frac{\partial^2}{\partial I_{K;\mu}\partial L_{\nu}}\langle\langle\hat{h}^{hfs}_{K};\hat{h}^{BO}\rangle\rangle_0
```

where appears the relativistic hyperfine operator

``` math
\hat{h}^{hfs}_{K}=-\sum_i\mathbf{m}_K\cdot\hat{\mathbf{B}}^{el}_{K}(i);\quad \hat{\mathbf{B}}^{el}_{K}(i)=-\frac{1}{4\pi\varepsilon_0 c^2}\frac{\mathbf{r}_{iK}\times ec\boldsymbol{\boldsymbol{\alpha}}}{r_{iK}^3},
```

expressed in terms of the nuclear magnetic dipole
$`\mathbf{m}_K= \gamma_K \mathbf{I}_K`$ and the operator
$`\hat{\mathbf{B}}^{el}_{K}`$ giving the magnetic field due to the
electrons at the nuclear position, and the first order correction to the
Born-Oppenheimer (BO) approximation

``` math
\hat{h}^{BO}=-\boldsymbol{\boldsymbol{\omega}}\cdot\hat{\mathbf{J}}_e;
\quad\boldsymbol{\boldsymbol{\omega}}=\mathbf{L}\cdot\mathbf{I}^{-1},
```

expressed in terms of the angular velocity
$`\boldsymbol{\boldsymbol{\omega}}`$ associated with the total angular
momentum of the electrons
$`\hat{\mathbf{J}}_e=\hat{\mathbf{L}}_e+\hat{\mathbf{S}}_e`$. Note that
the origin of the orbital angular momentum is the molecular center of
mass.

> [!NOTE]
> The current implementation gives by default (`PROPERTIES_.PRINT`
> values up to 3) results only for molecules in equilibrium.

Results are given in kHz, but for particular `PROPERTIES_.PRINT` values
they could also be given in ppm (to compare results with
`PROPERTIES_.SHIELDING`).

The total spin-rotation tensors, as well as their electronic (linear
response) and nuclear contributions are always given separately.

Using `PROPERTIES_.PRINT` 0 or 1, results are only given in kHz, whereas
employing `PROPERTIES_.PRINT` 2 or 3 they are also shown in ppm.

In addition, when `PROPERTIES_.PRINT` 1 or 3 are used, the
paramagnetic-like (e-e) and diamagnetic-like (e-p) parts of the linear
response contributions are given separately, together with results for
the $`\mathbf{L}`$ and $`\mathbf{S}`$ parts of the linear response.

Finally, employing `PROPERTIES_.PRINT` 4 expectation value and nuclear
contributions are given separately, with the inclusion of Thomas
precesion effects, in order to properly include contributions to nuclear
spin-rotations out of the equilibrium geometry of the molecular system.

For more details, the user is welcome to see the Tutorial Section.

Atomic centers may be restricted with `INTEGRALS_.SELECT` under
`**INTEGRALS`.
