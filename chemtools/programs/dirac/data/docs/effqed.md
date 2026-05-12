orphan  

# Effective QED potentials

QED effects are high-order relativistic effects that cannot be captured
in the framework of the no-pair approximation. A rough explanation of
the QED effects that contribute to the Lamb shift of bound-state
electrons is as follows:

- Vacuum polarization (VP): A charge in space is surrounded by virtual
  electron-positron pairs which contributes to its observed charge.
- Self-energy (SE): The electron drags along its electromagnetic field
  which contributes to its observed mass.

Effective QED potentials are an approximate way to include QED effects
in all-electron relativistic Hamiltonians. In the current implementation
(see Ref. Sunaga2022) the following potentials are available:

## Vacuum polarization

### Uehling potential

The Uehling potential is the dominant contribution to the vacuum
polarization. It is extracted from the term linear in nuclear charge
after regularization and renormalization. For a spherically symmetric
nuclear charge distribution it reads

``` math
\begin{aligned}
\varphi^\text{VP}_\text {Ueh} (\boldsymbol{x})= & \frac{Z e}{4 \pi \varepsilon_0 r_x} \bar{\lambda} \frac{2 \alpha}{3} \int_0^{\infty} r_y d r_y \rho^{\text {nuc. }}\left(r_y\right) \\ & \times\left[K_0\left(\frac{2}{\bar{\lambda}}\left|r_x-r_y\right|\right)-K_0\left(\frac{2}{\bar{\lambda}}\left|r_x+r_y\right|\right)\right]
\end{aligned}
```

where $`\bar{\lambda}=\hbar/mc`$ is the reduced Compton length and

``` math
K_0(x)=\int_1^{\infty} d \zeta e^{-x \zeta}\left(\frac{1}{\zeta^3}+\frac{1}{2 \zeta^5}\right) \sqrt{\zeta^2-1}.
```

## Electron self-energy

The electron self-energy effect is delocal and hence exchange-like.
However, some effective local potentials are available in the
literature. Two of them are implemented in DIRAC:

### 1) Flambaum and Ginges SE potential

The Flambaum-Ginges potential Flambaum2005 is derived from the linear
contribution to the self-energy process, but has been further modeled
and parametrized to to account for the full self-energy process to all
orders in $`(Z\alpha)`$. It splits into two terms, associated with
electric and magnetic form factors. The electric contribution is further
divided into a high-frequency (HF) and low-frequency (LF) part. The
high-frequency is properly derived from the electric form factor, but
comes with a cutoff parameter $`\lambda`$ to eliminate infrared
divergency, whereas the low-frequency is simply modeled as a
hydrogen-like function. The final forms are

``` math
\varphi^\text{SE}_\text{FG} = \varphi_{\text {mag }} + \varphi_{\text {HF }} + \varphi_{\text {LF}},
```

where

``` math
\varphi_{\mathrm{mag}} (\boldsymbol{x})=\frac{\alpha \hbar}{4 \pi m_e c} i {\gamma} \cdot \nabla_x\left[ \Phi(\boldsymbol{x}) \left(K_m\left(\frac{2 r_x}{\bar{\lambda}}\right)-1\right)\right],
```

``` math
\varphi_{\text {HF }}(\boldsymbol{x}, \lambda)=-\frac{\alpha}{\pi} A(Z,\boldsymbol{x})\Phi(\boldsymbol{x}) K_e\left(\frac{2 r_x}{\bar{\lambda}}\right),
```

``` math
\varphi_{\mathrm{LF}} (\boldsymbol{x})=-\frac{B(Z)}{e} Z^4 \alpha^5 m_e c^2 e^{-Z r_x / a_0}.
```

The functions $`K_m(x)`$ and $`K_e(x)`$ are as follows:

``` math
K_m(x)=\int_1^{\infty} d \zeta \frac{e^{-x \zeta}}{\zeta^2 \sqrt{\zeta^2-1}};
```

``` math
\begin{aligned}
K_e(x)= & \int_1^{\infty} d \zeta \frac{e^{-x \zeta}}{\sqrt{\zeta^2-1}}\left\{-\frac{3}{2}+\frac{1}{\zeta^2}+\left(1-\frac{1}{2 \zeta^2}\right)\right. \\ & \left.\times\left[\ln \left(\zeta^2-1\right)+  4 \ln \left(\frac{1}{Z \alpha}+\frac{1}{2}\right)   \right]\right\} .
\end{aligned}
```

$`A`$ and $`B`$ are fitting functions adjusted to reproduce the
self-energy corrections of hydrogen-like atoms. $`\Phi(\boldsymbol{x})`$
is the nuclear electrostatic potential. The above forms are strictly
only correct for a point-charge nuclei, that is, for
$`\Phi(\boldsymbol{x})=\frac{Z e}{4 \pi \varepsilon_0 r_x}`$. The
corresponding potentials for extended nuclei are properly obtained by
convolution with the nuclear charge distribution. However, the above
approximation has been reported to work well in the case of the Uehling
potential Artemyev1997.

### 2) Pyykko and Zhao SE potential

This is a Gaussian-type model potential where the parameters are fitted
to reproduce the orbital energy and M1 hyperfine splitting within the
range $`29 \leq Z \leq 83`$ Pyykko2003.

``` math
\varphi^\text{SE}_\mathrm{PZ}(\boldsymbol{x})=B(Z) e^{-\beta(Z) r_x^2}
```

$`B`$ and $`\beta`$ are fitting functions.

# How to use

Effective QED potentials, which are one-electron operators, should be
added to the one-electron Hamiltonian, as follows:

``` math
\hat{h}_D \rightarrow \hat{h}_D  -e \varphi_{\mathrm{effQED}}
```

``` math
\varphi_{\mathrm{effQED}}  =\sum_A^\text{nuc}\left(\varphi_A^{\mathrm{SE}}+\varphi_A^{\mathrm{VP}}\right)
```

The DIRAC implementation was obtained by grafting code from the
numerical atomic code GRASP GRASP onto the DFT of grid. See `*EFFQED`
for tuning options. The present input gives precise control, but is
admittedly somewhat cumbersome.

## Example 1: DCG Hamiltonian with Uehling and FG potentials (variational treatment)

<div class="literalinclude">

ueh_fg.inp

</div>

Although the $`\boldsymbol{\gamma}`$ operator appears in the expression
of the magnetic term above, the reader may find that the matrix part of
the magnetic term is `ALPHADOT` (i.e., this one-electron operator
involves the scalar product of $`\boldsymbol{\alpha}`$ and the scalar
operators `XSEFGMG`, `YSEFGMG`, and `ZSEFGMG`). This input is correct
because the $`\boldsymbol{\beta}`$ part in
$`\boldsymbol{\gamma}=\boldsymbol{\beta\alpha}`$ is already included in
the code for a technical reason. In this example, the molecular orbitals
are relaxed due to the QED effects. The contributions of effective QED
potentials are included here in the total energy.

## Example 2: Expectation values of Uehling and PZ potentials (perturbative treatment)

<div class="literalinclude">

ueh_pz_exp.inp

</div>

In this example, the expectation values of the effective QED potentials
are obtained using the molecular orbitals calculated based on the
Dirac-Coulomb-Gaunt (DCG) Hamiltonian. That is, effQED potentials are
treated as perturbations, and the corresponding energies are obtained
using first-order perturbation theory. In this case, the individual
contributions of effQED potentials to the total energy are printed out
in the property section.

# Note on the application to core-properties

The PZ potential was fitted to reproduce the QED effects on the
hyperfine coupling constant and orbital energies (1s and 2s orbitals),
and the fitting functions for the FG potential were adjusted to
reproduce the self-energy corrections to high s- and p-states of
hydrogenic ions.

However, we have confirmed that these potentials cannot accurately
account for QED effects on the orbital energies for core orbitals. In
fact, these potential provide different QED effects on the
parity-violating energy Sunaga2021.

Therefore, these potential would not accurately provide QED effects on
the core properties, such as XPS, Mössbauer, and NMR.

It has been confirmed that these SE potentials yield very similar
results in the case of valence properties Sunaga2022.
