orphan  

# Polarizable continuum model: some basic remarks

The possibility to perform solvent calculations is currently under
development in the `pcm` branch. We will exploit a *Continuum Solvation
Model* (CSM) namely the *Polarizable Continuum Model* (PCM). PCM is a
*focused model*: the solute (a single molecule or a cluster containing
the solute and some relevant solvent molecules) is described quantum
mechanically, while the solvent is approximated as a structureless
continuum whose interaction with the solute is mediated by its
permittivity, $`\varepsilon`$.

The solute is accomodated inside a molecular cavity, built as a set of
interlocking spheres centered on the atoms constituting the molecule
under investigation. The current implementation in DIRAC is limited to
an SCF description of the solute. For a more in-depth presentation of
the PCM, please refer to Tomasi2005, Mennucci2007 and references
therein. For a presentation of the details of the implementation in
DIRAC, please refer to DiRemigio2015.

## Basic Theory

In CSMs we write the Schrödinger equation for the solute as:

``` math
\left[ H_0 + V_{\sigma\rho}(\rho_{\mathrm{M}})\right] | \psi \rangle = E | \phi \rangle
```

where $`H_0`$ is the solute Hamiltonian *in vacuo* and
$`V_{\sigma\rho}(\rho_{\mathrm{M}})`$ is the solute-solvent interaction
potential, **which depends on the solute wavefunction** through the
first-order density matrix. This introduces a *nonlinearity* which must
be dealt with appropriately. In standard molecular electronic-structure
theory we write the energy as the expectation value of the Hamiltonian
($`| \phi \rangle`$ being a trial wave function):

``` math
\langle \phi | H_0 | \phi \rangle
```

An upper-bound estimate to the exact energy of the system can then be
obtained by a variational procedure from this functional. When
introducing a nonlinearity of the type above, the standard functional
**does not** lead to an upper-bound estimate to the exact energy upon
minimization. The appropriate functional is instead given by:

``` math
\langle \phi | H_0 + \frac{1}{2}V_{\sigma\rho} | \phi \rangle
```

which has the status of a *free energy* (an extensive justification of
this fact is given in Tomasi1994).

Thus in the PCM framework, the basic energetic quantity is the *free
energy of solvation* which is conveniently partitioned as follows
(Mennucci2007, Tomasi2005, Amovilli1998):

``` math
\Delta G_\mathrm{sol} = \Delta G_\mathrm{el} + G_\mathrm{cav} + G_\mathrm{dis} + G_\mathrm{rep} + \Delta G_\mathrm{Mm}
```

where:

- $`\Delta G_\mathrm{el}`$ accounts for the *electrostatic*
  solute-solvent interaction, arising from mutual polarization in the
  charge distributions;
- $`G_\mathrm{cav}`$ is the *cavitation* energy, needed to form the
  molecular cavity inside the continuum representing the solvent;
- $`G_\mathrm{dis}`$ is the *dispersion* energy, due to the
  solute-solvent dispesion interactions;
- $`G_\mathrm{rep}`$ is the *repulsion* energy, which accounts for Pauli
  repulsioni;
- $`\Delta G_\mathrm{Mm}`$ is due to molecular motion and accounts for
  *entropic* contributions to the free energy.

The electrostatic term is, usually, the largest contribution to the
solvation energy. Thus we will **exclusively** be concerned with its
calculation (see also Amovilli1998 for a discussion of the other terms).

> [!WARNING]
> The non-electrostatic terms are **not implemented** in DIRAC.

In CSMs the calculation of this term requires the solution of the
classical Poisson problem nested within the QM calculation. We will
define the expectation value of the solvent operator $`V_{\sigma\rho}`$
as the *polarization energy*:

``` math
U_\mathrm{pol} = \langle \psi | V_{\sigma\rho} | \psi \rangle
```

## Electrostatic term in the Polarizable Continuum Model

In the PCM, the solute-solvent electrostatic interaction is represented
by an *apparent surface function* (ASC) $`\sigma`$ spread on the cavity
surface. Using the integral equation formulation of the Poisson problem,
the apparent surface charge is obtained by solving an appropriate
integral equation relating $`\sigma`$ to the molecular electrostatic
potential (MEP) $`\varphi`$ evaluated on the cavity surface:

``` math
\mathcal{T}(\varepsilon_\mathrm{r})\sigma = -\mathcal{R}\varphi
```

The integral operators $`\mathcal{T}(\varepsilon_\mathrm{r})`$ and
$`\mathcal{R}`$ are defined as follows:

``` math
\mathcal{T}(\varepsilon_\mathrm{r}) = \left(2\pi\frac{\varepsilon + 1}{\varepsilon_r -1} - \mathcal{D}\right)\mathcal{S}
```
``` math
\mathcal{R} = 2\pi - \mathcal{D}
```

in terms of components of the Calderon projector (see Tomasi2005) This
equation can be solved numerically by discretization of the cavity
surface with a triangular mesh (the finite elements being called
*tesserae*). The solution of the electrostatic problem then amounts to
solving a linear system of equations:

``` math
\mathbf{T}\mathbf{q} = -\mathbf{R}\mathbf{v} \rightarrow \mathbf{q} = \mathbf{K}\mathbf{v}
```

where $`\mathbf{v}`$ and $`\mathbf{q}`$ are vectors of dimension equal
to the number of finite elements. They contain the MEP and the ASC,
respectively, sampled at the finite elements centroids. The polarization
energy can be expressed as the scalar product of the MEP and ASC sampled
at the cavity boundary:

``` math
U_\mathrm{pol} = \mathbf{q}\cdot\mathbf{v}
```

If we split the MEP into its nuclear and electronic parts
$`\mathbf{v} = \mathbf{v}^\mathrm{N} + \mathbf{v}^\mathrm{e}`$ , due to
the linearity of Poisson equation, the same separation can be achieved
for the ASC
$`\mathbf{q} = \mathbf{q}^\mathrm{N} + \mathbf{q}^\mathrm{e}`$.
Exploiting this separation the polarization energy can be rewritten as:

``` math
U_\mathrm{pol} = U_\mathrm{NN} + U_\mathrm{Ne} + U_\mathrm{eN} + U_\mathrm{ee} = U_\mathrm{NN} + 2U_\mathrm{eN} + U_\mathrm{ee}
```

where $`U_{xy} (x, y = \mathrm{e}, \mathrm{N})`$ is the interaction
energy between the $`x`$ charge distribution and the $`y`$-induced ASC.
We also exploited the self-adjointedness of $`\mathbf{K}`$ to get
$`U_\mathrm{Ne} = U_\mathrm{eN}`$.

## PCM-SCF

Nesting the PCM inside an SCF calculation requires the calculation of
the MEP and ASC at cavity points at every SCF iteration and the update
of the Fock matrix to account for the effect of the mutual
solute-solvent polarization. The "solvated" Fock matrix is written as:

``` math
f_{pq}= f_{pq}^\mathrm{vac} + \mathbf{q}\cdot\mathbf{v}_{pq}^\mathrm{e}
```

The PCM matrix elements are more explicitly given as:

``` math
\mathbf{q}\cdot\mathbf{v}_{pq}^\mathrm{e} = \sum_{I}^{N_\mathrm{ts}}q_Iv_{pq,I}^\mathrm{e}
```
``` math
v_{pq,I}^\mathrm{e} = \left\langle \phi_p \left| \frac{-1}{|\mathbf{r} - \mathbf{s}_I|} \right| \phi_q \right\rangle
```
