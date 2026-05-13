orphan  

# Average-of-configuration open-shell Hartree-Fock

This is a short introduction to the theory behind
average-of-configuration open-shell Hartree-Fock as implemented in
DIRAC. For a more complete description the reader may consult chapter 3
of the PhD thesis of Jørn Thyssen Thyssen2004 .

It should first be noted that there is no restricted open-shell
Hartree-Fock (ROHF) code in DIRAC. The reason is that spin-orbit
interaction couples spin and spatial degrees of freedom and make the
formalism much more complicated since one cannot exploit spin symmetry
alone for fixing the expansion coefficients in the reference
configuration state function (CSF) which serves as the trial function.

Instead of optimizing the energy for a single open-shell state, we shall
optimize the energy for a limited set of open-shell states.

## Energy expression

Suppose that we have a set of $`N_{det}`$ of Slater determinants,
constituting our N-particle basis. We next construct and diagonalize a
CI matrix in this basis. This gives $`N_{det}`$ solutions of the form

``` math
\left|\Psi_I\right> = \sum_{P=1}^{N_{det}} \left|\Phi_P\right> C_{PI}
```

We will now find the set of orbitals which minimizes the average energy

``` math
E_{av}=\frac{1}{N_{det}}\sum_{I=1}^{N_{det}}\left<\Psi_I\left|H\right|\Psi_I\right>
```

Inserting the expansion of the solutions in terms of Slater determinants
and using the fact that the expansion coefficients $`C_{PI}`$ are
elements of a unitary matrix we obtain

``` math
E_{av}=\frac{1}{N_{det}}\sum_{P=1}^{N_{det}}\sum_{Q=1}^{N_{det}}\left<\Phi_P\left|H\right|\Phi_Q\right>\sum_{I=1}^{N_{det}}C_{PI}^*C_{QI}
       =\frac{1}{N_{det}}\sum_{P=1}^{N_{det}}\left<\Phi_P\left|H\right|\Phi_P\right>
```

showing that the average can also be taken over the N-particle basis
itself.

## Introducing open shells and active electrons

The above average energy expression is a functional of the orbitals
entering the Slater determinants. We will make a distinction between :

- *inactive orbitals*, present in all Slater determinants, represented
  by indices $`ijkl`$
- *active orbitals*, present in some, but not all Slater determinants,
  represented by indices $`uvxy`$
- *secondary orbitals*, not present in any Slater determinant,
  represented by indices $`abcd`$

We shall also employ indices $`pqrs`$ for general orbitals.

In order to generate our N-particle basis for averaging we will
distribute the orbitals into a number of shells. Each shell $`S`$ is
specified by $`M_S`$ orbital and $`N_S`$ electrons. Inactive and
secondary shells have $`N_S=M_S`$ and $`N_S=0`$, respectively, whereas
active shells have $`N_S < M_S`$. We generate our N-electron basis by
distributing all active electrons in all possible ways within their
respective shells. The total energy can then be written in terms of
orbitals rather than Slater determinants as

``` math
E_{av}=\sum_S f_s \left\{\sum_{p\in S}\left(h_{pp}+\frac{1}{2}\sum_{S^{\prime}}Q^{S^{\prime}}_{pp}+\frac{1}{2}(a_S-1)Q^S_{pp}\right)\right\}
```

where we have introduced

- the fractional occupation $`f_S=\frac{N_S}{M_S}`$ of shell $`S`$
- the coupling coefficient $`a_S`$
  - $`a_S=\frac{M_S\left(N_S-1\right)}{N_S\left(M_S-1\right)}`$ for
    $`f_S\ne 0`$
  - $`a_S=1`$ for $`f_S = 0`$
- two-electron contributions
  $`Q^S_{pq}=f_S\displaystyle{\sum_{r\in S}\langle pr||qr\rangle}`$

## Orbital rotations

We will use a exponential parametrization for the rotation of orbitals

``` math
\tilde{\phi}_p = \sum_q\phi_q\left[exp(-\kappa)\right]_{pq}
```

where $`\kappa`$ is an anti-Hermitian matrix to ensure unitarity of the
transformation. The exponential parametrization allows for unconstrained
optimization (no Lagrange multipliers). It also allows the easy
identification of redundant variational parameters, that is, parameters
whose variation does not change the energy. In this particular case one
finds that rotations within shells are redundant and the corresponding
matrix elements $`\kappa_{pq}`$ can be set to zero.

## Gradient elements and off-diagonal blocks of the Fock matrix

The generally non-zero elements of the gradient vector are:

- inactive-secondary rotations:

``` math
\quad g_{ia}=h_{ai}+\sum_{S^{\prime}} Q_{ai}^{S^{\prime}}=F^I_{ai}
```

- active-secondary rotations:

``` math
\quad g_{ua}=f_U\left[F^I_{au}+(a_U-1)Q^U_{au}\right];\quad u\in U
```

- inactive-active rotations:

``` math
\quad g_{iu}=(1-f_U)\left[F^I_{ui}+f_U\alpha_UQ^U_{ui}\right];\quad u\in U
```

- inter-shell active-active rotations

``` math
\quad g_{uv}=(f_U-f_V)F^I_{vu}-(\alpha_U-1)f_UQ^U_{vu}-(\alpha_V-1)f_VQ^V_{vu};\quad u\in U,\ v\in V
```

where we have introduced

``` math
\quad \alpha_S = \frac{1-a_S}{1-f_S}
```

These gradient elements allow the definition of the off-diagonal
elements of the Fock matrix:

- inactive-secondary block:

``` math
\mathbb{F}_{ia}=F^I_{ia}
```

- active-secondary block:

``` math
\mathbb{F}_{ua}=F^I_{ua}+(a_U-1)Q^U_{ua};\quad u\in U
```

- inactive-active block:

``` math
\mathbb{F}_{iu}=F^I_{iu}+f_U\alpha_UQ^U_{iu};\quad u\in U
```

- inter-shell active-active rotations $`(U\ne V)`$:

``` math
\begin{aligned}
\mathbb{F}_{uv}=\left\{\begin{array}{ll}
F^I_{uv}+\frac{(a_U-1)}{(f_U-f_V)}Q^U_{uv}+\frac{(a_V-1)}{(f_V-f_U)}Q^V_{uv}&\mbox{for }f_U\ne f_V\\
(a_U-1)Q^U_{uv}+(a_V-1)Q^V_{uv}&\mbox{for }f_U = f_V\\\end{array}\right.
\end{aligned}
```

## Diagonal blocks of the Fock matrix

The diagonal blocks of the Fock matrix are *a priori* not related to
gradient elements and there is therefore freedom of choice in their
specification. The specific choice will not affect the total energy, but
will affect orbitals energies as well as convergence of the AOC HF
calculation.

In order to obtain a meaningful definition of the diagonal blocks of the
Fock matrix we will consider an extension of Koopmans' theorem to
average-of-configuration Hartree-Fock, that is, we consider average
energy after removal of an electron from a specific shell $`T`$ and
using the same orbital set as for the original N-electron system.

The energy difference becomes

``` math
E^N_{av}-E^{N-1}_{av}=\frac{1}{M_T}\sum_{t\in T}\left[h_{tt}+\sum_S Q^S_{tt}+(a_{T}-1)Q_{tt}^T\right]
```

If we now define the diagonal block of the Fock matrix corresponding to
shell $`T`$ as

``` math
\mathbb{F}_{pq} = h_{pq}+\sum_S Q^S_{pq}+(a_{T}-1)Q_{pq}^T
```

the ionization potential associated with the electron removal becomes

``` math
IP = E^{N-1}_{av}-E^N_{av}=-\frac{1}{M_T}\sum_{t\in T}\varepsilon_t
```

In the case of a degenerate shell we simply find

``` math
IP = E^{N-1}_{av}-E^N_{av}=-\varepsilon_t,\quad t\in T
```

identical to the original Koopmans' theorem, whereas one in the general
case gets an average over the orbital energies of the shell.

Based on these observations we define the diagonal blocks of the AOC
Fock matrix as

- inactive-inactive block:

``` math
\quad \mathbb{F}_{ij}=F^I_{ij}
```

- secondary-secondary block:

``` math
\quad \mathbb{F}_{ab}=F^I_{ab}
```

- active-active block:

``` math
\quad \mathbb{F}_{uv}=F^I_{uv}+(a_{U}-1)Q_{uv}^U;\quad u,v \in U
```

*These are the definitions employed in DIRAC12 and onwards* (and also
the definition found in the thesis of Jørn Thyssen Thyssen2004). In
previous versions the term $`(a_{U}-1)Q_{uv}^U`$ was missing from the
active-active block. Since $`Q_{uu}^U`$ is positive and for an open
shell $`a_U<1`$ removal of this term tend to shift orbital energies of
the open shell upwards.

Convergence problems typically occur when orbital energies between
shells have similar values such that the selection of occupied orbitals
for the construction of the Fock matrix becomes ambiguous. In a
closed-shell system this will for instance happen when the HOMO-LUMO gap
closes. The definition of the active-active block in pre-DIRAC12 version
(which was in fact an unintended omittal) can in some instances lead to
improved convergence. More specifically, this happens when the orbitals
of an open shell and the closed shell (or another open shell) are almost
degenerate. However, such situations are often symptomatic for a wrong
choice of partitioning of orbitals into closed and open shells.
Furthermore, the definition of the active-active block in pre-DIRAC12
versions tend to close the HOMO-LUMO gap which may hamper convergence.

## Level shift

Whenever there is almost degeneracy of orbitals between different shells
the recommended strategy is to exploit the freedom in the definition of
diagonal blocks of the Fock matrix and introduce a *level shift*
$`\lambda`$, that is

``` math
\quad F^U_{uv} \rightarrow F^U_{uv}+\lambda\delta_{uv}
```

The level shift of secondary (virtual) orbitals is controlled by the
keyword `SCF_.LSHIFT`, whereas open shells can be shifted using the
keyword `SCF_.OLEVEL`.

# Convergence issues

Open-shell systems tend to be more difficult to converge than
closed-shell ones, because of additional orbital classes and more
possibilities of near-degeneracies between orbital classes. It is
important to understand that DIRAC will generally order orbitals
according to energy. Furthermore, DIRAC starts by filling closed-shell
orbitals, then open-shell ones. In the case of Uranium
($`[Rn]5f^36d^17s^2`$) the closed-shell $`7s`$ orbitals will have higher
orbital energies than the $`6d`$ open-shell ones, and this may lead to
convergence problems. Fortunately, since DIRAC21 convergence of
open-shell atoms is unproblematic thanks to atomic supersymmetry, see
keyword `SCF_.KPSELE`.
