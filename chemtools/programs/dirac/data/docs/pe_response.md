orphan  

# PE-TDDFT calculations of excitation energies and solvent shifts

In this tutorial we investigate property calculations in solution
through the polarizable embedding (PE) model (Olsen2010, Olsen2011). The
tutorial builds on the implementation reported in Hedegaard2017 which
uses [PElib](https://gitlab.com/pe-software/pelib-public) to calculate
the environment contributions. This implementation allows calculation of
electric properties up to linear response (this corresponds to electric
properties up to second order), including an explicit environment.
Examples include electric dipole moments and polarizabilities. In
addition, the linear response module allows calculation of excitation
energies and transition moments, including the effect of a surrounding
environment through the PE model. Here we focus on TDDFT including a
solvent. A few comments regarding calculations of environment effect on
dipole moments and polarizabilities are given in the end of this
tutorial.

## TDDFT including an explicit environment through the PE model

The generalized eigenvalue problems that is solved when solving the
response equations

``` math
\left(E^{[2]} -\hbar\lambda S^{[2]}\right)X_m=0
```

which is completely equivalent to what is done in a vacuum calculation.
The main difference is in the electronic Hessian, $`E^{[2]}`$, which is
modified by the presence of an environment (cf. Eqs. 36-38) in
Hedegaard2017. The Hessian has the structure

``` math
\begin{aligned}
E^{[2]} =  \left(\begin{array}{cc}
                                \mathbf{A} & \mathbf{B} \\
                                \mathbf{B}^* & \mathbf{A}^{*}
                                \end{array} \right) ,
\end{aligned}
```

with matrix elements

``` math
\begin{aligned}
A_{ai;bj} = \langle 0 \vert [\hat{q}^{\dagger}_{ai},[\hat{q}_{bj},\hat{f}_0 + \hat{v}^{\mathrm{PE}}]] \vert 0 \rangle +
  \langle 0 \vert [\hat{q}^{\dagger}_{ai},\hat{v}^{\mathrm{ind}}]] \vert 0 \rangle \label{Hessian_2a} \\
B_{ai;bj} = \langle 0 \vert [\hat{q}_{ai},[\hat{q}_{bj},\hat{f}_0 + \hat{v}^{\mathrm{PE}}]] \vert 0 \rangle +
  \langle 0 \vert [\hat{q}_{ai},\hat{v}^{\mathrm{ind}}] \vert 0 \rangle .
\end{aligned}
```

The modification is manifested in the additional terms
$`\hat{v}^{\mathrm{PE}}`$ and $`\hat{v}^{\mathrm{ind}}`$, while
$`\hat{f}_0`$ is the usual Fock (or Kohn-Sham) operator. Note however,
that for Kohn-Sham theory contributions from the exchange-correlation
kernel will also be present (these are not shown in the above equation).
A brief, physical explaination of the the two additional terms is given
below

- $`\hat{v}^{\mathrm{PE}}=\hat{v}^{\mathrm{es}}+\hat{v}^{\mathrm{ind}}`$
  accounts for interactions between multipoles and ground-state density
  $`\hat{v}^{\mathrm{es}}`$ as well as mutual polarization between the
  ground-state density and the environment $`\hat{v}^{\mathrm{ind}}`$.
- $`\hat{v}^{\mathrm{ind}}`$ accounts for the differential interaction
  between the ground- and excited states with a polarizable environment.

A derivation and more thorough explanation of the $`E^{[2]}`$ term can
be found in Hedegaard2017 (cf. Eqs. 36-38).

To demonstrate how to calculate the effect of the environment with PE,
we use the same system as in the `PE-HF tutorial <pe_scf>`. The employed
`h2o.mol` file and potential `3_h2o.pot` can be found in that tutorial.
The input `pe_response.inp` is similar to a regular TDDFT calculation
but with an additional <span class="title-ref">.PELIB</span> keyword
under the <span class="title-ref">\*\*HAMILTONIAN</span> section:

    **DIRAC
    .TITLE
    H2O
    .WAVE FUNCTION
    .PROPERTIES
    **HAMILTONIAN
    .DFT
    B3LYP
    .PELIB        ! activate polarizable embedding
    **INTEGRALS
    *TWOINT
    .SCREEN
    1.0D-12
    *READIN
    .UNCONT
    **WAVE FUNCTIONS
    .SCF
    *SCF
    .CLOSED SHELL
    10
    **PROPERTIES
    *EXCITATION ENERGIES
    .EXCITA
    1 5
    .INTENS
    0
    **END OF

As for the case of PE-HF, the specification of
<span class="title-ref">.PELIB</span> will thus make the program
calculate the PE-specific terms; in case of linear response that is the
additional $`\hat{v}^{\mathrm{PE}}`$ and $`\hat{v}^{\mathrm{ind}}`$
terms. Note that use of the <span class="title-ref">.GSPOL</span>
keyword under the <span class="title-ref">\*PELIB</span> flag will
result in a calculation where only the term involving
$`\hat{v}^{\text{PE}}`$ is included (physically, this corresponds to a
frozen environment during the excitation process). Now we have all
ingredients for starting the calculation:

    pam --inp=pe_response.inp --mol=h2o.mol --pot=3_h2o.pot

The summary will look identical to a vacuum calculation:

    * Isotropic DL-DL non-zero oscillator strengths (f)
    ===================================================
    DL = dipole length
    Rate = Dipole radiation rate (s-1)
    Lifetime = corresponding radiation lifetime (s)

    Level  Frequency (eV)    f           Rate         Lifetime    Symmetry
    ------------------------------------------------------------------------
        1      7.00316    0.000001    1.26019E+03    7.93528E-04
        2      7.00317    0.000000    3.70356E+02    2.70010E-03
        3      7.00317    0.000002    3.30199E+03    3.02848E-04
        4      7.30170    0.049543    1.14615E+08    8.72484E-09
        5      8.53008    0.000000    5.32729E+01    1.87713E-02
    ------------------------------------------------------------------------
    Sum of oscillator strengths:      0.04955

Note that the first three excitations have very low oscillator strength.
These excitations corresponds to triplet excitations, which are
automatically included in a relativistic framework. The fourth
excitation is the observed one. We can calculate the electrostatic part
of the solvent shift (i.e. not including the solvent effect on the
geometry), by calculating the corresponding excitation energy without PE
and substracting the result from the PE result. We collect results with
and without PE (and the obtained solvent shift) in the table below.

| QM method (potential method) | Potential | PE   | vac. | shift |
|------------------------------|-----------|------|------|-------|
| B3LYP (B3LYP/6-31+G\*)       | 3 H2O     | 7.30 | 6.69 | 0.61  |
| Experimental                 |           | 8.2  | 7.4  | 0.8   |

The experimental value is taken from discussion given in
Christiansen2000. We note that

- The shift is a bit below the experimental one
- The excitation energies are somewhat underestimated (mainly due to the
  B3LYP functional).

To improve the model, we can either (i) expand the environment, (ii)
describe the environment more accurately or (iii) describe the QM system
more accurately (but currently DIRAC only includes PE with HF and DFT).
Finally, we might also (iv) expand the QM system. Performing these
calculations will be not be part of this tutorial. A table with selected
results from larger environments, potentials obtained with a more
accurate basis set (aug-cc-pVTZ) and a more accurate QM method are
compiled below. All these calculations are non-relativistic calculations
carried out with the Dalton program and taken from Hedegaard2016. The
"full CI" calculations are obtained with the density matrix
renormalization group (DMRG).

| QM method (potential method)          | Potential | PE   | vac. | shift |
|---------------------------------------|-----------|------|------|-------|
| Non. rel. B3LYP (B3LYP/6-31+G\*)      | 127 H2O   | 7.65 | 6.69 | 0.96  |
| Non. rel. full CI (B3LYP/6-31+G\*)    | 3 H2O     | 7.90 | 7.46 | 0.44  |
| Non. rel. full CI (B3LYP/6-31+G\*)    | 127 H2O   | 8.17 | 7.46 | 0.71  |
| Non. rel. B3LYP (B3LYP/aug-cc-pVTZ)   | 3 H2O     | 7.42 | 6.69 | 0.73  |
| Non. rel. B3LYP (B3LYP/aug-cc-pVTZ)   | 127 H2O   | 7.89 | 6.69 | 1.20  |
| Non. rel. full CI (B3LYP/aug-cc-pVTZ) | 3 H2O     | 7.95 | 7.46 | 0.49  |
| Non. rel. full CI (B3LYP/aug-cc-pVTZ) | 127 H2O   | 8.36 | 7.46 | 0.90  |

## Electric dipole moments and polarizabilities

It is also possible to calculate simple expectation values or other
linear response properties with the PElib module. The work-flow is
straightforward, and we only need to add the
<span class="title-ref">.PELIB</span> keyword as done for the TDDFT
calculation above. The input related to the property is identical to a
vacuum calculation. Examples of simple expectation values and linear
response properties that are currently implemented with PE is the
(electric) dipole moment and static or frequency-dependent
polarizabilities. Unlike in a non-relativistic framework, the
relativistic framework allow many magnetic properties to be obtained
simply as expectation values (e.g. properties related to electron
paramagnetic spectroscopy). All PE contributions to these properties
will be automatically included from the wave-function optimization and
will follow in a future release of an EPR module. Note also that linear
response properties that require London orbitals are also not currently
possible with PE. An example is given below for a dipole moment
calculation input file :

    **DIRAC
    .WAVE FUNCTION
    .PROPERTIES
    **HAMILTONIAN
    .PELIB
    **WAVE FUNCTION
    .SCF
    **PROPERTIES
    .DIPOLE
    *END OF

## Things to note

- Calculations with transformed two-component Hamiltonians are generally
  possibly by supplying the program with the requested model under
  <span class="title-ref">\*\*HAMILTONIAN</span>. In such a calculation,
  $`\hat{v}^{\mathrm{PE}}`$ is not transformed.
- Calculations with (linear) complex response or complex polarization
  propagator (CPP) is possible - see the tutorial "Complex response" (no
  other thing than activating CPP is needed to to PE-CPP calculations).
