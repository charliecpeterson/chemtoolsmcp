orphan  

<div class="index">

\*DFT

</div>

# \*DFT

## **Introduction**

In this link the directives for modifying the DFT calculation are
described (e.g. modifying grid, using ALDA and much more)

*typical input (here correcting LDA):*

    **HAMILTONIAN
    .DFT
    LDA
    *DFT
    .SAOP
    LBalpha

Single line following .SAOP : asymptotic potential (currently LB94 or
LBalpha).

The specification of the functional is described in connection with the
.DFT option in the `HAMILTONIAN_.DFT` section

The DFT module is described in references Schipper2000, Saue2002,
Fossgaard2003.

It uses standard non-relativistic functionals. This is justified in many
cases since various studies, like refs. Varga1999, Varga2000, Mayer1996,
indicate that relativistic corrections to the exchange-correlation
functionals have a negligible effect on spectroscopic constants;
however, for example for core properties further studies are necessary.

## **Integration algorithm and thresholds**

<div class="index">

.SCREENING

</div>

### .SCREENING

Specify the screening threshold used in the BLAS-3 integration scheme.

*Default:*

    .SCREENING
     1.0D-18

<div class="index">

.POINTWISE

</div>

### .POINTWISE

Switch to old (unscreened) BLAS-2 integration scheme (default before
DIRAC11) instead of the new BLAS-3 scheme (default since DIRAC11).

<div class="index">

.TINYDENS

</div>

### .TINYDENS

Specify the tiny density threshold. Grid points with a density smaller
than this threshold do not contribute to XC matrix elements.

*Default:*

    .TINYDENS
     1.0D-14

## **Asymptotic correction**

<div class="index">

.GRAC

</div>

### .GRAC

Activate the gradient regulated asymptotic correction scheme (GRAC),
Gruning2001, followed by two lines of input.

First line: Asymptotic potential (currently LB94 or LBalpha).

Second line (free format): Parameters α and β (see reference
Gruning2001) the ionization potential, and the threshold for difference
in HOMO eigenvalue below which asymptotic correction is switched on. A
sufficiently converged density is needed before the correction is
activated (before the bulk potential is shifted in order to reproduce
the desired IP (see reference Gruning2001)

*Typical input (here correcting PBE0):*

    **HAMILTONIAN
    .DFT
    PBE0
    *DFT
    .GRAC
    LB94
    0.5 40.0 0.79248 1.0D-6

<div class="index">

.SAOP!

</div>

### .SAOP!

Activate the statistical averaging of (model) orbital potentials (SAOP)
as defined in reference Schipper2000. This implies (and activates) the
ALDA kernel. The functional under `HAMILTONIAN_.DFT` is expected to be
GLLBhole. The asymptotic potential is LBalpha.

*Recommended input:*

    **HAMILTONIAN
    .DFT
    GLLBhole
    *DFT
    .SAOP!

<div class="index">

.SAOP

</div>

### .SAOP

The more general version of `DFT_.SAOP!`

Followed by a single line of input: asymptotic potential (currently LB94
or LBalpha).

*Typical input (here correcting LDA):*

    **HAMILTONIAN
    .DFT
    LDA
    *DFT
    .SAOP
    LBalpha

## **Spin magnetization TDDFT**

<div class="index">

.NOSAOP

</div>

### .NOSAOP

Turn off spin density contribution to XC response.

<div class="index">

.COLLINEAR

</div>

### .COLLINEAR

Use the collinear approximation as a definition of the spin density.

``` math
s={m_{z}}={\sum\limits_{i}}{ \phi_{i}^{\dagger}{{\Sigma}_{z}}{\phi_{i}} }
```

instead of the default noncollinear definition

``` math
s = \vert \mathbf{m} \vert
```

<div class="index">

.BETASIGMA

</div>

### .BETASIGMA

use

``` math
\mathbf{m}=\sum_{i}{\phi_{i}^{\dagger}{\beta}{\mathbf{\Sigma}}{\phi_{i}}}
```

instead of the default form

``` math
\mathbf{m} = \sum_{i} \phi_{i}^\dagger \mathbf{\Sigma} \phi_{i}
```

## **Adiabatic local density approximation (ALDA)**

<div class="index">

.ALDA

</div>

### .ALDA

Approximate all functional derivatives beyond the xc potential by SVWN
derivatives. For hybrid functionals exact exchange is switched off in
the solution of the response equation.

<div class="index">

.XALDA

</div>

### .XALDA

Same as `DFT_.ALDA` but keep the fraction of exact exchange of the xc
functional under `HAMILTONIAN_.DFT`.

<div class="index">

.ALDA+

</div>

### .ALDA+

Use `DFT_.ALDA+` only for the Hermitian contribution (density
contribution) and use the proper xc kernel for the anti-Hermitian part
(spin density contribution).

<div class="index">

.ALDA-

</div>

### .ALDA-

Use `DFT_.ALDA-` only for the anti-Hermitian contribution (spin density
contribution) and use the proper xc kernel for the Hermitian part
(density contribution).

<div class="index">

.XALDA+

</div>

### .XALDA+

Use `DFT_.XALDA+` only for the Hermitian contribution (density
contribution) and use the proper xc kernel for the anti-Hermitian part
(spin density contribution).

<div class="index">

.XALDA-

</div>

### .XALDA-

Use `DFT_.XALDA-` only for the anti-Hermitian contribution (spin density
contribution) and use the proper xc kernel for the Hermitian part
(density contribution).

## **Other functionality**

<div class="index">

.GAUNTSCALE

</div>

### .GAUNTSCALE

Scale Gaunt integrals (if included) with the same factor as for
Hartree-Fock exchange. This means that hybrid functionals include
fractional HF Gaunt interaction, and pure functionals no HF Gaunt
interaction at all. If this option is not given the Gaunt integrals will
be included to 100%, meaning that even an LDA calculation will include
full Hartree-Fock Gaunt interaction when the .GAUNT keyword is given in
`**HAMILTONIAN`.

<div class="index">

.OVLDIAG

</div>

### .OVLDIAG

Activate the overlap diagnostic for TD-DFT calculations of excitation
energies according to Peach2008. This is only available in the
development version.
