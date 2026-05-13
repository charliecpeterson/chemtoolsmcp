orphan  

<div class="index">

\*FDE

</div>

# \*FDE

Frozen density embedding (FDE) directives.

General information on the FDE method can be found in recent reviews
(e.g. Jacob2013, Gomes2012a), whereas details concerning the Dirac
implementation are found in Gomes2008, Hofener2012 and Hofener2013.

## **Data import**

If Dirac is used to calculate the subsystem of interest, data from the
subsystem(s) making up the environment must be fed into the program.

<div class="index">

.EMBPOT

</div>

### .EMBPOT

Specifies the name of the file containing the embedding potential over
an integration grid.

*Default:*

    .EMBPOT
     EMBPOT

<div class="index">

.FRDENS

</div>

### .FRDENS

Specifies the name of the file containing the "frozen" quantities
(integration grid, electrostatic potential, electrostatic potential due
to external fields (e.g. point charges), density and (x,y,z) components
of the density gradient).

*Default:*

    .FRDENS
     FRZDNS

## **Embedding potential**

If the `FDE_.EMBPOT` option was selected, the program will simple
compute matrix elements of the the AO basis used by the program, and add
it to the one-electron Fock matrix. Because of that, this option can be
used with any wavefunction in Dirac. The ground-state calculation in
this case corresponds to the "fixed potential" scheme of Gomes2008.

However, if `FDE_.EMBPOT` is not specified, the program implicitly
selects the `FDE_.UPDATE` option, and assumes a frozen density has been
provided (see `FDE_.FRDENS`).

<div class="index">

.UPDATE

</div>

### .UPDATE

This option requires that the embedding potential for the ground-state
be calculated using the "frozen" quantities and those for the subsystem
treated with Dirac. The non-additive contributions are calculted with
the orbital-free kinetic energy and exchange-correlation density
functionals specified under `FDE_.NAKEF` and `FDE_.NAXCF`, respectively.

Currently this option is only supported during the SCF procedure, so it
does not have any effect for correlated wavefunctions (e.g. MP2, CI,
CC).

The `FDE_.EMBPOT` and this option cannot be specified at the same time.

<div class="index">

.NAKEF

</div>

### .NAKEF

Specifies the orbital-free kinetic energy density functional used to
calculate the non-additive kinetic energy contributions to the
ground-state embedding potential (see `FDE_.UPDATE`) and/or response
contributions (see `FDE_.RSP` or `FDE_.RSPLDA`)

*Default:*

    .NAKEF
     kin_tf

Which corresponds to the Thomas-Fermi kinetic energy functional. Other
available functionals are: PW91k (kin_pw91).

<div class="index">

.NAXCF

</div>

### .NAXCF

Specifies the exchange-correlation density functional used to calculate
the non-additive kinetic energy contributions to the ground-state
embedding potential (see `FDE_.UPDATE`) and/or response contributions
(see `FDE_.RSP` or `FDE_.RSPLDA`)

*Default:*

    .NAXCF
     lda

Where lda is equivalent to specifying "slaterx=1.0 vwnc=1.0" (without
quotes). Other available functionals are: pbe (equivalent to "pbex=1.0
pbec=1.0"), blyp (equivalent to "beckex=1.0 lypc=1.0"), pp86 (equivalent
to "pw86x=1.0 p86c=1.0")

## **Response contributions**

For response-based approaches (TDHF, TDDFT), contributions from FDE to
the active subsystem can be included through the keywords below (the
default is to not include any). For correlated wavefunctions (e.g. MP2,
CI, CC) these do not yet have any effect.

These options can be used together with either `FDE_.UPDATE` or
`FDE_.EMBPOT`, but require that "frozen" quantities are present (see
`FDE_.FRDENS`)

<div class="index">

.RSP

</div>

### .RSP

Specifies that FDE response contributions should be calculated employing
the density functionals specified under `FDE_.NAKEF` and `FDE_.NAXCF`,
if any.

<div class="index">

.RSPLDA

</div>

### .RSPLDA

Specifies that the non-additive exchange-correlation and kinetic energy
FDE response contributions should be calculated with the LDA and
Thomas-Fermi functionals, respectively.

## **Data export**

Dirac can also provide ground-state data (density and density gradient,
electrostatic potential etc) from the subsystem in question over a grid,
so that these can be used by other programs.

<div class="index">

.GRIDOU

</div>

### .GRIDOU

> [!WARNING]
> Development version only.

Specifies the grid over which the quantities will be calculated and
exported, and enables the export. The input is a XYZW file, and the
output is in XML format. The original file is overwritten.

*Default:*

    .GRIDOU
     GRIDOUT 

<div class="index">

.LEVEL

</div>

### .LEVEL

> [!WARNING]
> Development version only.

Specifies which kind of wavefunction will be used to obtain the exported
quantities

*Default:*

    .LEVEL
     DHF 

which corresponds to SCF (Hartree-Fock, DFT) ones. Also supported: MP2.

<div class="index">

.EXONLY

</div>

### .EXONLY

> [!WARNING]
> Development version only.

This option forces the program to skip the calculation of any FDE
contributions, and proceed to exporting the

> [!WARNING]
> Development version only.

## **Magnetic properties with London atomic orbitals - FDE contributions to the property gradient**

In calculations of magnetic properties with London atomic orbitals
(LAOs), additional contributions from FDE to the active subsystem should
be included in the property gradient (:cite:
<span class="title-ref">olejniczak2017</span>).

``` math
E_{[1],emb}^{[B]} = - \int v_{emb}^I \Omega_{ia}^{B,I} - \iint w_{emb}^{I,I} \Omega_{ia}^{I} \Omega_{jj}^{B,I} - \iint w_{emb}^{I,II} \Omega_{ia}^{I} \Omega_{jj}^{B,II}
```

where $`\Omega_{pq}^{B}`$ is the first-order perturbed overlap
distribution, which in a basis of London orbitals consists of two terms

``` math
\Omega_{pq}^{B} = \frac{i}{2} (R_{MN} \times r) \Omega_{pq} + (T_{pt}^{B*} \Omega_{tq} + \Omega_{pt}T_{tq}^B)
```

"direct" LAO term (the first term) and "reorthonormalization" term (two
last terms in brackets).

FDE contributions to property gradient can be included by the keywords
presented below.

<div class="index">

.LAO11

</div>

### .LAO11

This keyword turns on the FDE-LAO contributions to the property gradient
dependent on the embedding potential ($`v_{emb}^I`$) and the uncoupled
embedding kernel ($`w_{emb}^{I,I}`$). This is the advised option. Other
contributions to the property gradient can be included or excluded by
using expert keywords (below).

<div class="index">

.LDPT

</div>

### .LDPT

This keyword turns on the term dependent on the embedding potential
($`v_{emb}^I`$) only. Calculations with the `FDE_.LDPT` keyword are less
demanding than calculations with the `FDE_.LAO11` keyword, but may lead
to inaccurate results, especially for heavy elements.

<div class="index">

.NOLDPT

</div>

### .NOLDPT

This keyword turns off the term dependent on the embedding potential
($`v_{emb}^I`$).

<div class="index">

.L11KR

</div>

### .L11KR

This keyword turns on the terms dependent on the uncoupled embedding
kernel ($`w_{emb}^{I,I}`$).

<div class="index">

.NL11KR

</div>

### .NL11KR

This keyword turns off the terms dependent on the uncoupled embedding
kernel ($`w_{emb}^{I,I}`$).

<div class="index">

.LDKR

</div>

### .LDKR

This keyword corresponds to the term dependent on the uncoupled
embedding kernel ($`w_{emb}^{I,I}`$), which involves only the "direct"
LAO contribution.

<div class="index">

.NOLDKR

</div>

### .NOLDKR

This keyword excludes the "direct" LAO contribution to the uncoupled
embedding kernel ($`w_{emb}^{I,I}`$).

<div class="index">

.LRKR

</div>

### .LRKR

This keyword corresponds to the term dependent on the uncoupled
embedding kernel ($`w_{emb}^{I,I}`$), which involves only the
"reorthonormalization" LAO contribution.

<div class="index">

.NOLRKR

</div>

### .NOLRKR

This keyword excludes the "reorthonormalization" LAO contribution to the
uncoupled embedding kernel ($`w_{emb}^{I,I}`$).

<div class="index">

.LAO12

</div>

### .LAO12

This keyword turns on the FDE-LAO contributions to the property gradient
dependent on the coupled embedding kernel ($`w_{emb}^{I,II}`$). This
option includes both, the term dependent on the non-additive part of the
coupled embedding kernel and the Coulomb term. Each of these terms can
be turned on by separate keywords (below). If this keyword is employed,
it is also necessary to import perturbed densities using the
`FDE_.PERTIM` keyword.

<div class="index">

.LAOFRZ

</div>

### .LAOFRZ

This keyword corresponds to the term dependent on the non-additive part
of embedding kernel involving the coupling between two subsystems
($`w_{emb}^{I,II}`$). In order to calculate this term, it is necessary
also to use the keyword `FDE_.PERTIM`.

<div class="index">

.LFCOUL

</div>

### .LFCOUL

This keyword corresponds to the term dependent on the Coulomb part of
embedding kernel involving the coupling between two subsystems
($`w_{emb}^{I,II}`$). In order to calculate this term, it is necessary
also to use the keyword `FDE_.PERTIM`.

<div class="index">

.PERTIM

</div>

### .PERTIM

This keyword should be used whenever `FDE_.LAOFRZ` keyword is present.
It commands the program to read the contributions ("direct" LAO
contribution and "reorthonormalization" contribution) to the perturbed
density in LAO basis from files.

*Default:*

    .PERTIM
     pertden_direct_lao.FINAL
     pertden_reorth_lao.FINAL 

<div class="index">

.FRZNOS

</div>

### .FRZNOS

This keyword means that no spin-density contributions to the perturbed
density will be used in FDE calculations.

## **Magnetizability with London atomic orbitals - FDE contributions to the expectation value of the magnetizability tensor**

The FDE calculations of the magnetizability tensor with London atomic
orbitals (LAOs), require the FDE-LAO contributions to the property
gradient (presented above) and the FDE-LAO contributions to the
expectation value of the magnetizability tensor. The latter involve the
terms dependent on the embedding potential ($`v_{emb}^I`$), uncoupled
embedding kernel ($`w_{emb}^{I,I}`$) and coupled embedding kernel
($`w_{emb}^{I,II}`$).

<div class="index">

.EMAFDE

</div>

### .EMAFDE

This keyword turns on all FDE-LAO contributions to the expectation value
part of the magnetizability tensor. This is the advised option. It is
possible to exclude each contribution by using expert keywords
(presented below).

<div class="index">

.MNOPOT

</div>

### .MNOPOT

This keyword turns off the contribution to the expectation value part of
the magnetizability tensor dependent on the embedding potential
($`v_{emb}^I`$).

<div class="index">

.MNOUKE

</div>

### .MNOUKE

This keyword turns off the contribution to the expectation value part of
the magnetizability tensor dependent on the uncoupled embedding kernel
($`w_{emb}^{I,I}`$).

<div class="index">

.MNONKE

</div>

### .MNONKE

This keyword turns off the contribution to the expectation value part of
the magnetizability tensor dependent on the non-additive part of the
coupled embedding kernel ($`w_{emb}^{I,II}`$).

## **Debug/expert options**

<div class="index">

.SKIPX

</div>

### .SKIPX

Specifies that the non-additive exchange-correlation contributions are
not to be calculated.

<div class="index">

.SKIPK

</div>

### .SKIPK

Specifies that the non-additive kinetic energy contributions are not to
be calculated.
