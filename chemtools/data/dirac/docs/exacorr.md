orphan  

# Highly parallelised relativistic Coupled Cluster Code

<div class="index">

\*\*EXACC

</div>

# \*\*EXACC

This section discusses the highly parallelised relativistic Coupled
Cluster Code (ExaCorr) Pototschnig2021.

In this release only computations using the X2C Hamiltonian (with either
`HAMILTONIAN_.X2Cmmf` or `HAMILTONIAN_.X2C`) are possible.

Originally based on the tensor libraries
[TAL-SH](https://github.com/DmitryLyakh/TAL_SH) and
[ExaTENSOR](https://github.com/ORNL-QCI/ExaTENSOR) by Dmitry Lyakh
(which support both CPU and GPU architectures), through the Tensor
Algebra Processing Primitives (TAPP TAPP2026) API the code now supports
additional tensor libraries:
[TBLIS](https://github.com/MatthewsResearchGroup/tblis) (CPU
architectures), and the [TAPP reference
implementation](https://github.com/TAPPorg/reference-implementation)
(CPU architectures, testing only).

The choice of tensor library is made at compile time, see the section on
[tensor library installation and
use](../../installation/tensor_libraries.html) for further information
on choosing the library and setting up parallel runs.

In all supported tensor libraries, the tensors used in ExaCorr are
currently kept in working memory. As such, sufficent RAM needs to be
available.

In ExaTENSOR the memory is distributed, so each additional node used in
the calculation will contribute its memory to the memory pool accessible
by the library. Currently, it is recommended to use enough nodes that
the tensors fit, but not substantially more. In the current release, if
the library runs out of memory the code will not stop but enter a
blocked state and calculations will not advance. So carefully control
the advancement of your calculations, stopping them if they appear to
hang.

For the other tensor libraries, calculations can only be carried out on
a single node and as such are bound to the memory available in the
underlying hardware. For TAL-SH runs a global memory pool size is
controlled by the keywodd `EXACC_.TALSH_BUFF`. In the case of TAL-SH and
ExaTENSOR, instructions/suggestions for tesing memory requirements can
be found in the `exacorr_talsh_memory` and `exacorr_exatensor_memory`
tests.

## **Mandatory keywords**

<div class="index">

.OCCUPIED

</div>

### .OCCUPIED

Defines occupied orbitals. Specification of list or energy range (see
`orbital_strings`).

    .OCCUPIED
    energy -1.0 0.0 0.00001

<div class="index">

.VIRTUAL

</div>

### .VIRTUAL

Defines virtual orbitals. Specification of list or energy range (see
`orbital_strings`).

    .VIRTUAL
    20..30

## **Optional keywords**

All of the following kewords are optional. For convenience, these have
been separated into different classes though in the input such
distictions are not made. For tractical examples please consult the
inputs for the exacorr tests.

## **General**

<div class="index">

.OCC_BETA

</div>

### .OCC_BETA

Can be used to specify a "high-spin" reference determinant with a
different number of "barred" occupied orbitals, than "unbarred" occupied
spinors. If .OCC_BETA is specified .OCCUPIED is interpreted as a list of
unbarred (alpha) spinors. NB: alpha and beta are used in a loose sense
in relativistic calculations to indicate the (un)barred spinors.

    .OCC_BETA
    energy -1.0 0.0 0.00001

<div class="index">

.VIR_BETA

</div>

### .VIR_BETA

Can be used to specify a different number of "barred" virtual orbitals
than "unbarred" occupied spinors. If .VIR_BETA is specified .VIRTUAL is
interpreted as a list of unbarred (alpha) spinors. NB: alpha and beta
are used in a loose sense in relativistic calculations to indicate the
(un)barred spinors.

    .VIR_BETA
    20..30

<div class="index">

.PRINT

</div>

### .PRINT

Print level.

*Default:*

    .PRINT
     0

<div class="index">

.EXATENSOR

</div>

### .EXATENSOR

This keyword activates the full multinode EXATENSOR library, which is
designed for massively parallel supercomputers. The additional
infrastructure needed for parallel communication makes this
implementation inefficient when used for single node runs. For such
purposes the use of the supported single-node tensor libraries is
recommended, as it avoids the computational overhead in setting up the
parallel environment, and for TAL-SH or some TAPP backends that may
enable the use of GPUs when available.

*Default:*

    Do not use EXATENSOR

<div class="index">

.LSHIFT

</div>

### .LSHIFT

Expert option: Level shift of orbital energies, ignored for values
smaller 0.

*Default:*

    .LSHIFT
     0.0D0

<div class="index">

.EXECUTOR

</div>

### .EXECUTOR

Selects the tensor library interface between TAPP (1), TAL-SH (2) and
EXATENSOR (3). In the case of TAPP, the tensor library backend
(reference implementation, TBLIS, etc) is preselected at compilation
time.

*Default:*

    .EXECUTOR
     2

<div class="index">

.AUTOKERNELS

</div>

### .AUTOKERNELS

Activates the use of automatically generated code for the solution of
amplitude, multipliers etc equations. The use of wavefunctions with
triples and higher excitations requires this keyword.

*Default:*

    Do not use automatically generated equations code 

<div class="index">

.AUTOSCHEME

</div>

### .AUTOSCHEME

Selects automatically generated code. Supported options are (1) for
mbauto mbauto and (2) for tenpi Brandejs2025.

*Default:*

    .AUTOSCHEME
     2

## **Integral transformation**

<div class="index">

.MOINT_SCHEME

</div>

### .MOINT_SCHEME

Expert option to choose another AO to MO integral transformation scheme.
Change at your own risk.

In TALSH only schemes 3 (default) and 42 (using Cholesky decompostion)
are available.

In ExaTensor schemes 1-4 and 42 are available with 42 using Cholesky
decompostion. Scheme 4 is default for ExaTensor as it reduces the memory
footprint by only keeping part of the AO integrals in memory. The other
methds keep all AO integrals in memeory. Scheme 0 prints the memory
requirements and attempts to allocate the memory without doing the
calculation.

*Default:*

    .MOINT_SCHEME
     3

<div class="index">

.CHOLESKY

</div>

### .CHOLESKY

Threshold to define the accuracy of the Cholesky decomposition (MOINT
scheme 42), resulting in inaccuracies of the computed energy of this
order of magnitude (in Hartree units).

*Default:*

    .CHOLESKY
     1.0D-9

## **Tensor library specific**

<div class="index">

.EXA_BLOCKSIZE

</div>

### .EXA_BLOCKSIZE

Expert option: Number to tune the parallel distribution (branching) of
the spinor spaces.

*Default:*

    .EXA_BLOCKSIZE
     75

<div class="index">

.TALSH_BUFF

</div>

### .TALSH_BUFF

Maximum memory (in gigabytes) used in TALSH, aim at about 80% of
available memory on your machine.

*Default:*

    .TALSH_BUFF
     50

## **Ground-state calculation general controls**

<div class="index">

.FF_PROP

</div>

### .FF_PROP

Carries out finite-field calculations for one or more operators (defined
in the `*PRPTRA` section) and one or more perturbation strenghts. The
input expects a first line with the number of properties and operators,
followed by blocks containing the operator label and a list of real and
imaginary components of the perturbations trenghts

    .FF_PROP
    2 3
    [operator label 1]
    re(P1), im(P1)
    re(P2), im(P2)
    re(P3), im(P3)
    [operator label 2]
    re(P4), im(P4)
    re(P5), im(P5)
    re(P6), im(P6)

where `P1,P2,...` are the perturbation strenghts. The perturbations are
only applied to the energy calculation, not to any subsequent steps.

*Example* The input below defines the `Heff_EDM` operator and carries
out finite-field calculations for two imaginary frequencies

    **MOLTRA
    .ACTIVE
    energy -1.00   2.0 0.001
    .NO4IND
    .PRPTRA
    *PRPTRA
    # eEDM effective operator
    .OPERATOR
     'Heff_EDM'
     iBETAGAM
     EDM
     COMFACTOR
     1.0D0
    **EXACC
    .OCCUPIED
    energy -1  0.0 0.001
    .VIRTUAL
    energy 0.0   2 0.001
    .OCC_BETA
    energy -1  -0.2 0.001
    .VIR_BETA
    energy -0.2   2 0.001
    .FF_PROP
    1 2
    Heff_EDM
    0.0000,0.00000
    0.0000, 0.005

*Default:*

    Finite field calculations are not enabled

<div class="index">

.TCONVERG

</div>

### .TCONVERG

Set convergence criteria (CC iterations, Lambda equations)

*Default:*

    .TCONVERG
     1.0D-9

<div class="index">

.NCYCLES

</div>

### .NCYCLES

Maximum number of allowed CC iterations to solve the CC and LAMBDA
equations.

*Default:*

    .NCYCLES
     30

<div class="index">

.LAMBDA

</div>

### .LAMBDA

Solve Lambda-equations, needs to be activated in order to compute the
one particle density matrix and molecular properties.

This calculation generates the file CCDENS, which contains the CC
ground-state density matrix in AO basis. In this release, CCDENS is used
by the property module to calculate ground-state expectation values.

If saved, CCDENS can be used in a property calculation (see
`PROPERTIES_.RDCCDM`) without the need to invoke this module.

*Default:*

    Lambda equations are not solved

<div class="index">

.GS2RDM

</div>

### .GS2RDM

Forms the 2-body reduced density matrix (2RDM) for the ground-state CCSD
wavefunction. When generated, the 2RDM and the 1RDM generated after
solving the Lambda-equations are used to calculate the correlation
energy (which can be compared to the one calculated with the CCSD
method).

The 2RDM is saved to a binary file in MO basis format (CCSD2RDM) written
to the scratch directory. If this file is of interest the user should
take action to copy it back to the work directory. There is no
functionality yet in DIRAC to transform the 2RDM to AO basis.

*Default:*

    2-body reduced density matrix is not formed

## **Wavefunctions for iterative methods**

The CCSD wavefunction is the general default for the code (ground-state
energies and expectation values, excited states and response
properties).

For ground-state amplitude and Lagrange multiplier calculations other
models can be chosen.

<div class="index">

.CC2

</div>

### .CC2

Performs a CC2 calculation instead of the default CCSD. Currently
supported only for energies.

*Default:*

    CC2 is not activated

<div class="index">

.CCDOUBLES

</div>

### .CCDOUBLES

Performs a CCD calculation instead of the default CCSD (switch off the
contributions of single excitations).

*Default:*

    CCDOUBLES is not activated

<div class="index">

.CCiT

</div>

### .CCiT

Defines CCSDT as the wavefunction for iterative ground-state amplitudes
Brandejs2025,mbauto and Lagrange multpliers Fabbro2025 calculations.

    .CCiT

<div class="index">

.CCiQ

</div>

### .CCiQ

Defines CCSDTQ as the wavefunction for iterative ground-state amplitudes
Brandejs2025,mbauto and Lagrange multpliers Fabbro2025 calculations.

    .CCiQ

<div class="index">

.CCiT1a

</div>

### .CCiT1a

> [!WARNING]
> This code is experimental.

Defines CCSDT-1a as the wavefunction for iterative grond-state
amplitudes. Currently only supported for amplitude code generated with
the mbauto tool.

    .AUTOSCHEME
    1
    .CCiT1a

<div class="index">

.CCiT1b

</div>

### .CCiT1b

> [!WARNING]
> This code is experimental.

Defines CCSDT-1b as the wavefunction for iterative grond-state
amplitudes. Currently only supported for amplitude code generated with
the mbauto tool.

    .AUTOSCHEME
    1
    .CCiT1b

<div class="index">

.CCiT2

</div>

### .CCiT2

> [!WARNING]
> This code is experimental.

Defines CCSDT-2 as the wavefunction for iterative grond-state
amplitudes. Currently only supported for amplitude code generated with
the mbauto tool.

    .AUTOSCHEME
    1
    .CCiT2

<div class="index">

.CCiT3

</div>

### .CCiT3

> [!WARNING]
> This code is experimental.

Defines CCSDT-3 as the wavefunction for iterative grond-state
amplitudes. Currently only supported for amplitude code generated with
the mbauto tool.

    .AUTOSCHEME
    1
    .CCiT3

## **Perturbative triples corrections for ground-state energies**

<div class="index">

.NOTRIPLES

</div>

### .NOTRIPLES

Deactivates computation of triples energy corrections. This is useful
for ExaTENSOR (as the current implementation is not efficient) and if
only excited states or properties are sought.

*Default:*

    CCSD(T) energies are calculated if CCSD wavefunctions are employed. 

<div class="index">

.TRIPL_BLOCK

</div>

### .TRIPL_BLOCK

Defines the number of blocks in the CCSD(T) calculation. Splitting the
evaluation of the triples correction to the energy reduces the memory
footprint of calculations at the expense of potentially longer execution
times.

    .TRIPL_BLOCK
    5

*Default:*

    .TRIPL_BLOCK
    1

## **Virtual space compression and natural orbital generation**

> [!WARNING]
> The virtual space compression code requires a two-step workflow: a
> first calculation to generate one of the flavors for the frozen
> virtual natural orbitals, and a second one restarting from these
> orbitals to carry out the CC calculations. For examples please consult
> the testset.

<div class="index">

.MP2NO

</div>

### .MP2NO

Perform a closed shell MP2 calculation and generate the frozen natural
orbitals (FNOs) Yuan2022 by diagonalization of the virtual-virtual block
of the MP2 density matrix, outputting the original occupied orbitals and
a set of truncated virtuals in AO basis (the FNOs are transformed into
AO basis and saved on file MP2NOs_AO, which has the same structure as
DFCOEF.)

The user must specify a threshold indicating a NO occupation number, and
orbitals with occupation below that will not be retained in the
truncated AO basis.

    .MP2NO
    1.0d-3

If one would like to use the new set including the FNOs to do
higher-level computation like CC, the MP2NOs_AO should be retrieved from
the work directory for the calculati, and used as a standard DFCOEF file
in subsequent calculations (currently a run in which FNOs are generated
and used in the same post-SCF calculation is not supported).

Apart from MP2NOs_AO, this option also ouputs the full, untruncated FNOs
space (MO basis) in file NOs_MO. The Fock matrix (FNO basis) is also
outputted to file FM_in_NatOrb.

The NOs_MO is provided so that users wishing to change the truncation
thresold above don't have to repeat the same MP2 calculation (If present
in the scratch directory, the NOs_MO will make the code skip the MP2
calculation).

*Default:*

    MP2FNO is not activated.

<div class="index">

.MP2NO++

</div>

### .MP2NO++

Performs a calculation to generate perturbed frozen virtual natural
orbitals (MP2FNO++) Le2026, based on the perturbation sensitive natural
spinors approach Chakraborty2025.

    .MP2NO++

*Default:*

    MP2NO++ is not activated.

<div class="index">

.NO_PROP

</div>

### .NO_PROP

Defines which properties will be taken into account in MP2FNO++
calculations. The number of properties considered is followed by their
index in the definition of the MO property operations in `*PRPTRA`.

    .NO_PROP
    3
    1 2 3

<div class="index">

.NO_WEIGHTED_SUM

</div>

### .NO_WEIGHTED_SUM

Defines whether to weight out properties in the generation of MP2FNO++.
When enabled,

    .NO_WEIGHTED_SUM
    1

the properties will be weighted by the number of properties (1/number of
properties defined in NO_PROP). Otherwise, all weights are set to 1.

*Default:*

    All properties used to define the MP2FNO++ have weights set to 1.

<div class="index">

.NO_OMEGA

</div>

### .NO_OMEGA

Defines the real and imaginary parts of the perturbing frequency
$`\omega`$ used in the MP2FNO++ calculation. For instance, to define a
perturbing frequency with 0.1 au for the real part and 0.003 for the
imaginary part:

    .NO_OMEGA
    0.100 0.003

The default is to carry out MP2FNO++ calculations for static
perturbations.

*Default:*

    .NO_OMEGA
    0.000 0.000

<div class="index">

.DONATORB

</div>

### .DONATORB

Calculate natural occupation numbers and orbitals in AO (quaternion)
basis, from a density matrix in the same basis (such as the one
generated by the first-order property code, which saved by default in
file CCDENS), and store them in file DFNOSAO.

This option assumes that CCDENS is in the work directory for the
calculation.

*Default:*

    DONATORB is not activated.

## **Laplace transform options**

<div class="index">

.MP2LAP

</div>

### .MP2LAP

Enables the use of the Laplace transform formalism in MP2 calculations.

*Default:*

    Laplace transform MP2 is disabled

<div class="index">

.MP2_LAP_NO

</div>

### .MP2_LAP_NO

Enables the use of the Laplace transform formalism in the generation of
MP2 frozen virtual natural orbitals.

*Default:*

    Laplace transform MP2 in FNOs generation is disabled

<div class="index">

.NLAPLACE

</div>

### .NLAPLACE

Selects the number of integration points for Laplace transform
calculations.

    .NLAPLACE
    15

*Default:*

    .NLAPLACE
    10

<div class="index">

.LAP_TRIPL

</div>

### .LAP_TRIPL

Enables the use of the Laplace transform formalism in CCSD(T)
calculations.

## **CC response theory calculations**

<div class="index">

.CCCILR

</div>

### .CCCILR

Calculates the orbital-unrelaxed coupled cluster linear response (LR)
function Yuan2024

``` math
\langle\langle A; B \rangle\rangle_{\omega_{B}}
```

where $`A`$ and $`B`$ represent one-body operators, and $`\omega_{B}`$
the perturbing frequency associated with $`B`$. Each of these operators
can correspond to electric or magnetic perturbations, though for the
latter it is only possible to carry out calculations in a common gauge
origin.

The current implementation can handle both static and
frequency-dependent (with either real or complex frequencies) response
calculations.

For linear response calculations two wave-function models can be used:
the standard linear response CC model (also referred to in the following
as CC-CC) and the EOM model (also referred to in the following as
CC-CI); we note EOM-LR is an approximation to LR-CC.

    .CCCILR
    1

where 1 indicates CC-CC type wave-function, and other numbers indicate
CC-CI type wave-function.

Apart from this keyword, additional inputs are needed to define a linear
response calculation : first users need also to define which operators
will be transformed to the MO basis, by specifying the appropriate input
for the `*PRPTRA` section of `**MOLTRA`. Second, the `EXACC_.LAMBDA`
keyword has to be used, to activate the calculation of ground-state
Lagrange multipliers.

<div class="index">

.CCRASYM

</div>

### .CCRASYM

Calculaties the orbital-unrelaxed coupled cluster linear response (LR)
function with the asymmetric formulation Yuan2024 for the CC (CC-CC)
approach.

With the asymmetric formulation only the responses to the $`B`$
perturbation are determined. As such, requing it can be useful when
there are more $`A`$ perturbations than $`B`$ ones.

    .CCRASYM

The asymmetric formulation, which is not defined for LR-EOM (CC-CI), is
disabled by default.

*Default:*

    CCRASYM is deactivated

<div class="index">

.CCCIQR

</div>

### .CCCIQR

Calculates the orbital-unrelaxed coupled cluster quadratic response (QR)
function for EOM (QR-EOMCC) Yuan2023 and coupled cluster (QR-CC)
Yuan2025. As for linear response, one must have in mind that QR-EOMCC is
an approximation to QR-CC

``` math
\langle\langle A;B,C\rangle\rangle_{\omega_{B},\omega_{C}}
```

As in the case of linear response, the code can currently be used in
static and frequency-dependent (with either real or complex frequencies)
response calculations.

    .CCCIQR
    1

1 indicates a QR-CC type calculation, and other numbers indicate a
QR-EOMCC type calculation.

The same requirements with respect to transforming operators to MO basis
and calculating ground-state Lagrange multipliers as for linear response
apply here.

<div class="index">

.LRFRE

</div>

### .LRFRE

Specify $`\omega_B`$ (in atomic unit) in the coupled cluster linear
response calculation

    .LRFRE
    3
    0.122,0.124,0.126
    0.010,0.010,0.010

The first line contains the number of frequencies to be used. The second
line indicates the real part of each frequency. The third line indicates
the imaginary part of each frequency.

<div class="index">

.QR1FRE

</div>

### .QR1FRE

Specify $`\omega_{B}`$ (in atomic unit) in the coupled cluster quadratic
response calculation

    .QR1FRE
    1
    0.1
    0.0

The setup for $`\omega_{B}`$ is the same as LRFRE above.

<div class="index">

.QR2FRE

</div>

### .QR2FRE

Specify $`\omega_{C}`$ (in atomic unit) in the coupled cluster quadratic
response calculation

    .QR2FRE
    1
    0.15
    0.0

The setup for $`\omega_{C}`$ is the same as LRFRE above.

<div class="index">

.AProp

</div>

### .AProp

Specify the $`A`$ operator(s) in linear and quadratic response function
calculations. This keyword must always be specified in response
calculations.

    .AProp
    3
    1,2,3

The first line contains the number of operators that will be considered
as $`A`$, and the second line specifies a list of numerical operator
indexes corresponding to the order in which these appear in the
`*PRPTRA` section of `**MOLTRA`.

<div class="index">

.BProp

</div>

### .BProp

Specify the $`B`$ operator(s) in linear and quadratic response function
calculations. This keyword must always be specified in response
calculations.

    .BProp
    3
    1,2,3

The input follows the same structure as that of `EXACC_.AProp` above.

We note that in the case of linear response, the code will calculate
response functions for all possible combinations of $`A`$, $`B`$
operators.

<div class="index">

.CProp

</div>

### .CProp

Specify the C operator in quadratic response functions.

    .CProp
    3
    1,2,3

The input follows the same structure as that of `EXACC_.AProp` above.

We note that the the code will calculate response functions for all
possible combinations of $`A`$, $`B`$ and $`C`$ operators.

<div class="index">

.DProp

</div>

### .DProp

Specify the D operator in two-photon absorption cross-section
calculations based on quadratic response, see `EXACC_.CCTPA` below for
details:

    .DProp
    3
    1,2,3

The input follows the same structure as that of `EXACC_.AProp` above.

<div class="index">

.NTA

</div>

### .NTA

Indicate the number of time-antisymmetric operators in the response
function. From this information, the code will determine if the response
function is real or imaginary (see for instance Saue2002a pages
390-393).

    .NTA
    1

*Default:*

    NTA is 0

This determination is of importance for instance in the case of response
functions containing an odd number of time-antisymmetric operators (e.g.
optical rotation in the case of linear response, of Verdet constants in
the case of quadratic response).

In future releases this determination will be done automatically and
this feature will be deprecated.

*Example:*

The fragment of input below demonstrates the definition of the
components of the dynamic ($`\omega_B = (0.1, 0.0)`$) dipole-dipole
polarizability. Note the definition of the transformation to MO of the
$`x,y,z`$ components of the dipole operator via the `*PRPTRA` section of
`**MOLTRA` (we have respectively indexes 1, 2 and 3 when defining the
$`A, B`$ operators), and the request for $`\Lambda`$-equation
calculations for the ground state.

Note that it is very important to define the same range of active
orbitals in `**MOLTRA` and in `**EXACC`, though in the former the range
is continous and in the latter it is subdivided into occupied and
virtual.

    **MOLTRA
    .ACTIVE
    3..10
    .NO4IND
    .PRPTRA
    *PRPTRA
    .OPERATOR
     XDIPLEN
     COMFACTOR
     1.000
    .OPERATOR
     YDIPLEN
     COMFACTOR
     1.000
    .OPERATOR
     ZDIPLEN
     COMFACTOR
     1.000
    **EXACC
    .OCCUPIED
    3..5
    .VIRTUAL
    6..10
    .EXEOM
    1
    .PRINT
    1
    .LAMBDA
    .CCCILR
    20
    .LRFRE
    1
    0.1
    0.0
    .AProp
    3
    1,2,3
    .BProp
    3
    1,2,3
    *CCDIAG
    .CONVERG
    1.0d-6
    .MAXSIZE
    20
    .MAXITER
    50

Note: When EOM or response modules are activated, a `*CCDIAG` section
should also be defined under `**EXACC`. This menu allows one to control
the parameters of the iterativel solver in EOM or response calculations
(see Yuan2024 and Yuan2024b for a discussion of implementation details).
The options available can be found under the `*CCDIAG` section from
`**RELCC`.

<div class="index">

.CCTPA

</div>

### .CCTPA

Evaluate the coupled-cluster two-photon scattering amplitudes
$`S_{AB,CD}`$, obtained from quadratic response and EOM-EE left and
right eigenvectors:

``` math
S_{AB,CD}=T_{AB}^{0f}(\omega)T_{CD}^{f0}(\omega) = \frac{1}{2} [T_{AB}^{0f}(\omega)T_{CD}^{f0}(\omega) +  (T_{CD}^{0f}(\omega)T_{AB}^{f0}(\omega))^{*}]
```

``` math
T_{AB}^{f0}(\omega) = \sum_{n}\left[ \frac{\bra{f}\hat{A}\ket{n}\bra{n}\hat{B}\ket{0}}{\omega_{n}-{(\omega+i\gamma})}+\frac{\bra{f}\hat{B}\ket{n}\bra{n}\hat{A}\ket{0}}{\omega_{n}-(\omega^\prime+i\gamma)} \right]
```

    .CCTPA
    20
    4

where the fist line indicates the type of wavefunction (here CC-CI; for
CC-CC, which is currently not supported, the number would be equal to
one), and the second line specifies the number of the $`\langle f |`$
obtained from an EOM-EE calculation.

The $`\omega, \omega^\prime`$ frequencies do not need to be specified
since for this keyword they are both taken to be equal to half of the
excitation energy of the target state $`\langle f |`$:

``` math
\omega = \omega^\prime = \frac{1}{2}\omega_{f}
```

*Example:*

requesting CC-CI (EOM) two-photon scattering amplitudes between the
reference state (ground state) and the target state (EOM-EE excited
state number 4 out of 5 requested):

    .LAMBDA
    .CCTPA
    20
    4
    .QR1FRE
    1
    0.0
    0.0
    .AProp
    1
    1
    .BProp
    1
    3
    .CProp
    1
    1
    .DProp
    1
    3
    *EXEOM
    .NROOTS
    5
    .FLAVOR
    EE

The $`\omega, \omega^\prime`$ frequencies do not need to be specified
since for this keyword they are both taken to be equal to half of the
excitation energy of the target state $`\langle f |`$:

``` math
\omega = \omega^\prime = \frac{1}{2}\omega_{f}
```

> [!WARNING]
> This code is experimental. At the moment, we only can do calculations
> for the target state, which is non-degenerate, and r. Only CC-CI (EOM)
> version works, CC-CC is in development.

## **Excited state calculation options**

<div class="index">

.CIS

</div>

### .CIS

Activates the CIS module to obtain singly excited states Yuan2026.
Currently the code will form and diagonalize the full CI singles matrix,
so it is not necessary to specify the number of states to calculate.

    .CIS

*Default:*

    CIS is not activated.

<div class="index">

.CIS(D)

</div>

### .CIS(D)

Activates the CIS(D) module Yuan2026 to obtain perturbation corrections
to a number of CIS excited state energies. The following input requests
corrections for the 10 lowest excited states.

    .CIS(D)
    10

*Default:*

    CIS(D) is not activated.

<div class="index">

.EXEOM

</div>

### .EXEOM

Activate the equation of motion (EOMCC) module Yuan2024b. This option
can be used to obtain excitation energies (EOM-EE), singly ionized
(EOM-IP) and electron attached (EOM-EA) states starting from a closed
shell reference and at the CCSD level, by solving for the right-hand
eigenvalues and eigenvectors of the appropriate similarity-transformed
Hamiltonian. Users should specify the number of eom runs.

    .EXEOM
    1

*Default:*

    EXEOM is not activated.

<div class="index">

\*EXEOM

</div>

# \*EXEOM

If the EOM calculation is activated, This menu controls the parameters
for excited state calculations with the equation of motion coupled
cluster (EOM-CC) theory.

For EOM-CC, currently the implementation supports the excitation energy
(EE), single electron attachment (EA) and detachment (IP) models for
CCSD wavefunctions only. Furthermore, in the current release is not
possible to calculate oscillator strenghts.

Note that there is no default, if no options are selected no
calculations will be performed.

When EOM or response modules are activated, a `*CCDIAG` section should
also be defined under `**EXACC`. This menu allows one to control the
parameters of the iterativel solver in EOM or response calculations (see
Yuan2024 and Yuan2024b for a discussion of implementation details). The
options available can be found under the `*CCDIAG` section from
`**RELCC`.

## **Mandatory keywords**

<div class="index">

.NROOTS

</div>

### .NROOTS

Specify the number of roots in the calculation.

    .NROOTS
    4

<div class="index">

.FLAVOR

</div>

### .FLAVOR

Specify the flavor of EOM: EE (excitation energy), EA (single electron
attachment energies), and IP (single electron detachment energies) :

    .FLAVOR
    EE
