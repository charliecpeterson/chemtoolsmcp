orphan  

# X2C and local X2C

This tutorial briefly demonstrates the keywords and their usage to
invoke the exact two-component (X2C) Hamiltonian and a local X2C variant
of it in Dirac. The current available implementation and other features
which will be part of the next release Dirac2014 are described in
Knecht2014.

A recent and comprehensive review on the topic of relativistic
Hamiltonians for chemistry can be found in Saue2011 by Trond Saue. The
central idea of any decoupling approach is to generate a two-component
Hamiltonian that reproduces the positive-energy spectrum of the parent
four-component Hamiltonian in the best possible way. This goal may be
achieved by a unitary transformation $`\mathbf{U}`$ that will formally
decouple the large and small components by block-diagonalization:

``` math
\begin{aligned}
{\hat{h}_{bd}} = 
\mathbf{U}^{\dagger}
\left[\begin{array}{cc}
\hat{h}_{LL} & \hat{h}_{LS} \\
\hat{h}_{SL} & \hat{h}_{SS} \end{array}\right] 
\mathbf{U} = 
\left[\begin{array}{cc}
\hat{h}_{++} & 0 \\
     0           & \hat{h}_{--} \end{array}\right],
\end{aligned}
```

Popular choices to obtain a unitary transformation matrix $`\mathbf{U}`$
are in particular discussed in Section 3 of Saue2011 while a detailed
abstract of the actual implementation of the X2C approach in Dirac is
given in the last part of Section 3.3 in the same review paper. Property
operators are transformed automatically on the fly to two-components
within the X2C approach in Dirac in order to avoid the so-called
picture-change errors. For a detailed discussion on the latter topic
with the example of charge densities at a heavy nucleus see for example
Knecht2011 by Knecht and co-workers. Transforming the the electronic
Hamiltonian based on the above prescription

``` math
{\hat{h}^{X2C}} = {\hat{h}_{++}} = 
\left[\mathbf{U}^{\dagger}
\hat{h}^{4c}\mathbf{U}\right]_{++}
```

it is mandatory to also transform the parent four-component
property-operator $`\Omega^{4c}`$ using the same unitary transformation
matrix $`\mathbf{U}`$

``` math
{\Omega^{2c}} = 
\left[\mathbf{U}^{\dagger}
\Omega^{4c}\mathbf{U}\right]_{++}
```

since otherwise we would introduce (significant) errors ('picture-change
errors') by simply using the approximate relation

``` math
{\Omega^{2c}} \approx \left[\Omega^{4c}\right]_{LL} .
```

## X2C

### X2C with spin-same-orbit two-electron corrections

To invoke the X2C Hamiltonian $`\hat{h}^{X2C}`$ in Dirac which is based
on the diagonalization of the 1-electron bare-nucleus Dirac Hamiltonian,
use the keyword .X2C under the input deck \*\*HAMILTONIAN:

    **HAMILTONIAN
    .X2C

This includes by default atomic-mean-field two-electron spin-same-orbit
corrections from the AMFI module.

### X2C with spin-same-orbit and spin-other-orbit two-electron corrections

To add spin-other-orbit corrections (Gaunt term of the Breit
interaction) use:

    **HAMILTONIAN
    .X2C
    .GAUNT

### spinfree X2C

In the spinfree case one should use:

    **HAMILTONIAN
    .X2C
    .SPINFREE

## local X2C

For large systems the diagonalisation of the Dirac Hamiltonian --- which
is an essential step in the derivation of the unitary transformation
matrix $`\mathbf{U}_{mol}`$ --- may become a bottleneck for the full
molecule (hence to stress this fact we have introduced the explicit
label *mol*). This bottleneck can be avoided by extracting the coupling
at the level of individual atoms as proposed in a working protocol by
Peng and Reiher Peng2012. In Dirac2013 we have implemented the *diagonal
local approximation to the unitary decoupling transformation* (DLU
scheme) of Peng2012. The central idea behind this local scheme is to
approximate $`\mathbf{U}_{mol}`$ by a superposition of $`i=1,...,N_A`$
atomic (or in principle also fragment) matrices $`\mathbf{U}_{i}`$ where
a decoupling matrix is calculated for each *unique* atom type.

``` math
\mathbf{U}_{mol} \approx \mathbf{U}_{i} \oplus  \mathbf{U}_{i+1} \oplus \mathbf{U}_{i+2} \oplus \ldots \mathbf{U}_{N_A} ,
```

In order to benefit from the above DLU approach in Dirac which at
present works *only for C1 symmetry* we need to first carry out the
atomic calculations and save the atomic $`\mathbf{U}_{i}`$ matrices.
This can be done as follows. We run an atomic X2C transformation for
each *unique* atom in our molecule (here we assume a dyall.v2z basis):

    **DIRAC
    .TITLE
     atomic X2C - U matrix construction
    **HAMILTONIAN
    .X2C
    **WAVE FUNCTION
    *SCF
    **MOLECULE
    *BASIS
    .DEFAULT 
    dyall.v2z
    *SYMMETRY
    .NOSYM
    **INTEGRALS
    *READIN
    .UNCONTRACT
    **END OF

Note the use of .NOSYM to enforce C1 symmetry in the atomic run even for
an atom. A possible corresponding xyz input for an atom (here Cl) would
look like:

    1
    chlorine atom XYZ file
    Cl 0.0 0.0 0.0

We save the file X2CMAT containing the atomic $`\mathbf{U}_{i}`$ matrix
and rename it according to the atomic number of the respective atom,
here X2CMAT.017 for chlorine (Z=17). For mercury (Z=80) the appropriate
ending would be X2CMAT.080 and similar for other atoms:

    $ ./pam --inp=atom.inp --mol=carbon.xyz --get=X2CMAT
    $ mv X2CMAT X2CMAT.006

After we have created all atomic $`\mathbf{U}_{i}`$ matrices for each
*unique* atom type in our molecule --- let's assume to work on Hg(Cl
$`_{2`$ ) $`_{6}`$ where we would need a transformation matrix for Hg
and Cl --- we are ready to setup the molecular X2C calculation based on
the local/atomic approach of the decoupling unitary $`\mathbf{U}_{i}`$
matrices. The relevant part of the input would then read:

    **DIRAC
    .TITLE
     local X2C based on atomic U matrices
    .WAVE FUNCTION
    **HAMILTONIAN
    .X2C
    *X2C
    .fragX2C
    **WAVE FUNCTION
    .SCF
    *SCF
    ...
    ...
    **MOLECULE
    *BASIS
    .DEFAULT 
    dyall.v2z
    *SYMMETRY
    .NOSYM
    **INTEGRALS
    *READIN
    .UNCONTRACT
    **END OF

This input should be used with the following pam line:

    $ ./pam ... --put=X2CMAT.* --mw=200

In this DLU approach proposed by Peng and Reiher Peng2012 the molecular
$`\mathbf{U}_{mol}`$ matrix will then be approximated as described
above. It works with spinfree as well as all spin-dependent (spin-orbit)
Hamiltonian schemes and includes a picture-change transformation of all
molecular four-component property operators $`\Omega^{4c}_{mol}`$.
