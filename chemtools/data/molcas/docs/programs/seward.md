<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/seward.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.52. SEWARD

[previous](<scf.html> "4.2.51. SCF") | [next](<single_aniso.html> "4.2.53. SINGLE_ANISO") | [index](<../../genindex.html> "General Index")

# 4.2.52. SEWARD¶

  * Analytic integration

    * Description

    * Dependencies

    * Files

      * Input Files

      * Output files

    * Input

      * General keywords

      * Keywords associated to one-electron integrals

      * Additional keywords for property calculations

      * Keywords for two-electron integrals

      * Keywords associated to electron–molecule scattering calculations within the framework of the \\(R\\)-matrix method

      * The basis set label and the all electron basis set library

  * Numerical integration

    * Description

    * Input

  * Relativistic operators

    * Using the Douglas–Kroll–Hess Hamiltonian

    * Douglas–Kroll–Hess transformed properties

    * Using the X2C/Barysz–Sadlej–Snijders Hamiltonian

    * Local approximation to relativistic decoupling




The SEWARD module generates one- and two-electron integrals needed by other programs. The two-electron integrals may optionally be Cholesky decomposed. In addition, it will serve as the input parser for parameters related to the specification of the quadrature grid used in numerical integration in association with DFT and reaction field calculations.

In the following three subsections we will in detail describe the input parameters for analytic integration, numerical integration, and relativistic operators.

## 4.2.52.1. Analytic integration¶

Any conventional ab initio quantum chemical calculation starts by computing overlap, kinetic energy, nuclear attraction and electron repulsion integrals. These are used repeatedly to determine the optimal wave function and the total energy of the system under investigation. Finally, to compute various properties of the system additional integrals may be needed, examples include multipole moments and field gradients.

### 4.2.52.1.1. Description¶

SEWARD is able to compute the following integrals:

  * kinetic energy,

  * nuclear attraction,

  * two electron repulsion (optionally Cholesky decomposed),

  * \\(n\\)th (default \\(n\\)=2) order moments (overlap, dipole moment, etc.),

  * electric field (generated at a given point by all charges in the system),

  * electric field gradients (spherical gradient operators),

  * linear momentum (velocity),

  * orbital angular momentum,

  * relativistic mass-velocity correction (1st order),

  * one-electron Darwin contact term,

  * one-electron relativistic no-pair Douglas–Kroll,

  * diamagnetic shielding,

  * spherical well potential (Pauli repulsion),

  * ECP and PP integrals,

  * modified kinetic energy and multipole moment integrals (integration on a finite sphere centered at the origin) for use in the variational \\(R\\)-matrix approach,

  * external field (represented by a large number of charges and dipoles),

  * angular momentum products, and

  * atomic mean-field integrals (AMFI) for spin–orbit coupling.




_Note that while_ SEWARD _compute these integrals the input to select them and their settings are put in the input of_ GATEWAY.

In general these integrals will be written to a file, possibly in the form of Cholesky vectors (two-electron integrals only). However, SEWARD can also compute the orbital contributions and total components of these properties if provided with orbital coefficients and orbital occupation numbers.

To generate the one- and two-electron integrals SEWARD uses two different integration schemes. Repulsion type integrals (two-electron integrals, electric field integrals, etc.) are evaluated by the reduced multiplication scheme of the Rys quadrature [[170](<../../references.html#id123> "R. Lindh, U. Ryu, B. Liu. J. Chem. Phys., 95 \(1991\) 5889-5897.")]. All other integrals are computed by the Gauss–Hermite quadrature. SEWARD use spherical Gaussians as basis functions, the only exception to this are the diffuse/polarization functions of the 6-31G family of basis sets. The double coset [[171](<../../references.html#id96> "E. R. Davidson. J. Chem. Phys., 62 \(1975\) 400-403.")] formalism is used to treat symmetry. SEWARD is especially designed to handle ANO-type basis sets, however, segmented basis sets are also processed.

At present the following limitations are built into SEWARD:

Max number of unique basis functions: | 2000  
---|---  
Max number of symmetry independent centers: | 500  
Highest angular momentum: | 15  
Highest symmetry point group: | \\(D_{2h}\\)  
  
### 4.2.52.1.2. Dependencies¶

SEWARD usually runs after program GATEWAY. In the same time, any input used in GATEWAY can be placed into SEWARD input. However, it is recommended to specify all information about the molecule and the basis set in GATEWAY input.

SEWARD does normally not depend on any other code, except of GATEWAY. There are two exceptions however. The first one is when SEWARD is used as a property module. In these cases the file INPORB has to have been generated by a wave function code. The second case, which is totally transparent to the user, is when SEWARD picks up the new Cartesian coordinates generated by SLAPAF during a geometry optimization.

### 4.2.52.1.3. Files¶

#### 4.2.52.1.3.1. Input Files¶

Apart form the standard input file SEWARD will use the following input files: RYSRW, ABDATA, RUNFILE, INPORB (for calculation of properties) ([Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)). In addition, SEWARD uses the following files:

BASLIB
    

The default directory for one-particle basis set information. This directory contains files which are part of the program system and could be manipulated by the user in accordance with the instructions in [Section 4.2.59](<../basis_library.html#ug-sec-the-basis-set-libraries>) and following subsections. New basis set files can be added to this directory by the local Molcas administrator.

QRPLIB
    

Library for numerical mass-velocity plus Darwin potentials (used for ECPs).

#### 4.2.52.1.3.2. Output files¶

In addition to the standard output file SEWARD may generate the following files: ONEINT, ORDINT, CHVEC, CHRED, CHORST, CHOMAP, CHOR2F ([Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)).

### 4.2.52.1.4. Input¶

Below follows a description of the input to SEWARD. Observe that if nothing else is requested SEWARD will by default compute the overlap, the dipole, the quadrupole, the nuclear attraction, the kinetic energy, the one-electron Hamiltonian, and the two-electron repulsion integrals.

The input for each module is preceded by its name like:
    
    
    &SEWARD
    

Argument(s) to a keyword, either individual or composed by several entries, can be placed in a separated line or in the same line separated by a semicolon. If in the same line, the first argument requires an equal sign after the name of the keyword.

#### 4.2.52.1.4.1. General keywords¶

TITLe
    

One line of title card follows.

TEST
    

SEWARD will only process the input and generate a non-zero return code.

ONEOnly
    

SEWARD will not compute the two-electron integrals.

NODKroll
    

SEWARD will not compute Douglas–Kroll integrals.

DIREct
    

Prepares for later integral-direct calculations. As with keyword OneOnly, SEWARD will evaluate no two-electron integrals.

EXPErt
    

Sets “expert mode”, in which various default settings are altered. Integral-direct calculations will be carried out if the two-electron integral file is unavailable.

CHOLesky
    

SEWARD will Cholesky decompose the two-electron integrals using default configuration (in particular, the decomposition threshold is 1.0d-4) of the decomposition driver. The decomposition threshold can be changed using keyword THRC. Default is to not decompose.

1CCD
    

SEWARD will Cholesky decompose the two-electron integrals using the one-center approximation. The decomposition threshold can be changed using keyword THRC. Default is to not decompose.

THRCholesky
    

Specify decomposition threshold for Cholesky decomposition of two-electron integrals on the next line. Default value: 1.0d-4.

SPAN
    

Specify span factor (0 \\(<\\) span \\(\leq\\) 1) for Cholesky decomposition of two-electron integrals on the next line. Span=1 implies full pivoting, Span=0 means no pivoting. If the span factor is too low, numerical errors may cause negative diagonal elements and force the program to quit; if the span factor is too large, the execution time may increase. Default value: 1.0d-2.

LOW Cholesky
    

SEWARD will Cholesky decompose the two-electron integrals using low accuracy (threshold 1.0d-4) configuration of the decomposition driver. Default is to not decompose.

MEDIum Cholesky
    

SEWARD will Cholesky decompose the two-electron integrals using medium accuracy (threshold 1.0d-6) configuration of the decomposition driver. Default is to not decompose.

HIGH Cholesky
    

SEWARD will Cholesky decompose the two-electron integrals using high-accuracy (threshold 1.0d-8) configuration of the decomposition driver. Default is to not decompose.

FAKE CD/RI
    

If CD/RI vectors are already available, SEWARD will not redo work!

JMAX
    

The integer entry on the next line is the highest rotational quantum number for which SEWARD will compute the rotational energy within the rigid rotor model. The default value is 5.

SYMMetry
    

See the the description in the manual for the program GATEWAY.

BASIs Set
    

See the the description in the manual for the program GATEWAY.

ZMAT
    

See the the description in the manual for the program GATEWAY.

XBAS
    

See the the description in the manual for the program GATEWAY.

XYZ
    

See the the description in the manual for the program GATEWAY.

NOGUessorb
    

Disable automatic generation of starting orbitals with the GuessOrb procedure.

NODElete
    

Do not delete any orbitals automatically.

SDELete
    

Set the threshold for deleting orbitals based on the eigenvalues of the overlap matrix. All eigenvalues with eigenvectors below this threshold will be deleted. If you want no orbitals deleted use keyword NODElete.

TDELete
    

Set the threshold for deleting orbitals based on the eigenvalues of the kinetic energy matrix. All eigenvalues with eigenvectors above this threshold will be deleted. If you want no orbitals deleted use keyword NODElete.

VERBose
    

Force SEWARD to print a bit more verbose.

EMBEdding
    

Reads in an embedding potential from a file. It can also write out the density and the electrostatic potential on a grid. It is a block keyword which _must_ be ended with ENDEmbedding.

The subkeywords are:

EMBInput —
    

Specifies the path to the file which contains the embedding potential (e.g. `EMBI=myEmbPot.dat`). The file contains a potential given on a grid. It has the number of grid points in the first line. Then in five columns data for each grid point is given (x, y, z, weight of this grid point, value of the potential). Default is `EMBPOT`.

OUTGrid —
    

Specifies the path to a file containing a grid on which the output is produced. It is only needed if you want to have the data on a grid different from the one given in EMBInput. The columns for the potential and weights are not needed in this file (and also not read in).

WRDEnsity —
    

Switches on the calculation of the final electron density on a grid. The output file path must be specified along with this keyword (e.g. `WRDE=myDens.dat`).

WREPotential —
    

Switches on the calculation of the electrostatic potential on a grid. The output file path must be specified along with this keyword (e.g. `WREP=myESP.dat`).

ENDEmbedding —
    

Ends the EMBEdding section. This keyword _must_ be present.

The EMBEdding feature is currently only supported by the SCF part of Molcas.

#### 4.2.52.1.4.2. Keywords associated to one-electron integrals¶

MULTipoles
    

Entry which specifies the highest order of the multipole for which integrals will be generated. The default center for the dipole moment operator is the origin. The default center for the higher order operators is the center of the nuclear mass. The default is to do up to quadrupole moment integrals (2).

CENTer
    

This option is used to override the default selection of the origin of the multipole moment operators. On the first entry add an integer entry specifying the number of multipole moment operators for which the origin of expansion will be defined. Following this, one entry for each operator, the order of the multipole operator and the coordinates of the center (in au) of expansion are specified. The default is the origin for 0th and 1st multipoles, and the center of mass for higher-order multipoles.

RELInt
    

Requests the computation of mass-velocity and one-electron Darwin contact term integrals for the calculation of a first order correction of the energy with respect to relativistic effects.

RELAtivistic
    

Request arbitrary scalar relativistic Douglas–Kroll–Hess (DKH) correction to the one-electron Hamiltonian and the so-called picture-change correction to the property integrals (multipole moments and electronic potential related properties). An argument of the form `RXXPyy` follows. Here XX represents the order of the DKH correction to the one-electron Hamiltonian and yy the order of the picture-change correction. The character P denotes the parameterization used in the DKH procedure. The possible parametrizations P of the unitary transformation used in the DKH transformation supported by Molcas are:

`P=O`: Optimum parametrization (OPT)

`P=E`: Exponential parametrization (EXP)

`P=S`: Square-root parametrization (SQR)

`P=M`: McWeeny parametrization (MCW)

`P=C`: Cayley parametrization (CAY)

Hence, the proper keyword for the 4th order relativistically corrected one-electron Hamiltonian and 3rd order relativistically corrected property integrals in the EXP parameterization would read as R04E03. If yy is larger than XX it is set to XX. If yy is omitted it will default to same value as XX. Recommended orders and parametrization is R02O02. Since the EXP parameterization employs a fast algorithm, it is recommended for high order DKH transformation.

RLOCal
    

Request local approximation to the relativistic exact decoupling approaches such as X2C, BSS and DKH. This option cannot be used together with point group symmetry.

Grid Input
    

Specification of numerical quadrature parameters, consult the numerical quadrature section of this manual.

#### 4.2.52.1.4.3. Additional keywords for property calculations¶

VECTors
    

Requests a property calculation. For this purpose a file, INPORB, must be available, which contains the MO’s and occupation numbers of a wave function. A custom filename can be given with FileOrb.

FILEorb
    

The next line specifies the filename containing the orbitals and occupation numbers from which the properties will be computed. By default a file named INPORB will be used.

ORBCon
    

The keyword will force SEWARD to produce a list of the orbital contributions to the properties being computed. The default is to generate a compact list.

THRS
    

The real entry on the following line specifies the threshold for the occupation number of an orbital in order for the OrbCon option to list the contribution of that orbital to a property. The default is 1.0d-6.

ORBAll
    

When OrbCon is present, the keyword will force SEWARD to produce a list of the computed properties of all orbitals (including unoccupied orbitals), and the properties are not weighted by orbital occupation numbers. The total electronic and nuclear contributions printed are the same as those printed by using OrbCon without OrbAll.

#### 4.2.52.1.4.4. Keywords for two-electron integrals¶

NOPAck
    

The two-electron integrals will not be packed. The default is to pack the two-electron integrals.

PKTHre
    

An entry specifies the desired accuracy for the packing algorithm, the default is 1.0d-14.

THREshold
    

Threshold for writing integrals to disk follows. The default is 1.0d-14.

CUTOff
    

Threshold for ignoring the calculation of integrals based on the pair prefactor follows. The default is 1.0d-16.

#### 4.2.52.1.4.5. Keywords associated to electron–molecule scattering calculations within the framework of the \\(R\\)-matrix method¶

This section contains keyword which control the radial numerical integration of the diffuse basis functions describing the scattered electrons in the variational \\(R\\)-matrix approach. The activation of this option is controlled by that the center of the diffuse basis is assigned the unique atom label DBAS.

RMAT
    

Radius of the \\(R\\)-matrix sphere (in bohr). This sphere is centered at the coordinate origin. The default is 10 bohr.

RMEA
    

Absolute precision in radial integration. The default is 1d-9.

RMER
    

Relative precision in radial integration. The default is 1d-14.

RMQC
    

Effective charge of the target molecule. This is the effective charge seen by the incident electron outside of the \\(R\\)-matrix sphere. The default is 0d0.

RMDI
    

Effective dipole of the target molecule. This is the effective dipole seen by the incident electron outside of the \\(R\\)-matrix sphere. The default is (0d0,0d0,0d0).

RMEQ
    

Minimal value of the effective charge of the target molecule to be considered. This is also the minimal value of the components of the effective dipole to be considered. Default is 1d-8

RMBP
    

Parameter used for test purposes in the definition of the Bloch term. Default is 0d0.

CELL
    

Defines the three vectors of the unit cell (\\(\vec{e}_1\\), \\(\vec{e}_2\\), \\(\vec{e}_3\\)). The optional keyword _angstrom_ before the definition of vectors would read data in Å. Must consist of three entries (four in the case of Å) which correspond to coordinates of the vectors. All the atoms which are defined after that key are considered as the atoms of the cell.

SPREad
    

Three integer numbers \\(n_1\\), \\(n_2\\), \\(n_3\\) which define the spread of the unit cell along the unit cell vectors. For example, `0 0 2` would add all cell’s atoms translated on \\(-2\vec{e}_3\\), \\(-\vec{e}_3\\), \\(\vec{e}_3\\), \\(2\vec{e}_3\\). This key must be placed **before** the definition of the unit cell atoms.

Below follows an input for the calculation of integrals of a carbon atom. The comments in the input gives a brief explanation of the subsequent keywords.
    
    
    &SEWARD
    Title= This is a test deck!
    * Remove integrals from a specific irreps
    Skip= 0 0 0 0 1 1 1 1
    * Requesting only overlap integrals.
    Multipole= 0
    * Request integrals for diamagnetic shielding
    DSHD= 0.0 0.0 0.0; 1; 0.0 0.0 0.0
    * Specify a title card
    * Request only one-electron integrals to be computed
    OneOnly
    * Specify group generators
    Symmetry= X Y Z
    * Enable an inline basis set
    Expert
    * Specify basis sets
    Basis set
    C.ANO-L...6s5p3d2f.
    Contaminant d
    C  0.0 0.0 0.0
    End of basis
    

#### 4.2.52.1.4.6. The basis set label and the all electron basis set library¶

The label, which defines the basis set for a given atom or set of atoms, is given as input after the keyword Basis set. It has the following general structure (notice that the last character is a period):
    
    
    atom.type.author.primitive.contracted.aux1.aux2.
    

where the different identifiers have the following meaning:

`atom`
    

Specification of the atom by its chemical symbol.

`type`
    

Gives the type of basis set (ANO, STO, ECP, etc.) according to specifications given in the basis set library, _vide supra_. Observe that the upper cased character of the type label defines the file name in the basis directory.

`author`
    

First author in the publication where that basis set appeared.

`primitive`
    

Specification of the primitive set (e.g. 14s9p4d3f).

`contracted`
    

Specification of the contracted set to be selected. Some basis sets allow only one type of contraction, others all types up to a maximum. The first basis functions for each angular momentum is then used. **Note** , for the basis set types ANO and ECP, on-the-fly decontraction of the most diffuse functions are performed in case the number of contracted functions specified in this field exceeds what formally is specified in the library file.

`aux1`
    

Extra field that can be used to identify further variants. For an ECP, if the type is not `ECP`, this field must be `ECP`.

`aux2`
    

Specification of the type of AIMP, for instance, to choose between different embedding AIMP’s. non-relativistic and relativistic core AIMP’s.

Only the identifiers `atom`, `type`, and `contracted` have to be included in the label. The others can be left out. However, the periods have to be kept. Example — the basis set label “`C.ano-s...4s3p2d.`” will by Molcas be interpreted as “`C.ano-s.Pierloot.10s6p3d.4s3p2d.`”, which is the first basis set in the ANO-S file in the basis directory that fulfills the specifications given.

More information about basis set format can be found in the section Advanced examples.

## 4.2.52.2. Numerical integration¶

Various Density Functional Theory (DFT) models can be used in Molcas. Energies and analytical gradients are available for all DFT models. In DFT the exact exchange present in HF theory is replaced by a more general expression, the exchange-correlation functional, which accounts for both the exchange energy, \\(E_{\text{X}} [P]\\) and the electron correlation energy, \\(E_{\text{C}} [P]\\).

### 4.2.52.2.1. Description¶

We shall now describe briefly how the exchange and correlation energy terms look like. The functionals used in DFT are integrals of some function of the electron density and optionally the gradient of the electron density

\\[E_{\text{X}}[P] = \int f(\rho_{\alpha}(r), \rho_{\beta}(r), \nabla \rho_{\alpha}(r), \nabla \rho_{\beta}(r))\;dr\\]

The various DFT methods differ in which function, \\(f\\), is used for \\(E_{\text{X}}[P]\\) and for \\(E_{\text{C}}[P]\\). In Molcas pure DFT methods are supported, together with hybrid methods, in which the exchange functional is a linear combination of the HF exchange and a functional integral of the above form. The latter are evaluated by numerical quadrature. In the SEWARD input the parameters for the numerical integration can be set up. In the SCF and RASSCF inputs the keywords for using different functionals can be specified. Names for the various pure DFT models are given by combining the names for the exchange and correlation functionals.

The DFT gradients has been implemented for both the fixed and the moving grid approach [[172](<../../references.html#id130> "B. G. Johnson, P. M. W. Gill, J. A. Pople. J. Chem. Phys., 98 \(1993\) 5612-5626."), [173](<../../references.html#id58> "N. C. Handy, D. J. Tozer, G. J. Laming, C. W. Murray, R. D. Amos. Isr. J. Chem., 33 \(1993\) 331-344."), [174](<../../references.html#id137> "J. Baker, J. Andzelm, A. Scheiner, B. Delley. J. Chem. Phys., 101 \(1994\) 8894-8902.")]. The latter is known to be translationally invariant by definition and is recommended in geometry optimizations.

### 4.2.52.2.2. Input¶

Below follows a description of the input to the numerical integration utility in the SEWARD input.

Compulsory keywords

GRID Input
    

This marks the beginning of the input to the numerical integration utility.

END Of Grid-Input
    

This marks the end of the input to the numerical integration utility.

Optional keywords

GRID
    

It specifies the quadrature quality. The possible indexes that can follow are COARSE, SG1GRID, FINE, ULTRAFINE, following the Gaussian98 convention. Default is FINE.

RQUAd
    

It specifies the radial quadrature scheme. Options are LOG3 (Mura and Knowles) [[175](<../../references.html#id142> "M. E. Mura, P. J. Knowles. J. Chem. Phys., 104 \(1996\) 9848-9858.")], BECKE (Becke) [[176](<../../references.html#id114> "A. D. Becke. J. Chem. Phys., 88 \(1988\) 2547-2553.")], MHL (Murray et al.) [[177](<../../references.html#id223> "C. W. Murray, N. C. Handy, G. J. Laming. Mol. Phys., 78 \(1993\) 997-1014.")], TA (Treutler and Ahlrichs, defined for \\(\ce{H}\\)\--\\(\ce{Kr}\\)) [[178](<../../references.html#id138> "O. Treutler, R. Ahlrichs. J. Chem. Phys., 102 \(1995\) 346-354.")], and LMG (Lindh et al.) [[179](<../../references.html#id265> "R. Lindh, P.-Å. Malmqvist, L. Gagliardi. Theor. Chem. Acc., 106 \(2001\) 178-187.")], respectively. The default is MHL.

GGL
    

It activates the use of Gauss and Gauss–Legendre angular quadrature. Default is to use the Lebedev angular grid.

LEBEdev
    

It turns on the Lebedev angular grid.

LOBAtto
    

It activates the use of Lobatto angular quadrature. Default is to use the Lebedev angular grid.

LMAX
    

It specifies the angular grid size. Default is 29.

NGRId
    

It specifies the maximum number of grid points to process at one instance. Default is 128 grid points.

NOPRuning
    

It turns off the the angular pruning. Default is to prune.

NR
    

It is followed by the number of radial grid points. Default is 75 radial grid points.

NQDIrect
    

Recompute the values of the AOs on every SCF iteration. Default is to write them on disk on the first iteration and then retrieve them.

FIXEd grid
    

Use a fixed grid in the evaluation of the gradient. This corresponds to using the grid to numerically evaluate the analytic gradient expression. Default is to use a moving grid.

MOVIng grid
    

Use a moving grid in the evaluation of the gradient. This correspond to evaluating the gradient of the numerical expression of the DFT energy. This is the default.

RTHReshold
    

It is followed by the value for the the radial threshold. Default value is 1.0D-13.

T_Y
    

Threshold for screening in the assembling of the integrals. Note that in the SCF module the value is optionally adjusted to be the lower of the input or a value one magnitude tighter than the energy threshold. Default value is 1.0D-11.

NOSCreening
    

Turn off any screening in the numerical integration.

CROWding
    

The crowding factor, according to MHL, used in the pruning of the angular grid close to the nuclei. Default value 3.0.

The SCF and RASSCF programs have their own keywords to decide which functionals to use in a DFT calculation.

Below follows an example of a DFT calculation with two different functionals.
    
    
    &GATEWAY
    Basis set
    H.3-21G.....
    H1 0.0  0.0 0.0
    End of basis
    
    &SEWARD
    Grid input
     RQuad= Log3; nGrid= 50000; GGL; lMax= 26; Global
    End of Grid Input
    
    &SCF; Occupations=1; KSDFT=LDA5; Iterations= 1 1
    
    &SCF; Occupations= 1; KSDFT=B3LYP; Iterations= 1 1
    

## 4.2.52.3. Relativistic operators¶

The current different implementation of all relativistic operators in Molcas as described in the following sections has been programmed and tested in Ref. [[180](<../../references.html#id269> "D. Peng, M. Reiher. Theor. Chem. Acc., 131 \(2012\) 1081\(1–20\).")]

### 4.2.52.3.1. Using the Douglas–Kroll–Hess Hamiltonian¶

For all-electron calculations, the preferred way is to use the scalar-relativistic Douglas–Kroll–Hess (DKH) Hamiltonian, which, in principle, is available up to arbitrary order in Molcas; for actual calculations, however, the standard 2nd order is usually fine, but one may use a higher order than 8th order by default to be on the safe side.

The arbitrary-order Hamiltonian is activated by setting
    
    
    RXXPyy
    

somewhere in the SEWARD input, where the `XX` denotes the order of the DKH Hamiltonian in the external potential. I.e., for the standard 2nd-order Hamiltonian you may use `R02O`. Note in particular that the parametrization `P` does not affect the Hamiltonian up to fourth order. Therefore, as long as you run calculations with DKH Hamiltonians below 5th order you may use any symbol for the parametrization as they would all yield the same results.

The possible parametrizations `P` of the unitary transformation used in the DKH transformation supported by Molcas are:

`P=O`: Optimum parametrization (OPT)

`P=E`: Exponential parametrization (EXP)

`P=S`: Square-root parametrization (SQR)

`P=M`: McWeeny parametrization (MCW)

`P=C`: Cayley parametrization (CAY)

Up to fourth order (`XX`=04) the DKH Hamiltonian is independent of the chosen parametrization. Higher-order DKH Hamiltonians depend slightly on the chosen parametrization of the unitary transformations applied in order to decouple the Dirac Hamiltonian. Since the EXP parameterization employs a fast algorithm [[181](<../../references.html#id174> "D. Peng, K. Hirao. J. Chem. Phys., 130 \(2009\) 044102\(1–10\).")], it is recommended for high-order DKH transformation.

For details on the arbitrary-order DKH Hamiltonians see [[182](<../../references.html#id153> "M. Reiher, A. Wolf. J. Chem. Phys., 121 \(2004\) 2037-2047.")] with respect to theory, [[183](<../../references.html#id152> "M. Reiher, A. Wolf. J. Chem. Phys., 121 \(2004\) 10945-10956.")] with respect to aspects of implementation, and [[184](<../../references.html#id267> "M. Reiher. Theor. Chem. Acc., 116 \(2006\) 241-252.")] with respect to general principles of DKH. The current version of Molcas employs different algorithms, but the polynomial cost scheme of the DKH implementation as described in [[181](<../../references.html#id174> "D. Peng, K. Hirao. J. Chem. Phys., 130 \(2009\) 044102\(1–10\).")] is used as the default algorithm. The implementation in MOLCAS has been presented in [[180](<../../references.html#id269> "D. Peng, M. Reiher. Theor. Chem. Acc., 131 \(2012\) 1081\(1–20\).")].

For details on the different parametrizations of the unitary transformations see [[185](<../../references.html#id150> "A. Wolf, M. Reiher, B. A. Hess. J. Chem. Phys., 117 \(2002\) 9215-9226.")].

### 4.2.52.3.2. Douglas–Kroll–Hess transformed properties¶

As mentioned above, four-component molecular property operators need to be DKH transformed as well when going from a four-component to a two- or one-component description; the results do not coincide with the well-known corresponding nonrelativistic expressions for a given property but are properly picture change corrected.

The transformation of electric-field-like molecular property operators can be carried out for any order smaller or equal to the order chosen for the scalar-relativistic DKH Hamiltonian. In order to change the default transformation of order 2, you may concatenate the input for the DKH Hamiltonian by 2 more numbers specifying the order in the property,
    
    
    RxxPyy
    

where `yy` denotes the order of the Hamiltonian starting with first order 01. The DKH transformation is then done automatically for all one-electron electric-field-like one-electron property matrices.

Also note that the current implementation of both the Hamiltonian and the property operators is carried out in the full, completely decontracted basis set of the molecule under consideration. The local nature of the relativistic contributions is not yet exploited and hence large molecules may require considerable computing time for all higher-order DKH transformations.

For details on the arbitrary-order DKH properties see [[186](<../../references.html#id162> "A. Wolf, M. Reiher. J. Chem. Phys., 124 \(2006\) 064102\(1–11\).")] with respect to theory and [[180](<../../references.html#id269> "D. Peng, M. Reiher. Theor. Chem. Acc., 131 \(2012\) 1081\(1–20\)."), [187](<../../references.html#id163> "A. Wolf, M. Reiher. J. Chem. Phys., 124 \(2006\) 064103\(1–10\).")] with respect to implementation aspects.

### 4.2.52.3.3. Using the X2C/Barysz–Sadlej–Snijders Hamiltonian¶

Exact decoupling of the relativistic Dirac Hamiltonian can be achieved with infinite-order approaches, such as the so-called X2C (exact-two-component) and BSS (Barysz–Sadlej–Snijders) method. In Molcas, both methods are available for all-electron calculations. The evaluation of transformation matrices employs a non-iterative scheme.

The exact decoupling Hamiltonian is activated by setting either `RX2C` or `RBSS` somewhere in the SEWARD input, where `RX2C` and `RBSS` denote using the scalar (one-component) X2C and BSS Hamiltonian respectively. The one-electron Hamiltonian as well as the property integrals will be transformed according to the given exact decoupling method. In other words, all property integrals are by default picture change corrected.

The computation time of the X2C/BSS method is almost the same as of the DKH method at 8th order, while X2C is a little bit faster than BSS since the additional free-particle Foldy–Wouthuysen transformation is skipped in the X2C approach [[180](<../../references.html#id269> "D. Peng, M. Reiher. Theor. Chem. Acc., 131 \(2012\) 1081\(1–20\).")]. For molecules including only light atoms, the DKH method with low orders (< 8) is enough to account for the relativistic effects.

The differences between different exact decoupling relativistic methods are very small compared to errors introduced by other approximations, such as the basis set incompleteness, approximate density functionals, etc. Therefore, any exact decoupling model is acceptable for the treatment of relativistic effects in molecular calculations.

For details on the exact decoupling approaches see [[180](<../../references.html#id269> "D. Peng, M. Reiher. Theor. Chem. Acc., 131 \(2012\) 1081\(1–20\).")] with respect to theories and comparison of numerical results, [[188](<../../references.html#id159> "W. Kutzelnigg, W. Liu. J. Chem. Phys., 123 \(2005\) 241102\(1–4\)."), [189](<../../references.html#id161> "W. Liu, D. Peng. J. Chem. Phys., 125 \(2006\) 044102\(1–10\)."), [190](<../../references.html#id166> "D. Peng, W. Liu, Y. Xiao, L. Cheng. J. Chem. Phys., 127 \(2007\) 104106\(1–15\).")] for the X2C method, and [[191](<../../references.html#id69> "M. Barysz, A. J. Sadlej, J. G. Snijders. Int. J. Quantum Chem., 65 \(1997\) 225-239."), [192](<../../references.html#id54> "D. Kędziera, M. Barysz. Chem. Phys. Lett., 446 \(2007\) 176-181.")] for the BSS method.

### 4.2.52.3.4. Local approximation to relativistic decoupling¶

The computational cost for relativistic transformations increases rapidly if the molecule becomes larger. The local DLU scheme [[193](<../../references.html#id154> "D. Peng, M. Reiher. J. Chem. Phys., 136 \(2012\) 244108.")] was proposed to reduce the computational cost based on the atomic decomposition of the (unitary) decoupling transformation. It is important to note that the DLU scheme can be applied to any kind of relativistic approaches mentioned above (i.e., DKH, BSS, and X2C). It was found [[193](<../../references.html#id154> "D. Peng, M. Reiher. J. Chem. Phys., 136 \(2012\) 244108.")] that the DLU approximation introduces very small errors for total energies, which are less than 0.01 milli-hartree for molecules including heavy atoms. The local approximation is activated by setting RLOCal somewhere in the SEWARD input.

The direct local approximation to the Hamiltonian, called DLH in Ref. [[193](<../../references.html#id154> "D. Peng, M. Reiher. J. Chem. Phys., 136 \(2012\) 244108.")] may be activated by setting `RLOCal=DLH`. However, as DLH is not superior to the DLU scheme, but introduces slightly larger errors, it is not recommended.

The picture-change effect for molecular properties is automatically taken care of when a local approximation is employed for the transformed operator. The default order of DKH transformed properties is set to the same as the order of the DKH Hamiltonian.

It is important to note the local approximation cannot be used together with point group symmetry in the current implementation. Because the relativistic transformation is applied in the molecular orbital (MO) representation instead of the atomic orbital (AO) representation. Thus, the program will report an error and exit if symmetry is used.

### Table of Contents

  * [1\. Introduction](<../../intro.html>)
  * [2\. Installation Guide](<../../installation.guide/ig.html>)
  * [3\. Short Guide to Molcas](<../../tutorials/tut.html>)
  * [4\. User’s Guide](<../ug.html>)
    * [4.1. The Molcas environment](<../env-main.html>)
    * [4.2. Programs](<../programs.html>)
      * [4.2.1. ALASKA](<alaska.html>)
      * [4.2.2. AVERD](<averd.html>)
      * [4.2.3. CASPT2](<caspt2.html>)
      * [4.2.4. CASVB](<casvb.html>)
      * [4.2.5. CCSDT](<ccsdt.html>)
      * [4.2.6. CHCC](<chcc.html>)
      * [4.2.7. CHT3](<cht3.html>)
      * [4.2.8. CMOCORR ¤](<cmocorr.html>)
      * [4.2.9. CPF](<cpf.html>)
      * [4.2.10. DIMERPERT ¤](<dimerpert.html>)
      * [4.2.11. DMRGSCF](<dmrgscf.html>)
      * [4.2.12. DYNAMIX](<dynamix.html>)
      * [4.2.13. EMBQ ¤](<embq.html>)
      * [4.2.14. ESPF (+ QM/MM interface)](<espf.html>)
      * [4.2.15. EXPBAS](<expbas.html>)
      * [4.2.16. EXTF](<extf.html>)
      * [4.2.17. FALCON ¤](<falcon.html>)
      * [4.2.18. FALSE](<false.html>)
      * [4.2.19. FFPT](<ffpt.html>)
      * [4.2.20. GATEWAY](<gateway.html>)
      * [4.2.21. GENANO](<genano.html>)
      * [4.2.22. GEO ¤](<geo.html>)
      * [4.2.23. GRID_IT](<grid_it.html>)
      * [4.2.24. GUESSORB](<guessorb.html>)
      * [4.2.25. GUGA](<guga.html>)
      * [4.2.26. GUGACI](<gugaci.html>)
      * [4.2.27. GUGADRT](<gugadrt.html>)
      * [4.2.28. LEVEL](<level.html>)
      * [4.2.29. LOCALISATION](<localisation.html>)
      * [4.2.30. LOPROP](<loprop.html>)
      * [4.2.31. MBPT2](<mbpt2.html>)
      * [4.2.32. MCKINLEY (a.k.a. DENALI)](<mckinley.html>)
      * [4.2.33. MCLR](<mclr.html>)
      * [4.2.34. MCPDFT](<mcpdft.html>)
      * [4.2.35. MKNEMO ¤](<mknemo.html>)
      * [4.2.36. MOTRA](<motra.html>)
      * [4.2.37. MPPROP](<mpprop.html>)
      * [4.2.38. MPSSI](<mpssi.html>)
      * [4.2.39. MRCI](<mrci.html>)
      * [4.2.40. MULA](<mula.html>)
      * [4.2.41. NEMO ¤](<nemo.html>)
      * [4.2.42. NEVPT2](<nevpt2.html>)
      * [4.2.43. NUMERICAL_GRADIENT](<numerical_gradient.html>)
      * [4.2.44. POLY_ANISO](<poly_aniso.html>)
      * [4.2.45. QMSTAT](<qmstat.html>)
      * [4.2.46. QUATER](<quater.html>)
      * [4.2.47. RASSCF](<rasscf.html>)
      * [4.2.48. RASSI](<rassi.html>)
      * [4.2.49. RHODYN](<rhodyn.html>)
      * [4.2.50. RPA](<rpa.html>)
      * [4.2.51. SCF](<scf.html>)
      * 4.2.52. SEWARD
      * [4.2.53. SINGLE_ANISO](<single_aniso.html>)
      * [4.2.54. SLAPAF](<slapaf.html>)
      * [4.2.55. SURFACEHOP](<surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * [4.2.57. VIBROT](<vibrot.html>)
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<scf.html> "4.2.51. SCF") | [next](<single_aniso.html> "4.2.53. SINGLE_ANISO") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/seward.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
