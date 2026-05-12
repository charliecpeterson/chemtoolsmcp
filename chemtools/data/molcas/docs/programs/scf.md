<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/scf.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.51. SCF

[previous](<rpa.html> "4.2.50. RPA") | [next](<seward.html> "4.2.52. SEWARD") | [index](<../../genindex.html> "General Index")

# 4.2.51. SCF¶

  * Description

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Basic general keywords

    * Advanced general keywords

      * DFT functionals:

    * Keywords for direct calculations

    * Limitations

    * Input examples




## 4.2.51.1. Description¶

The SCF program of the Molcas program system generates closed-shell Hartree–Fock, open-shell UHF, and Kohn–Sham DFT wave functions.

The construction of the Fock matrices is either done conventionally from the two-electron integral file ORDINT, which was generated in a previous step by the SEWARD integral code, or alternatively (only for closed shell calculations) integral-direct by recomputing all the two-electron integrals when needed [[138](<../../references.html#id84> "J. Almlöf, K. Faegri, Jr., K. Korsell. J. Comput. Chem., 3 \(1982\) 385-399.")]. The later route is recommended for large basis sets or molecules, when the two-electron integral file would become extensively large. It is automatically taken, when the SCF program cannot find any ORDINT file in the work directory. The direct Fock matrix construction employs an efficient integral prescreening scheme, which is based on differential densities [[139](<../../references.html#id86> "D. Cremer, J. Gauss. J. Comput. Chem., 7 \(1986\) 274-282."), [140](<../../references.html#id88> "M. Häser, R. Ahlrichs. J. Comput. Chem., 10 \(1989\) 104-111.")]: only those AO integrals are computed, where the estimated contractions with the related differential density matrix elements give significant (Coulomb or exchange) contributions to the (differential) two-electron part of the Fock matrix. Integral prescreening is performed at two levels, (i) at the level of shell quadruples, and (ii) at the level of individual primitive Gaussians. Prescreening at the level of contracted functions is not supported, because this would be inefficient in the context of a general contraction scheme. In order to work with differential density and Fock matrices, a history of these entities over previous iterations has to be kept. All these matrices are partly kept in memory, and partly held on disk. The SCF program either works with simple differences of the actual and the previous density, or alternatively with minimized densities, obtained from linear combinations of the actual density and all the previous minimized densities.

Besides the conventional and the fully-direct algorithms there is also a semi-direct path, which allows for the storage of some of the AO integrals during the first iteration, which then are retrieved from disk in subsequent iterations. That path is taken, if the keyword DISK with an appropriate argument specifying the amount of AO integrals to store is found on the input stream. The semi-direct path is recommended for medium sized problems, where the two-electron integral file would become a bit too large (but not orders of magnitude).

The program contains a feature that allows you to make the orbitals partially populated during the aufbau procedure. This feature is not primarily intended to accelerate the convergence but rather to ensure that you do get convergence in difficult cases. The orbitals are populated with with electrons according to

\\[\eta_i=\frac{2}{1+e^{(\varepsilon_i-\varepsilon_f)/kT}}\\]

where \\(\varepsilon_i\\) is the orbital energy of orbital \\(i\\) and \\(\varepsilon_f\\) is the Fermi energy. In this “Fermi aufbau” procedure the temperature is slowly lowered until it reaches a minimum value and then kept constant until a stable closed shell configuration is determined. Then normal SCF iterations will be performed with the selected closed shell configuration. For systems that are not really closed shell systems, for example diradicals, you might end up in the situation that the program does not find any stable closed shell configuration. In that case it will continue to optimize the closed shell energy functional with partial occupation numbers. If this is the case, this is probably what you want, and such orbitals would be ideal as starting orbitals for an MCSCF calculation.

The initial orbital guess is either obtained by diagonalizing the bare nuclei Hamiltonian, from an initial guess produced by the module GUESSORB or from orbitals of a previous Hartree–Fock SCF calculation. These starting orbitals are automatically located in the order

  1. SCF orbitals from a previous calculation located in the RUNFILE

  2. SCF orbitals from a previous calculation located in a formatted orbitals file, INPORB.

  3. initial guess orbitals from module GUESSORB located in the RUNFILE and




The program has three types of convergence accelerating schemes: (i) dynamic damping [[141](<../../references.html#id24> "G. Karlström. Chem. Phys. Lett., 67 \(1979\) 348-350.")], (ii) the \\(C^2\\)-DIIS method using the orbital gradient as error vector [[142](<../../references.html#id64> "H. Sellers. Int. J. Quantum Chem., 45 \(1993\) 31-41.")], and (iii) a combined second-order update/\\(C^2\\)-DIIS procedure. The latter eliminates the Brillouin violating elements of the Fock matrix by proper orbital rotations and hence avoids diagonalization of the Fock matrix: the approximate inverse Hessian is updated (BFGS) in a first step, and then the new orbital displacement vector is obtained from the updated Hessian using \\(C^2\\)-DIIS extrapolation [[143](<../../references.html#id201> "T. H. Fischer, J. Almlöf. J. Phys. Chem., 96 \(1992\) 9768-9774.")]. Dynamic damping gives substantial improvements in highly anharmonic regions of the energy hyper surface, while the second-order update/\\(C^2\\)-DIIS procedure exhibits excellent convergence for less anharmonic regions. By default, dynamic damping is used during the first few iterations. When the change in the density between two subsequent iterations drops below a certain threshold the second-order update/\\(C^2\\)-DIIS procedure kicks in. It is also possible to use the older first order \\(C^2\\)-DIIS scheme instead of the second-order update/\\(C^2\\)-DIIS procedure by setting the density threshold for the latter to zero in the corresponding input card (keyword QNRThreshold).

By default SCF behaves in different ways depending on what kind of start orbitals are found according to

  1. No start orbitals are found. In this case the core Hamiltonian is diagonalized and these orbitals are used as start. The “Fermi aufbau” procedure is used until a stable configuration is found.

  2. Start orbitals from GUESSORB are found. In this case the HOMO–LUMO gap is analyzed and if it is small the “Fermi aufbau” procedure is used until a stable configuration is found. Otherwise the configuration suggested by GUESSORB is used.

  3. Start orbitals from a previous SCF calculation is found. The configuration from the previous SCF calculation is used, unless some problem is detected such as partial occupation numbers from an unconverged calculation. In the latter case “Fermi aufbau” is used.

  4. Start orbitals from an INPORB is in the same way as for start orbitals from an SCF calculation, see last point.




This behavior can be changed by suitable keywords described below.

One of the main objects of the SCF program in the context of the Molcas program system is to generate starting orbitals for subsequent MCSCF calculations. Two options are available to improve the canonical Hartree–Fock orbitals in this respect.

(i) It is possible to specify pseudo occupation numbers that are neither zero nor two, thus simulating to some extent an open shell system. The resulting wavefunction does not have any physical meaning, but will provide better starting orbitals for open shell systems.

(ii) Usually, the lowest virtual canonical Hartree–Fock orbitals are too diffuse as correlating orbitals in an MCSCF calculation. If the keyword IVO is encountered in the input stream, the SCF program will diagonalize the core Hamiltonian matrix within the virtual space and write the resulting more compact eigenvectors to the SCFORB and RUNFILE files, rather than the virtual eigenvectors of the Fock matrix. It should be noted, that this option must never be used, if the SCF wave function itself is used subsequently as a reference function: no MP2 or coupled cluster calculations after an SCF run with IVO!

A further method to generate starting orbitals for MCSCF calculations is to perform an SCF calculation for a slightly positively charged moiety.

## 4.2.51.2. Dependencies¶

The SCF program requires the one-electron integral file ONEINT and the communications file RUNFILE, which contains among others the basis set specifications processed by SEWARD. For conventional (not integral-direct) runs the two-electron integral file ORDINT is required as well. All these files are generated by a preceding SEWARD run.

## 4.2.51.3. Files¶

Below is a list of the files that are used/created by the program SCF.

### 4.2.51.3.1. Input files¶

SCF will use the following input files: ONEINT, ORDINT, RUNFILE, INPORB (for more information see [Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)).

### 4.2.51.3.2. Output files¶

SCFORB
    

SCF orbital output file. Contains the canonical Hartree–Fock orbitals for closed shell calculations. If the IVO option was specified, the virtual orbitals instead are those that diagonalize the bare nuclei Hamiltonian within that subspace.

UHFORB
    

Contains the canonical Hartree–Fock orbitals for open shell calculations.

UNAORB
    

This file is produced if you make a UHF calculation and it contain natural orbitals.

MD_SCF
    

Molden input file for molecular orbital analysis.

## 4.2.51.4. Input¶

Below follows a description of the input to SCF.

The input for each module is preceded by its name like:
    
    
    &SCF
    

Argument(s) to a keyword, either individual or composed by several entries, can be placed in a separated line or in the same line separated by a semicolon. If in the same line, the first argument requires an equal sign after the name of the keyword.

### 4.2.51.4.1. Basic general keywords¶

Below is a list of keywords that should cover the needs of most users.

TITLe
    

One line for the title

UHF
    

Use this keyword to run Unrestricted Hartree–Fock code. Note that current implementation of UHF code has some restrictions, and not all features of SCF program are supported.

HFC
    

Requests the computation of hyperfine coupling tensor matrix on each atom using spin polarization in the calculated spin unrestricted wavefunctions, has to be used with the keyword UHF.

ZSPIN
    

Use this keyword to specify the difference in the number of \\(\alpha\\) and \\(\beta\\) electrons in the system. The default is 0 or 1 depending on if there is an even or odd number of electrons. Any value different from 0 requires the UHF keyword. This keyword is not needed when you specify the number of electrons with the keyword OCCUpied.

SPIN
    

Alternative way of specifying the electronic spin of the system. The keyword is followed by an integer giving the value of spin multiplicity (\\(2S+1\\)). Default is 1 (singlet) or 2 (doublet) depending on if there is an even or odd number of electrons.

RS-Rfo
    

Use this keyword to optimize the SCF orbitals using the restricted step rational function optimization (RS-RFO) procedure. Default is the use of the quasi-Newton-Raphson C2-DIIS procedure.

S-GEk
    

Use this keyword to optimize the SCF orbitals using the restricted variance optimization (RVO) procedure, based on a subspace gradient-enhanced Kriging (S-GEK) surrogate model. Default is the use of the quasi-Newton-Raphson C2-DIIS procedure.

EXPAnd
    

Select method for subspace expansion in conjunction with the S-GEK method (see above). Possible values are: `1`: DIIS, `2`: BFGS, `3`: RS-RFO. The default is `1`.

KSDFT
    

Use this keyword to do density functional theory calculations. This keyword should be followed by a functional keyword. Use pymolcas help_func to see a list of available keywords, you can also specify a [Libxc](<https://www.tddft.org/programs/libxc/>) functional name, or a number \\(N\\) followed by \\(N\\) lines, each of them containing a weight factor and a Libxc functional name (or `HF_X` for exact exchange). Examples (all three should be equivalent):
    
    
    KSDFT=B3LYP                 * A functional keyword
    
    
    
    KSDFT=HYB_GGA_XC_B3LYP      * A Libxc functional name
    
    
    
    KSDFT=5                     * Five components with their weights
          0.20 HF_X             * Keyword for exact exchange
          0.08 XC_LDA_X         * Libxc functional names
          0.72 XC_GGA_X_B88     *  .
          0.19 XC_LDA_C_VWN_RPA *  .
          0.81 XC_GGA_C_LYP     *  .
    

DFCF
    

Use this keyword to scale the exchange terms and/or correlation terms of a density functional. This keyword should be followed by the scaling factor for the exchange terms and the scaling factor for the correlation terms, separated by a space. If the values are 1.0 (default), then the original density functional is used. For an HLE-type functional, use 1.25 (for exchange) and 0.5 (for correlation). Example: DFCF=1.25 0.5

CHARge
    

Use this keyword to set the number of electrons in the system. This number is defined by giving the net charge of the system. If this keyword is not specified, the molecule is assumed to have net charge zero. The input is given as
    
    
    Charge=n
    

where `n` is the charge of the system.

OCCUpied
    

Use this keyword to set the number of electrons in the system. This number is defined by giving the number of electron pairs per irreducible representation of the subgroup of \\(D_{2h}\\) used in the calculation. You can use one and only one of the keywords, CHARge and OCCUpied for this purpose. If neither of these keywords are specified CHARge is assumed with a net charge of zero. It should be noted that the “Fermi aufbau” procedure is not used when you specify this keyword. The input for one of the point groups \\(D_2\\), \\(C_{2h}\\) or \\(C_{2v}\\) is given as
    
    
    OCCUpied= n1 n2 n3 n4
    

where `n1` is the number of electron pairs (occupied orbitals) in the first irreducible representation, etc.

If UHF keyword was specified, occupation numbers must be specified in two lines: for alpha and beta spins

FERMi
    

Use this keyword to specify that you want to use the “Fermi aufbau” procedure for the first few iterations to ensure convergence. The orbitals will be partially populated according to a Fermi population. The input is gives as
    
    
    Fermi= m
    

where `m` is the temperature parameter according to

  * `m=0`: No temperature is used. Not recommended.

  * `m=1`: A low temperature is used and will yield swift convergence for well behaved systems.

  * `m=2`: A medium low temperature is used and will yield swift and safe convergence for most systems. This is the default value.

  * `m=3`: A medium temperature is used and you will obtain good convergence for closed shell systems. If the system is not a closed shell system, the temperature dependent aufbau procedure may not terminate. This will result in a density matrix with fractional occupation numbers.

  * `m=4`: A medium high temperature is used and the temperature dependent aufbau procedure will most probably not terminate. This is useful for generating starting orbitals for an MCSCF calculation.

  * `m=5`: A high temperature is used. Behaves as m=4 only more so.




It should be noted that only dynamic damping is used until the program has found a stable closed shell configuration. When this has happened the more efficient methods: the ordinary \\(C^2\\)-DIIS and the second order update/\\(C^2\\)-DIIS procedure, are enabled.

NOFErmi
    

Disable the “Fermi aufbau” procedure, even if it is automatically enabled. This will require some initial orbitals to be present (typically, by default, the orbitals generated by GUESSORB). If the occupations found with the input orbitals do not match the required number of electrons and multiplicity, the occupations will be set by populating the orbitals in energy order across all irreps.

CHOLesky
    

SCF will use Cholesky (or RI/DF) representation of the two-electron integrals to compute the corresponding contributions to the Fock or KS matrices. The default (LK) algorithm is used. The configuration may be tailored using the ChoInput section. Default is to not use Cholesky unless the Cholesky (or RI/DF) representation of the two-electron integrals has been produced by SEWARD.

CHOInput
    

This marks the start of an input section for modifying the default settings of the Cholesky SCF. Below follows a description of the associated options. The options may be given in any order, and they are all optional except for ENDChoinput which marks the end of the CHOInput section.

  * NoLK Available only within ChoInput. Deactivates the “Local Exchange” (LK) screening algorithm [[126](<../../references.html#id169> "F. Aquilante, T. B. Pedersen, R. Lindh. J. Chem. Phys., 126 \(2007\) 194106\(1–11\).")] in computing the Fock matrix. The loss of speed compared to the default algorithm can be substantial, especially for electron-rich systems. Default is to use LK.

  * DMPK Available only within ChoInput. Modifies the thresholds used in the LK screening. The keyword takes as argument a (double precision) floating point (non-negative) number used as correction factor for the LK screening thresholds. The default value is 0.045d0. A smaller value results in a slower but more accurate calculation.

**Note:** The default choice of the LK screening thresholds is tailored to achieve as much as possible an accuracy of the converged SCF energy consistent with the choice of the Cholesky decomposition threshold.

  * NODEcomposition Available only within ChoInput. Deactivates the Cholesky decomposition of the AO 1-particle density matrix. The Exchange contribution to the Fock matrix is therefore computed using occupied canonical orbitals instead of (localized) “Cholesky MOs” [[99](<../../references.html#id164> "F. Aquilante, T. B. Pedersen, A. Sánchez de Merás, H. Koch. J. Chem. Phys., 125 \(2006\) 174101\(1–7\).")]. This choice tends to lower the performances of the LK screening. Default is to perform this decomposition in order to obtain the Cholesky MOs.

  * TIME Activates printing of the timings of each task of the Fock matrix build. Default is to not show these timings.

  * MEMFraction Set the fraction of memory to use as global Cholesky vector buffer. Default: for serial runs 0.0d0; for parallel runs 0.3d0.



CONStraints
    

Performs a Constrained (Natural Orbitals) SCF calculation, available only in combination with Cholesky or RI integral representation. An example of input for the keyword CONS is the following:
    
    
    CONStraints
     2  3
     1 -1
     1  1  1
    
    ADDCorrelation
    pbe
    
    SAVErage
    

The keyword CONS has two compulsory arguments: the number of constrained NOs (in each irrep) to be used in the CNO-SCF calculation, followed by one line per irrep specifying the spin configuration of the so-called (+) wavelet (-1 \\(\rightarrow\\) beta, 1 \\(\rightarrow\\) alpha) The OPTIONAL keyword ADDC is used to include a correlation energy correction through a DFT functional specified as argument (LDA, LDA5, PBE and BLYP available at the moment) The OPTIONAL keyword SAVE forces the program to use spin-averaged wavelets.

OFEMbedding
    

Performs a Orbital-Free Embedding (OFE)SCF calculation, available only in combination with Cholesky or RI integral representation. The runfile of the environment subsystem renamed AUXRFIL is required. An example of input for the keyword OFEM is the following:
    
    
    OFEMbedding
     ldtf/pbe
    dFMD
     1.0   1.0d2
    FTHAw
     1.0d-4
    

The keyword OFEM requires the specification of two functionals in the form fun1/fun2, where fun1 is the functional used for the Kinetic Energy (available functionals: Thomas–Fermi, with acronym LDTF, and the NDSD functional), and where fun2 is the xc-functional (LDA, LDA5, PBE and BLYP available at the moment). The OPTIONAL keyword dFMD has two arguments: first, the fraction of correlation potential to be added to the OFE potential (zero for KSDFT and one for HF); second, the exponential decay factor for this correction (used in PES calculations). The OPTIONAL keyword FTHA is used in a freeze-and-thaw cycle (EMIL Do While) to specify the (subsystems) energy convergence threshold.

ITERations
    

Specifies the maximum number of iterations. The default is 400 which is also the largest number you can specify.

CORE
    

The starting vectors are obtained from a diagonalization of the core Hamiltonian.

LUMORB
    

The starting vectors are taken from a previous SCFORB file called INPORB.

FILEORB
    

The starting vectors are taken from a previous SCFORB file, specified by user.

GSSRunfile
    

The starting vectors are taken from the orbitals produced by GUESSORB.

HLGAp
    

This keyword is used to make the program level shift the virtual orbitals in such a way that the HOMO–LUMO gap is at least the value specified on the next line. This will help convergence in difficult cases but may lead to that it converges to an excited configuration. A suitable value is 0.2.

### 4.2.51.4.2. Advanced general keywords¶

SCRAmble
    

This keyword will make the start orbitals slightly scrambled, accomplished by making a few small random orbital rotations. How much the orbitals are scrambled is determined by the parameter read on the next entry. A reasonable choice for this parameter is 0.2 which correspond to maximum rotation angle of \\(\arcsin 0.2\\). Using this keyword may be useful for UHF calculations with same number of \\(\alpha\\) and \\(\beta\\) electrons that are not closed shell cases.

ORBItals
    

Specifies the number of orbitals in the subspace of the full orbital space defined by the basis set, in which the SCF energy functional is optimized. The size of this subspace is given for each of the irreducible representations of the subgroup of \\(D_{2h}\\). If this keyword is not specified when starting orbitals are read, the full orbital space is assumed. The keyword takes as argument _nIrrep_ (# of irreps) integers. **Note** that this keyword is only meaningful when the SCF program is fed with input orbitals (cf. LUMORB).

FROZen
    

Specifies the number of orbitals not optimized during iterative procedure. The size of this subspace is given for each of the irreducible representations of the subgroup of \\(D_{2h}\\). If this keyword is not specified the number of frozen orbitals is set to zero for each irreducible representation. If the starting vectors are obtained from a diagonalization of the bare nuclei Hamiltonian the atomic orbitals with the lowest one-electron energy are frozen. If molecular orbitals are read from INPORB the frozen orbitals are those that are read in first in each symmetry. The keyword takes as argument _nIrrep_ (# of irreps) integers.

OVLDelete
    

Specifies the threshold for deleting near linear dependence in the basis set. The eigenvectors of the overlap matrix with eigenvalues less than that threshold are removed from the orbital subspace, and do not participate in the optimization procedure. The default value is 1.0d-5. The keyword takes as argument a (double precision) floating point number. Note that the SCFORB file will contain the deleted orbitals as a complementary set to the actual SCF orbitals! In future use of this orbital file the complementary set should always be deleted from use.

PRORbitals
    

Specifies which orbitals are to be printed in the log file (standard output). The keyword takes as argument two integers. The possible values are:

0 — No orbitals printed.

1 — orbitals with orbital energies smaller than \\(2E_{\text{HOMO}}-E_{\text{LUMO}}\\) are printed.

2 — followed by real number (ThrEne); orbitals with orbital energies smaller than ThrEne are printed.

Default value is 1.

Second (optional) argument specifies a format:

0 — No orbitals printed

1 — Print only one-electron energies and occupation numbers

2 — Short print format

3 — Extended print format

Default value is 3 for small numbers of MOs and 2 for number of MOs > 256.

PRLScf
    

Specifies the general print level of the calculation. An integer has to be supplied as argument. The default value, 1, is recommended for production calculations.

THREsholds
    

Specifies convergence thresholds. Four individual thresholds are specified as arguments, which have to be fulfilled simultaneously to reach convergence: EThr, DThr and FThr specify the maximum permissible difference in energy, density matrix elements and Fock matrix elements, respectively, in the last two iterations. The DltNTh finally specifies the norm of the orbital displacement vector used for the orbital rotations in the second-order/\\(C^2\\)-DIIS procedure. The corresponding values are read in the order given above. The default values are 1.0d-9, 1.0d-4, 1.5d-4, and 0.1d-2, respectively. **Note** that these thresholds automatically define the threshold used in the direct Fock matrix construction to estimate individual contributions to the Fock matrix such that the computed energy will have an accuracy that is better than the convergence threshold.

NODIis
    

Disable the DIIS convergence acceleration procedure.

DIISthr
    

Set the threshold on the change in density, at which the DIIS procedure is turned on. The keyword takes as argument a (double precision) floating point number. The default value is 0.15.

QNRThr
    

Set the threshold on the change in density, at which the second-order/\\(C^2\\)-DIIS procedure kicks in. The keyword takes as argument a (double precision) floating point number. The default value is 0.15.

**Note:** the change in density has to drop under both the DIISthr and the QNRThr threshold, for the second-order/\\(C^2\\)-DIIS to be activated. If the latter is set to zero the older first order \\(C^2\\)-DIIS procedure will be used instead.

C1DIis
    

Use \\(C^1\\)-DIIS for convergence acceleration rather than \\(C^2\\)-DIIS which is the default (not recommended).

NODAmp
    

Disable the Damping convergence acceleration procedure.

OCCNumbers
    

Gives the option to specify occupation numbers other than 0 and 2. This can be useful for generating starting orbitals for open shell cases. It should be noted however, that it is still the closed shell SCF energy functional that is optimized, thus yielding unphysical energies. Occupation numbers have to be provided for all occupied orbitals. In the case of UHF calculation occupation numbers should be specified on two different entries: for alpha and beta spin.

IVO
    

Specifies that the virtual orbitals are to be improved for subsequent MCSCF calculations. The core Hamiltonian is diagonalized within the virtual orbital subspace, thus yielding as compact orbitals as possible with the constraint that they have to be orthogonal to the occupied orbitals. **Note** that this option must not be used whenever the Hartree–Fock wavefunction itself is used as a reference in a subsequent calculation.

NOMInimization
    

Program will use density differences \\(D^{(k)}-D^{(k-1)}\\) rather than minimized differences.

ONEGrid
    

Disable use of a smaller intermediate grid in the integration of the DFT functional during the first SCF iterations.

RFPErt
    

This keyword will add a constant reaction field perturbation to the bare nuclei Hamiltonian. The perturbation is read from RUNOLD (if not present defaults to RUNFILE) and is the latest self consistent perturbation generated by one of the programs SCF or RASSCF.

STAT
    

This keyword will add an addition print outs with statistic information.

For calculations of a molecule in a reaction field see [Section 4.2.20.1.5](<gateway.html#ug-sec-rfield>) of the present manual and [Section 5.1.6](<../../advanced.examples/ex-rc.html#tut-sec-cavity>) of the examples manual.

#### 4.2.51.4.2.1. DFT functionals:¶

Below is a partial list of the keywords for DFT functionals currently implemented in the package. Note that most [Libxc](<https://www.tddft.org/programs/libxc/>) functionals are available too.

LSDA, LDA, SVWN
    

Vosko, Wilk, and Nusair 1980 correlation functional fitting the RPA solution to the uniform electron gas [[144](<../../references.html#id17> "S. H. Vosko, L. Wilk, M. Nusair. Can. J. Phys., 58 \(1980\) 1200-1211.")] (functional III in the paper).

LSDA5, LDA5, SVWN5
    

Functional V from the VWN80 paper [[144](<../../references.html#id17> "S. H. Vosko, L. Wilk, M. Nusair. Can. J. Phys., 58 \(1980\) 1200-1211.")] which fits the Ceperley–Alder solution to the uniform electron gas.

HFB
    

Becke’s 1988 exchange functional which includes the Slater exchange along with corrections involving the gradient of the density [[145](<../../references.html#id237> "A. D. Becke. Phys. Rev. A, 38 \(1988\) 3098-3100.")].

HFS
    

\\(\rho^{4/3}\\) with the theoretical coefficient of 2/3 also known as Local Spin Density exchange [[146](<../../references.html#id234> "P. Hohenberg, W. Kohn. Phys. Rev., 136 \(1964\) B864-B871."), [147](<../../references.html#id235> "W. Kohn, L. J. Sham. Phys. Rev., 140 \(1965\) A1133-A1138."), [148](<../../references.html#id273> "J. C. Slater. Quantum Theory of Molecular and Solids. Vol. 4. The Self-Consistent Field for Molecular and Solids. McGraw–Hill, 1974.")].

HFB86
    

Becke’s 1986 two-parameter exchange functional which includes the Slater exchange along with corrections involving the gradient of the density [[149](<../../references.html#id108> "A. D. Becke. J. Chem. Phys., 84 \(1986\) 4524-4529."), [150](<../../references.html#id168> "A. D. Becke, E. R. Johnson. J. Chem. Phys., 127 \(2007\) 124108\(1–8\).")].

HFO
    

Handy’s stand-alone OPTX exchange functional [[151](<../../references.html#id228> "N. C. Handy, A. J. Cohen. Mol. Phys., 99 \(2001\) 403-412.")].

BLYP
    

Becke’s 1988 exchange functional which includes the Slater exchange along with corrections involving the gradient of the density [[145](<../../references.html#id237> "A. D. Becke. Phys. Rev. A, 38 \(1988\) 3098-3100.")]. Correlation functional of Lee, Yang, and Parr, which includes both local and non-local terms [[152](<../../references.html#id238> "C. Lee, W. Yang, R. G. Parr. Phys. Rev. B, 37 \(1988\) 785-789."), [153](<../../references.html#id31> "B. Miehlich, A. Savin, H. Stoll, H. Preuss. Chem. Phys. Lett., 157 \(1989\) 200-206.")].

BPBE
    

Becke’s 1988 exchange functional which includes the Slater exchange along with corrections involving the gradient of the density [[145](<../../references.html#id237> "A. D. Becke. Phys. Rev. A, 38 \(1988\) 3098-3100.")] , combined with the GGA correlation functional by Perdew, Burke and Ernzerhof [[154](<../../references.html#id240> "J. P. Perdew, K. Burke, M. Ernzerhof. Phys. Rev. Lett., 77 \(1996\) 3865-3868.")].

B3LYP
    

Becke’s 3 parameter functional [[155](<../../references.html#id131> "A. D. Becke. J. Chem. Phys., 98 \(1993\) 5648-5652.")] with the form

\\[A E_{\text{X}}^{\text{Slater}} + (1-A) E_{\text{X}}^{\text{HF}} + B \Delta E_{\text{X}}^{\text{Becke}} + E_{\text{C}}^{\text{VWN}} + C \Delta E_{\text{C}}^{\text{non-local}}\\]

where the non-local correlation functional is the LYP functional and the VWN is functional III (not functional V). The constants \\(A\\), \\(B\\), \\(C\\) are those determined by Becke by fitting to the G1 molecule set, namely \\(A\\)=0.80, \\(B\\)=0.72, \\(C\\)=0.81.

B3LYP5
    

Becke’s 3 parameter functional [[155](<../../references.html#id131> "A. D. Becke. J. Chem. Phys., 98 \(1993\) 5648-5652.")] with the form

\\[A E_{\text{X}}^{\text{Slater}} + (1-A) E_{\text{X}}^{\text{HF}} + B \Delta E_{\text{X}}^{\text{Becke}} + E_{\text{C}}^{\text{VWN}} + C \Delta E_{\text{C}}^{\text{non-local}}\\]

where the non-local correlation functional is the LYP functional and the VWN is functional V. The constants \\(A\\), \\(B\\), \\(C\\) are those determined by Becke by fitting to the G1 molecule set, namely \\(A\\)=0.80, \\(B\\)=0.72, \\(C\\)=0.81.

B2PLYP_SCF
    

Grimme’s double-hybrid density functional [[156](<../../references.html#id160> "S. Grimme. J. Chem. Phys., 124 \(2006\) 034108\(1–16\).")] based on Becke’s 1988 exchange and LYP correlation GGA functionals with the form

\\[A E_{\text{X}}^{\text{Slater}} + (1-A) E_{\text{X}}^{\text{HF}} + A \Delta E_{\text{X}}^{\text{Becke}} + C E_{\text{C}}^{\text{LYP}} + (1-C) E_{\text{C}}^{\text{PT2}}\\]

The constants \\(A\\), and \\(C\\) are \\(A\\)=0.47, \\(C\\)=0.73. The SCF program computes only the DFT part of the B2PLYP energy. In order to get the PT2 term, one has to run the MBPT2 program on converged B2PLYP orbitals, and scale the MP2 correlation energy by the factor \\((1-C)\\)=0.27.

B86LYP
    

Becke’s 1986 exchange [[149](<../../references.html#id108> "A. D. Becke. J. Chem. Phys., 84 \(1986\) 4524-4529."), [150](<../../references.html#id168> "A. D. Becke, E. R. Johnson. J. Chem. Phys., 127 \(2007\) 124108\(1–8\).")] functional combined with the LYP correlation [[152](<../../references.html#id238> "C. Lee, W. Yang, R. G. Parr. Phys. Rev. B, 37 \(1988\) 785-789."), [153](<../../references.html#id31> "B. Miehlich, A. Savin, H. Stoll, H. Preuss. Chem. Phys. Lett., 157 \(1989\) 200-206.")].

BWig
    

Becke’s 1988 GGA exchange functional combined with the local Wigner correlation functional [[157](<../../references.html#id57> "P. A. Stewart, P. M. W. Gill. J. Chem. Soc. Faraday Trans., 91 \(1995\) 4337-4341.")].

GLYP
    

Gill’s 1996 GGA exchange functional [[158](<../../references.html#id224> "P. M. W. Gill. Mol. Phys., 89 \(1996\) 433-445.")] combined with the LYP correlation [[152](<../../references.html#id238> "C. Lee, W. Yang, R. G. Parr. Phys. Rev. B, 37 \(1988\) 785-789."), [153](<../../references.html#id31> "B. Miehlich, A. Savin, H. Stoll, H. Preuss. Chem. Phys. Lett., 157 \(1989\) 200-206.")].

OLYP
    

Handy’s OPTX exchange functional [[151](<../../references.html#id228> "N. C. Handy, A. J. Cohen. Mol. Phys., 99 \(2001\) 403-412.")] combined with the LYP correlation [[152](<../../references.html#id238> "C. Lee, W. Yang, R. G. Parr. Phys. Rev. B, 37 \(1988\) 785-789."), [153](<../../references.html#id31> "B. Miehlich, A. Savin, H. Stoll, H. Preuss. Chem. Phys. Lett., 157 \(1989\) 200-206.")].

OPBE
    

Handy’s OPTX exchange functional [[151](<../../references.html#id228> "N. C. Handy, A. J. Cohen. Mol. Phys., 99 \(2001\) 403-412.")] combined with the PBE correlation:cite:PBE:96.

O3LYP
    

A hybrid density functional based on the OPTX exchange [[159](<../../references.html#id48> "W.-M. Hoe, A. J. Cohen, N. C. Handy. Chem. Phys. Lett., 341 \(2001\) 319-328.")] , with the form

\\[A E_{\text{X}}^{\text{HF}} + B E_{\text{X}}^{\text{Slater}} + C \Delta E_{\text{X}}^{\text{OPTX}} + 0.19 E_{\text{C}}^{\text{VWN}} + 0.81 \Delta E_{\text{C}}^{\text{LYP}}\\]

The constants \\(A\\), \\(B\\), \\(C\\) are as provided in Ref. [[159](<../../references.html#id48> "W.-M. Hoe, A. J. Cohen, N. C. Handy. Chem. Phys. Lett., 341 \(2001\) 319-328.")]: \\(A\\)=0.1161, \\(B\\)=0.9262, \\(C\\)=0.8133.

KT3
    

The exchange-correlation functional by Keal and Tozer, 2004 [[160](<../../references.html#id51> "M. J. Allen, T. W. Keal, D. J. Tozer. Chem. Phys. Lett., 380 \(2003\) 70-77."), [161](<../../references.html#id156> "T. W. Keal, D. J. Tozer. J. Chem. Phys., 121 \(2004\) 5654-5660.")].

TLYP
    

\\[E_{\text{X}}^{\text{HF}} + E_{\text{C}}^{\text{non-local}}\\]

where the non-local correlation functional is the LYP functional.

PBE
    

The Perdew, Burke, Ernzerhof GGA functional 1996 [[154](<../../references.html#id240> "J. P. Perdew, K. Burke, M. Ernzerhof. Phys. Rev. Lett., 77 \(1996\) 3865-3868.")].

PBE0
    

The Perdew, Ernzerhof, Burke non-empirical hybrid functional 1996 [[162](<../../references.html#id143> "J. P. Perdew, M. Ernzerhof, K. Burke. J. Chem. Phys., 105 \(1996\) 9982-9985.")].

PBEsol
    

The Perdew et al. 2008 modification of PBE for solids.

RGE2
    

The regularized gradient approximation (RGE2) exchange functional by Ruzsinszky, Csonka, and Scuseria, 2009 that contains higher-power s terms in the exchange functional, as compared to the PBEsol. It is coupled with the PBEsol correlation [[163](<../../references.html#id178> "A. Ruzsinszky, G. I. Csonka, G. E. Scuseria. J. Chem. Theory Comput., 5 \(2009\) 763-769.")].

PTCA
    

The correlation functional by Tognetti, Cortona, and Adamo combined with the PBE exchange [[164](<../../references.html#id172> "V. Tognetti, P. Cortona, C. Adamo. J. Chem. Phys., 128 \(2008\) 034101\(1–8\).")].

SSB
    

The exchange functional SSB-sw by Swart, Solà, and Bickelhaupt, 2008 [[165](<../../references.html#id175> "M. Swart, M. Solà, F. M. Bickelhaupt. J. Chem. Phys., 131 \(2009\) 049103\(1–9\).")] that switches between the OPTX exchange for small values of the reduced density gradient and the PBE exchange for the large ones. It is combined with the PBE correlation functional.

M06
    

The M06 functional of the Minnesota 2006 class of functionals by Zhao and Truhlar [[166](<../../references.html#id165> "Y. Zhao, D. G. Truhlar. J. Chem. Phys., 125 \(2006\) 194101\(1–18\)."), [167](<../../references.html#id213> "Y. Zhao, D. G. Truhlar. J. Phys. Chem. A, 110 \(2006\) 13126-13130."), [168](<../../references.html#id268> "Y. Zhao, D. G. Truhlar. Theor. Chem. Acc., 120 \(2008\) 215-241."), [169](<../../references.html#id3> "Y. Zhao, D. G. Truhlar. Acc. Chem. Res., 41 \(2008\) 157-167.")].

M06L
    

The M06-L functional of the Minnesota 2006 class of functionals by Zhao and Truhlar [[166](<../../references.html#id165> "Y. Zhao, D. G. Truhlar. J. Chem. Phys., 125 \(2006\) 194101\(1–18\)."), [167](<../../references.html#id213> "Y. Zhao, D. G. Truhlar. J. Phys. Chem. A, 110 \(2006\) 13126-13130."), [168](<../../references.html#id268> "Y. Zhao, D. G. Truhlar. Theor. Chem. Acc., 120 \(2008\) 215-241."), [169](<../../references.html#id3> "Y. Zhao, D. G. Truhlar. Acc. Chem. Res., 41 \(2008\) 157-167.")].

M06HF
    

The M06-HF functional of the Minnesota 2006 class of functionals by Zhao and Truhlar [[166](<../../references.html#id165> "Y. Zhao, D. G. Truhlar. J. Chem. Phys., 125 \(2006\) 194101\(1–18\)."), [167](<../../references.html#id213> "Y. Zhao, D. G. Truhlar. J. Phys. Chem. A, 110 \(2006\) 13126-13130."), [168](<../../references.html#id268> "Y. Zhao, D. G. Truhlar. Theor. Chem. Acc., 120 \(2008\) 215-241."), [169](<../../references.html#id3> "Y. Zhao, D. G. Truhlar. Acc. Chem. Res., 41 \(2008\) 157-167.")].

M062X
    

The M06-2X functional of the Minnesota 2006 class of functionals by Zhao and Truhlar [[166](<../../references.html#id165> "Y. Zhao, D. G. Truhlar. J. Chem. Phys., 125 \(2006\) 194101\(1–18\)."), [167](<../../references.html#id213> "Y. Zhao, D. G. Truhlar. J. Phys. Chem. A, 110 \(2006\) 13126-13130."), [168](<../../references.html#id268> "Y. Zhao, D. G. Truhlar. Theor. Chem. Acc., 120 \(2008\) 215-241."), [169](<../../references.html#id3> "Y. Zhao, D. G. Truhlar. Acc. Chem. Res., 41 \(2008\) 157-167.")].

### 4.2.51.4.3. Keywords for direct calculations¶

_Note_ again that the threshold for contributions to the Fock matrix depends on the convergence thresholds mentioned above. The choice between the conventional and direct SCF methods is based on the presence of a two-electron integral file (file ORDINT). The keyword Direct in the SEWARD input controls that no two-electron integral file is to be generated and that integral direct algorithms can be used in subsequent modules. Thus, _the choice between conventional and direct SCF is done already in the input for the integral program_ SEWARD. The direct (or semi-direct) path will be taken whenever there are no two-electron integrals available.

CONVentional
    

This option will override the automatic choice between the conventional and the direct SCF algorithm such that the conventional method will be executed regardless of the status of the ORDINT file.

DISK
    

This option enables/disables the semi-direct algorithm. It requires two arguments which specifies the max Mbyte of integrals that are written on disk during the first iteration (and retrieved later in subsequent iterations) and the size of the corresponding I/O buffer in kbyte. The default values are 2000 MByte and 512 kByte. In case the specified disk space is zero and the I/O buffer is different from zero it will default to a semi-direct SCF with in-core storage of the integrals. The size of the memory for integrals storage is the size of the I/O buffer. If the size of the disk is non-zero and the I/O buffer size is zero the latter will be reset to the default value.

THIZe
    

This option specifies a threshold for two-electron integrals. Only integrals above this threshold (but not necessarily all of those) are kept on disk for the semi-direct algorithm. The keyword takes as argument a (double precision) floating point number. Default value is 1.0D-6.

SIMPle
    

If this option is specified, only a simple prescreening scheme, based solely on the estimated two-electron integral value will be employed (no density involved).

### 4.2.51.4.4. Limitations¶

The limitations on the number of basis functions are the same as specified for SEWARD.

### 4.2.51.4.5. Input examples¶

First we have the bare minimum of input. This will work well for almost all systems containing an even number of electrons.
    
    
    &SCF
    

The next example is almost as simple. Here we have an open shell case, i.e. you have an odd number of electrons in the neutral system and you need to generate starting orbitals for RASSCF. In this case we recommend that you perform a calculation on the cation with the input below.
    
    
    &SCF; Charge= 1
    

The next example explains how to run UHF code for a nitrogen atom:
    
    
    &SCF; UHF; ZSPIN=3
    

The next example is a bit more elaborate and show how to use a few of the keywords. The system is water that have the electron configuration \\(\text{1a}_1^2 \text{2a}_1^2 \text{3a}_1^2 \text{1b}_1^2 \text{1b}_2^2\\).
    
    
    &SCF; Title= Water molecule. Experimental equilibrium geometry. The symmetries are a1, b2, b1 and a2.
    Occupied= 3 1 1 0
    Threshold= 0.5D-9 0.5D-6 0.5D-6 0.5D-5
    * semi-direct algorithm writing max 128k words (1MByte) to disk
    * the size of the I/O buffer by default (512 kByte)
    Disk= 1 0
    Ivo
    

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
      * 4.2.51. SCF
      * [4.2.52. SEWARD](<seward.html>)
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

[previous](<rpa.html> "4.2.50. RPA") | [next](<seward.html> "4.2.52. SEWARD") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/scf.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
