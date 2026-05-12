<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/rassi.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.48. RASSI

[previous](<rasscf.html> "4.2.47. RASSCF") | [next](<rhodyn.html> "4.2.49. RHODYN") | [index](<../../genindex.html> "General Index")

# 4.2.48. RASSI¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Keywords

    * Input example




The RASSI (RAS State Interaction) program forms overlaps and other matrix elements of the Hamiltonian and other operators over a wave function basis, which consists of RASSCF wave functions, each with an individual set of orbitals. It is extensively used for computing dipole oscillator strengths, but any one-electron operator, for which the SEWARD has computed integrals to the ORDINT file, can be used, not just dipole moment components.

Also, it solves the Schrödinger equation projected on the space spanned by these wave functions, i.e., it forms non-interacting linear combinations of the input state functions, and computes matrix elements over the resulting eigenbasis as well.

Finally, using these spin-free eigenstates as a basis, it can compute spin–orbit interaction matrix elements, diagonalize the resulting matrix, and compute various matrix elements over the resulting set of spin–orbit eigenstates.

If only matrix elements of some one-electron operator(s), such as the dipole transition moments, are required, the calculation of Hamiltonian matrix elements and the transformation to the eigenbasis of this matrix can be skipped. However, if any states have the same symmetry and different orbitals, it is desirable to use the transitions strengths as computed between properly non-interacting and orthonormal states. The reason is that the individually optimized RASSCF states are interacting and non-orthogonal, and the main error in the computed transition matrix elements is the difference in electronic dipole moment times the overlap of any two states involved. For excited states, the overlap is often in the order of 10%.

Please note: Due to the increasing number of calculations done with a hundred input states, or more, there has been a demand to change the output. Until Molcas 6.2, the default assumption has been to print all expectation values and matrix elements that can be computed from the selection of one-electron integrals. From 6.4, this is requested by keywords, see the keyword list below for XVIN, XVES, XVSO, MEIN, MEES, and MESO.

Apart from computing oscillator strengths, overlaps and Hamiltonian matrix elements can be used to compute electron transfer rates, or to form quasi-diabatic states and reexpress matrix elements over a basis of such states.

The CSF space of a RASSCF wave function is closed under deexcitation. For any given pair of RASSCF wave functions, this is used in the way described in reference [[127](<../../references.html#id62> "P.-Å. Malmqvist. Int. J. Quantum Chem., 30 \(1986\) 479-494.")] to allow the pair of orbital sets to be transformed to a biorthonormal pair, while simultaneously transforming the CI expansion coefficients so that the wave functions remain unchanged. The basic principles are the same as in the earlier program [[109](<../../references.html#id30> "P.-Å. Malmqvist, B. O. Roos. Chem. Phys. Lett., 155 \(1989\) 189-194.")], but is adapted to allow RASSCF as well as CASSCF wave functions. It uses internally a Slater determinant expansion. It can now use spin-dependent operators, including the AMFI spin–orbit operator, and can compute matrix elements over spin–orbit states, i.e. the eigenstates of the sum of the spin-free hamiltonian and the spin–orbit operator.

One use of the RASSI eigenstates is to resolve ambiguities due to the imperfect description of highly excited states. Association between individually optimized states and the exact electronic eigenstates is often not clear, when the calculation involves several or many excited states. The reason is that the different states each use a different set of orbitals. The State Interaction calculation gives an unambiguous set of non-interacting and orthonormal eigenstates to the projected Schrödinger equation, and also the overlaps between the original RASSCF wave functions and the eigenstates. The latter is a very efficient diagnostic, since it describes the RASSCF states in terms of one single wave-function basis set.

To make the last point clear, assume the following situation: We have performed three RASSCF calculations, one where we optimize for the lowest state, one for the first excited state, and one for the 2nd excited state in the same symmetry. The active orbitals are fairly much mixed around, so a simple inspection of the CI coefficient is insufficient for comparing the states. Assume that for each state, we have calculated the three lowest CI roots. It can now happen, that the 2nd root of each calculation is a fair approximation to the exact 2nd eigenstate, and the same with the 3rd, or possibly that the order gets interchanged in one or two of the calculation. In that case, a RASSI calculation with these 9 states will give three improved solutions close to the original ones, and of course 6 more that are considered to be the removed garbage. The overlaps will confirm that each of the input states consists mainly of one particular out of the three lowest eigenstates. This situation is the one we usually assume, if no further information is available.

However, it happens that the active orbitals of the three calculations do not span approximately the same space. The orbital optimization procedure has made a qualitatively different selection of correlating orbitals for the three different calculation. Then the RASSI calculation may well come out with 4 lowest roots that overlap strongly with the original RASSCF states. This may change the assignments and may also give valuable information about the importance of some state. The natural orbitals of the eigenstates will show that the active space used in the RASSCF was in some way inappropriate.

Another bothersome situation is also solved by the RASSI method. The analysis of the original states in terms of RASSI eigenstates may reveal that the three optimized RASSCF states consists mainly of TWO low RASSI eigenstates! This is because the RASSCF optimization equations are non-linear and may sometimes offer spurious extra solutions. Two of the calculations are in this case to be regarded qualitatively, as two different (local) solutions that approximate (imperfectly) the same excited state. Also in this case, the natural orbitals will probably offer a clue to how to get rid of the problem. Extra solutions rarely occur for low states in CASSCF calculations, provided a generous active space can be afforded. Problems occur when the active space is too small, and in particular with general RASSCF calculations.

A further application is the preparation of a suitable orbital basis for a subsequent CI calculation. Note that such an application also allows the use of badly converged RASSCF wave functions, or of RASSCF wave functions containing multiple minima solutions close to a common exact eigenstate. In effect, the RASSI program cleans up the situation by removing the errors due to bad convergence (pushing the errors into a garbage part of the spectrum). This requires that the set of input states (9 in this example) provides flexibility enough to remove at least a major part of the error. As one would expect, this is usually true: The erratic non-convergent, or the too slowly convergent, error mode is to a large extent spanned by the few lowest RASSCF wave functions.

Finally, there are situations where there is no problem to obtain adiabatic RASSCF solutions, but where it is still imperative to use RASSI natural orbitals in a subsequent CI. Consider the case of transition metal chemistry, where there is in general two or more electronic states involved. These states are supposed to interact strongly, at least within some range of interatomic distances. Here, an MCSCF solution, such as RASSCF, will have at least two very different solutions, one associated with each configuration of the transition metal atom. Using one set of orbitals, one electronic state has a reasonably described potential energy curve, while other states get pushed far up in energy. Using another set of orbitals, another state gets correctly described. In no calculation with a single orbital set do we obtain the avoided crossings, where one switches from one diabatic state to another. The only way to accomplish this is via a RASSI calculation. In this case, it is probably necessary also to shift the energies of the RASSCF states to ensure that the crossing occur at the correct places. The shifts can be determined by correcting the atomic spectrum in the separated-atoms limit.

Note, however, that most of the problems described above can be solved by performing state-averaged RASSCF calculations.

## 4.2.48.1. Dependencies¶

The RASSI program needs one or more JOBIPH files produced by the RASSCF program. Also, it needs a ONEINT file from SEWARD, with overlap integrals and any one-electron property integrals for the requested matrix elements. If Hamiltonian matrix elements are used, also the ORDINT file is needed.

## 4.2.48.2. Files¶

### 4.2.48.2.1. Input files¶

ORDINT*
    

Ordered two-electron integral file produced by the SEWARD program. In reality, this is up to 10 files in a multi-file system, named ORDINT, ORDINT1,…,ORDINT9. This is necessary on some platforms in order to store large amounts of data.

ONEINT
    

The one-electron integral file from SEWARD

JOBnnn
    

A number of JOBIPH files from different RASSCF jobs. An older naming convention assumes file names JOB001, JOB002, etc. for these files. They are automatically linked to default files named $Project.JobIph, $Project.JobIph01, $Project.JobIph02, etc. in directory $WorkDir, unless they already exist as files or links before the program starts. You can set up such links yourself, or else you can specify file names to use by the keyword IPHNames.

JOBIPHnn
    

A number of JOBIPH files from different RASSCF jobs. The present naming convention assumes file names JOBIPH, JOBIPH01, etc. for such files, when created by subsequent RASSCF runs, unless other names were specified by input. They are automatically linked to default files named $Project.JobIph, $Project.JobIph01, $Project.JobIph02, etc. in directory $WorkDir, unless they already exist as files or links before the program starts. You can set up such links yourself, or else you can specify file names to use by the keyword IPHNames.

### 4.2.48.2.2. Output files¶

SIORBnn
    

A number of files containing natural orbitals, (numbered sequentially as SIORB01, SIORB02, etc.)

BIORBnnmm
    

A number of files containing binatural orbitals for the transition between states `nn` and `mm`. Each such file contains pairs of orbitals, in the same format as the \\(\alpha\\) and \\(\beta\\) components of UHF orbitals. The file for transition to state `nn`=2 from state `mm`=1 will be named BIORB.2_1.

TOFILE
    

This output is only created if TOFIle is given in the input. It will contain the transition density matrix computed by RASSI. Currently, this file is only used as input to QMSTAT.

EIGV
    

Like TOFILE this file is only created if TOFIle is given in the input. It contains auxiliary information that is picked up by QMSTAT.

NTORB and MD_NTO
    

This output is only created if NTOCalc is given in the input. The files will contain natural transition orbitals in INPORB (NTORB) and Molden (MD_NTO) formats.

## 4.2.48.3. Input¶

This section describes the input to the RASSI program in the Molcas program system, with the program name:
    
    
    &RASSI
    

When a keyword is followed by additional mandatory lines of input, this sequence cannot be interrupted by a comment line. The first 4 characters of keywords are decoded. An unidentified keyword makes the program stop.

### 4.2.48.3.1. Keywords¶

CHOInput
    

This marks the start of an input section for modifying the default settings of the Cholesky RASSI. Below follows a description of the associated options. The options may be given in any order, and they are all optional except for ENDChoinput which marks the end of the CHOInput section.

  * NoLK Available only within ChoInput. Deactivates the “Local Exchange” (LK) screening algorithm [[126](<../../references.html#id169> "F. Aquilante, T. B. Pedersen, R. Lindh. J. Chem. Phys., 126 \(2007\) 194106\(1–11\).")] in computing the Fock matrix. The loss of speed compared to the default algorithm can be substantial, especially for electron-rich systems. Default is to use LK.

  * DMPK Available only within ChoInput. Modifies the thresholds used in the LK screening. The keyword takes as argument a (double precision) floating point (non-negative) number used as correction factor for the LK screening thresholds. The default value is 1.0d-1. A smaller value results in a slower but more accurate calculation.

**Note:** the default choice of the LK screening thresholds is tailored to achieve as much as possible an accuracy of the RASSI energies consistent with the choice of the Cholesky decomposition threshold.

  * NODEcomposition Available only within ChoInput. The inactive Exchange contribution to the Fock matrix is computed using inactive canonical orbitals instead of (localized) “Cholesky MOs”. This choice is effective only in combination with the LK screening. Default is to use Cholesky MOs. **Note:** the Cholesky MOs in RASSI are computed by decomposing the density type supermatrix \\(\mat{D}=(\mat{C}_A, \mat{C}_B)(\mat{C}_A, \mat{C}_B)^{\text{T}}\\) where \\(\mat{C}\\) is the corresponding canonical MOs matrix for the state \\(A\\) and \\(B\\).

  * PSEUdo When computing the coupling between 2 different states A and B, only for the first state we use pure Cholesky MOs. The invariance of the Fock matrix is then ensured by rotating the orbitals of B according to the orthogonal matrix defined in A through the Cholesky localization. These orbitals used for B are therefore called “pseudo Cholesky MOs”.

  * TIME Activates printing of the timings of each task of the Fock matrix build. Default is to not show these timings.

  * MEMFraction Set the fraction of memory to use as global Cholesky vector buffer. Default: for serial runs 0.0d0; for parallel runs 0.3d0.



MEIN
    

Demand for printing matrix elements of all selected one-electron properties, over the input RASSCF wave functions.

MEES
    

Demand for printing matrix elements of all selected one-electron properties, over the spin-free eigenstates.

MESO
    

Demand for printing matrix elements of all selected one-electron properties, over the spin–orbit states.

PROPerty
    

Replace the default selection of one-electron operators, for which matrix elements and expectation values are to be calculated, with a user-supplied list of operators.

From the lines following the keyword the selection list is read by the following _FORTRAN_ code:
    
    
    READ({*},{*}) NPROP,(PNAME(I),ICOMP(I),I=1,NPROP)
    

NPROP is the number of selected properties, PNAME(I) is a character string with the label of this operator on SEWARD’s one-electron integral file, and ICOMP(I) is the component number.

The default selection is to use dipole and/or velocity integrals, if these are available in the ONEINT file. This choice is replaced by the user-specified choice if the PROP keyword is used. Note that the character strings are read using list directed input and thus must be within single quotes, see sample input below. For a listing of presently available operators, their labels, and component conventions, see SEWARD program description.

SOCOupling
    

Enter a positive threshold value. Spin–orbit interaction matrix elements over the spin components of the spin-free eigenstates will be printed, unless smaller than this threshold. The value is given in cm\\(^{-1}\\) units. The keyword is ignored unless an SO hamiltonian is actually computed.

SOPRoperty
    

Enter a user-supplied selection of one-electron operators, for which matrix elements and expectation values are to be calculated over the spin–orbit eigenstates. This keyword has no effect unless the SPIN keyword has been used. Format: see PROP keyword.

SPINorbit
    

Spin–orbit interaction matrix elements will be computed. Provided that the ONEL keyword was not used, the resulting Hamiltonian including the spin–orbit coupling, over a basis consisting of all the spin components of wave functions constructed using the spin-free eigenstates, will be diagonalized. NB: For this keyword to have any effect, the SO integrals must have been computed by SEWARD! See AMFI keyword in SEWARD documentation.

ONEL or ONEE
    

The two-electron integral file will not be accessed. No Hamiltonian matrix elements will be calculated, and only matrix elements for the original RASSCF wave functions will be calculated.

J-VAlue
    

For spin–orbit calculations: The output lines with energy for each spin–orbit state will be annotated with the approximate J (= L + S) quantum numbers. J is a well-defined quantum number only for isolated atoms but approximate J-values may be useful also for transition metal complexes, etc.

OMEGa
    

For spin–orbit calculations: The output lines with energy for each spin–orbit state will be annotated with the approximate Omega (projection of J) quantum number. Omega is a well-defined quantum number only for linear molecules but approximate Omega values may be useful also otherwise (similar to J-values).

NROF jobiphs
    

Number of JOBIPH files used as input. This keyword should be followed by the number of states to be read from each JOBIPH. Further, one line per JOBIPH is required with a list of the states to be read from the particular file. See sample input below. Alternatively, the first line can contain the number of JOBIPH used as input followed by the word “`ALL`”, indicating that all states will be taken from each file. In this case no further lines are required. For JOBIPH file names, see the Files section. Note: If this keyword is missing, then by default all files named “JOB001”, “JOB002”, etc. will be used, and all states found on these files will be used.

SUBSets
    

In many cases, RASSI is used to compute the transition moments between a set of initial states (for example the ground state) and a set of final states. This keyword allows to restrict the computation of transition moments between the two sets and not within each set, thus saving time and reducing the output size. This also affects data written to rassi.h5. The keyword is followed by the index where the two sets split (assuming energy ordering). For a calculation between one ground state and several excited states, SUBSets should be 1. Default is to compute the transition moments between all states. SUBS always refers to the index of the relevant non-relativistic state; it is automatically translated to the corresponding SO-coupled state if a SO-RASSI run is performed.

NFINal
    

In cases of spin–orbit coupling and high spin multiplicities (for example in lanthanides), the SUBSets keyword alone may not be enough to reduce the computational effort to an acceptable level. In this case one can use NFINal to specify the maximum number of SO-coupled states considered in the second subset. For example, to compute the luminescence between the first quintet state and the seven lower-lying septet multiplets, use SUBS=7 and NFIN=1.

IPHNames
    

Followed by one entry for each JOBIPH file to be used, with the name of each file. Note: This keyword presumes that the number of JOBIPH files have already been entered using keyword NROF. For default JOBIPH file names, see the Files section. The names will be truncated to 8 characters and converted to uppercase.

SHIFt
    

The next entry or entries gives an energy shift for each wave function, to be added to diagonal elements of the Hamiltonian matrix. This may be necessary e.g. to ensure that an energy crossing occurs where it should. NOTE: The number of states must be known (See keyword NROF) before this input is read. In case the states are not orthonormal, the actual quantity added to the Hamiltonian is `0.5D0*(ESHFT(I)+ESHFT(J))*OVLP(I,J)`. This is necessary to ensure that the shift does not introduce artificial interactions. SHIFT and HDIAG can be used together.

HDIAg
    

The next entry or entries gives an energy for each wave function, to replace the diagonal elements of the Hamiltonian matrix. Non-orthogonality is handled similarly as for the SHIFT keyword. SHIFT and HDIAG can be used together.

NATOrb
    

The next entry gives the number of eigenstates for which natural orbitals will be computed. They will be written, formatted, commented, and followed by natural occupancy numbers, on one file each state. For file names, see the Files section. The format allows their use as standard orbital input files to other Molcas programs.

SONOrb
    

This computes the spin–orbit natural orbitals (SO-NOs) for spin–orbit coupled states. performs the transition dipole moment (TDM) partitioning study based on the obtained SO-NTOs. It takes an integer number specifying the number of requested SO-NOs, followed by the same number of integers specifying the spin–orbit (SO) coupled states.

BINAtorb
    

The next entry gives the number of transitions for which binatural orbitals will be computed. Then a line should follow for each transition, with the two states involved. The orbitals and singular values provide a singular value decomposition of a transition density matrix [[128](<../../references.html#id340> "P. Å. Malmqvist, V. Veryazov. Mol. Phys., 110\[19-20\] \(2012\) 2455–2464.")]. The bra and ket orbitals are written followed by the singular values in the usual UHF format used by other Molcas programs.

ORBItals
    

Print out the Molecular Orbitals read from each JOBIPH file.

OVERlaps
    

Print out the overlap integrals between the various orbital sets.

CIPRint
    

Print out the CI coefficients read from JOBIPH.

THRS
    

The next line gives the threshold for printing CI coefficients. The default value is 0.05.

CIH5
    

Add CI coefficients and occupation vectors in Slater determinant basis as well as molecular orbitals (both original and biorthonormally transformed) to the HDF5 file. If coupled with CIPRint and ORBItals keywords print them also to output file. Needed for the interface to SCAMPI program. Note that it can be enabled only if no more than two JOBIPH files are computed at a time.

DIPRint
    

The next entry gives the threshold for printing dipole intensities. Default is 1.0D-5.

QIPRint
    

The next entry gives the threshold for printing quadrupole intensities. Default is 1.0D-5. Will overwrite any value chosen for dipole intensities.

RSPR
    

The next entry gives the threshold for printing reduced rotatory strength intensities. Default is 1.0D-7.

QIALl
    

Print all quadrupole intensities.

CD
    

Compute rotatory strengths (for circular dichroism) from the multipole expansion of transition moments.

TINTensities
    

Activate the computation of transition intensities (oscillator strengths and rotatory strengths) using the non-relativistic Hamiltonian with the explicit Coulomb-field vector operator (\\(A\\)) in the weak field approximation.

TIGRoup
    

Group the states close in energy for the purpose of computing transition intensities wi the exponential operator (TINTensities keyword). A single wave vector will be used for all transitions to the states in the group. This is a good approximation when the energy difference between the states in a group is negligible with respect to the energy of the transition. The keyword reads a real value, that is the maximum relative difference for transitions in a group with respect to the average energy. This keyword requires the use of SUBSets and TINTensities.

IIORder
    

Set the order of the Lebedev grids used in the isotropic integration of transition intensities in association with the TINT option. Default value is 5, the maximum is 131.

PRRAw
    

Print the raw directions for the exact semi-classical intensities (see the TINT keyword).

PRWEighted
    

Print the weighted directions for the exact semi-classical intensities (see the TINT keyword).

DIREction
    

Define the direction of the incident light for which we will compute transition moments and oscillator strengths. The keyword is followed by an integer \\(n\\), the number of directions, and then \\(n\\) lines with three real numbers each specifying the direction. The values do not need to be normalized.

POLArization
    

Define the direction of the polarization of the incident light, see DIREction. The keyword is followed by three real numbers specifying the components of a vector (not necessarily normalized), the polarizarion direction is defined by orthogonalizing this vector with each vector specified in DIREction. Currently, this keyword only works with the oscillator strengths computed with the TINTensities keyword.

RFPErt
    

RASSI will read from RUNOLD (if not present defaults to RUNFILE) a response field contribution and add it to the Fock matrix.

HCOM
    

The spin-free Hamiltonian is computed.

HEXT
    

The spin-free Hamiltonian is read from the input instead of being computed. It is read from the following few lines, as a triangular matrix: One element of the first row, two from the next, etc., as list-directed input of reals.

HEFF
    

A spin-free effective Hamiltonian is read from JOBIPH instead of being computed. It must have been computed by an earlier program. Presently, this is done by a multi-state calculation using CASPT2. In the future, other programs may add dynamic correlation estimates in a similar way. This keyword is not needed if the input file is in HDF5 format. Note that using HEFF or EJOB can significantly speed up the RASSI job by avoiding the explicit computation of the Hamiltonian.

EJOB
    

The spin-free effective Hamiltonian’s diagonal is filled with energies read from a JOBIPH or JOBMIX file. If an effective Hamiltonian is read (using HEFF or reading from an HDF5 file), the diagonal elements are taken from the stored Hamiltonian; this can be useful for using the SS-CASPT2 energies from a MS-CASTP2 calculation. The off-diagonal elements are approximated as \\(H_{ij} \approx \frac{1}{2} S_{ij}(H_{ii}+H_{ij})\\), where \\(S_{ij}\\) is the overlap between two states; so if the input states are orthogonal, the effective Hamiltonian will be diagonal. Note that using HEFF or EJOB can significantly speed up the RASSI job by avoiding the explicit computation of the Hamiltonian.

TOFIle
    

Signals that a set of files with data from RASSI should be created. This keyword is necessary if QMSTAT is to be run afterwards.

XVIN
    

Demand for printing expectation values of all selected one-electron properties, for the input RASSCF wave functions.

XVES
    

Demand for printing expectation values of all selected one-electron properties, for the spin-free eigenstates.

XVSO
    

Demand for printing expectation values of all selected one-electron properties, for the spin–orbit states.

EPRG
    

This computes the g matrix and principal g values for the states lying within the energy range supplied on the next line. A value of 0.0D0 or negative will select only the ground state, a value E will select all states within energy E of the ground state. The states should be ordered by increasing energy in the input. The angular momentum and spin–orbit coupling matrix elements need to be available (use keywords SPIN and PROP). For a more detailed description see ref [[129](<../../references.html#id19> "S. Vancoillie, P.-Å. Malmqvist, K. Pierloot. ChemPhysChem, 8 \(2007\) 1803-1815.")].

MAGN
    

This computes the magnetic moment and magnetic susceptibility. On the next two lines you have to provide the magnetic field and temperature data. On the first line put the number of magnetic field steps, the starting field (in tesla), size of the steps (in tesla), and an angular resolution for sampling points in case of powder magnetization (for a value of 0.0d0 the powder magnetization is deactivated). The second line reads the number of temperature steps, the starting temperature (K), and the size of the temperature steps (K). The angular momentum and spin–orbit coupling matrix elements need to be available (use keywords SPIN and PROP). For a more detailed description see ref [[130](<../../references.html#id215> "S. Vancoillie, L. Rulíšek, F. Neese, K. Pierloot. J. Phys. Chem. A, 113 \(2009\) 6149-6157.")].

HOP
    

Enables a trajectory surface hopping (TSH) algorithm which allow non-adiabatic transitions between electronic states during molecular dynamics simulation with DYNAMIX program. The algorithm computes the scalar product of the amplitudes of different states in two consecutive steps. If the scalar product deviates from the given threshold a transition between the states is invoked by changing the root for the gradient computation. The current implementation is working only with SA-CASSCF.

STOVerlaps
    

Computes only the overlaps between the input states.

TRACk
    

Tries to follow a particular root during an optimization. Needs two JOBIPH files (see NrOfJobIphs) with the same number of roots. The first file corresponds to the current iteration, the second file is the one from the previous iteration (taken as a reference). With this keyword RASSI selects the root from the first JOBIPH with highest overlap with the root that was selected in the previous iteration. It also needs MDRlxRoot, rather than RlxRoot, to be specified in RASSCF. No other calculations are done by RASSI when Track is specified.

DQVD
    

Perfoms DQΦ diabatization [[131](<../../references.html#id299> "C. E. Hoyer, X. Xu, D. Ma, L. Gagliardi, D. G. Truhlar. J. Chem. Phys., 141\[11\] \(2014\) 114104.")] by using properties that are computed with RASSI. Seven properties must be computed with RASSI in order for this keyword to work (\\(x\\), \\(y\\), \\(z\\), \\(xx\\), \\(yy\\), \\(zz\\), \\(1/r\\)), they will be automatically selected with the default input if the corresponding integrals are available (see keywords MULT and EPOT in GATEWAY). At present, this keyword also requires ALPHa and BETA, where ALPHa is the parameter in front of \\(rr\\) and BETA is the parameter in front of \\(1/r\\). When ALPHa and BETA are equal to zero, this method reduces to Boys localized diabatization [[132](<../../references.html#id300> "J. E. Subotnik, S. Yeganeh, R. J. Cave, M. A. Ratner. J. Chem. Phys., 129\[24\] \(2008\) 244101.")]. At present, this method only works for one choice of origin for each quantity.

ALPHa
    

ALPHa is the prefactor for the quadrupole term in DQΦ diabatization. This keyword must be used in conjunction with DQVD and BETA. You must specify a real number (e.g. \\(\alpha = 1.0\\) not \\(\alpha = 1\\)).

BETA
    

BETA is the prefactor for the electrostatic potential term in DQΦ diabatization. This keyword must be used in conjunction with DQVD and ALPHa. You must specify a real number (e.g. \\(\beta = 1.0\\) not \\(\beta = 1\\)).

TRDI
    

Prints out the components and the module of the transition dipole vector. Only vectors with sizes large than 1.0D-4 a.u. are printed. See also the TDMN keyword.

TRDC
    

Prints out COMPLEX valued components of the transition dipole vector for spin–orbit calculations, otherwise functionally equivalent to TRDI and TDMN.

TDMN
    

Prints out the components and the module of the transition dipole vector. On the next line, the minimum size, in a.u., for the dipole vector to be printed must be given.

TRD1
    

Prints the 1-electron (transition) densities to ASCII files and to the HDF5 file rassi.h5.

TRD2
    

Prints the 1-/2-electron (transition) densities to ASCII files.

TDM
    

If this keyword is given, and if HDF5 support is enabled, the 1-electron transition (spin) density matrix between every pair of states in the current calculation will be computed and stored in the HDF5 file (use SUBSets to restrict to a subset of states). Use this to prepare WFA runs or visualisation with Pegamoid.

DYSOn
    

Enables calculation of Dyson amplitudes (an approximation of photo-electron intensities) between states that differ by exactly one in their number of electrons. Dyson amplitudes are correctly obtained from a biorthonormally transformed orbital sets as described in [[133](<../../references.html#id356> "B. N. C. Tenorio, A. Ponzi, S. Coriani, P. Decleva. Molecules, 27 \(2022\) 1203.")].

Calculations are performed for spin-free states, and for spin–orbit coupled states if the keyword SPINorbit has also been specified. Note that spin–orbit coupled amplitudes are per default obtained from an approximation where a transformation is applied directly to the spin-free amplitudes rather than the Dyson orbitals, which may severly impact the accuracy. For a complete calculation also for spin–orbit states see the DYSExport keyword.

DYSExport
    

Requires the DYSOn keyword and enables exportation of Dyson orbitals (from which Dyson amplitudes are obtained). The next line specifies the number (starting from the first) of spin-free and spin–orbit states (two numbers, both mandatory) for which the exportation will be done. Note that the ordering of spin-free states depends on the ordering of JOBfiles, whereas spin–orbit states are always energy ordered.

Dyson amplitudes for the spin–orbit states are here correctly obtained from a transformation of the Dyson orbitals (as opposed to the amplitudes, see DYSOn keyword), but only for the specified number of initial states. Note that this calculation may be time consuming, i.e. the number of initial states should be limited.

DCHS
    

Computes spectral intensity of double-core hole states similar to Dyson norm (see [[134](<../../references.html#id358> "B. N. C. Tenorio, P. Decleva, S. Coriani. J. Chem. Phys., 155\[13\] \(2021\) 131101.")]). Double core hole wave functions are generated with the DEXS keyword on RASSCF input (See DEXS keyword). The next line specifies the orbital number of the double-core hole (normally it is 1, that is, the first active orbital).

TDYSOn
    

Prints Auger density matrices to ASCII files (see [[135](<../../references.html#id357> "B. N. C. Tenorio, T. A. Voss, S. I. Bokarev, P. Decleva, S. Coriani. J. Chem. Theory Comput., 18\[7\] \(2022\) 4387-4407.")]). Required to run AUGER-OCA program found in the Tools/ folder. Requires the DYSOn keyword. It starts by an integer number specifying the number of scattering centers, followed by the same number of lines. Each line contains strings with the type of Auger scattering centers. An example for the computation of Auger matrix elements of carbon K-edge is “TDYS = 1; C 1s”.

RHODyn
    

Required to run RHODYN program. Enable saving pure spin–orbit coupling Hamiltonian and SO Dyson amplitudes (not squared!) to HDF5 file of RASSI. Keywords SPINorbit, MESO, XVES, XVSO, DYSOn are required to print corresponding properties.

NTOCalc
    

Enables natural transition orbital (NTO) calculation of two states from two JobIph files (which can be identical to each other). The NTO calculations can be performed for states with different spatial symmetries. To perform an NTO calculation, two JobIph files, which by convention are named JOB001 and JOB002, are needed. Since NTO calculations are performed usually between the ground state and an excited state, JOB001 is used to provide the information for the ground state, and JOB002 is used to provide the information for excited states. This way of storing information was chosen so that NTO calculations can be performed either for states with the same symmetry or states with different symmetries, but in the former case, if two states are obtained in a single SA-CASSCF or SA-RASSCF calculation, one may make a copy of the JobIph file to get the second JobIph file. The two states are specified in the keyword NROF to tell the program for which two states the NTO calculation is to be performed. The NTO files are named as $Project.NTOrb.SF.I_J.Spin.NTOType, which has the same format as .ScfOrb or .RasOrb, where Spin is a for alpha NTOs and b for beta NTOs, and where I and J are the RASSI states between which the NTOs are calculated, and where NTOType is PART for particle NTOs and HOLE for hole NTOs. In addition, Molden files for the orbitals named $Project.nto.molden.SF.I_J.Spin.NTOType are also generated. One may search for `Nr of states` in the RASSI part of the output and the three lines after this information tell the correspondence of the RASSI states (in the line starting with `State:`) with the actual states (in the line starting with `Root nr:`) in each JobIph file (in the line starting with `JobIph:`). If the states for which the NTO calculation is performed are singlets, only the alpha NTOs are printed out. For more information and examples of this method, please refer to the Minnesota OpenMolcas webpage1.

1
    

<https://comp.chem.umn.edu/openmolcas/>

SONT
    

This computes the spin–orbit natural transition orbitals (SO-NTOs) for two spin–orbit coupled states, and it also performs the transition dipole moment (TDM) partitioning study based on the obtained SO-NTOs. It starts by an integer number specifying the number of requested SO-NTO pairs, followed by the same number of lines. Each line contains two integers for the two spin–orbit (SO) coupled states. An input example has been shown below.

ARGU
    

This minimizes the imaginary component of the calculated SO-NTOs. The keyword SONT is needed.

EPRA
    

This computes the hyperfine tensor matrix and the principal magnetic axes values for the ground spin–orbit state. The hyperfine and spin–orbit coupling matrix elements are required upon calculation (use keywords SPIN and PROP). For the hyperfine matrix elements, either the spin-dependent (ASD) or the paramagnetic spin orbital (PSOP) part is needed, while in most cases both are recommended for the same atom. See reference for details [[84](<../../references.html#id352> "R. Feng, T. J. Duignan, J. Autschbach. J. Chem. Theory Comput., 17 \(2021\) 255-268.")].

AFCC
    

This computes the Fermi contact contribution of the total hyperfine coupling matrix. The keyword EPRA is needed. The spin-dependent (ASD) part of the hyperfine matrix elements is needed.

ASDC
    

This computes the spin-dipolar contribution of the total hyperfine coupling matrix. The keyword EPRA is needed. The spin-dependent (ASD) part of the hyperfine matrix elements is needed.

FCSD
    

This computes the spin-dependent contribution of the total hyperfine coupling matrix. The keyword EPRA is needed. The spin-dependent (ASD) part of the hyperfine matrix elements is needed.

APSO
    

This computes the paramagnetic spin orbital contribution of the total hyperfine coupling matrix. The keyword :EPRA is needed. The paramagnetic spin orbital (PSOP) part of the hyperfine matrix elements is needed.

ATSA
    

This keyword activates the pseudospin approach to compute the same hyperfine constants as EPRA. For Kramers pair ground states this keyword is optional, otherwise (non-Kramers pair ground state) it is needed. See reference for details [[84](<../../references.html#id352> "R. Feng, T. J. Duignan, J. Autschbach. J. Chem. Theory Comput., 17 \(2021\) 255-268.")]. The keyword EPRA is needed.

MONA
    

This keyword indicates that the properties of monomer A were calculated in the respective RASSI section of the Frenkel exciton protocol. This is important for the creation of the TDMs in the common basis of the two monomers. The geometry of monomer A must always be in the first place in the BSSE section.

MONB
    

This keyword indicates that the properties of monomer B were calculated in the respective RASSI section of the Frenkel exciton protocol. This is important for the creation of the TDMs in the common basis of the two monomers. The geometry of monomer B must always be in the second place in the BSSE section.

EXCItonics
    

This keyword initiates the calculation of the Frenkel exciton coupling elements between two monomers, the excitonic eigenvectors, eigenenergies and the absorption spectrum. Has to be put in the second RASSI section of the Frenkel exciton protocol.

EXAList
    

Number of initial states of monomer A in the Frenkel exciton calculation, followed by the list of these states in the next line. This keyword requires a proper use of the Frenkel exciton protocol and should be called in the second of the two RASSI sections.

EXBList
    

Number of initial states of monomer B in the Frenkel exciton calculation, followed by the list of these states in the next line. This keyword requires a proper use of the Frenkel exciton protocol and should be called in the second of the two RASSI sections.

### 4.2.48.3.2. Input example¶
    
    
    >>COPY  "Jobiph file 1" JOB001
    >>COPY  "Jobiph file 2" JOB002
    >>COPY  "Jobiph file 3" JOB003
    
    &RASSI
    NR OF JOBIPHS= 3 4 2 2    --- 3 JOBIPHs. Nr of states from each.
    1 2 3 4; 3 4; 3 4         --- Which roots from each JOBIPH.
    CIPR; THRS= 0.02
    Properties= 4; 'MltPl  1'  1   'MltPl  1'  3    'Velocity'  1 'Velocity'  3
    * This input will compute eigenstates in the space
    * spanned by the 8 input functions. Assume only the first
    * 4 are of interest, and we want natural orbitals out
    NATO= 4
    

An NTO input example using the JobIph file from a state-averaged calculation is as follows:
    
    
    >>COPY  "Jobiph file 1" JOB001
    >>COPY  "Jobiph file 2" JOB002
    
    &RASSI
    NTOC
    Nr of JobIphs=2 1 1
    1; 2
    *This NTO calculation is performed for the ground state and the first
    *excited state of the previous calculation done in the &RASSCF module.
    

An SO-NTO input example from three singlets and two triplets:
    
    
    >>COPY  $Project.JobIph.s0s1s2 JOB001
    >>COPY  $Project.JobIph.t1t2 JOB002
    
    &RASSI
    Nr of JobIphs
    2 3 2
    1 2 3
    1 2
    SPINorbit
    ARGU *This minimizes the imaginary component of SO-NTOs
    SONT
    3
    1 2
    1 3
    2 3
    *Three pairs of SO-NTOs are requested, between SO state 1 and 2,
    *SO state 1 and 3, and SO state 2 and 3.
    *Note that the states are SO coupled states.
    

An illustrative hyperfine calculation input for a diatomic molecule:
    
    
    >>COPY "Jobiph file 1" JOB001
    
    &RASSI
    Nr of JobIphs
    1 4
    1 2 3 4
    SPIN
    EPRA
    AFCC
    ASDC
    FCSD
    APSO
    ATSA
    PROPerties
    18
    'ASD    1' 1
    'ASD    1' 2
    'ASD    1' 3
    'ASD    1' 4
    'ASD    1' 5
    'ASD    1' 6
    'ASD    2' 1
    'ASD    2' 2
    'ASD    2' 3
    'ASD    2' 4
    'ASD    2' 5
    'ASD    2' 6
    'PSOP   1' 1
    'PSOP   1' 2
    'PSOP   1' 3
    'PSOP   2' 1
    'PSOP   2' 2
    'PSOP   2' 3
    * Note that the strings following PROP have to be of sizes of 8, each
    * followed by an integer number for the property component.
    * The last digit of the string is the atom number.
    * Note that there are 6 ASD and 3 PSOP components for each atom, respectively.
    * One has to include all 6 of ASD components to obtain principle
    * spin-dependent hyperfine contributions, and one has to include all 3 of PSOP
    * components to obtain principle paramagnetic spin orbital contributions.
    

It is also possible to calculate only the non-relativistic part of the spin–dependent hyperfine contributions:
    
    
    &RASSI
    Nr of JobIphs
    1 4
    1 2 3 4
    SPIN
    EPRA
    AFCC
    ASDC
    FCSD
    APSO
    ATSA
    PROPerties
    12
    'ASDO   1' 1
    'ASDO   1' 2
    'ASDO   1' 3
    'ASDO   1' 4
    'ASDO   1' 5
    'ASDO   1' 6
    'ASDO   2' 1
    'ASDO   2' 2
    'ASDO   2' 3
    'ASDO   2' 4
    'ASDO   2' 5
    'ASDO   2' 6
    * Note that 'ASD' is now 'ASDO' for the non-relativistic integrals.
    

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
      * 4.2.48. RASSI
      * [4.2.49. RHODYN](<rhodyn.html>)
      * [4.2.50. RPA](<rpa.html>)
      * [4.2.51. SCF](<scf.html>)
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

[previous](<rasscf.html> "4.2.47. RASSCF") | [next](<rhodyn.html> "4.2.49. RHODYN") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/rassi.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
