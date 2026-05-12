<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/localisation.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.29. LOCALISATION

[previous](<level.html> "4.2.28. LEVEL") | [next](<loprop.html> "4.2.30. LOPROP") | [index](<../../genindex.html> "General Index")

# 4.2.29. LOCALISATION¶

  * Description

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Optional general keywords

    * Limitations

    * Input examples




## 4.2.29.1. Description¶

The LOCALISATION program of the Molcas program system generates localised occupied orbitals according to one of the following procedures: Pipek–Mezey [[95](<../../references.html#id118> "J. Pipek, P. G. Mezey. J. Chem. Phys., 90 \(1989\) 4916-4926.")], Boys [[96](<../../references.html#id242> "S. F. Boys. Rev. Mod. Phys., 32 \(1960\) 296-299."), [97](<../../references.html#id243> "J. M. Foster, S. F. Boys. Rev. Mod. Phys., 32 \(1960\) 300-302.")], Edmiston–Ruedenberg [[98](<../../references.html#id244> "C. Edmiston, K. Ruedenberg. Rev. Mod. Phys., 35 \(1963\) 457-465.")], or Cholesky [[99](<../../references.html#id164> "F. Aquilante, T. B. Pedersen, A. Sánchez de Merás, H. Koch. J. Chem. Phys., 125 \(2006\) 174101\(1–7\).")]. Orthonormal, linearly independent, local orbitals may also be generated from projected atomic orbitals (Cholesky PAOs) [[99](<../../references.html#id164> "F. Aquilante, T. B. Pedersen, A. Sánchez de Merás, H. Koch. J. Chem. Phys., 125 \(2006\) 174101\(1–7\).")].

Orbital localisation makes use of the fact that a Hartree-Fock wave function is invariant under unitary transformations of the occupied orbitals,

\\[\tilde{C}_{\mu i} = \sum_j C_{\mu j} \mat{U}_{ji} ,\\]

where \\(\mat{U}\\) is unitary (i.e. orthogonal for real orbitals). The same is true for the inactive or active orbitals in a CASSCF wave function. Whereas the Pipek–Mezey [[95](<../../references.html#id118> "J. Pipek, P. G. Mezey. J. Chem. Phys., 90 \(1989\) 4916-4926.")], Boys [[96](<../../references.html#id242> "S. F. Boys. Rev. Mod. Phys., 32 \(1960\) 296-299."), [97](<../../references.html#id243> "J. M. Foster, S. F. Boys. Rev. Mod. Phys., 32 \(1960\) 300-302.")], and Edmiston–Ruedenberg [[98](<../../references.html#id244> "C. Edmiston, K. Ruedenberg. Rev. Mod. Phys., 35 \(1963\) 457-465.")] procedures define \\(\mat{U}\\) through an iterative maximisation of a localisation functional, the Cholesky orbitals are simply defined through the Cholesky decomposition of the one-electron density, i.e.

\\[\sum_i \tilde{C}_{\mu i}\tilde{C}_{\nu i} = P_{\mu\nu} = \sum_i C_{\mu i} C_{\mu i} .\\]

Cholesky orbitals are thus not optimum localised orbitals by any of the Pipek–Mezey, Boys, or Edmiston–Ruedenberg measures, but rather inherit locality from the density matrix, see [[99](<../../references.html#id164> "F. Aquilante, T. B. Pedersen, A. Sánchez de Merás, H. Koch. J. Chem. Phys., 125 \(2006\) 174101\(1–7\).")] for details.

Although these localisation schemes are mostly meant for localising occupied orbitals (except for PAOs which are defined for the virtual orbitals), the LOCALISATION program will attempt to localise any set of orbitals that the user specifies. This means that it is possible to mix occupied and virtual orbitals and thereby break the Hartree–Fock invariance. The default settings, however, do not break the invariance.

For Pipek–Mezey, Boys, and Edmiston–Ruedenberg localisations, iterative optimisations are carried out. We use the \\(\eta\\)-steps of Subotnik _et al._ [[100](<../../references.html#id157> "J. E. Subotnik, Y. Shao, W. Liang, M. Head-Gordon. J. Chem. Phys., 121 \(2004\) 9220-9229.")] for Edmiston–Ruedenberg, whereas the traditional Jacobi sweeps (consecutive two-by-two orbital rotations) [[95](<../../references.html#id118> "J. Pipek, P. G. Mezey. J. Chem. Phys., 90 \(1989\) 4916-4926."), [100](<../../references.html#id157> "J. E. Subotnik, Y. Shao, W. Liang, M. Head-Gordon. J. Chem. Phys., 121 \(2004\) 9220-9229.")] are employed for the Pipek–Mezey and Boys schemes.

## 4.2.29.2. Dependencies¶

The LOCALISATION program requires the one-electron integral file ONEINT and the communications file RUNFILE, which contains, among other data, the basis set specifications processed by GATEWAY and SEWARD. In addition, the Edmiston–Ruedenberg procedure requires the presence of Cholesky decomposed two-electron integrals produced by SEWARD.

## 4.2.29.3. Files¶

Below is a list of the files that are used/created by the program LOCALISATION.

### 4.2.29.3.1. Input files¶

LOCALISATION will use the following input files: ONEINT, RUNFILE, INPORB. For Edmiston–Ruedenberg localisation, it also needs CHVEC, CHRED and CHORST files (for more information see [Section 4.1.1.2](<../env-overview.html#ug-sec-files-list>)).

### 4.2.29.3.2. Output files¶

LOCORB
    

Localised orthonormal orbital output file. Note that LOCORB contains all orbitals (localised as well as non-localised according to the input specification).

DPAORB
    

Linearly dependent nonorthonormal projected atomic orbital output file (only produced for PAO runs).

IPAORB
    

Linearly independent nonorthonormal projected atomic orbital output file (only produced for PAO runs).

RUNFILE
    

Communication file for subsequent programs.

MD_LOC
    

Molden input file for molecular orbital analysis.

## 4.2.29.4. Input¶

Below follows a description of the input to LOCALISATION. The LOCALISATION program section of the Molcas input is bracketed by a preceding program reference
    
    
    &LOCALISATION
    

### 4.2.29.4.1. Optional general keywords¶

FILEorb
    

The next line specifies the filename containing the input orbitals that will be localised. By default a file named INPORB will be used.

NORBitals
    

The following line specifies the number of orbitals to localise in each irreducible representation. The default is to localise all occupied orbitals as specified in the INPORB input file, except for PAO runs where all the virtual orbitals are treated by default.

NFROzen
    

The following line specifies the number of orbitals to freeze in each irreducible representation. The default is not to freeze any orbitals, except for the localisations of the virtual space (see keywords PAO and VIRTual) where the default is to freeze all occupied orbitals (occupation number different from zero, as reported in the INPORB file).

FREEze
    

Implicit frozen core option. The default is not to freeze any orbitals, except for the localisations of the virtual space (see keywords PAO and VIRTual) where the default is to freeze all occupied orbitals (occupation number different from zero, as reported in the INPORB file). The definition of core orbitals is taken from program SEWARD.

OCCUpied
    

Requests that the occupied orbitals should be localised. This is the default except for PAO where the default is virtual.

VIRTual
    

Requests that the virtual orbitals should be localised. The default is to localise the occupied orbitals, except for PAO where the default is virtual.

ALL
    

Requests that all orbitals should be localised. The default is to localise the occupied orbitals, except for PAO where the default is virtual.

PIPEk-Mezey
    

Requests Pipek–Mezey localisation. This is the default.

BOYS
    

Requests Boys localisation. The default is Pipek–Mezey.

EDMIston-Ruedenberg
    

Requests Edmiston–Ruedenberg localisation. The default is Pipek–Mezey. Note that this option requires that the Cholesky (or RI/DF) representation of the two-electron integrals has been produced by SEWARD.

CHOLesky
    

Requests Cholesky localisation (non-iterative). The default is Pipek–Mezey. This and PAO are the only options that can handle point group symmetry. The decomposition threshold is by default 1.0d-8 but may be changed through the THREshold keyword.

PAO
    

Requests PAO localisation (non-iterative) using Cholesky decomposition to remove linear dependence. The default is Pipek–Mezey. This and Cholesky are the only options that can handle point group symmetry. The decomposition threshold is by default 1.0d-8 but may be changed through the THREshold keyword.

SKIP
    

Leaves the input orbitals unchanged. It is turned off by default.

ITERations
    

The following line specifies the maximum number of iterations to be used by the iterative localisation procedures. The default is 300.

THREshold
    

The following line specifies the convergence threshold used for changes in the localisation functional. The default is 1.0d-6. For Cholesky and PAO methods, it is the decomposition threshold and the default is 1.0d-8.

THRGradient
    

The following line specifies the convergence threshold used for the gradient of the localisation functional. The default is 1.0d-2.

THRRotations
    

The following line specifies the screening threshold used in the Jacobi sweep optimisation algorithm. The default is 1.0d-10.

CHOStart
    

Requests that iterative localisation procedures use Cholesky orbitals as initial orbitals. The default is to use the orbitals from INPORB directly.

ORDEr
    

Requests that the localised orbitals are ordered in the same way as the Cholesky orbitals would be. This is mainly useful when comparing orbitals from different localisation schemes. The ordering is done according to maximum overlap with the Cholesky orbitals. The default is not to order.

DOMAin
    

Requests orbital domains and pair domains are set up and analysed. The default is not to set up domains.

THRDomain
    

The following line specifies two thresholds to be used in defining orbital domains. The first is the Mulliken population threshold such that atoms are included in the domain until the population (divided by 2) is larger than this number (default: 9.0d-1). The second threshold is used for the Pulay completeness check of the domain (default: 2.0d-2).

THRPairdomain
    

The following line specifies three thresholds to be used for classifying pair domains: R1, R2, and R3. (Defaults: 1.0d-10, 1.0d1, and 1.5d1.) If R is the smallest distance between two atoms in the pair domain (union of the individual orbital domains), then pair domains are classified according to: R \\(\leq\\) R1: strong pair, R1 \\(<\\) R \\(\leq\\) R2: weak pair, R2 \\(<\\) R \\(\leq\\) R3: distant pair, and R3 \\(<\\) R: very distant pair.

LOCNatural orbitals
    

This keyword is used to select atoms for defining the localised natural orbitals (LNOs), thus a set of localised orbitals with well-defined occupation numbers. All other options specified in the LOCALISATION program input apply (e.g., input orbitals, localisation method, etc.). On the next line give the number of atoms that identify the region of interest and the threshold used to select the localised orbitals belonging to this region (recommended values > 0.2 and < 1). An additional line gives the names of the (symmetry unique) atoms as defined in the SEWARD input. The keyword LOCN is used to define suitable occupation numbers for RASSCF active orbitals that have been localised. It has proven useful in Effective Bond Order (EBO) analysis. Here is a sample input for a complex containing an iron-iron multiple bond.
    
    
    LOCN
    2  0.3
    Fe1  Fe2
    

In this example, the (localised) orbitals constructed by the LOCALISATION program are subdivided in two groups: those having less than 0.3 total Mulliken population on the two iron atoms, and the remaining orbitals, obviously localised on the iron-iron region. The resulting density matrices for the two subsets of orbitals are then diagonalised separately and the corresponding (localised) natural orbitals written to LOCORB with the proper occupation numbers. Note that the two sets of LNOs are mutually non-orthogonal.

LOCCanonical orbitals
    

This keyword is used to select atoms for defining the localised canonical orbitals (LCOs), thus a set of localised orbitals with well-defined orbital energies (eigenvalues of a local Fock matrix). Please, refer to the analogous keyword LOCN in this manual for more details and input examples.

### 4.2.29.4.2. Limitations¶

The limitations on the number of basis functions are the same as specified for SEWARD.

### 4.2.29.4.3. Input examples¶

This input is an example of the Boys localisation of the CO molecule. Note that no symmetry should not be used in any calculation of localised orbitals except for Cholesky and PAO orbitals.
    
    
    &GATEWAY
    Coord = $MOLCAS/Coord/CO.xyz
    Basis = STO-3G
    Group = C1
    
    &SEWARD ; &SCF
    
    &LOCALISATION
    Boys
    

This input is an example of the Projected Atomic Orbital localisation of the virtual orbitals of the CO molecule. The threshold for the Cholesky decomposition that removes linear dependence is set to 1.0d-14.
    
    
    &GATEWAY
    Coord = $MOLCAS/Coord/CO.xyz
    Basis = STO-3G
    Group = C1
    
    &SEWARD ; &SCF
    
    &LOCALISATION
    PAO
    Threshold = 1.0d-14
    

This input is an example of the Cholesky localisation (using default 1.0d-8 as threshold for the decomposition) of the valence occupied orbitals of the CO molecule. Orbital domains are set up and analysed.
    
    
    &GATEWAY
    Coord = $MOLCAS/Coord/CO.xyz
    Basis = STO-3G
    Group = C1
    
    &SEWARD ; &SCF
    
    &LOCALISATION
    Cholesky
    Freeze
    Domain
    

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
      * 4.2.29. LOCALISATION
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

[previous](<level.html> "4.2.28. LEVEL") | [next](<loprop.html> "4.2.30. LOPROP") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/localisation.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
