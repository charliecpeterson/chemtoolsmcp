<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/wfa.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.58. WFA

[previous](<vibrot.html> "4.2.57. VIBROT") | [next](<../basis_library.html> "4.2.59. The Basis Set Libraries") | [index](<../../genindex.html> "General Index")

# 4.2.58. WFA¶

Warning

This program requires a submodule.

  * Installation

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Keywords

    * Input example

    * Large jobs

  * Output

    * State/difference density matrix analysis (SCF/RASSCF/RASSI)

    * Transition density matrix analysis (RASSI)




The WFA program of the Molcas program system provides various visual and quantitative wavefunction analysis methods. It is based on the libwfa [[223](<../../references.html#id319> "F. Plasser, M. Wormit, S. A. Mewes, B. Thomitzni, A. Dreuw. libwfa: Wave-function analysis tool library for quantum chemical applications."), [224](<../../references.html#id320> "F. Plasser, A. I. Krylov, A. Dreuw. Wiley Interdiscip. Rev.: Comput. Mol. Sci., 12 \(2022\) e1595.")] wavefunction analysis library. The interface to Molcas is described in Ref. [[225](<../../references.html#id321> "F. Plasser, S. A. Mewes, A. Dreuw, L. González. J. Chem. Theory Comput., 13\[11\] \(2017\) 5343-5353.")].

The program computes natural transition orbitals (NTO) [[226](<../../references.html#id311> "R. L. Martin. J. Chem. Phys., 118\[11\] \(2003\) 4775-4777."), [227](<../../references.html#id309> "F. Plasser, M. Wormit, A. Dreuw. J. Chem. Phys., 141\[2\] \(2014\) 024106.")], which provide a compact description of one-electron excited states. Natural difference orbitals (NDO) [[227](<../../references.html#id309> "F. Plasser, M. Wormit, A. Dreuw. J. Chem. Phys., 141\[2\] \(2014\) 024106.")] can be computed to visualize many-body effects and orbital relaxation effects [[228](<../../references.html#id310> "F. Plasser, S. A. Bäppler, M. Wormit, A. Dreuw. J. Chem. Phys., 141\[2\] \(2014\) 024107.")]. A module for the statistical analysis of exciton wavefunctions is included [[229](<../../references.html#id313> "S. A. Bäppler, F. Plasser, M. Wormit, A. Dreuw. Phys. Rev. A, 90\[5\] \(2014\) 052521."), [230](<../../references.html#id312> "F. Plasser, B. Thomitzni, S. A. Bäppler, J. Wenzel, D. R. Rehn, M. Wormit, A. Dreuw. J. Comput. Chem., 36\[21\] \(2015\) 1609-1620.")], which provides various quantitative descriptors to describe the excited states. Output is printed for the 1-electron transition density matrix (1TDM) and for the 1-electron difference density matrix (1DDM). A decomposition into local and charge transfer contributions on different chromophores is possible through the charge transfer number analysis [[231](<../../references.html#id314> "F. Plasser, H. Lischka. J. Chem. Theory Comput., 8\[8\] \(2012\) 2777-2789.")], which has been integrated into Molcas recently. Postprocessing is possible through the external [TheoDORE](<https://theodore-qc.sourceforge.net/>) [[232](<../../references.html#id315> "F. Plasser. J. Chem. Phys., 152\[8\] \(2020\) 084108.")] program.

WFA supports full use of spatial symmetry and can analyse transitions between different spin multiplicities and particle numbers.

## 4.2.58.1. Installation¶

The WFA module is currently not installed by default. Its installation occurs via CMake. It requires a working HDF5 installation (including C++ bindings) and access to the include files of the Armadillo C++ linear algebra library. In the current settings, external BLAS/LAPACK libraries have to be used. Use, e.g., the following commands for installation:
    
    
    FC=ifort cmake -D LINALG=MKL -D WFA=ON -D ARMADILLO_INC=../armadillo-7.300.0/include ..
    

To obtain the required libraries, you can use on Ubuntu:
    
    
    sudo apt install libhdf5-dev libhdf5-cpp-103
    

Alternatively, you can link against the dynamic HDF5 libraries distributed along with [Anaconda](<https://www.anaconda.com/>).

## 4.2.58.2. Dependencies¶

The WFA program requires HDF5 files, which are written by either SCF, RASSCF, or RASSI. In the case of RASSI, the TDM (or TRD1) keyword has to be activated.

## 4.2.58.3. Files¶

### 4.2.58.3.1. Input files¶

WFAH5
    

All information that the WFA program needs is contained in this HDF5 file. The name can be adjusted with the H5FIle option.

### 4.2.58.3.2. Output files¶

WFAH5
    

The orbital coefficients of NOs, NTOs, and NDOs are written to the same HDF5 file that is also used for input.

*.om
    

These are input files for the external TheoDORE program.

OmFrag.txt
    

Input file for TheoDORE.

For a seamless interface to TheoDORE, you can also create the tden_summ.txt file via
    
    
    grep '^|' molcas.log > tden_summ.txt
    

The NOs, NTOs, and NDOs on the HDF5 file can be accessed via [Pegamoid](<https://pypi.org/project/Pegamoid/>). Alternatively, the orbitals can be converted to Molden format via the [Molpy program](<https://github.com/felixplasser/molpy>). Call, e.g.:
    
    
    penny molcas.rassi.h5 --wfaorbs molden
    

## 4.2.58.4. Input¶

The input for the WFA module is preceded by:
    
    
    &WFA
    

### 4.2.58.4.1. Keywords¶

Basic Keywords:

H5FIle
    

Specifies the name of the HDF5 file used for reading and writing (e.g. $Project.scf.h5, $Project.rasscf.h5, $Project.rassi.h5). You either have to use this option or rename the file of interest to WFAH5.

WFALevel
    

Select how much output is produced (0-4, default: 3).

CTNUmmode
    

Specifies what properties are computed in a [TheoDORE](<https://theodore-qc.sourceforge.net/>)-style fragment-based analysis (0-3, default: 1). This requires defining fragments via ATLIsts.

0 — none

1 — Basic: POS, PR, DEL, CT, CTnt

2 — Extended: POS, POSi, POSf, PR, PRi, PRf, DEL, COH, CT, CTnt

3 — For transition metal complexes: POSi, POSf, PR, CT, MC, LC, MLCT, LMCT, LLCT

The definition of the descriptors is provided [here](<https://sourceforge.net/p/theodore-qc/wiki/Transition%20density%20matrix%20analysis/attachment/Om_desc.pdf>). For a more fine-grained input use PROPlist.

ATLIsts
    

Define the fragments in a [TheoDORE](<https://theodore-qc.sourceforge.net/>)-style analysis. _Note:_ If symmetry is turned on, then Molcas may reorder the atoms. In this case it is essential to take the order Molcas produced (seen for example in the Molden files).

The first entry is the number of fragments. Then enter the atomic indices of the fragment followed by a *. Example:
    
    
    ATLISTS
    2
    1 2 4 *
    3 *
    

_Note:_ This input can be generated automatically via TheoDORE by suppling a file with coordinates coord.mol and running
    
    
    theodore theoinp -a coord.mol
    

REFState
    

Index of the reference state for 1TDM and 1DDM analysis (default: 1).

Advanced keywords for fine grain output options and debug information:

MULLiken
    

Activate Mulliken population analysis (also for CT numbers).

LOWDin
    

Activate Löwdin population analysis (also for CT numbers).

NXO
    

Activate NO, NTO, and NDO analysis.

EXCIton
    

Activate exciton and multipole analysis.

DOCTnumbers
    

Activate charge transfer number analysis and creation of *.om files.

H5ORbitals
    

Print the NOs, NTOs, and/or NDOs to the HDF file.

PROPlist
    

Manual input of properties to be printed out in a [TheoDORE](<https://theodore-qc.sourceforge.net/>)-style fragment based analysis. Use only if CTNUMMODE does not provide what you want.

Enter as a list followed by a *, e.g.
    
    
    PROPLIST
    Om POS PR CT COH CTnt *
    

The full list of descriptors is provided [here](<https://sourceforge.net/p/theodore-qc/wiki/Transition%20density%20matrix%20analysis/attachment/Om_desc.pdf>).

DEBUg
    

Print debug information.

ADDInfo
    

Add info for verification runs with pymolcas verify.

### 4.2.58.4.2. Input example¶
    
    
    * Analysis of SCF job
    &SCF
    
    &WFA
    H5file = $Project.scf.h5
    
    
    
    * Analysis of RASSCF job
    * Reduced output
    &RASSCF
    
    &WFA
    H5file = $Project.rasscf.h5
    wfalevel = 1
    
    
    
    * Analysis of RASSI job, use the TDM keyword
    &RASSI
    EJOB
    TDM
    
    &WFA
    H5file = $Project.rassi.h5
    ATLISTS
    2
    1 2 4 *
    3 *
    

### 4.2.58.4.3. Large jobs¶

The computational effort spent in RASSI and the size of the file $Project.rassi.h5 scale with the square of the number of states included in the computation. This can be a severe bottleneck. To reduce the time spent in RASSI use the HEFF or EJOB keywords; these will cause RASSI to read in the Hamiltonian rather than recomputing it. To reduce the output to the file $Project.rassi.h5 use SUBSets = 1. Note that this only works if the reference state is the first state treated by RASSI (and that is always possible if the states are reordered appropriately via NROF).
    
    
    &RASSI
    TDM
    EJOB
    SUBSets = 1
    
    &WFA
    H5file = $Project.rassi.h5
    REFState = 1
    

Subsequently you may reduce the file size by repacking the HDF5 file:
    
    
    h5repack -f GZIP=5 $Project.rassi.h5 $Project.rassi-repack.h5 && rm $Project.rassi.h5
    

Alternatively, you can avoid the quadratic scaling in RASSI by processing states in batches specified via the NROF keyword. As an extreme example, you can iterate over individual states using the following input (here the 10 states of JOB002 are analysed using the first state of JOB001 as reference):
    
    
    >> FOREACH IST in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    
    &RASSI
    TDM
    NROF
    2 1 1
    1
    $IST
    
    >> COPY $Project.rassi.h5 $Project.rassi.$IST.h5
    
    &WFA
    h5file=$Project.rassi.$IST.h5
    
    >> ENDDO
    

## 4.2.58.5. Output¶

### 4.2.58.5.1. State/difference density matrix analysis (SCF/RASSCF/RASSI)¶
    
    
    RASSCF analysis for state 2 (3) A
    

or
    
    
    RASSI analysis for state S1
    

Descriptor | Explanation  
---|---  
`n_u` | Number of unpaired electrons \\(n_u=\sum_i\min(n_i, 2-n_i)\\) [[227](<../../references.html#id309> "F. Plasser, M. Wormit, A. Dreuw. J. Chem. Phys., 141\[2\] \(2014\) 024106."), [233](<../../references.html#id318> "M. Head-Gordon. Chem. Phys. Lett., 372\[3-4\] \(2003\) 508-511.")]  
`n_u,nl` | Number of unpaired electrons \\(n_{u,nl}=\sum_i n_i^2(2-n_i)^2\\)  
`PR_NO` | NO participation ratio \\(\text{PR}_{\text{NO}}\\)  
`p_D` and `p_A` | Promotion number \\(p_D\\) and \\(p_A\\)  
`PR_D` and `PR_A` | D/A participation ratio \\(\text{PR}_D\\) and \\(\text{PR}_A\\)  
`Dipole moment [D]` | Dipole moment (and its Cartesian components)  
`RMS size of the density [Ang]` | Root-mean-square size of the overall electron density  
`<r_h> [Ang]` | Mean position of detachment density \\(\vec{d}_D\\) [[230](<../../references.html#id312> "F. Plasser, B. Thomitzni, S. A. Bäppler, J. Wenzel, D. R. Rehn, M. Wormit, A. Dreuw. J. Comput. Chem., 36\[21\] \(2015\) 1609-1620.")]  
`<r_e> [Ang]` | Mean position of attachment density \\(\vec{d}_A\\)  
`|<r_e - r_h>| [Ang]` | Linear D/A distance \\(\vec{d}_{D\rightarrow A} = \vec{d}_A - \vec{d}_D\\)  
`Hole size [Ang]` | RMS size of detachment density \\(\sigma_D\\)  
`Electron size [Ang]` | RMS size of attachment density \\(\sigma_A\\)  
  
### 4.2.58.5.2. Transition density matrix analysis (RASSI)¶
    
    
    RASSI analysis for transition from state 1 to 2 (S0-S1)
    

Output listing | Explanation  
---|---  
`Leading SVs` | Largest NTO occupation numbers  
`Sum of SVs (Omega)` | \\(\Omega\\), Sum of NTO occupation numbers  
`PR_NTO` | NTO participation ratio \\(\text{PR}_{\text{NTO}}\\) [[231](<../../references.html#id314> "F. Plasser, H. Lischka. J. Chem. Theory Comput., 8\[8\] \(2012\) 2777-2789.")]  
`Entanglement entropy (S_HE)` | \\(S_{H|E}=-\sum_i\lambda_i\log_2\lambda_i\\) [[234](<../../references.html#id316> "F. Plasser. J. Chem. Phys., 144\[19\] \(2016\) 194107.")]  
`Nr of entangled states (Z_HE)` | \\(Z_{HE}=2^{S_{H|E}}\\)  
`Renormalized S_HE/Z_HE` | Replace \\(\lambda_i\rightarrow \lambda_i/\Omega\\)  
`omega` | Norm of the 1TDM \\(\Omega\\), single-exc. character  
`QTa` / `QT2` | Sum over absolute (\\(Q^t_a\\)) or squared (\\(Q^t_2\\)) transition charges as measure for ionic character [[235](<../../references.html#id322> "S. A. do Monte, R. F. K. Spada, R. L. R. Alves, L. Belcher, R. Shepard, H. Lischka, F. Plasser. J. Phys. Chem. A, 127\[46\] \(2023\) 9842-9852.")]  
`LOC` / `LOCa` | Local contributions: Trace of the \\(\Omega\\) matrix with respect to basis functions (LOC) or squareroots of the values (LOCa)  
`<Phe>` | Expec. value of the particle-hole permutation operator, measuring de-excitations [[236](<../../references.html#id317> "P. Kimber, F. Plasser. Phys. Chem. Chem. Phys., 22\[11\] \(2020\) 6058-6080.")]  
`Trans. dipole moment [D]` | Transition dipole moment (and its Cartesian components)  
`Transition <r^2> [a.u.]` | Transition matrix element of \\(x^2+y^2+z^2\\) (and its Cartesian components)  
`<r_h> [Ang]` | Mean position of hole \\(\langle\vec{x}_h\rangle_{\text{exc}}\\) [[230](<../../references.html#id312> "F. Plasser, B. Thomitzni, S. A. Bäppler, J. Wenzel, D. R. Rehn, M. Wormit, A. Dreuw. J. Comput. Chem., 36\[21\] \(2015\) 1609-1620.")]  
`<r_e> [Ang]` | Mean position of electron \\(\langle\vec{x}_e\rangle_{\text{exc}}\\)  
`|<r_e - r_h>| [Ang]` | Linear e/h distance \\(\vec{d}_{h\rightarrow e} = \langle\vec{x}_e - \vec{x}_h\rangle_{\text{exc}}\\)  
`Hole size [Ang]` | RMS hole size: \\(\sigma_h = (\langle\vec{x}_h^2\rangle_{\text{exc}} - \langle\vec{x}_h\rangle_{\text{exc}}^2)^{1/2}\\)  
`Electron size [Ang]` | RMS electron size: \\(\sigma_e = (\langle\vec{x}_e^2\rangle_{\text{exc}} - \langle\vec{x}_e\rangle_{\text{exc}}^2)^{1/2}\\)  
`RMS electron-hole separation [Ang]` | \\(d_{\text{exc}} = (\langle \left|\vec{x}_e - \vec{x}_h\right|^2\rangle_{\text{exc}})^{1/2}\\) [[229](<../../references.html#id313> "S. A. Bäppler, F. Plasser, M. Wormit, A. Dreuw. Phys. Rev. A, 90\[5\] \(2014\) 052521.")]  
`Covariance(r_h, r_e) [Ang^2]` | \\(\text{COV}\left(\vec{x}_h,\vec{x}_e\right) = \langle\vec{x}_h\cdot\vec{x}_e\rangle_{\text{exc}} - \langle\vec{x}_h\rangle_{\text{exc}}\cdot\langle\vec{x}_e\rangle_{\text{exc}}\\)  
`Correlation coefficient` | \\(R_{eh} = \text{COV}\left(\vec{x}_h,\vec{x}_e\right)/\sigma_h\cdot\sigma_e\\) [[230](<../../references.html#id312> "F. Plasser, B. Thomitzni, S. A. Bäppler, J. Wenzel, D. R. Rehn, M. Wormit, A. Dreuw. J. Comput. Chem., 36\[21\] \(2015\) 1609-1620.")]  
`Center-of-mass size` | \\((\langle \left|\vec{x}_e + \vec{x}_h\right|^2\rangle_{\text{exc}}-\langle \vec{x}_e + \vec{x}_h\rangle_{\text{exc}}^2)^{1/2}\\)  
  
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
      * [4.2.52. SEWARD](<seward.html>)
      * [4.2.53. SINGLE_ANISO](<single_aniso.html>)
      * [4.2.54. SLAPAF](<slapaf.html>)
      * [4.2.55. SURFACEHOP](<surfacehop.html>)
      * [4.2.56. SYMMETRIZE](<symmetrize.html>)
      * [4.2.57. VIBROT](<vibrot.html>)
      * 4.2.58. WFA
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<vibrot.html> "4.2.57. VIBROT") | [next](<../basis_library.html> "4.2.59. The Basis Set Libraries") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/wfa.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
