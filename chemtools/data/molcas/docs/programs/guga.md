<!-- Source: https://molcas.gitlab.io/OpenMolcas/sphinx/users.guide/programs/guga.html -->

[ ](<../../index.html>)

[Molcas Manual (version v26.02-275-g15bc02d80)](<../../index.html>)

4.2.25. GUGA

[previous](<guessorb.html> "4.2.24. GUESSORB") | [next](<gugaci.html> "4.2.26. GUGACI") | [index](<../../genindex.html> "General Index")

# 4.2.25. GUGA¶

  * Dependencies

  * Files

    * Input files

    * Output files

  * Input

    * Keywords

    * Input example




The GUGA program generates coupling coefficients used in the MRCI and the CPF programs in Direct CI calculations [[71](<../../references.html#id21> "B. Roos. Chem. Phys. Lett., 15 \(1972\) 153-159.")]. These coupling coefficients are evaluated by the Graphical Unitary Group Approach [[72](<../../references.html#id59> "I. Shavitt. Int. J. Quantum Chem., 12-S11 \(1977\) 131-148."), [73](<../../references.html#id60> "I. Shavitt. Int. J. Quantum Chem., 14-S12 \(1978\) 5-32."), [74](<../../references.html#id98> "P. E. M. Siegbahn. J. Chem. Phys., 72 \(1980\) 1647-1656.")], for wavefunctions with at most two electrons excited from a set of reference configurations. The program was written by P. E. M. Siegbahn, Institute of Physics, Stockholm University, Sweden. Only the MRCI program can use several reference configurations. The reference configurations can be specified as a list, where the occupation numbers are given for each active orbital (see below) in each reference configuration, or as a Full CI the space defined by the active orbitals. In the GUGA, MRCI and CPF programs, the orbitals are classified as follows: Frozen, Inactive, Active, Secondary, and Deleted orbitals. Within each symmetry type, they follow this order. For the GUGA program, only the inactive and active orbitals are relevant.

  * **Inactive:** Inactive orbitals are doubly occupied in all reference configurations, but excitations out of this orbital space are allowed in the final CI wavefunction, i.e., they are correlated but have two electrons in all _reference_ configurations. Since only single and double excitations are allowed, there can be no more than two holes in the active orbitals. Using keyword NoCorr (See input description) a subset of the inactive orbitals can be selected, and at most a single hole is then allowed in the selected set. This allows the core-polarization part of core-valence correlation, while preventing large but usually inaccurate double-excitation core correlation.

  * **Active:** Active orbitals are those which may have different occupation in different reference configurations. Using keyword OneOcc (See input description) a restriction may be imposed on some selection of active orbitals, so that the selected orbitals are always singly occupied. This may be useful for transition metal compounds or for deep inner holes.




## 4.2.25.1. Dependencies¶

The GUGA program does not depend on any other program for its execution.

## 4.2.25.2. Files¶

### 4.2.25.2.1. Input files¶

The GUGA program does not need any input files apart from the file of input keywords.

### 4.2.25.2.2. Output files¶

CIGUGA
    

This file contains the coupling coefficients that are needed in subsequent CI calculations. For information about how these coefficients are structured you are referred to the source code [[74](<../../references.html#id98> "P. E. M. Siegbahn. J. Chem. Phys., 72 \(1980\) 1647-1656.")]. The theoretical background for the coefficient can be found in Refs [[72](<../../references.html#id59> "I. Shavitt. Int. J. Quantum Chem., 12-S11 \(1977\) 131-148."), [73](<../../references.html#id60> "I. Shavitt. Int. J. Quantum Chem., 14-S12 \(1978\) 5-32."), [74](<../../references.html#id98> "P. E. M. Siegbahn. J. Chem. Phys., 72 \(1980\) 1647-1656.")] and references therein.

## 4.2.25.3. Input¶

This section describes the input to the GUGA program in the Molcas program system, with the program name:
    
    
    &GUGA
    

### 4.2.25.3.1. Keywords¶

Formally, there are no compulsory keyword. Obviously, some input must be given for a meaningful calculation.

TITLe
    

The line following this keyword is treated as title line

SPIN
    

The spin degeneracy number, i.e. 2S+1. The value is read from the line following the keyword, in free format. The default value is 1, meaning a singlet wave function.

ELECtrons
    

The number of electrons to be correlated in the CI of CPF calculation. The value is read from the line following the keyword, in free format. Note that this number should include the nr of electrons in inactive orbitals. An alternative input specification is NACTEL. Default: Twice nr of inactive orbitals.

NACTel
    

The number of electrons in active orbitals in the reference configurations. The value is read from the line following the keyword, in free format. Note that this number includes only the of electrons in active orbitals. An alternative input specification is ELECTRONS. Default: Zero.

INACtive
    

The number of inactive orbitals, i.e. orbitals that have occupation numbers of 2 in all reference configurations. Specified for each of the symmetries. The values are read from the line following the keyword, in free format.

ACTIve
    

The number of active orbitals, i.e. orbitals that have varying occupation numbers in the reference configurations. Specified for each of the symmetries. The values are read from the line following the keyword, in free format.

At least one of the Inactive or Active keywords must be present for a meaningful calculation. If one of them is left out, the default is 0 in all symmetries.

ONEOcc
    

Specify a number of active orbitals per symmetry that are required to have occupation number one in all configurations. These orbitals are the first active orbitals. The input is read from the line after the keyword, in free format.

NOCOrr
    

Specify the number of inactive orbitals per symmetry out of which at most one electron (total) is excited. These orbitals are the first inactive orbitals. The input is read from the line after the keyword, in free format.

REFErence
    

Specify selected reference configurations. The additional input that is required usually spans more than one line. The first line after the keyword contains the number of reference configurations, and the total number of active orbitals, and these two numbers are read by free format. Thereafter the input has one line per reference configuration, specifying the occupation number for each of the active orbitals, read by 80I1 format. Note that Reference and CIall are mutually exclusive.

CIALl
    

Use a Full CI within the subspace of the active orbitals as reference configurations. The symmetry of the wavefunction must be specified. The value is read from the line following the keyword, in free format. Note that CIall and Reference are mutually exclusive. One of these two alternatives must be chosen for a meaningful calculation.

FIRSt
    

Perform a first order calculation, i.e. only single excitations from the reference space. No additional input is required.

NONInteracting space
    

By default, those double excitations from inactive to virtual orbitals are excluded, where the inactive and virtual electrons would couple to a resulting triplet. With the NonInteracting Space option, such ‘non-interacting’ configurations are included as well.

PRINt
    

Printlevel of the program. Default printlevel (0) produces very little output. Printlevel 5 gives some information that may be of interest. The value is read from the line following the keyword, in free format.

### 4.2.25.3.2. Input example¶
    
    
    &GUGA
    Title
     Water molecule. 2OH correlated.
    Electrons =     4
    Spin      =     1
    Active    =     2    2    0    0
    Interacting space
    Reference
        3    4
      2020 ; 0220 ; 2002
    

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
      * 4.2.25. GUGA
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
      * [4.2.58. WFA](<wfa.html>)
      * [4.2.59. The Basis Set Libraries](<../basis_library.html>)
    * [4.3. GUI](<../tools.html>)
  * [5\. Advanced Examples and Annexes](<../../advanced.examples/ae.html>)



### Search

[previous](<guessorb.html> "4.2.24. GUESSORB") | [next](<gugaci.html> "4.2.26. GUGACI") | [index](<../../genindex.html> "General Index")

[Get PDF](<../../../Manual.pdf>) | [Show Source](<../../_sources/users.guide/programs/guga.rst.txt>)

(C) Copyright 2017–2025, MOLCAS Team. Created using [Sphinx](<https://www.sphinx-doc.org/>) 4.5.0. 
